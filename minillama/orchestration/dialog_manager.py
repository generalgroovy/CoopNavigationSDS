"""Main controller for a single dialog run. It coordinates turns, route interpretation, GUI events, and final dialog results.
"""
import time

from minillama.caller.responder import TemplateAgentAResponder
from minillama.assistant.pipeline import DialogState
from minillama.orchestration.dialog_result import DialogResult
from minillama.orchestration.route_memory import RouteProposalMemory
from minillama.analysis.metrics import TASK_TERMS, COMPARISON_TERMS, COOPERATION_TERMS
from minillama.analysis.route_interpreter import NaturalRouteInterpreter
from minillama.network.route_planner import (
    estimate_route_time,
    fmt_time,
    optimal_time_route,
    route_line_change_count,
    route_line_sequence,
    route_duration_text,
    route_is_valid,
    route_station_sequence,
)
from minillama.network.route_constraints import (
    OBJECTIVE_MODE_LABELS,
    CONSTRAINT_LABELS,
    acceptable_duration_limit,
    optimal_constraint_route,
    probability_class,
    route_allowed_modes,
    route_constraint_gap,
    route_constraint_status,
    route_has_near_capacity,
    route_near_capacity_count,
    stage_viability_report,
    route_transfer_miss_probability,
    normalize_objective_mode,
    stated_constraint_keys,
    unsatisfied_constraint_keys,
)
from minillama.speech.io import SpeechTransport

DEFAULT_MAX_TURN_ELAPSED_SEC = 2.0
HARD_MAX_TURN_ELAPSED_SEC = 20.0


def route_reaches_goal(stations, scenario):
    """Route reaches goal function for this module's MVC responsibility.
    
    Args:
        stations: Input value used by `route_reaches_goal`; see the function signature and caller context for the expected type.
        scenario: Input value used by `route_reaches_goal`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return (
        bool(stations)
        and stations[0] == scenario["start_station"]
        and stations[-1] == scenario["destination_station"]
    )


def route_duration_min(stations, scenario, allowed_modes=None):
    """Return total route duration in minutes when the route can be scheduled."""
    estimate = estimate_route_time(
        stations,
        scenario["start_time_min"],
        scenario["transfer_time_min"],
        allowed_modes=allowed_modes,
    )
    if not estimate:
        return None
    arrival, _ = estimate
    return arrival - scenario["start_time_min"]


def agent_b_model_name(agent_b_plugin):
    """Return the model/plugin name used for Agent B metrics."""
    if hasattr(agent_b_plugin, "model_adapter"):
        return getattr(agent_b_plugin.model_adapter, "name", "unknown-model")
    if hasattr(agent_b_plugin, "pipeline"):
        return getattr(agent_b_plugin.pipeline.model_adapter, "name", "unknown-model")
    return getattr(agent_b_plugin, "name", type(agent_b_plugin).__name__)


def constraint_gap_missed(gap, transfer_tolerance=1):
    """Return whether a valid proposal is worse than the stated constraint baseline."""
    if not gap:
        return False
    return (
        gap.get("duration_gap_min", 0) > 0
        or gap.get("line_change_gap", 0) > int(transfer_tolerance)
        or gap.get("near_capacity_gap", gap.get("fullness_gap", 0)) > 0
        or gap.get("risk_unviable", False)
    )


def agent_a_ended_conversation(text):
    """Return whether Agent A has naturally closed the call."""
    lower = (text or "").lower()
    return "thanks" in lower and any(term in lower for term in ("i'll take", "i will take", "choose", "that works"))


class DialogManager:
    """Controller for one two-agent dialog. It advances turns, interprets routes, and emits UI/metric events.
    """
    def __init__(
        self,
        test_case,
        agent_b_plugin,
        num_turns,
        route_interpreter=None,
        speech_transport=None,
        agent_a_responder=None,
        monitor=None,
        invalid_route_limit=2,
        constraint_miss_limit=2,
        transfer_tolerance=1,
        metric_snapshot_interval=1,
        max_turn_elapsed_sec=DEFAULT_MAX_TURN_ELAPSED_SEC,
        metric_config=None,
        agent_a_objective_mode=None,
    ):
        """  init   method for this module's MVC responsibility.
        
        Args:
            test_case: Input value used by `__init__`; see the function signature and caller context for the expected type.
            agent_b_plugin: Input value used by `__init__`; see the function signature and caller context for the expected type.
            num_turns: Input value used by `__init__`; see the function signature and caller context for the expected type.
            route_interpreter: Input value used by `__init__`; see the function signature and caller context for the expected type.
            speech_transport: Input value used by `__init__`; see the function signature and caller context for the expected type.
            agent_a_responder: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.test_case = test_case
        self.agent_b_plugin = agent_b_plugin
        self.num_turns = num_turns
        self.route_interpreter = route_interpreter or NaturalRouteInterpreter()
        self.speech_transport = speech_transport or SpeechTransport()
        self.agent_a_responder = agent_a_responder or TemplateAgentAResponder()
        self.monitor = monitor
        self.invalid_route_limit = invalid_route_limit
        self.constraint_miss_limit = constraint_miss_limit
        self.transfer_tolerance = max(0, int(transfer_tolerance))
        self.metric_snapshot_interval = max(1, int(metric_snapshot_interval or 1))
        self.max_turn_elapsed_sec = min(
            HARD_MAX_TURN_ELAPSED_SEC,
            max(1.0, float(max_turn_elapsed_sec or DEFAULT_MAX_TURN_ELAPSED_SEC)),
        )
        self.metric_config = dict(metric_config or {})
        self.agent_a_objective_mode = normalize_objective_mode(agent_a_objective_mode)

    def run(self, event_queue):
        """Run method for this module's MVC responsibility.
        
        Args:
            event_queue: Input value used by `run`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        speech_turns = []
        timing_turns = []
        nlu_turns = []
        runtime_events = []
        candidate_events = []
        warning_count = 0
        conversation_word_count = 0
        task_term_count = 0
        comparison_term_count = 0
        cooperation_term_count = 0

        def ingest_conversation_text(text):
            """Update running conversation statistics for the latest utterance."""
            nonlocal conversation_word_count, task_term_count, comparison_term_count, cooperation_term_count
            conversation_word_count += len(text.split())
            words = text.lower().split()
            task_term_count += sum(1 for word in words if word in TASK_TERMS)
            comparison_term_count += sum(1 for word in words if word in COMPARISON_TERMS)
            cooperation_term_count += sum(1 for word in words if word in COOPERATION_TERMS)

        def elapsed_since_ready():
            """Return elapsed conversation time after both agents are ready."""
            return time.time() - start_wall if start_wall is not None else 0.0

        def cap_turn_timing(generation_sec, speech_sec):
            """Cap effective turn timing while preserving raw measurements for diagnostics."""
            raw_generation_sec = max(0.0, float(generation_sec or 0.0))
            raw_speech_sec = max(0.0, float(speech_sec or 0.0))
            raw_turn_elapsed_sec = raw_generation_sec + raw_speech_sec
            effective_generation_sec = min(raw_generation_sec, self.max_turn_elapsed_sec)
            remaining_sec = max(self.max_turn_elapsed_sec - effective_generation_sec, 0.0)
            effective_speech_sec = min(raw_speech_sec, remaining_sec)
            effective_turn_elapsed_sec = effective_generation_sec + effective_speech_sec
            return {
                "generation_sec": effective_generation_sec,
                "speech_sec": effective_speech_sec,
                "turn_elapsed_sec": effective_turn_elapsed_sec,
                "raw_generation_sec": raw_generation_sec,
                "raw_speech_sec": raw_speech_sec,
                "raw_turn_elapsed_sec": raw_turn_elapsed_sec,
                "turn_capped": raw_turn_elapsed_sec > self.max_turn_elapsed_sec,
                "max_turn_elapsed_sec": self.max_turn_elapsed_sec,
            }

        def log_conversation_step(speaker, utterance, turn_number, generation_sec, speech_sec, parsed_route=None, has_station_mentions=False):
            """Record one conversation step with the current evaluation state."""
            if not self.monitor:
                return

            timing = cap_turn_timing(generation_sec, speech_sec)
            parsed_route_valid = route_is_valid(parsed_route, allowed_modes=planning_allowed_modes) if parsed_route else False
            parsed_route_goal = route_reaches_goal(parsed_route, scenario) if parsed_route_valid else False
            payload = {
                "turn": turn_number,
                "speaker": speaker,
                "utterance": utterance,
                "generation_sec": round(timing["generation_sec"], 6),
                "speech_sec": round(timing["speech_sec"], 6),
                "turn_latency_sec": round(timing["turn_elapsed_sec"], 6),
                "turn_elapsed_sec": round(timing["turn_elapsed_sec"], 6),
                "raw_generation_sec": round(timing["raw_generation_sec"], 6),
                "raw_speech_sec": round(timing["raw_speech_sec"], 6),
                "raw_turn_elapsed_sec": round(timing["raw_turn_elapsed_sec"], 6),
                "turn_capped": timing["turn_capped"],
                "max_turn_elapsed_sec": round(timing["max_turn_elapsed_sec"], 6),
                "message_count": len(conversation),
                "word_count": conversation_word_count,
                "task_terms": task_term_count,
                "comparison_terms": comparison_term_count,
                "cooperation_terms": cooperation_term_count,
                "route": list(parsed_route or []),
                "route_valid": parsed_route_valid,
                "route_reaches_goal": parsed_route_goal,
                "has_station_mentions": has_station_mentions,
                "candidate_routes": len(route_memory.candidates),
                "route_revisions": route_revision_count,
                "best_duration": best_duration,
                "warnings": warning_count,
                "route_status": "Correct" if parsed_route_goal else "Partial" if parsed_route_valid else "Invalid",
            }
            self.monitor.log_step(
                turn=turn_number,
                speaker=speaker,
                utterance=utterance,
                metrics=payload,
            )

        def record_runtime_event(phase, event_type, payload=None):
            """Capture raw execution data; metrics are computed only after the dialog."""
            runtime_events.append(
                {
                    "index": len(runtime_events) + 1,
                    "elapsed_sec": round(elapsed_since_ready(), 6),
                    "phase": phase,
                    "event_type": event_type,
                    "payload": dict(payload or {}),
                }
            )

        def emit_speech_telemetry(trace, latency_sec):
            """Send source/transcript telemetry for automatic speech recognition metrics."""
            payload = {
                "speaker": trace.speaker,
                "generated_text": trace.generated_text,
                "outgoing_text": trace.outgoing_text,
                "incoming_transcript": trace.incoming_transcript,
                "source_text": trace.outgoing_text,
                "transcript": trace.incoming_transcript,
                "latency_sec": latency_sec,
                "simulated_duration_sec": trace.simulated_duration_sec,
                "audio": trace.audio,
                "outgoing_enabled": trace.outgoing_enabled,
                "incoming_enabled": trace.incoming_enabled,
                "tts_engine": trace.tts_engine,
                "asr_engine": trace.asr_engine,
                "pattern_key": trace.pattern_key,
                "mode": trace.mode,
                "pipeline_ok": trace.pipeline_ok,
                "failure_reason": trace.failure_reason,
                "diagnostics": trace.diagnostics or {},
            }
            speech_turns.append(payload)
            event_queue.put(
                (
                    "telemetry",
                    "speech",
                    payload,
                )
            )

        def emit_timing_telemetry(speaker, generation_sec, speech_sec, turn_number=None):
            """Send turn timing telemetry for latency metrics."""
            timing = cap_turn_timing(generation_sec, speech_sec)
            payload = {
                "turn": turn_number,
                "speaker": speaker,
                "generation_sec": timing["generation_sec"],
                "speech_sec": timing["speech_sec"],
                "turn_latency_sec": timing["turn_elapsed_sec"],
                "turn_elapsed_sec": timing["turn_elapsed_sec"],
                "raw_generation_sec": timing["raw_generation_sec"],
                "raw_speech_sec": timing["raw_speech_sec"],
                "raw_turn_elapsed_sec": timing["raw_turn_elapsed_sec"],
                "turn_capped": timing["turn_capped"],
                "max_turn_elapsed_sec": timing["max_turn_elapsed_sec"],
            }
            timing_turns.append(payload)
            event_queue.put(
                (
                    "telemetry",
                    "timing",
                    payload,
                )
            )

        def emit_nlu_telemetry(speaker, text, parsed_route, has_station_mentions):
            """Send semantic parsing telemetry for language understanding and dialog-state metrics."""
            valid = route_is_valid(parsed_route, allowed_modes=planning_allowed_modes)
            payload = {
                "speaker": speaker,
                "text": text,
                "has_station_mentions": has_station_mentions,
                "parsed_route": parsed_route,
                "route_valid": valid,
                "route_reaches_goal": route_reaches_goal(parsed_route, scenario) if valid else False,
            }
            nlu_turns.append(payload)
            event_queue.put(
                (
                    "telemetry",
                    "nlu",
                    payload,
                )
            )

        def build_agent_turn_segments():
            """Build per-agent timing segments for research analysis."""
            segments = []
            for index, (speaker, utterance) in enumerate(conversation, start=1):
                timing = next((turn for turn in timing_turns if turn.get("turn") == index and turn.get("speaker") == speaker), {})
                speech = speech_turns[index - 1] if index - 1 < len(speech_turns) else {}
                nlu = next((turn for turn in nlu_turns if turn.get("speaker") == speaker and turn.get("text") == utterance), {})
                audio = speech.get("audio") if isinstance(speech.get("audio"), dict) else {}
                segments.append({
                    "turn": index,
                    "speaker": speaker,
                    "utterance": utterance,
                    "word_count": len(utterance.split()),
                    "character_count": len(utterance),
                    "generation_sec": round(float(timing.get("generation_sec", 0.0) or 0.0), 6),
                    "speech_sec": round(float(timing.get("speech_sec", 0.0) or 0.0), 6),
                    "turn_latency_sec": round(float(timing.get("turn_latency_sec", 0.0) or 0.0), 6),
                    "turn_elapsed_sec": round(float(timing.get("turn_elapsed_sec", timing.get("turn_latency_sec", 0.0)) or 0.0), 6),
                    "raw_generation_sec": round(float(timing.get("raw_generation_sec", timing.get("generation_sec", 0.0)) or 0.0), 6),
                    "raw_speech_sec": round(float(timing.get("raw_speech_sec", timing.get("speech_sec", 0.0)) or 0.0), 6),
                    "raw_turn_elapsed_sec": round(float(timing.get("raw_turn_elapsed_sec", timing.get("turn_elapsed_sec", timing.get("turn_latency_sec", 0.0))) or 0.0), 6),
                    "turn_capped": bool(timing.get("turn_capped", False)),
                    "max_turn_elapsed_sec": round(float(timing.get("max_turn_elapsed_sec", self.max_turn_elapsed_sec) or self.max_turn_elapsed_sec), 6),
                    "pipeline_mode": speech.get("mode"),
                    "tts_engine": speech.get("tts_engine"),
                    "asr_engine": speech.get("asr_engine"),
                    "audio_duration_sec": audio.get("duration_sec"),
                    "audio_played": audio.get("played"),
                    "asr_repair_used": (speech.get("diagnostics") or {}).get("asr_repair_used"),
                    "route_valid": nlu.get("route_valid"),
                    "route_reaches_goal": nlu.get("route_reaches_goal"),
                    "parsed_route": nlu.get("parsed_route"),
                })
            return segments

        def summarize_agent_turn_segments(segments):
            """Aggregate per-agent timing segment metrics."""
            summaries = {}
            for speaker in ("Agent A", "Agent B"):
                speaker_segments = [segment for segment in segments if segment["speaker"] == speaker]
                turn_count = len(speaker_segments)
                elapsed_values = [segment["turn_elapsed_sec"] for segment in speaker_segments]
                generation_values = [segment["generation_sec"] for segment in speaker_segments]
                speech_values = [segment["speech_sec"] for segment in speaker_segments]
                word_count = sum(segment["word_count"] for segment in speaker_segments)
                summaries[speaker] = {
                    "speaker": speaker,
                    "turn_count": turn_count,
                    "word_count": word_count,
                    "mean_words_per_turn": round(word_count / turn_count, 6) if turn_count else 0.0,
                    "total_generation_sec": round(sum(generation_values), 6),
                    "total_speech_sec": round(sum(speech_values), 6),
                    "total_turn_elapsed_sec": round(sum(elapsed_values), 6),
                    "mean_turn_elapsed_sec": round(sum(elapsed_values) / turn_count, 6) if turn_count else 0.0,
                    "max_turn_elapsed_sec": round(max(elapsed_values), 6) if elapsed_values else 0.0,
                    "min_turn_elapsed_sec": round(min(elapsed_values), 6) if elapsed_values else 0.0,
                    "turn_over_budget_count": sum(1 for segment in speaker_segments if segment.get("turn_capped")),
                    "max_turn_budget_sec": self.max_turn_elapsed_sec,
                }
            return summaries

        scenario = dict(self.test_case.scenario)
        scenario["agent_a_objective_mode"] = self.agent_a_objective_mode
        persona = self.test_case.persona
        planning_allowed_modes = route_allowed_modes(scenario, persona)
        constraint_route = optimal_constraint_route(scenario, persona, objective_mode=self.agent_a_objective_mode)
        acceptable_time_limit = acceptable_duration_limit(scenario, persona, constraint_route=constraint_route)
        stage_options = stage_viability_report(
            scenario,
            persona,
            transfer_tolerance=self.transfer_tolerance,
        )
        reference_arrival, reference_steps = optimal_time_route(
            scenario["start_station"],
            scenario["destination_station"],
            scenario["start_time_min"],
            scenario["transfer_time_min"],
            allowed_modes=planning_allowed_modes,
        )
        preflight_viability = {
            "reference_route_available": reference_arrival is not None and bool(reference_steps),
            "constraint_route_available": constraint_route is not None,
            "start_station": scenario["start_station"],
            "destination_station": scenario["destination_station"],
            "allowed_modes": planning_allowed_modes,
            "reference_route": route_station_sequence(reference_steps),
            "constraint_route": constraint_route.route if constraint_route else [],
            "constraint_duration_min": constraint_route.duration_min if constraint_route else None,
            "constraint_label": constraint_route.label if constraint_route else None,
            "acceptable_duration_limit_min": acceptable_time_limit,
            "stage_option_viability": stage_options,
        }
        start_wall = None
        route_memory = RouteProposalMemory()
        best_route = []
        best_duration = None
        best_turn = None
        best_rank = (-1, float("-inf"))
        route_revision_count = 0
        invalid_route_count = 0
        constraint_miss_count = 0
        time_frame_miss_count = 0
        early_stop_reason = None

        def early_stop_message(reason):
            if reason == "invalid_route_limit":
                return "I will stop here; the route suggestions still are not connected from start to destination."
            if reason == "time_frame_miss_limit":
                return "I will stop here unsatisfied; the routes reached the destination but stayed too slow."
            if reason == "constraint_miss_limit":
                return "I will stop here semi-satisfied; I have a route, but it still misses one of my constraints."
            if reason == "turn_limit":
                return "I will stop here unsatisfied; we ran out of turns before settling the route."
            return "I will stop here."

        def prior_route_satisfies_current_goals(stated_keys):
            """Return whether an earlier candidate already satisfies validity, time, and stated constraints."""
            for candidate in route_memory.candidates:
                route = candidate.get("route")
                duration = candidate.get("duration")
                if not route or not route_reaches_goal(route, scenario):
                    continue
                if acceptable_time_limit is not None and (duration is None or duration > acceptable_time_limit):
                    continue
                estimate = estimate_route_time(
                    route,
                    scenario["start_time_min"],
                    scenario["transfer_time_min"],
                    allowed_modes=planning_allowed_modes,
                )
                if not estimate:
                    continue
                _arrival, prior_steps = estimate
                statuses = route_constraint_status(
                    prior_steps,
                    persona,
                    scenario,
                    stated_keys,
                    transfer_tolerance=self.transfer_tolerance,
                    constraint_route=constraint_route,
                )
                if not unsatisfied_constraint_keys(statuses):
                    return True
            return False

        start_wall = time.time()
        event_queue.put(("timer_start", start_wall))
        speech_started_at = time.time()
        opening_trace = self.speech_transport.transmit_trace(
            "Agent A",
            self.test_case.opening_utterance(),
        )
        opening_transcript = opening_trace.incoming_transcript
        opening_speech_sec = max(time.time() - speech_started_at, opening_trace.simulated_duration_sec)
        emit_speech_telemetry(opening_trace, opening_speech_sec)
        conversation = [("Agent A", opening_transcript)]
        ingest_conversation_text(opening_transcript)

        event_queue.put(("system", f"Test case: {self.test_case.name}"))
        event_queue.put(("system", f"Persona: {persona['name']}"))
        event_queue.put(("system", f"Scenario: {scenario['name']}"))
        event_queue.put(("system", f"Objective mode: {OBJECTIVE_MODE_LABELS[self.agent_a_objective_mode]}"))
        event_queue.put(("system", f"Speech transport: {self.speech_transport.description}"))
        record_runtime_event("preflight", "viability_check", preflight_viability)
        if not stage_options["all_stage_requirements_satisfied"]:
            event_queue.put((
                "warning",
                "Current scenario lacks the configured number of viable suboptimal alternatives for every conversation stage.",
            ))
        if constraint_route:
            event_queue.put(
                (
                    "system",
                    (
                        "Constraint baseline: "
                        f"{' to '.join(constraint_route.route)} "
                        f"({constraint_route.duration_min} minutes, "
                        f"{constraint_route.line_change_count} changes, "
                        f"{'near capacity' if constraint_route.has_near_capacity else 'not near capacity'}, "
                        f"{probability_class(constraint_route.delay_probability)} delay risk, "
                        f"{constraint_route.label})"
                    ),
                )
            )
            event_queue.put((
                "system",
                (
                    "Agent A transfer tolerance: "
                    f"up to {constraint_route.line_change_count + self.transfer_tolerance} changes "
                    f"({self.transfer_tolerance} over the constraint baseline)."
                ),
            ))
        event_queue.put(("message", conversation[0][0], conversation[0][1]))
        log_conversation_step(
            "Agent A",
            conversation[0][1],
            len(conversation),
            0.0,
            opening_speech_sec,
        )
        emit_timing_telemetry("Agent A", 0.0, opening_speech_sec, turn_number=len(conversation))
        record_runtime_event("opening", "agent_a_opening", {"turn": len(conversation), "text": conversation[0][1]})

        route_round = 0
        while len(conversation) < self.num_turns:
            route_round += 1
            state = DialogState(
                self.test_case,
                conversation,
                route_round - 1,
                scenario_override=scenario,
                persona_override=persona,
            )
            generation_started_at = time.time()
            reply_b = self.agent_b_plugin.run_agent_b(state)
            generation_sec = time.time() - generation_started_at
            speech_started_at = time.time()
            reply_trace = self.speech_transport.transmit_trace("Agent B", reply_b)
            reply_transcript = reply_trace.incoming_transcript
            speech_sec = max(time.time() - speech_started_at, reply_trace.simulated_duration_sec)
            emit_speech_telemetry(reply_trace, speech_sec)
            emit_timing_telemetry("Agent B", generation_sec, speech_sec, turn_number=len(conversation) + 1)
            conversation.append(("Agent B", reply_transcript))
            ingest_conversation_text(reply_transcript)
            event_queue.put(("message", "Agent B", reply_transcript))

            parsed_route = self.route_interpreter.interpret_reply(reply_transcript, scenario)
            has_station_mentions = self.route_interpreter.has_station_mentions(reply_transcript)
            emit_nlu_telemetry("Agent B", reply_transcript, parsed_route, has_station_mentions)
            log_conversation_step(
                "Agent B",
                reply_transcript,
                len(conversation),
                generation_sec,
                speech_sec,
                parsed_route=parsed_route,
                has_station_mentions=has_station_mentions,
            )
            record_runtime_event(
                "agent_b_reply",
                "route_parse",
                {
                    "turn": len(conversation),
                    "parsed_route": parsed_route,
                    "has_station_mentions": has_station_mentions,
                },
            )

            if route_is_valid(parsed_route, allowed_modes=planning_allowed_modes):
                duration = route_duration_min(parsed_route, scenario, allowed_modes=planning_allowed_modes)
                if duration is not None:
                    _, candidate_steps = estimate_route_time(
                        parsed_route,
                        scenario["start_time_min"],
                        scenario["transfer_time_min"],
                        allowed_modes=planning_allowed_modes,
                    )
                    candidate_reaches_goal = route_reaches_goal(parsed_route, scenario)
                    active_constraint_keys = stated_constraint_keys(conversation)
                    active_constraint_status = route_constraint_status(
                        candidate_steps,
                        persona,
                        scenario,
                        active_constraint_keys,
                        transfer_tolerance=self.transfer_tolerance,
                        constraint_route=constraint_route,
                    )
                    active_constraint_misses = unsatisfied_constraint_keys(active_constraint_status)
                    candidate_time_frame_satisfied = (
                        acceptable_time_limit is None
                        or duration <= acceptable_time_limit
                    )
                    prior_goal_match = prior_route_satisfies_current_goals(active_constraint_keys)
                    if route_memory.already_seen(parsed_route):
                        warning_count += 1
                        gap = route_constraint_gap(candidate_steps, duration, constraint_route)
                        event_queue.put(("warning", "Repeated route proposal ignored; compare a different route."))
                        candidate_event = {
                            "turn": route_round,
                            "route": parsed_route,
                            "duration": duration,
                            "decision": "repeat",
                            "best_duration": best_duration,
                            "previous_best": best_duration,
                            "constraint_duration": constraint_route.duration_min if constraint_route else None,
                            "stated_constraints": active_constraint_keys,
                            "constraint_status": active_constraint_status,
                            "unsatisfied_constraints": active_constraint_misses,
                            "acceptable_duration_limit_min": acceptable_time_limit,
                            "time_frame_satisfied": candidate_time_frame_satisfied,
                            "prior_route_satisfies_current_goals": prior_goal_match,
                            **gap,
                        }
                        candidate_events.append(candidate_event)
                        event_queue.put(("candidate", candidate_event))
                    else:
                        is_new_route = parsed_route != best_route
                        candidate = route_memory.record(route_round, parsed_route, duration, best_duration)
                        gap = route_constraint_gap(candidate_steps, duration, constraint_route)
                        candidate.update(
                            {
                                "constraint_duration": constraint_route.duration_min if constraint_route else None,
                                "constraint_route": constraint_route.route if constraint_route else [],
                                "stated_constraints": active_constraint_keys,
                                "constraint_status": active_constraint_status,
                                "unsatisfied_constraints": active_constraint_misses,
                                "acceptable_duration_limit_min": acceptable_time_limit,
                                "time_frame_satisfied": candidate_time_frame_satisfied,
                                "prior_route_satisfies_current_goals": prior_goal_match,
                                **gap,
                            }
                        )
                        if (
                            candidate_reaches_goal
                            and not candidate_time_frame_satisfied
                            and not prior_goal_match
                        ):
                            time_frame_miss_count += 1
                            event_queue.put(("warning", f"Route reaches the destination but exceeds {acceptable_time_limit} minutes."))
                        if (
                            active_constraint_keys
                            and candidate_reaches_goal
                            and active_constraint_misses
                        ):
                            constraint_miss_count += 1
                            labels = ", ".join(CONSTRAINT_LABELS.get(key, key) for key in active_constraint_misses)
                            event_queue.put(("warning", f"Route is valid but misses stated constraints: {labels}."))
                        candidate_rank = (1 if candidate_reaches_goal else 0, -duration)
                        if best_route and is_new_route:
                            route_revision_count += 1

                        if candidate_rank > best_rank:
                            best_route = parsed_route
                            best_duration = duration
                            best_turn = route_round
                            best_rank = candidate_rank
                            event_queue.put(("route", best_route))

                        candidate["best_duration"] = best_duration
                        candidate_events.append(dict(candidate))
                        event_queue.put(("candidate", candidate))
            elif has_station_mentions:
                warning_count += 1
                invalid_route_count += 1
                event_queue.put(("warning", "Station names mentioned, but no connected spoken route was inferred."))
                record_runtime_event(
                    "agent_b_reply",
                    "invalid_route",
                    {"turn": len(conversation), "text": reply_transcript, "parsed_route": parsed_route},
                )

            if invalid_route_count >= self.invalid_route_limit:
                early_stop_reason = "invalid_route_limit"
            elif time_frame_miss_count >= self.invalid_route_limit:
                early_stop_reason = "time_frame_miss_limit"
            elif constraint_miss_count >= self.constraint_miss_limit:
                early_stop_reason = "constraint_miss_limit"

            if len(conversation) < self.num_turns:
                generation_started_at = time.time()
                reply_a = (
                    early_stop_message(early_stop_reason)
                    if early_stop_reason
                    else self.agent_a_responder.reply(route_round - 1, persona, scenario, conversation)
                )
                generation_sec = time.time() - generation_started_at
                speech_started_at = time.time()
                reply_trace = self.speech_transport.transmit_trace("Agent A", reply_a)
                reply_transcript = reply_trace.incoming_transcript
                speech_sec = max(time.time() - speech_started_at, reply_trace.simulated_duration_sec)
                emit_speech_telemetry(reply_trace, speech_sec)
                emit_timing_telemetry("Agent A", generation_sec, speech_sec, turn_number=len(conversation) + 1)
                conversation.append(("Agent A", reply_transcript))
                ingest_conversation_text(reply_transcript)
                event_queue.put(("message", "Agent A", reply_transcript))
                log_conversation_step(
                    "Agent A",
                    reply_transcript,
                    len(conversation),
                    generation_sec,
                    speech_sec,
                )
                record_runtime_event(
                    "agent_a_reply",
                    "constraint_state",
                    {
                        "turn": len(conversation),
                        "stated_constraints": stated_constraint_keys(conversation),
                        "early_stop_reason": early_stop_reason,
                    },
                )
                if agent_a_ended_conversation(reply_transcript):
                    early_stop_reason = "agent_a_closed"
            if early_stop_reason:
                break

        if early_stop_reason is None and len(conversation) >= self.num_turns and not agent_a_ended_conversation(conversation[-1][1]):
            early_stop_reason = "turn_limit"

        end_wall = time.time()
        runtime_sec = elapsed_since_ready()

        estimate = estimate_route_time(
            best_route,
            scenario["start_time_min"],
            scenario["transfer_time_min"],
            allowed_modes=planning_allowed_modes,
        ) if best_route else None

        if estimate:
            displayed_arrival, displayed_steps = estimate
            displayed_duration = displayed_arrival - scenario["start_time_min"]
        else:
            displayed_arrival = None
            displayed_duration = None
            displayed_steps = []

        route_valid = route_is_valid(best_route, allowed_modes=planning_allowed_modes)
        reaches_goal = route_reaches_goal(best_route, scenario)
        route_correct = route_valid and reaches_goal
        displayed_near_capacity_count = route_near_capacity_count(displayed_steps)
        displayed_has_near_capacity = route_has_near_capacity(displayed_steps)
        displayed_transfer_miss_probability = route_transfer_miss_probability(displayed_steps)
        reference_duration = (
            reference_arrival - scenario["start_time_min"]
            if reference_arrival is not None
            else None
        )
        reference_route = route_station_sequence(reference_steps)
        reference_line_sequence = route_line_sequence(reference_steps)
        reference_line_change_count = route_line_change_count(reference_steps)
        constraint_duration = constraint_route.duration_min if constraint_route else None
        constraint_line_sequence = constraint_route.line_sequence if constraint_route else []
        constraint_line_change_count = constraint_route.line_change_count if constraint_route else None
        constraint_near_capacity_count = constraint_route.near_capacity_count if constraint_route else None
        constraint_has_near_capacity = constraint_route.has_near_capacity if constraint_route else None
        constraint_delay_probability = constraint_route.delay_probability if constraint_route else None
        constraint_transfer_miss_probability = constraint_route.transfer_miss_probability if constraint_route else None
        constraint_gap = route_constraint_gap(displayed_steps, displayed_duration, constraint_route)
        displayed_delay_probability = max((step.get("delay_probability", 0.0) for step in displayed_steps), default=None)
        displayed_delay_risk_class = probability_class(displayed_delay_probability)
        displayed_transfer_risk_class = probability_class(displayed_transfer_miss_probability)
        constraint_delay_risk_class = probability_class(constraint_delay_probability)
        constraint_transfer_risk_class = probability_class(constraint_transfer_miss_probability)
        displayed_line_sequence = route_line_sequence(displayed_steps)
        displayed_line_change_count = route_line_change_count(displayed_steps)
        agent_turn_segments = build_agent_turn_segments()
        agent_timing_summary = summarize_agent_turn_segments(agent_turn_segments)
        final_stated_constraints = stated_constraint_keys(conversation)
        final_constraint_status = route_constraint_status(
            displayed_steps,
            persona,
            scenario,
            final_stated_constraints,
            transfer_tolerance=self.transfer_tolerance,
            constraint_route=constraint_route,
        )
        final_unsatisfied_constraints = unsatisfied_constraint_keys(final_constraint_status)
        time_frame_satisfied = (
            acceptable_time_limit is None
            or (displayed_duration is not None and displayed_duration <= acceptable_time_limit)
        )
        if early_stop_reason in {"invalid_route_limit", "time_frame_miss_limit", "turn_limit"} or not route_correct:
            conversation_outcome = "unsatisfied"
        elif not time_frame_satisfied or final_unsatisfied_constraints:
            conversation_outcome = "semi_satisfied"
        else:
            conversation_outcome = "satisfied"

        metrics = (
            f"Test case:             {self.test_case.name}\n"
            f"Persona:               {persona['name']}\n"
            f"Scenario:              {scenario['name']}\n"
            f"Messages:              {len(conversation)}\n"
            f"Start time:            {fmt_time(scenario['start_time_min'])}\n"
            f"Start:                 {scenario['start_station']}\n"
            f"Destination:           {scenario['destination_station']}\n"
            f"Displayed route:       {' to '.join(best_route) if best_route else 'None'}\n"
            f"Displayed line sequence: {' to '.join(displayed_line_sequence) if displayed_line_sequence else 'None'}\n"
            f"Displayed line changes: {displayed_line_change_count if displayed_line_sequence else 'None'}\n"
            f"Displayed arrival:     {fmt_time(displayed_arrival) if displayed_arrival else 'None'}\n"
            f"Displayed duration:    {str(displayed_duration) + ' minutes' if displayed_duration is not None else 'None'}\n"
            f"Acceptable duration:   {str(acceptable_time_limit) + ' minutes' if acceptable_time_limit is not None else 'None'}\n"
            f"Time frame satisfied:  {time_frame_satisfied}\n"
            f"Duration breakdown:    {route_duration_text(displayed_steps)}\n"
            f"Near capacity:         {'yes' if displayed_has_near_capacity else 'no'} ({displayed_near_capacity_count} segments)\n"
            f"Reference route:       {' to '.join(reference_route) if reference_route else 'None'}\n"
            f"Reference line sequence: {' to '.join(reference_line_sequence) if reference_line_sequence else 'None'}\n"
            f"Reference line changes: {reference_line_change_count if reference_line_sequence else 'None'}\n"
            f"Reference duration:    {str(reference_duration) + ' minutes' if reference_duration is not None else 'None'}\n"
            f"Constraint target:     {constraint_route.label if constraint_route else 'None'}\n"
            f"Constraint route:      {' to '.join(constraint_route.route) if constraint_route else 'None'}\n"
            f"Constraint line sequence: {' to '.join(constraint_line_sequence) if constraint_line_sequence else 'None'}\n"
            f"Constraint line changes: {constraint_line_change_count if constraint_line_change_count is not None else 'None'}\n"
            f"Constraint duration:   {str(constraint_duration) + ' minutes' if constraint_duration is not None else 'None'}\n"
            f"Constraint capacity:   {('yes' if constraint_has_near_capacity else 'no') if constraint_has_near_capacity is not None else 'None'} ({constraint_near_capacity_count if constraint_near_capacity_count is not None else 'None'} segments)\n"
            f"Transfer tolerance:    {self.transfer_tolerance} extra changes\n"
            f"Allowed modes:         {', '.join(planning_allowed_modes) if planning_allowed_modes else 'all modes'}\n"
            f"Delay risk:            {displayed_delay_risk_class}\n"
            f"Transfer miss risk:    {displayed_transfer_risk_class}\n"
            f"Constraint delay risk: {constraint_delay_risk_class}\n"
            f"Constraint transfer risk: {constraint_transfer_risk_class}\n"
            f"Constraint gap:        {constraint_gap.get('duration_gap_min', 'None')} minutes, {constraint_gap.get('line_change_gap', 'None')} changes, {constraint_gap.get('near_capacity_gap', 'None')} near-capacity segments, risk viable {not constraint_gap.get('risk_unviable', False)}\n"
            f"Stage option viability: {stage_options['all_stage_requirements_satisfied']} ({stage_options['required_suboptimal_option_count']} required suboptimal options)\n"
            f"Candidate routes:      {len(route_memory.candidates)}\n"
            f"Route revisions:       {route_revision_count}\n"
            f"Best candidate turn:   {best_turn if best_turn is not None else 'None'}\n"
            f"Invalid route count:   {invalid_route_count}\n"
            f"Time frame miss count: {time_frame_miss_count}\n"
            f"Constraint miss count: {constraint_miss_count}\n"
            f"Early stop reason:     {early_stop_reason or 'None'}\n"
            f"Conversation outcome:  {conversation_outcome}\n"
            f"Stated constraints:    {', '.join(final_stated_constraints) if final_stated_constraints else 'None'}\n"
            f"Unsatisfied constraints: {', '.join(final_unsatisfied_constraints) if final_unsatisfied_constraints else 'None'}\n"
            f"Route valid:           {route_valid}\n"
            f"Route reaches goal:    {reaches_goal}\n"
            f"Route correct:         {route_correct}\n"
            f"Mean turn elapsed:     {str(round(sum(turn.get('turn_elapsed_sec', turn.get('turn_latency_sec', 0.0)) for turn in timing_turns) / len(timing_turns), 4)) + ' seconds' if timing_turns else 'None'}\n"
            f"Maximum turn budget:   {self.max_turn_elapsed_sec:.1f} seconds\n"
            f"Agent A mean turn:     {agent_timing_summary['Agent A']['mean_turn_elapsed_sec']} seconds\n"
            f"Agent B mean turn:     {agent_timing_summary['Agent B']['mean_turn_elapsed_sec']} seconds\n"
            f"Runtime:               {runtime_sec:.2f} seconds\n"
            f"Speech pipeline:       {self.speech_transport.description}\n"
        )

        event_queue.put(("metrics", metrics))
        record_runtime_event("final", "retrospective_metrics_ready", {"turns": len(conversation)})
        event_queue.put(("done",))

        return DialogResult(
            condition_id=self.test_case.key,
            test_case_key=self.test_case.key,
            persona_key=self.test_case.persona_key,
            scenario_key=self.test_case.scenario_key,
            speech_pattern_key=getattr(
                self.speech_transport.asr_engine,
                "pattern_key",
                self.speech_transport.asr_engine.name,
            ),
            model_name=agent_b_model_name(self.agent_b_plugin),
            conversation=conversation,
            route=best_route,
            route_steps=displayed_steps,
            route_valid=route_valid,
            route_reaches_goal=reaches_goal,
            route_correct=route_correct,
            route_duration_min=displayed_duration,
            runtime_sec=runtime_sec,
            metrics_text=metrics,
            extra={
                "speech_transport": self.speech_transport.description,
                "agent_a_responder": self.agent_a_responder.name,
                "agent_b_plugin": getattr(self.agent_b_plugin, "name", type(self.agent_b_plugin).__name__),
                "agent_a_objective_mode": self.agent_a_objective_mode,
                "messages": len(conversation),
                "candidate_routes": len(route_memory.candidates),
                "route_revisions": route_revision_count,
                "best_candidate_turn": best_turn,
                "invalid_route_count": invalid_route_count,
                "time_frame_miss_count": time_frame_miss_count,
                "constraint_miss_count": constraint_miss_count,
                "early_stop_reason": early_stop_reason,
                "conversation_outcome": conversation_outcome,
                "stated_constraints": final_stated_constraints,
                "constraint_status": final_constraint_status,
                "unsatisfied_constraints": final_unsatisfied_constraints,
                "acceptable_duration_limit_min": acceptable_time_limit,
                "time_frame_satisfied": time_frame_satisfied,
                "preflight_viability": preflight_viability,
                "stage_option_viability": stage_options,
                "reference_duration_min": reference_duration,
                "displayed_line_sequence": displayed_line_sequence,
                "displayed_line_changes": displayed_line_change_count,
                "reference_line_sequence": reference_line_sequence,
                "reference_line_changes": reference_line_change_count,
                "constraint_target": constraint_route.label if constraint_route else None,
                "constraint_route": constraint_route.route if constraint_route else [],
                "constraint_duration_min": constraint_duration,
                "constraint_line_sequence": constraint_line_sequence,
                "constraint_line_changes": constraint_line_change_count,
                "constraint_near_capacity": constraint_has_near_capacity,
                "constraint_near_capacity_count": constraint_near_capacity_count,
                "transfer_tolerance": self.transfer_tolerance,
                "allowed_modes": planning_allowed_modes,
                "constraint_delay_probability": constraint_delay_probability,
                "constraint_transfer_miss_probability": constraint_transfer_miss_probability,
                "route_delay_risk_class": displayed_delay_risk_class,
                "route_transfer_miss_risk_class": displayed_transfer_risk_class,
                "constraint_delay_risk_class": constraint_delay_risk_class,
                "constraint_transfer_miss_risk_class": constraint_transfer_risk_class,
                "constraint_duration_gap_min": constraint_gap.get("duration_gap_min"),
                "constraint_line_change_gap": constraint_gap.get("line_change_gap"),
                "constraint_fullness_gap": constraint_gap.get("fullness_gap"),
                "constraint_near_capacity_gap": constraint_gap.get("near_capacity_gap"),
                "constraint_delay_probability_gap": constraint_gap.get("delay_probability_gap"),
                "constraint_transfer_miss_probability_gap": constraint_gap.get("transfer_miss_probability_gap"),
                "risk_unviable": constraint_gap.get("risk_unviable", False),
                "warning_count": warning_count,
                "route_near_capacity": displayed_has_near_capacity,
                "route_near_capacity_count": displayed_near_capacity_count,
                "route_transfer_miss_probability": displayed_transfer_miss_probability,
                "mean_turn_elapsed_sec": round(
                    sum(turn.get("turn_elapsed_sec", turn.get("turn_latency_sec", 0.0)) for turn in timing_turns) / len(timing_turns),
                    6,
                ) if timing_turns else None,
                "max_turn_elapsed_sec": round(
                    max((turn.get("turn_elapsed_sec", turn.get("turn_latency_sec", 0.0)) for turn in timing_turns), default=0.0),
                    6,
                ),
                "max_turn_budget_sec": self.max_turn_elapsed_sec,
                "turn_over_budget_count": sum(1 for turn in timing_turns if turn.get("turn_capped")),
                "agent_turn_segments": agent_turn_segments,
                "agent_timing_summary": agent_timing_summary,
                "candidate_events": candidate_events,
                "speech_turns": speech_turns,
                "timing_turns": timing_turns,
                "nlu_turns": nlu_turns,
                "runtime_events": runtime_events,
                "metric_config": self.metric_config,
            },
        )
