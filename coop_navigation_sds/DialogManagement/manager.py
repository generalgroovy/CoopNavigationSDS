"""Main controller for a single dialog run.

It coordinates turns, route interpretation, trace events, and final results.
"""
import time

from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import TemplateAgentAResponder
from coop_navigation_sds.NaturalLanguageGeneration.assistant.pipeline import DialogState
from coop_navigation_sds.DialogManagement.result import DialogResult
from coop_navigation_sds.DialogManagement.memory import RouteProposalMemory
from coop_navigation_sds.EvaluationMetrics.metrics import TASK_TERMS, COMPARISON_TERMS, COOPERATION_TERMS
from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
from coop_navigation_sds.DialogManagement.stages import ConversationStage
from coop_navigation_sds.TransportNetwork.routes import (
    estimate_route_time,
    fmt_time,
    optimal_time_route,
    route_line_change_count,
    route_line_sequence,
    route_duration_text,
    route_is_valid,
    route_path_text_from_steps,
    route_step_details,
    route_station_sequence,
)
from coop_navigation_sds.TransportNetwork.constraints import (
    OBJECTIVE_MODE_LABELS,
    OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,
    CONSTRAINT_LABELS,
    acceptable_duration_limit,
    available_agent_a_constraints,
    optimal_constraint_route,
    layered_optimal_routes,
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
from coop_navigation_sds.DialogManagement.speech_pipeline import SpeechTransport

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
        metric_tiers=None,
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
        self.metric_tiers = dict(metric_tiers or {})
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
        phase_timings = []
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

        def emit_phase(turn_number, speaker, phase, **payload):
            """Emit one concise console-facing pipeline phase event."""
            event_queue.put((
                "phase",
                {
                    "turn": turn_number,
                    "speaker": speaker,
                    "phase": phase,
                    **payload,
                },
            ))

        def emit_phase_timing(
            turn_number,
            speaker,
            trace,
            generation_sec,
            speech_sec,
            nlu_sec=None,
            dialogue_management_sec=None,
        ):
            """Record a non-overlapping processing view plus audio stimulus duration."""
            pipeline_wall_sec = float(trace.pipeline_latency_sec or 0.0)
            nlu_value = float(nlu_sec or 0.0)
            management_value = float(dialogue_management_sec or 0.0)
            processing_sec = (
                float(generation_sec or 0.0)
                + pipeline_wall_sec
                + nlu_value
                + management_value
            )
            payload = {
                "turn": turn_number,
                "speaker": speaker,
                "natural_language_generation_sec": round(float(generation_sec or 0.0), 6),
                "text_to_speech_processing_sec": round(float(trace.tts_latency_sec or 0.0), 6),
                "audio_duration_sec": round(float(trace.simulated_duration_sec or 0.0), 6),
                "automatic_speech_recognition_processing_sec": round(float(trace.asr_latency_sec or 0.0), 6),
                "natural_language_understanding_sec": round(nlu_value, 6) if nlu_sec is not None else None,
                "dialogue_management_sec": round(management_value, 6) if dialogue_management_sec is not None else None,
                "speech_pipeline_wall_sec": round(pipeline_wall_sec, 6),
                "observed_turn_sec": round(float(generation_sec or 0.0) + float(speech_sec or 0.0) + nlu_value + management_value, 6),
                "accounted_processing_sec": round(processing_sec, 6),
                "audio_included_in_speech_pipeline_wall": bool(
                    getattr(
                        getattr(self.speech_transport, "config", None),
                        "realtime_enabled",
                        False,
                    )
                ),
            }
            phase_timings.append(payload)
            event_queue.put(("telemetry", "phase_timing", payload))
            record_runtime_event("timing", "turn_phase_breakdown", payload)

        def emit_speech_telemetry(trace, latency_sec, generation_sec=0.0, turn_number=None):
            """Send source/transcript telemetry for automatic speech recognition metrics."""
            turn_number = turn_number or (len(speech_turns) + 1)
            payload = {
                "turn": turn_number,
                "speaker": trace.speaker,
                "generated_text": trace.generated_text,
                "outgoing_text": trace.outgoing_text,
                "incoming_transcript": trace.incoming_transcript,
                "raw_asr_transcript": (trace.diagnostics or {}).get(
                    "raw_asr_transcript", trace.incoming_transcript
                ),
                "misinterpreted_tokens": (trace.diagnostics or {}).get(
                    "misinterpreted_tokens", []
                ),
                "transcript_corrections": (trace.diagnostics or {}).get(
                    "transcript_corrections", []
                ),
                "agent_input_transcript": trace.incoming_transcript,
                "source_text": trace.outgoing_text,
                "transcript": trace.incoming_transcript,
                "latency_sec": latency_sec,
                "tts_latency_sec": trace.tts_latency_sec,
                "asr_latency_sec": trace.asr_latency_sec,
                "pipeline_latency_sec": trace.pipeline_latency_sec,
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
                "console_phases_emitted": True,
            }
            emit_phase(
                turn_number,
                trace.speaker,
                "NLG",
                text=trace.generated_text,
                latency_sec=generation_sec,
            )
            emit_phase(
                turn_number,
                trace.speaker,
                "TTS",
                text=trace.outgoing_text,
                engine=trace.tts_engine,
                latency_sec=trace.tts_latency_sec,
                audio=trace.audio,
            )
            emit_phase(
                turn_number,
                trace.speaker,
                "ASR",
                text=trace.incoming_transcript,
                engine=trace.asr_engine,
                latency_sec=trace.asr_latency_sec,
            )
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

        def emit_nlu_telemetry(speaker, text, parsed_route, has_station_mentions, latency_sec=None, turn_number=None):
            """Send semantic parsing telemetry for language understanding and dialog-state metrics."""
            valid = route_is_valid(parsed_route, allowed_modes=planning_allowed_modes)
            payload = {
                "turn": turn_number,
                "speaker": speaker,
                "text": text,
                "has_station_mentions": has_station_mentions,
                "parsed_route": parsed_route,
                "route_valid": valid,
                "route_reaches_goal": route_reaches_goal(parsed_route, scenario) if valid else False,
                "latency_sec": latency_sec,
            }
            nlu_turns.append(payload)
            emit_phase(
                turn_number or len(speech_turns),
                speaker,
                "NLU",
                text=text,
                parsed_route=parsed_route,
                route_valid=valid,
                latency_sec=latency_sec,
            )
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
        available_constraints = available_agent_a_constraints(persona, scenario)
        configured_constraint_limit = int(
            scenario.get("maximum_progressive_constraints", 2) or 0
        )
        maximum_progressive_constraints = (
            len(available_constraints)
            if configured_constraint_limit <= 0
            else min(len(available_constraints), configured_constraint_limit)
        )
        minimum_compared_routes = max(
            1, int(scenario.get("minimum_compared_routes", 2) or 1)
        )
        # Connectivity is judged from what Agent A has revealed. Private
        # ticket and walking constraints are evaluated only after being stated.
        planning_allowed_modes = ("metro", "tram", "bus", "walking")
        lazy_calculations = {}

        def get_constraint_route():
            """Compute the constraint-aware baseline only when a route decision needs it."""
            if "constraint_route" not in lazy_calculations:
                lazy_calculations["constraint_route"] = optimal_constraint_route(
                    scenario,
                    persona,
                    objective_mode=self.agent_a_objective_mode,
                )
            return lazy_calculations["constraint_route"]

        def get_acceptable_time_limit():
            """Compute the acceptable duration threshold at first use."""
            if "acceptable_time_limit" not in lazy_calculations:
                lazy_calculations["acceptable_time_limit"] = acceptable_duration_limit(
                    scenario,
                    persona,
                    constraint_route=get_constraint_route(),
                )
            return lazy_calculations["acceptable_time_limit"]

        def get_stage_options():
            """Compute stage-viability evidence only for final protocol output."""
            if "stage_options" not in lazy_calculations:
                lazy_calculations["stage_options"] = stage_viability_report(
                    scenario,
                    persona,
                    transfer_tolerance=self.transfer_tolerance,
                    max_constraints=maximum_progressive_constraints,
                )
            return lazy_calculations["stage_options"]

        def get_layered_optima():
            """Calculate the baseline for validity, time, and three constraint layers."""
            if "layered_optima" not in lazy_calculations:
                lazy_calculations["layered_optima"] = layered_optimal_routes(
                    scenario,
                    persona,
                    transfer_tolerance=self.transfer_tolerance,
                    max_constraints=min(3, maximum_progressive_constraints),
                )
            return lazy_calculations["layered_optima"]

        def get_reference_route():
            """Compute the unconstrained reference route only when metrics need it."""
            if "reference_route" not in lazy_calculations:
                lazy_calculations["reference_route"] = optimal_time_route(
                    scenario["start_station"],
                    scenario["destination_station"],
                    scenario["start_time_min"],
                    scenario["transfer_time_min"],
                    allowed_modes=planning_allowed_modes,
                )
            return lazy_calculations["reference_route"]

        def build_preflight_viability():
            """Build protocol evidence after lazy route calculations have been requested."""
            reference_arrival, reference_steps = get_reference_route()
            constraint_route = get_constraint_route()
            stage_options = get_stage_options()
            acceptable_time_limit = get_acceptable_time_limit()
            return {
                "reference_route_available": reference_arrival is not None and bool(reference_steps),
                "constraint_route_available": constraint_route is not None,
                "start_station": scenario["start_station"],
                "destination_station": scenario["destination_station"],
                "allowed_modes": list(planning_allowed_modes or ()),
                "ticket_modes": list(scenario.get("ticket_modes", ())),
                "max_walking_min": scenario.get("max_walking_min"),
                "reference_route": route_station_sequence(reference_steps),
                "reference_path": route_path_text_from_steps(reference_steps),
                "constraint_route": constraint_route.route if constraint_route else [],
                "constraint_path": (
                    route_path_text_from_steps(constraint_route.steps)
                    if constraint_route else None
                ),
                "constraint_duration_min": constraint_route.duration_min if constraint_route else None,
                "constraint_label": constraint_route.label if constraint_route else None,
                "acceptable_duration_limit_min": acceptable_time_limit,
                "stage_option_viability": stage_options,
                "optimal_routes_by_layer": get_layered_optima(),
            }
        start_wall = None
        route_memory = RouteProposalMemory()
        best_route = []
        best_duration = None
        best_turn = None
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
                acceptable_time_limit = get_acceptable_time_limit()
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
                    constraint_route=get_constraint_route(),
                )
                if not unsatisfied_constraint_keys(statuses):
                    return True
            return False

        def select_best_candidate(stated_keys):
            """Select by validity, acceptable arrival, then revealed constraints."""
            ranked = []
            for candidate in route_memory.candidates:
                route = candidate.get("route") or []
                duration = candidate.get("duration")
                estimate = estimate_route_time(
                    route,
                    scenario["start_time_min"],
                    scenario["transfer_time_min"],
                    allowed_modes=planning_allowed_modes,
                )
                if not estimate:
                    continue
                _arrival, steps = estimate
                reaches = route_reaches_goal(route, scenario)
                time_ok = (
                    reaches
                    and duration is not None
                    and (get_acceptable_time_limit() is None or duration <= get_acceptable_time_limit())
                )
                statuses = route_constraint_status(
                    steps,
                    persona,
                    scenario,
                    stated_keys,
                    transfer_tolerance=self.transfer_tolerance,
                    constraint_route=get_constraint_route(),
                )
                misses = unsatisfied_constraint_keys(statuses)
                rank = (
                    int(reaches),
                    int(time_ok),
                    int(time_ok and not misses),
                    -len(misses),
                    -(duration if duration is not None else 10**9),
                )
                ranked.append((rank, candidate, statuses, misses))
            return max(ranked, key=lambda item: item[0]) if ranked else None

        def refresh_best_candidate(stated_keys):
            nonlocal best_route, best_duration, best_turn, route_revision_count
            selection = select_best_candidate(stated_keys)
            if selection is None:
                return False
            _rank, candidate, statuses, misses = selection
            selected_route = list(candidate["route"])
            changed = bool(best_route and selected_route != best_route)
            if changed:
                route_revision_count += 1
            best_route = selected_route
            best_duration = candidate.get("duration")
            best_turn = candidate.get("turn")
            candidate["selected_constraint_status"] = statuses
            candidate["selected_unsatisfied_constraints"] = misses
            return changed

        def phase_objective_satisfied(stated_keys):
            """Require a valid, timely route satisfying every revealed constraint."""
            return prior_route_satisfies_current_goals(stated_keys)

        def authoritative_stage(stated_keys):
            """Derive the next stage from completed objectives, not utterance wording."""
            if not phase_objective_satisfied(stated_keys):
                return ConversationStage.PROPOSAL if not stated_keys else ConversationStage.REFINEMENT
            if (
                self.agent_a_objective_mode == OBJECTIVE_SHORTEST_WITH_CONSTRAINTS
                and len(stated_keys) < maximum_progressive_constraints
            ):
                return ConversationStage.REFINEMENT
            return ConversationStage.CONFIRMATION

        def guard_agent_a_progress(reply, before_keys):
            """Block constraint revelation or closure until the active objective is met."""
            after_keys = stated_constraint_keys([*conversation, ("Agent A", reply)])
            new_keys = [key for key in after_keys if key not in before_keys]
            objective_satisfied = phase_objective_satisfied(before_keys)
            final_caller_turn = len(conversation) + 1 >= self.num_turns
            if len(new_keys) > 1:
                return (
                    "Let's handle one requirement at a time. "
                    "Please keep working on the current objective."
                )
            if new_keys and not objective_satisfied:
                return (
                    "That does not meet the current objective yet. "
                    "Please give another route that does."
                )
            if (
                agent_a_ended_conversation(reply)
                and (
                    not objective_satisfied
                    or (
                        self.agent_a_objective_mode == OBJECTIVE_SHORTEST_WITH_CONSTRAINTS
                        and len(before_keys) < maximum_progressive_constraints
                        and not final_caller_turn
                    )
                )
            ):
                return (
                    "I am not ready to choose yet. "
                    "Please satisfy the current route requirement first."
                )
            if (
                agent_a_ended_conversation(reply)
                and len(route_memory.candidates) < minimum_compared_routes
                and len(conversation) + 1 < self.num_turns
            ):
                return (
                    "That option works. Compare one distinct viable route "
                    "before I choose."
                )
            return reply

        start_wall = time.time()
        event_queue.put(("timer_start", start_wall))
        speech_started_at = time.time()
        opening_trace = self.speech_transport.transmit_trace(
            "Agent A",
            self.test_case.opening_utterance(),
        )
        opening_transcript = opening_trace.incoming_transcript
        opening_speech_sec = max(time.time() - speech_started_at, opening_trace.simulated_duration_sec)
        emit_speech_telemetry(
            opening_trace,
            opening_speech_sec,
            generation_sec=0.0,
            turn_number=1,
        )
        conversation = [("Agent A", opening_transcript)]
        ingest_conversation_text(opening_transcript)

        record_runtime_event("startup", "route_calculations_deferred", {
            "reference_route": "deferred_until_final_metrics",
            "constraint_baseline": "deferred_until_route_evaluation",
            "stage_viability": "deferred_until_final_protocol",
        })
        event_queue.put(("message", conversation[0][0], conversation[0][1]))
        log_conversation_step(
            "Agent A",
            conversation[0][1],
            len(conversation),
            0.0,
            opening_speech_sec,
        )
        emit_timing_telemetry("Agent A", 0.0, opening_speech_sec, turn_number=len(conversation))
        emit_phase_timing(
            len(conversation),
            "Agent A",
            opening_trace,
            0.0,
            opening_speech_sec,
        )
        record_runtime_event("opening", "agent_a_opening", {"turn": len(conversation), "text": conversation[0][1]})

        route_round = 0
        while len(conversation) < self.num_turns:
            route_round += 1
            active_keys_before_b = stated_constraint_keys(conversation)
            state = DialogState(
                self.test_case,
                conversation,
                route_round - 1,
                scenario_override=scenario,
                persona_override=persona,
                stage_override=authoritative_stage(active_keys_before_b),
            )
            event_queue.put(("stage", state.stage))
            record_runtime_event(
                "dialogue_management",
                "stage_entered",
                {
                    "turn": len(conversation) + 1,
                    "stage": state.stage.value,
                    "response_focus": state.context.response_focus,
                },
            )
            generation_started_at = time.time()
            reply_b = self.agent_b_plugin.run_agent_b(state)
            generation_sec = time.time() - generation_started_at
            speech_started_at = time.time()
            reply_trace = self.speech_transport.transmit_trace("Agent B", reply_b)
            reply_transcript = reply_trace.incoming_transcript
            speech_sec = max(time.time() - speech_started_at, reply_trace.simulated_duration_sec)
            emit_speech_telemetry(
                reply_trace,
                speech_sec,
                generation_sec=generation_sec,
                turn_number=len(conversation) + 1,
            )
            emit_timing_telemetry("Agent B", generation_sec, speech_sec, turn_number=len(conversation) + 1)
            conversation.append(("Agent B", reply_transcript))
            ingest_conversation_text(reply_transcript)
            event_queue.put(("message", "Agent B", reply_transcript))

            nlu_started_at = time.perf_counter()
            parsed_route = self.route_interpreter.interpret_reply(reply_transcript, scenario)
            has_station_mentions = self.route_interpreter.has_station_mentions(reply_transcript)
            nlu_latency_sec = time.perf_counter() - nlu_started_at
            emit_nlu_telemetry(
                "Agent B",
                reply_transcript,
                parsed_route,
                has_station_mentions,
                latency_sec=round(nlu_latency_sec, 6),
                turn_number=len(conversation),
            )
            dialogue_management_started_at = time.perf_counter()
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
                    "dialog_stage": state.stage.value,
                    "response_focus": state.context.response_focus,
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
                    constraint_route = get_constraint_route()
                    acceptable_time_limit = get_acceptable_time_limit()
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
                            "path": route_path_text_from_steps(candidate_steps),
                            "route_steps": route_step_details(candidate_steps),
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
                        candidate = route_memory.record(route_round, parsed_route, duration, best_duration)
                        gap = route_constraint_gap(candidate_steps, duration, constraint_route)
                        candidate.update(
                            {
                                "route_steps": route_step_details(candidate_steps),
                                "path": route_path_text_from_steps(candidate_steps),
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
                        previous_route = list(best_route)
                        refresh_best_candidate(active_constraint_keys)
                        if best_route != previous_route:
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
            elif time_frame_miss_count > self.constraint_miss_limit:
                early_stop_reason = "time_frame_miss_limit"
            elif constraint_miss_count >= self.constraint_miss_limit:
                early_stop_reason = "constraint_miss_limit"

            dialogue_management_sec = time.perf_counter() - dialogue_management_started_at
            emit_phase(
                len(conversation),
                "Agent B",
                "DM",
                latency_sec=dialogue_management_sec,
                text="Route and dialogue state updated",
            )
            emit_phase_timing(
                len(conversation),
                "Agent B",
                reply_trace,
                generation_sec,
                speech_sec,
                nlu_sec=nlu_latency_sec,
                dialogue_management_sec=dialogue_management_sec,
            )

            if len(conversation) < self.num_turns:
                generation_started_at = time.time()
                reply_a = (
                    early_stop_message(early_stop_reason)
                    if early_stop_reason
                    else self.agent_a_responder.reply(route_round - 1, persona, scenario, conversation)
                )
                if not early_stop_reason:
                    reply_a = guard_agent_a_progress(
                        reply_a,
                        stated_constraint_keys(conversation),
                    )
                generation_sec = time.time() - generation_started_at
                speech_started_at = time.time()
                reply_trace = self.speech_transport.transmit_trace("Agent A", reply_a)
                reply_transcript = reply_trace.incoming_transcript
                speech_sec = max(time.time() - speech_started_at, reply_trace.simulated_duration_sec)
                emit_speech_telemetry(
                    reply_trace,
                    speech_sec,
                    generation_sec=generation_sec,
                    turn_number=len(conversation) + 1,
                )
                emit_timing_telemetry("Agent A", generation_sec, speech_sec, turn_number=len(conversation) + 1)
                emit_phase_timing(
                    len(conversation) + 1,
                    "Agent A",
                    reply_trace,
                    generation_sec,
                    speech_sec,
                )
                conversation.append(("Agent A", reply_transcript))
                ingest_conversation_text(reply_transcript)
                event_queue.put(("message", "Agent A", reply_transcript))
                previous_route = list(best_route)
                refresh_best_candidate(stated_constraint_keys(conversation))
                if best_route != previous_route:
                    event_queue.put(("route", best_route))
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
                        "phase_objective_satisfied": phase_objective_satisfied(
                            stated_constraint_keys(conversation)
                        ),
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
        refresh_best_candidate(stated_constraint_keys(conversation))

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
        reference_arrival, reference_steps = get_reference_route()
        constraint_route = get_constraint_route()
        acceptable_time_limit = get_acceptable_time_limit()
        stage_options = get_stage_options()
        preflight_viability = build_preflight_viability()
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

        mean_turn_elapsed = (
            round(
                sum(turn.get("turn_elapsed_sec", turn.get("turn_latency_sec", 0.0)) for turn in timing_turns)
                / len(timing_turns),
                4,
            )
            if timing_turns
            else None
        )
        metrics = (
            "[Run]\n"
            f"Case: {self.test_case.scenario_key}\n"
            f"Persona: {persona['name']}\n"
            f"Outcome: {conversation_outcome}\n"
            f"Messages: {len(conversation)}\n"
            "[Task]\n"
            f"Journey: {scenario['start_station']} {fmt_time(scenario['start_time_min'])} -> {scenario['destination_station']}\n"
            f"Route: {' -> '.join(best_route) if best_route else 'none'}\n"
            f"Duration: {displayed_duration if displayed_duration is not None else 'none'} min\n"
            f"Duration limit: {acceptable_time_limit if acceptable_time_limit is not None else 'none'} min\n"
            f"Arrival: {fmt_time(displayed_arrival) if displayed_arrival is not None else 'none'}\n"
            f"Valid: {route_correct}\n"
            f"Constraints: {', '.join(final_stated_constraints) if final_stated_constraints else 'none'}\n"
            f"Unmet constraints: {', '.join(final_unsatisfied_constraints) if final_unsatisfied_constraints else 'none'}\n"
            "[Comparison]\n"
            f"Optimal duration: {reference_duration if reference_duration is not None else 'none'} min\n"
            f"Candidates: {len(route_memory.candidates)}\n"
            f"Revisions: {route_revision_count}\n"
            f"Near capacity: {'yes' if displayed_has_near_capacity else 'no'}\n"
            f"Delay risk: {displayed_delay_risk_class}\n"
            "[Execution]\n"
            f"Invalid proposals: {invalid_route_count}\n"
            f"Slow proposals: {time_frame_miss_count}\n"
            f"Constraint misses: {constraint_miss_count}\n"
            f"Stop reason: {early_stop_reason or 'none'}\n"
            f"Mean turn: {mean_turn_elapsed if mean_turn_elapsed is not None else 'none'} s\n"
            f"Runtime: {runtime_sec:.2f} s\n"
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
                "configured_num_turns": self.num_turns,
                "maximum_progressive_constraints": maximum_progressive_constraints,
                "minimum_compared_routes": minimum_compared_routes,
                "speech_playback_enabled": bool(
                    getattr(getattr(self.speech_transport, "config", None), "playback_enabled", False)
                ),
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
                "allowed_lines": [],
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
                "phase_timings": phase_timings,
                "nlu_turns": nlu_turns,
                "runtime_events": runtime_events,
                "metric_config": self.metric_config,
                "metric_tiers": self.metric_tiers,
            },
        )
