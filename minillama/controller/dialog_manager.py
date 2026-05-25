"""Main controller for a single dialog run. It coordinates turns, route interpretation, GUI events, and final dialog results.
"""
import time

from minillama.agent_a.agent_a_responder import TemplateAgentAResponder
from minillama.agent_b.pipeline import DialogState
from minillama.controller.dialog_result import DialogResult
from minillama.controller.route_memory import RouteProposalMemory
from minillama.evaluation.metrics import TASK_TERMS, COMPARISON_TERMS, COOPERATION_TERMS
from minillama.evaluation.route_interpreter import NaturalRouteInterpreter
from minillama.model.route_planner import (
    estimate_route_time,
    fmt_time,
    optimal_time_route,
    route_line_change_count,
    route_line_sequence,
    route_duration_text,
    route_is_valid,
    route_station_sequence,
)
from minillama.model.route_constraints import optimal_constraint_route, route_constraint_gap
from minillama.agent_b.speech_io import SpeechTransport


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


def route_duration_min(stations, scenario):
    """Return total route duration in minutes when the route can be scheduled."""
    estimate = estimate_route_time(
        stations,
        scenario["start_time_min"],
        scenario["transfer_time_min"],
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
        metric_snapshot_interval=1,
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
        self.metric_snapshot_interval = max(1, int(metric_snapshot_interval or 1))

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
        metric_snapshots = []
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

        def log_conversation_step(speaker, utterance, turn_number, generation_sec, speech_sec, parsed_route=None, has_station_mentions=False):
            """Record one conversation step with the current evaluation state."""
            if not self.monitor:
                return

            parsed_route_valid = route_is_valid(parsed_route) if parsed_route else False
            parsed_route_goal = route_reaches_goal(parsed_route, scenario) if parsed_route_valid else False
            payload = {
                "turn": turn_number,
                "speaker": speaker,
                "utterance": utterance,
                "generation_sec": round(generation_sec, 6),
                "speech_sec": round(speech_sec, 6),
                "turn_latency_sec": round(generation_sec + speech_sec, 6),
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

        def build_metric_snapshot(turn_number, phase, speaker, parsed_route=None):
            """Build a compact periodic metric snapshot for runtime logs."""
            parsed_route_valid = route_is_valid(parsed_route) if parsed_route else False
            parsed_route_goal = route_reaches_goal(parsed_route, scenario) if parsed_route_valid else False
            return {
                "turn": turn_number,
                "phase": phase,
                "speaker": speaker,
                "elapsed_sec": round(time.time() - start_wall, 6),
                "message_count": len(conversation),
                "word_count": conversation_word_count,
                "task_terms": task_term_count,
                "comparison_terms": comparison_term_count,
                "cooperation_terms": cooperation_term_count,
                "candidate_routes": len(route_memory.candidates),
                "route_revisions": route_revision_count,
                "best_route": list(best_route),
                "best_duration": best_duration,
                "best_candidate_turn": best_turn,
                "warning_count": warning_count,
                "invalid_route_count": invalid_route_count,
                "constraint_miss_count": constraint_miss_count,
                "route_valid": parsed_route_valid,
                "route_reaches_goal": parsed_route_goal,
            }

        def emit_metric_snapshot(turn_number, phase, speaker, parsed_route=None, force=False):
            """Emit periodic metric snapshots to the GUI stream and logger."""
            if not force and turn_number % self.metric_snapshot_interval != 0:
                return
            snapshot = build_metric_snapshot(turn_number, phase, speaker, parsed_route=parsed_route)
            metric_snapshots.append(snapshot)
            event_queue.put(("metric_snapshot", snapshot))

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

        def emit_timing_telemetry(speaker, generation_sec, speech_sec):
            """Send turn timing telemetry for latency metrics."""
            payload = {
                "speaker": speaker,
                "generation_sec": generation_sec,
                "speech_sec": speech_sec,
                "turn_latency_sec": generation_sec + speech_sec,
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
            valid = route_is_valid(parsed_route)
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

        scenario = self.test_case.scenario
        persona = self.test_case.persona
        constraint_route = optimal_constraint_route(scenario, persona)
        start_wall = time.time()
        route_memory = RouteProposalMemory()
        best_route = []
        best_duration = None
        best_turn = None
        best_rank = (-1, float("-inf"))
        route_revision_count = 0
        invalid_route_count = 0
        constraint_miss_count = 0
        early_stop_reason = None

        def constraint_missed(gap):
            """Return whether a valid proposal is worse than the stated constraint baseline."""
            return any(
                gap.get(key, 0) > 0
                for key in (
                    "duration_gap_min",
                    "line_change_gap",
                    "fullness_gap",
                    "delay_probability_gap",
                )
            )

        def early_stop_message(reason):
            if reason == "invalid_route_limit":
                return "I will stop here; the route suggestions still are not connected from start to destination."
            if reason == "constraint_miss_limit":
                return "I will stop here; the options still ignore the constraints I gave."
            return "I will stop here."

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
        event_queue.put(("system", f"Speech transport: {self.speech_transport.description}"))
        if constraint_route:
            event_queue.put(
                (
                    "system",
                    (
                        "Constraint baseline: "
                        f"{' to '.join(constraint_route.route)} "
                        f"({constraint_route.duration_min} minutes, "
                        f"{constraint_route.line_change_count} changes, "
                        f"{constraint_route.average_fullness} percent full, "
                        f"{round(constraint_route.delay_probability * 100)} percent delay risk, "
                        f"{constraint_route.label})"
                    ),
                )
            )
        event_queue.put(("message", conversation[0][0], conversation[0][1]))
        log_conversation_step(
            "Agent A",
            conversation[0][1],
            len(conversation),
            0.0,
            opening_speech_sec,
        )
        emit_metric_snapshot(len(conversation), "opening", "Agent A", force=True)

        route_round = 0
        while len(conversation) < self.num_turns:
            route_round += 1
            state = DialogState(self.test_case, conversation, route_round - 1)
            generation_started_at = time.time()
            reply_b = self.agent_b_plugin.run_agent_b(state)
            generation_sec = time.time() - generation_started_at
            speech_started_at = time.time()
            reply_trace = self.speech_transport.transmit_trace("Agent B", reply_b)
            reply_transcript = reply_trace.incoming_transcript
            speech_sec = max(time.time() - speech_started_at, reply_trace.simulated_duration_sec)
            emit_speech_telemetry(reply_trace, speech_sec)
            emit_timing_telemetry("Agent B", generation_sec, speech_sec)
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
            emit_metric_snapshot(len(conversation), "agent_b_reply", "Agent B", parsed_route=parsed_route)

            if route_is_valid(parsed_route):
                duration = route_duration_min(parsed_route, scenario)
                if duration is not None:
                    _, candidate_steps = estimate_route_time(
                        parsed_route,
                        scenario["start_time_min"],
                        scenario["transfer_time_min"],
                    )
                    candidate_reaches_goal = route_reaches_goal(parsed_route, scenario)
                    constraints_already_stated = bool(best_route)
                    if route_memory.already_seen(parsed_route):
                        warning_count += 1
                        gap = route_constraint_gap(candidate_steps, duration, constraint_route)
                        event_queue.put(("warning", "Repeated route proposal ignored; compare a different route."))
                        event_queue.put(
                            (
                                "candidate",
                                {
                                    "turn": route_round,
                                    "route": parsed_route,
                                    "duration": duration,
                                    "decision": "repeat",
                                    "best_duration": best_duration,
                                    "previous_best": best_duration,
                                    "constraint_duration": constraint_route.duration_min if constraint_route else None,
                                    **gap,
                                },
                            )
                        )
                    else:
                        is_new_route = parsed_route != best_route
                        candidate = route_memory.record(route_round, parsed_route, duration, best_duration)
                        gap = route_constraint_gap(candidate_steps, duration, constraint_route)
                        candidate.update(
                            {
                                "constraint_duration": constraint_route.duration_min if constraint_route else None,
                                "constraint_route": constraint_route.route if constraint_route else [],
                                **gap,
                            }
                        )
                        if (
                            constraints_already_stated
                            and candidate_reaches_goal
                            and constraint_missed(gap)
                        ):
                            constraint_miss_count += 1
                            event_queue.put(("warning", "Route is valid but misses stated constraints."))
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
                        event_queue.put(("candidate", candidate))
            elif has_station_mentions:
                warning_count += 1
                invalid_route_count += 1
                event_queue.put(("warning", "Station names mentioned, but no connected spoken route was inferred."))

            if invalid_route_count >= self.invalid_route_limit:
                early_stop_reason = "invalid_route_limit"
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
                emit_timing_telemetry("Agent A", generation_sec, speech_sec)
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
                emit_metric_snapshot(len(conversation), "agent_a_reply", "Agent A")
            if early_stop_reason:
                break

        end_wall = time.time()

        estimate = estimate_route_time(
            best_route,
            scenario["start_time_min"],
            scenario["transfer_time_min"],
        ) if best_route else None

        if estimate:
            displayed_arrival, displayed_steps = estimate
            displayed_duration = displayed_arrival - scenario["start_time_min"]
        else:
            displayed_arrival = None
            displayed_duration = None
            displayed_steps = []

        route_valid = route_is_valid(best_route)
        reaches_goal = route_reaches_goal(best_route, scenario)
        route_correct = route_valid and reaches_goal
        average_route_fullness = round(
            sum(step.get("fullness", 0) for step in displayed_steps) / len(displayed_steps),
            1,
        ) if displayed_steps else None
        reference_arrival, reference_steps = optimal_time_route(
            scenario["start_station"],
            scenario["destination_station"],
            scenario["start_time_min"],
            scenario["transfer_time_min"],
        )
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
        constraint_average_fullness = constraint_route.average_fullness if constraint_route else None
        constraint_delay_probability = constraint_route.delay_probability if constraint_route else None
        constraint_gap = route_constraint_gap(displayed_steps, displayed_duration, constraint_route)
        displayed_line_sequence = route_line_sequence(displayed_steps)
        displayed_line_change_count = route_line_change_count(displayed_steps)

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
            f"Duration breakdown:    {route_duration_text(displayed_steps)}\n"
            f"Average crowding:      {str(average_route_fullness) + ' percent' if average_route_fullness is not None else 'None'}\n"
            f"Reference route:       {' to '.join(reference_route) if reference_route else 'None'}\n"
            f"Reference line sequence: {' to '.join(reference_line_sequence) if reference_line_sequence else 'None'}\n"
            f"Reference line changes: {reference_line_change_count if reference_line_sequence else 'None'}\n"
            f"Reference duration:    {str(reference_duration) + ' minutes' if reference_duration is not None else 'None'}\n"
            f"Constraint target:     {constraint_route.label if constraint_route else 'None'}\n"
            f"Constraint route:      {' to '.join(constraint_route.route) if constraint_route else 'None'}\n"
            f"Constraint line sequence: {' to '.join(constraint_line_sequence) if constraint_line_sequence else 'None'}\n"
            f"Constraint line changes: {constraint_line_change_count if constraint_line_change_count is not None else 'None'}\n"
            f"Constraint duration:   {str(constraint_duration) + ' minutes' if constraint_duration is not None else 'None'}\n"
            f"Constraint crowding:   {str(constraint_average_fullness) + ' percent' if constraint_average_fullness is not None else 'None'}\n"
            f"Constraint delay risk: {str(round(constraint_delay_probability * 100, 1)) + ' percent' if constraint_delay_probability is not None else 'None'}\n"
            f"Constraint gap:        {constraint_gap.get('duration_gap_min', 'None')} minutes, {constraint_gap.get('line_change_gap', 'None')} changes, {constraint_gap.get('fullness_gap', 'None')} fullness, {constraint_gap.get('delay_probability_gap', 'None')} delay\n"
            f"Candidate routes:      {len(route_memory.candidates)}\n"
            f"Route revisions:       {route_revision_count}\n"
            f"Best candidate turn:   {best_turn if best_turn is not None else 'None'}\n"
            f"Invalid route count:   {invalid_route_count}\n"
            f"Constraint miss count: {constraint_miss_count}\n"
            f"Early stop reason:     {early_stop_reason or 'None'}\n"
            f"Route valid:           {route_valid}\n"
            f"Route reaches goal:    {reaches_goal}\n"
            f"Route correct:         {route_correct}\n"
            f"Runtime:               {end_wall - start_wall:.2f} seconds\n"
            f"Speech pipeline:       {self.speech_transport.description}\n"
        )

        event_queue.put(("metrics", metrics))
        emit_metric_snapshot(len(conversation), "final", "system", force=True)
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
            runtime_sec=end_wall - start_wall,
            metrics_text=metrics,
            extra={
                "speech_transport": self.speech_transport.description,
                "agent_a_responder": self.agent_a_responder.name,
                "agent_b_plugin": getattr(self.agent_b_plugin, "name", type(self.agent_b_plugin).__name__),
                "messages": len(conversation),
                "candidate_routes": len(route_memory.candidates),
                "route_revisions": route_revision_count,
                "best_candidate_turn": best_turn,
                "invalid_route_count": invalid_route_count,
                "constraint_miss_count": constraint_miss_count,
                "early_stop_reason": early_stop_reason,
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
                "constraint_average_fullness": constraint_average_fullness,
                "constraint_delay_probability": constraint_delay_probability,
                "constraint_duration_gap_min": constraint_gap.get("duration_gap_min"),
                "constraint_line_change_gap": constraint_gap.get("line_change_gap"),
                "constraint_fullness_gap": constraint_gap.get("fullness_gap"),
                "constraint_delay_probability_gap": constraint_gap.get("delay_probability_gap"),
                "warning_count": warning_count,
                "average_route_fullness": average_route_fullness,
                "speech_turns": speech_turns,
                "timing_turns": timing_turns,
                "nlu_turns": nlu_turns,
                "metric_snapshots": metric_snapshots,
            },
        )
