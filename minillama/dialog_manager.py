"""Main controller for a single dialog run. It coordinates turns, route interpretation, GUI events, and final dialog results.
"""
import time

from minillama.agent_a_responder import TemplateAgentAResponder
from minillama.pipeline import DialogState
from minillama.dialog_result import DialogResult
from minillama.route_memory import RouteProposalMemory
from minillama.route_interpreter import NaturalRouteInterpreter
from minillama.route_planner import (
    estimate_route_time,
    fmt_time,
    optimal_time_route,
    route_duration_text,
    route_is_valid,
    route_station_sequence,
)
from minillama.speech_io import SpeechTransport


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

    def run(self, event_queue):
        """Run method for this module's MVC responsibility.
        
        Args:
            event_queue: Input value used by `run`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        def emit_speech_telemetry(speaker, source_text, transcript, latency_sec):
            """Send source/transcript telemetry for ASR-facing GUI metrics."""
            event_queue.put(
                (
                    "telemetry",
                    "speech",
                    {
                        "speaker": speaker,
                        "source_text": source_text,
                        "transcript": transcript,
                        "latency_sec": latency_sec,
                    },
                )
            )

        def emit_timing_telemetry(speaker, generation_sec, speech_sec):
            """Send turn timing telemetry for latency metrics."""
            event_queue.put(
                (
                    "telemetry",
                    "timing",
                    {
                        "speaker": speaker,
                        "generation_sec": generation_sec,
                        "speech_sec": speech_sec,
                        "turn_latency_sec": generation_sec + speech_sec,
                    },
                )
            )

        def emit_nlu_telemetry(speaker, text, parsed_route, has_station_mentions):
            """Send semantic parsing telemetry for NLU/DST metrics."""
            valid = route_is_valid(parsed_route)
            event_queue.put(
                (
                    "telemetry",
                    "nlu",
                    {
                        "speaker": speaker,
                        "text": text,
                        "has_station_mentions": has_station_mentions,
                        "parsed_route": parsed_route,
                        "route_valid": valid,
                        "route_reaches_goal": route_reaches_goal(parsed_route, scenario) if valid else False,
                    },
                )
            )

        scenario = self.test_case.scenario
        persona = self.test_case.persona
        speech_started_at = time.time()
        opening_signal, opening_transcript = self.speech_transport.transmit(
            "Agent A",
            self.test_case.opening_utterance(),
        )
        emit_speech_telemetry("Agent A", opening_signal.text, opening_transcript, time.time() - speech_started_at)
        conversation = [("Agent A", opening_transcript)]

        event_queue.put(("system", f"Test case: {self.test_case.name}"))
        event_queue.put(("system", f"Persona: {persona['name']}"))
        event_queue.put(("system", f"Scenario: {scenario['name']}"))
        event_queue.put(("system", f"Speech transport: {self.speech_transport.description}"))
        event_queue.put(("message", conversation[0][0], conversation[0][1]))

        start_wall = time.time()
        route_memory = RouteProposalMemory()
        best_route = []
        best_duration = None
        best_turn = None
        route_revision_count = 0

        for turn in range(self.num_turns):
            state = DialogState(self.test_case, conversation, turn)
            generation_started_at = time.time()
            reply_b = self.agent_b_plugin.run_agent_b(state)
            generation_sec = time.time() - generation_started_at
            speech_started_at = time.time()
            reply_signal, reply_transcript = self.speech_transport.transmit("Agent B", reply_b)
            speech_sec = time.time() - speech_started_at
            emit_speech_telemetry("Agent B", reply_signal.text, reply_transcript, speech_sec)
            emit_timing_telemetry("Agent B", generation_sec, speech_sec)
            conversation.append(("Agent B", reply_transcript))
            event_queue.put(("message", "Agent B", reply_transcript))

            parsed_route = self.route_interpreter.interpret_reply(reply_transcript, scenario)
            has_station_mentions = self.route_interpreter.has_station_mentions(reply_transcript)
            emit_nlu_telemetry("Agent B", reply_transcript, parsed_route, has_station_mentions)

            if route_is_valid(parsed_route):
                duration = route_duration_min(parsed_route, scenario)
                if duration is not None:
                    if route_memory.already_seen(parsed_route):
                        event_queue.put(("warning", "Repeated route proposal ignored; compare a different route."))
                        event_queue.put(
                            (
                                "candidate",
                                {
                                    "turn": turn + 1,
                                    "route": parsed_route,
                                    "duration": duration,
                                    "decision": "repeat",
                                    "best_duration": best_duration,
                                    "previous_best": best_duration,
                                },
                            )
                        )
                        continue

                    is_new_route = parsed_route != best_route
                    candidate = route_memory.record(turn + 1, parsed_route, duration, best_duration)

                    if best_duration is None or duration < best_duration:
                        if best_route and is_new_route:
                            route_revision_count += 1
                        best_route = parsed_route
                        best_duration = duration
                        best_turn = turn + 1
                        event_queue.put(("route", best_route))

                    candidate["best_duration"] = best_duration
                    event_queue.put(("candidate", candidate))
            elif has_station_mentions:
                event_queue.put(("warning", "Station names mentioned, but no connected spoken route was inferred."))

            if turn < self.num_turns - 1:
                generation_started_at = time.time()
                reply_a = self.agent_a_responder.reply(turn, persona, scenario, conversation)
                generation_sec = time.time() - generation_started_at
                speech_started_at = time.time()
                reply_signal, reply_transcript = self.speech_transport.transmit("Agent A", reply_a)
                speech_sec = time.time() - speech_started_at
                emit_speech_telemetry("Agent A", reply_signal.text, reply_transcript, speech_sec)
                emit_timing_telemetry("Agent A", generation_sec, speech_sec)
                conversation.append(("Agent A", reply_transcript))
                event_queue.put(("message", "Agent A", reply_transcript))

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

        metrics = (
            f"Test case:             {self.test_case.name}\n"
            f"Persona:               {persona['name']}\n"
            f"Scenario:              {scenario['name']}\n"
            f"Messages:              {len(conversation)}\n"
            f"Start time:            {fmt_time(scenario['start_time_min'])}\n"
            f"Start:                 {scenario['start_station']}\n"
            f"Destination:           {scenario['destination_station']}\n"
            f"Displayed route:       {' -> '.join(best_route) if best_route else 'None'}\n"
            f"Displayed arrival:     {fmt_time(displayed_arrival) if displayed_arrival else 'None'}\n"
            f"Displayed duration:    {str(displayed_duration) + ' min' if displayed_duration is not None else 'None'}\n"
            f"Duration breakdown:    {route_duration_text(displayed_steps)}\n"
            f"Reference route:       {' -> '.join(reference_route) if reference_route else 'None'}\n"
            f"Reference duration:    {str(reference_duration) + ' min' if reference_duration is not None else 'None'}\n"
            f"Candidate routes:      {len(route_memory.candidates)}\n"
            f"Route revisions:       {route_revision_count}\n"
            f"Best candidate turn:   {best_turn if best_turn is not None else 'None'}\n"
            f"Route valid:           {route_valid}\n"
            f"Route reaches goal:    {reaches_goal}\n"
            f"Route correct:         {route_correct}\n"
            f"Runtime:               {end_wall - start_wall:.2f}s\n"
        )

        event_queue.put(("metrics", metrics))
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
                "reference_duration_min": reference_duration,
            },
        )
