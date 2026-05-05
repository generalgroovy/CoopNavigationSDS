import time

from agents import generate_agent_a_template
from dialog_result import DialogResult
from pipeline import DialogState
from route_interpreter import NaturalRouteInterpreter
from route_planner import (
    estimate_route_time,
    fmt_time,
    route_duration_text,
    route_is_valid,
)
from speech_io import SpeechTransport


def route_reaches_goal(stations, scenario):
    return (
        bool(stations)
        and stations[0] == scenario["start_station"]
        and stations[-1] == scenario["destination_station"]
    )


class DialogManager:
    def __init__(self, test_case, pipeline, num_turns, route_interpreter=None, speech_transport=None):
        self.test_case = test_case
        self.pipeline = pipeline
        self.num_turns = num_turns
        self.route_interpreter = route_interpreter or NaturalRouteInterpreter()
        self.speech_transport = speech_transport or SpeechTransport()

    def run(self, event_queue):
        scenario = self.test_case.scenario
        persona = self.test_case.persona
        _, opening_transcript = self.speech_transport.transmit(
            "Agent A",
            self.test_case.opening_utterance(),
        )
        conversation = [("Agent A", opening_transcript)]

        event_queue.put(("system", f"Test case: {self.test_case.name}"))
        event_queue.put(("system", f"Persona: {persona['name']}"))
        event_queue.put(("system", f"Scenario: {scenario['name']}"))
        event_queue.put(("system", f"Speech transport: {self.speech_transport.description}"))
        event_queue.put(("message", conversation[0][0], conversation[0][1]))

        start_wall = time.time()
        last_valid_route = []

        for turn in range(self.num_turns):
            state = DialogState(self.test_case, conversation, turn)
            reply_b = self.pipeline.run_agent_b(state)
            _, reply_transcript = self.speech_transport.transmit("Agent B", reply_b)
            conversation.append(("Agent B", reply_transcript))
            event_queue.put(("message", "Agent B", reply_transcript))

            parsed_route = self.route_interpreter.interpret_reply(reply_transcript, scenario)

            if route_is_valid(parsed_route):
                last_valid_route = parsed_route
                event_queue.put(("route", last_valid_route))
            elif self.route_interpreter.has_station_mentions(reply_transcript):
                event_queue.put(("warning", "Station names mentioned, but no connected spoken route was inferred."))

            if turn < self.num_turns - 1:
                reply_a = generate_agent_a_template(turn, persona, scenario)
                _, reply_transcript = self.speech_transport.transmit("Agent A", reply_a)
                conversation.append(("Agent A", reply_transcript))
                event_queue.put(("message", "Agent A", reply_transcript))

        end_wall = time.time()

        estimate = estimate_route_time(
            last_valid_route,
            scenario["start_time_min"],
            scenario["transfer_time_min"],
        ) if last_valid_route else None

        if estimate:
            displayed_arrival, displayed_steps = estimate
            displayed_duration = displayed_arrival - scenario["start_time_min"]
        else:
            displayed_arrival = None
            displayed_duration = None
            displayed_steps = []

        route_valid = route_is_valid(last_valid_route)
        reaches_goal = route_reaches_goal(last_valid_route, scenario)
        route_correct = route_valid and reaches_goal

        metrics = (
            f"Test case:             {self.test_case.name}\n"
            f"Persona:               {persona['name']}\n"
            f"Scenario:              {scenario['name']}\n"
            f"Messages:              {len(conversation)}\n"
            f"Start time:            {fmt_time(scenario['start_time_min'])}\n"
            f"Start:                 {scenario['start_station']}\n"
            f"Destination:           {scenario['destination_station']}\n"
            f"Displayed route:       {' -> '.join(last_valid_route) if last_valid_route else 'None'}\n"
            f"Displayed arrival:     {fmt_time(displayed_arrival) if displayed_arrival else 'None'}\n"
            f"Displayed duration:    {str(displayed_duration) + ' min' if displayed_duration is not None else 'None'}\n"
            f"Duration breakdown:    {route_duration_text(displayed_steps)}\n"
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
            model_name=getattr(self.pipeline.model_adapter, "name", "unknown-model"),
            conversation=conversation,
            route=last_valid_route,
            route_steps=displayed_steps,
            route_valid=route_valid,
            route_reaches_goal=reaches_goal,
            route_correct=route_correct,
            route_duration_min=displayed_duration,
            runtime_sec=end_wall - start_wall,
            metrics_text=metrics,
            extra={
                "speech_transport": self.speech_transport.description,
                "messages": len(conversation),
            },
        )
