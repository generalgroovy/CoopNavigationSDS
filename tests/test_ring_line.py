import unittest
import re

from coop_navigation_sds.TransportNetwork.network import (
    LINES,
    ADJACENCY,
    station_transfer_time_min,
)
from coop_navigation_sds.TransportNetwork.constraints import (
    ConstraintRoute,
    optimal_constraint_route,
    probability_class,
    probability_class_allowed,
    route_constraint_gap,
    route_near_capacity_count,
)
from coop_navigation_sds.TransportNetwork.routes import (
    estimate_route_time,
    line_direction_sequences,
    line_allowed,
    line_mode,
    route_duration_breakdown,
    route_path_text_from_steps,
    route_step_details,
    route_text_from_steps,
    transfer_miss_probability,
    transfer_time_at_station,
)


class RingLineTests(unittest.TestCase):
    def test_metro_m1_ring_uses_two_opposing_directions(self):
        ring_name = "M1"
        self.assertEqual(LINES[ring_name]["mode"], "metro")

        sequences = line_direction_sequences(ring_name)
        self.assertEqual(len(sequences), 2)
        self.assertGreaterEqual(len(sequences[0]), 3)
        self.assertEqual(sequences[0][0], sequences[0][-1])
        self.assertEqual(sequences[0][:-1], list(reversed(sequences[1][:-1])))

    def test_same_line_through_station_has_no_intermediate_wait_or_transfer(self):
        self.assertEqual(transfer_time_at_station("Golf", "M1", "M1", 4), 0)
        self.assertEqual(transfer_time_at_station("Golf", None, "M1", 4), 0)

        ring_stops = LINES["M1"]["stops"]
        estimate = estimate_route_time(ring_stops[:3], 480, 4, allowed_modes=("metro",))
        self.assertIsNotNone(estimate)
        _, steps = estimate
        for previous, current in zip(steps, steps[1:]):
            if previous["line"] == current["line"]:
                self.assertEqual(current["transfer"], 0)
                self.assertEqual(current["wait"], 0)
        proposal = route_text_from_steps(steps)
        complete_path = route_path_text_from_steps(steps)
        self.assertRegex(proposal, r"metro line M\d+ from \w+ to \w+")
        self.assertIn("minutes", proposal)
        self.assertEqual(complete_path.count("-->"), 1)
        self.assertIn(f"--{steps[0]['mode']} {steps[0]['line']}", complete_path)
        self.assertTrue(complete_path.endswith(steps[-1]["to"]))
        if len(steps) > 1:
            self.assertIn(steps[0]["to"], complete_path)
        for step in route_step_details(steps):
            self.assertEqual(
                set(step),
                {"step_index", "from_station", "to_station", "line", "transport_type"},
            )
            self.assertTrue(step["from_station"])
            self.assertTrue(step["to_station"])
            self.assertTrue(step["line"])
            self.assertIn(step["transport_type"], {"metro", "tram", "bus", "walking"})

    def test_walking_leg_uses_minutes_and_named_stations_without_a_line(self):
        steps = [{
            "from": "Alpha", "to": "Bravo", "line": "Walking", "mode": "walking",
            "depart": 480, "arrive": 485, "wait": 0,
        }]

        proposal = route_text_from_steps(steps)
        complete_path = route_path_text_from_steps(steps)
        details = route_step_details(steps)

        self.assertIn("Walk 5 minutes from Alpha to Bravo", proposal)
        self.assertEqual(complete_path, "Alpha --walk 5 min--> Bravo")
        self.assertIsNone(details[0]["line"])
        self.assertEqual(details[0]["transport_type"], "walking")

    def test_complete_path_condenses_intermediate_stations_on_same_line(self):
        steps = [
            {"from": "Bravo", "to": "Charlie", "line": "T1", "mode": "tram", "depart": 480, "arrive": 484, "wait": 0},
            {"from": "Charlie", "to": "Delta", "line": "T1", "mode": "tram", "depart": 484, "arrive": 488, "wait": 0},
            {"from": "Delta", "to": "Gamma", "line": "T1", "mode": "tram", "depart": 488, "arrive": 492, "wait": 0},
        ]

        self.assertEqual(
            route_path_text_from_steps(steps),
            "Bravo --tram T1 (Charlie, Delta)--> Gamma",
        )

    def test_startup_constraint_route_is_available_for_proposal_comparison(self):
        scenario = {
            "start_station": "Alpha",
            "destination_station": "Echo",
            "start_time_min": 480,
            "transfer_time_min": 4,
        }
        persona = {
            "preferences": {
                "priority": "fast route",
                "switching": "avoid unnecessary line changes",
                "fullness": "dislikes very full train cars",
            }
        }

        target = optimal_constraint_route(scenario, persona)
        self.assertIsNotNone(target)
        self.assertEqual(target.route[0], "Alpha")
        self.assertEqual(target.route[-1], "Echo")

        estimate = estimate_route_time(target.route, 480, 4)
        self.assertIsNotNone(estimate)
        arrival, steps = estimate
        gap = route_constraint_gap(steps, arrival - 480, target)
        self.assertEqual(gap["duration_gap_min"], 0)
        self.assertEqual(gap["line_change_gap"], 0)
        self.assertEqual(gap["near_capacity_gap"], 0)

    def test_fullness_gap_is_binary_near_capacity_difference(self):
        target = ConstraintRoute(
            route=["Alpha", "Charlie"],
            steps=[],
            duration_min=10,
            line_sequence=["Ring"],
            line_change_count=0,
            average_fullness=20,
            near_capacity_count=0,
            has_near_capacity=False,
            delay_probability=0.0,
            transfer_miss_probability=0.0,
            score=(),
            label="avoid near capacity",
        )
        high_capacity_steps = [
            {"line": "Ring", "fullness": 86, "delay_probability": 0.0},
            {"line": "Ring", "fullness": 20, "delay_probability": 0.0},
        ]
        gap = route_constraint_gap(high_capacity_steps, target.duration_min, target)

        self.assertEqual(route_near_capacity_count(high_capacity_steps), 1)
        self.assertEqual(gap["near_capacity_gap"], 1)
        self.assertEqual(gap["fullness_gap"], 1)

    def test_all_required_transport_modes_are_present(self):
        service_labels = {data.get("mode") for data in LINES.values()}

        self.assertEqual(service_labels, {"metro", "tram", "bus", "walking"})

    def test_ticket_filter_blocks_unavailable_transport_modes(self):
        bus_line = next(name for name, data in LINES.items() if data["mode"] == "bus")

        self.assertFalse(line_allowed(bus_line, ("tram", "metro")))
        self.assertTrue(line_allowed(bus_line, ("tram", "bus")))
        self.assertEqual(line_mode(bus_line), "bus")

    def test_network_contains_no_pseudo_edges(self):
        station_modes = {}
        for station in ADJACENCY:
            station_modes[station] = {
                line_mode(line)
                for _next_station, line, _travel in ADJACENCY[station]
            }

        self.assertTrue(station_modes)
        self.assertTrue(all(modes.issubset({"metro", "tram", "bus", "walking"}) for modes in station_modes.values()))
        self.assertTrue(all("walking" in modes for modes in station_modes.values()))

    def test_risk_is_reported_as_general_class(self):
        self.assertEqual(probability_class(0.1), "low")
        self.assertEqual(probability_class(0.3), "medium")
        self.assertEqual(probability_class(0.6), "high")
        self.assertTrue(probability_class_allowed(0.3, 0.32))
        self.assertFalse(probability_class_allowed(0.6, 0.32))

    def test_station_specific_transfer_time_and_risk_apply_on_line_change(self):
        station_lines = [
            line_name for line_name, data in LINES.items()
            if line_name != "Walking" and "Golf" in data["stops"]
        ]
        self.assertGreaterEqual(len(station_lines), 2)
        previous_line, next_line = station_lines[:2]
        transfer = transfer_time_at_station("Golf", previous_line, next_line, 2)
        self.assertEqual(transfer, station_transfer_time_min("Golf"))
        self.assertGreater(transfer_miss_probability("Golf", next_line, transfer, 0, 480), 0.0)
        self.assertEqual(transfer_miss_probability("Golf", previous_line, 0, 0, 480), 0.0)
