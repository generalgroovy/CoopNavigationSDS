import unittest

from minillama.network.metro_data import LINES, ADJACENCY, station_transfer_time_min
from minillama.network.route_constraints import (
    ConstraintRoute,
    nearby_walking_links,
    optimal_constraint_route,
    probability_class,
    probability_class_allowed,
    route_constraint_gap,
    route_near_capacity_count,
)
from minillama.network.route_planner import (
    estimate_route_time,
    line_direction_sequences,
    line_mode,
    route_duration_breakdown,
    route_text_from_steps,
)


class RingLineTests(unittest.TestCase):
    def test_ring_line_uses_one_canonical_sequence(self):
        ring_name = next((name for name, data in LINES.items() if data.get("kind") == "Ring"), None)
        self.assertIsNotNone(ring_name)

        sequences = line_direction_sequences(ring_name)
        self.assertEqual(len(sequences), 2)
        self.assertGreaterEqual(len(sequences[0]), 3)
        self.assertEqual(sequences[0][0], sequences[0][-1])
        self.assertEqual(sequences[1][0], sequences[1][-1])

    def test_same_line_through_station_has_no_intermediate_wait_or_transfer(self):
        estimate = estimate_route_time(["Alpha", "Bravo", "Ivy"], 480, 4)
        self.assertIsNotNone(estimate)
        _, steps = estimate

        self.assertEqual([step["line"] for step in steps], ["Ring", "Ring"])
        self.assertEqual(steps[1]["depart"], steps[0]["arrive"])
        self.assertEqual(steps[1]["wait"], 0)
        self.assertEqual(steps[1]["transfer"], 0)
        self.assertEqual(route_duration_breakdown(steps)["transfer"], 0)

        proposal = route_text_from_steps(steps)
        self.assertEqual(proposal.count("Take Ring"), 1)
        self.assertIn("Take Ring from Alpha to Ivy", proposal)
        self.assertIn("It takes", proposal)

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
            mode_sequence=["metro"],
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

    def test_lines_have_transport_modes_for_ticket_constraints(self):
        modes = {data.get("mode") for data in LINES.values()}

        self.assertIn("metro", modes)
        self.assertIn("tram", modes)
        self.assertIn("bus", modes)

    def test_ticket_mode_filter_blocks_disallowed_mode(self):
        estimate = estimate_route_time(["Alpha", "Hotel"], 480, 2, allowed_modes=("metro",))

        self.assertIsNone(estimate)
        self.assertEqual(line_mode("Diagonal-SE-1"), "bus")

    def test_network_contains_bus_only_stations(self):
        station_modes = {}
        for station in ADJACENCY:
            station_modes[station] = {
                line_mode(line)
                for _next_station, line, _travel in ADJACENCY[station]
            }

        bus_only = [station for station, modes in station_modes.items() if modes - {"walking"} == {"bus"}]

        self.assertTrue(bus_only)

    def test_risk_is_reported_as_general_class(self):
        self.assertEqual(probability_class(0.1), "low")
        self.assertEqual(probability_class(0.3), "medium")
        self.assertEqual(probability_class(0.6), "high")
        self.assertTrue(probability_class_allowed(0.3, 0.32))
        self.assertFalse(probability_class_allowed(0.6, 0.32))

    def test_walking_links_are_available_for_persona_constraints(self):
        links = nearby_walking_links(max_minutes=8, limit=4)

        self.assertTrue(links)
        self.assertLessEqual(max(link[0] for link in links), 8)

    def test_station_specific_transfer_time_and_risk_apply_on_line_change(self):
        estimate = estimate_route_time(["Alpha", "Golf", "November"], 480, 2)
        self.assertIsNotNone(estimate)
        _, steps = estimate

        self.assertEqual(steps[1]["transfer"], station_transfer_time_min("Golf"))
        self.assertGreater(steps[1]["transfer_miss_probability"], 0.0)
        self.assertEqual(steps[0]["transfer_miss_probability"], 0.0)
