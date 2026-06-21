import unittest
import re

from coop_navigation_sds.TransportNetwork.network import (
    ADJACENCY,
    LINES,
    PUBLIC_TRANSPORT_MODES,
    STATIONS,
    STATION_PUBLIC_MODES,
    TRAVEL_TIMES,
    line_segment_key,
)
from coop_navigation_sds.TransportNetwork.constraints import optimal_constraint_route
from coop_navigation_sds.TransportNetwork.routes import optimal_time_route
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import PERSONAS
from coop_navigation_sds.TransportNetwork import DEFAULT_TEST_CASE, get_test_case


class NetworkStructureTests(unittest.TestCase):
    def test_public_line_names_encode_transport_type(self):
        limits = {"metro": 20, "tram": 25, "bus": 30}
        prefixes = {"metro": "M", "tram": "T", "bus": "B"}
        for line_name, data in LINES.items():
            mode = data.get("mode")
            if mode not in prefixes:
                continue
            match = re.fullmatch(rf"{prefixes[mode]}(\d+)", line_name)
            self.assertIsNotNone(match)
            self.assertLessEqual(int(match.group(1)), limits[mode])
    def test_every_station_has_exactly_two_public_modes_plus_walking(self):
        served = {station for data in LINES.values() for station in data["stops"]}

        self.assertEqual(set(STATIONS), served)
        self.assertEqual({data.get("mode") for data in LINES.values()}, {"metro", "tram", "bus", "walking"})
        for station in STATIONS:
            self.assertEqual(len(STATION_PUBLIC_MODES[station]), 2)
            self.assertTrue(set(STATION_PUBLIC_MODES[station]).issubset(PUBLIC_TRANSPORT_MODES))
            self.assertTrue(any(LINES[line]["mode"] == "walking" for _next, line, _minutes in ADJACENCY[station]))
        coverage = {
            mode: sum(mode in STATION_PUBLIC_MODES[station] for station in STATIONS)
            for mode in PUBLIC_TRANSPORT_MODES
        }
        self.assertGreaterEqual(coverage["bus"], coverage["tram"])
        self.assertGreater(coverage["tram"], coverage["metro"])

    def test_no_pseudo_links_exist_and_network_remains_connected(self):
        pseudo_edges = [
            (station, neighbor, line_name)
            for station, links in ADJACENCY.items()
            for neighbor, line_name, _minutes in links
            if line_name.startswith("Walk ")
        ]

        self.assertEqual(pseudo_edges, [])
        for start in STATIONS[:5]:
            for destination in STATIONS[-5:]:
                arrival, steps = optimal_time_route(start, destination, 8 * 60, 2)
                self.assertIsNotNone(arrival)
                self.assertTrue(steps)

    def test_line_travel_times_are_line_specific(self):
        sampled = {}
        for line_name, data in LINES.items():
            if len(data["stops"]) < 2:
                continue
            a, b = data["stops"][0], data["stops"][1]
            sampled[line_name] = TRAVEL_TIMES[line_segment_key(line_name, a, b)]

        self.assertGreaterEqual(len(sampled), 3)
        self.assertTrue(all(minutes >= 1 for minutes in sampled.values()))

    def test_generated_lines_have_varied_station_coverage(self):
        stop_counts = [len(data["stops"]) for data in LINES.values()]

        self.assertGreater(max(stop_counts), min(stop_counts))

    def test_default_scenario_has_constraint_viable_routes_for_research_personas(self):
        base_case = get_test_case(DEFAULT_TEST_CASE)
        for persona_key in ("focused_commuter", "crowd_averse_rider", "delay_sensitive_traveler"):
            route = optimal_constraint_route(base_case.scenario, PERSONAS[persona_key])
            self.assertIsNotNone(route, persona_key)
            self.assertEqual(route.route[0], base_case.scenario["start_station"])
            self.assertEqual(route.route[-1], base_case.scenario["destination_station"])


if __name__ == "__main__":
    unittest.main()
