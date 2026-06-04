import unittest
from collections import Counter, defaultdict

from minillama.network.config import STATION_CLASS_ACCESS_MODES, STATION_CLASS_RATIOS
from minillama.network.metro_data import ADJACENCY, LINES, STATION_CLASSES, STATIONS, TRAVEL_TIMES, line_segment_key
from minillama.network.route_constraints import optimal_constraint_route
from minillama.network.route_planner import optimal_time_route
from minillama.caller.config import PERSONAS
from minillama.scenarios import DEFAULT_TEST_CASE, get_test_case


class NetworkStructureTests(unittest.TestCase):
    def test_station_classes_match_access_rules_and_ratios(self):
        counts = Counter(STATION_CLASSES.values())
        total = len(STATIONS)

        self.assertEqual(set(counts), {1, 2, 3})
        for station_class, ratio in STATION_CLASS_RATIOS.items():
            self.assertLessEqual(abs((counts[station_class] / total) - ratio), 0.08)

        served_modes = defaultdict(set)
        for _line_name, data in LINES.items():
            mode = data.get("mode")
            for station in data["stops"]:
                station_class = STATION_CLASSES[station]
                self.assertIn(mode, STATION_CLASS_ACCESS_MODES[station_class])
                served_modes[station_class].add(mode)

        self.assertIn("metro", served_modes[1])
        self.assertIn("tram", served_modes[1])
        self.assertIn("bus", served_modes[1])
        self.assertNotIn("metro", served_modes[2])
        self.assertIn("tram", served_modes[2])
        self.assertEqual(served_modes[3], {"bus"})

    def test_walking_links_exist_but_transport_network_remains_connected(self):
        walking_edges = [
            (station, neighbor, line_name)
            for station, links in ADJACENCY.items()
            for neighbor, line_name, _minutes in links
            if line_name.startswith("Walk ")
        ]

        self.assertGreater(len(walking_edges), 0)
        for start in STATIONS[:5]:
            for destination in STATIONS[-5:]:
                arrival, steps = optimal_time_route(start, destination, 8 * 60, 2)
                self.assertIsNotNone(arrival)
                self.assertTrue(steps)

    def test_line_travel_times_are_line_specific_and_have_mode_scaling(self):
        sampled = {}
        for line_name, data in LINES.items():
            if len(data["stops"]) < 2:
                continue
            a, b = data["stops"][0], data["stops"][1]
            sampled.setdefault(data.get("mode"), TRAVEL_TIMES[line_segment_key(line_name, a, b)])

        self.assertIn("metro", sampled)
        self.assertIn("tram", sampled)
        self.assertIn("bus", sampled)
        self.assertLessEqual(sampled["metro"], sampled["bus"])
        self.assertLessEqual(sampled["tram"], sampled["bus"])

    def test_bus_lines_cover_more_stations_than_tram_and_metro(self):
        by_mode = defaultdict(list)
        for data in LINES.values():
            by_mode[data.get("mode")].append(len(data["stops"]))

        bus_average = sum(by_mode["bus"]) / len(by_mode["bus"])
        tram_average = sum(by_mode["tram"]) / len(by_mode["tram"])
        metro_average = sum(by_mode["metro"]) / len(by_mode["metro"])

        self.assertGreater(bus_average, tram_average)
        self.assertGreater(tram_average, metro_average)

    def test_default_scenario_has_constraint_viable_routes_for_research_personas(self):
        base_case = get_test_case(DEFAULT_TEST_CASE)
        for persona_key in ("focused_commuter", "crowd_averse_rider", "delay_sensitive_traveler"):
            route = optimal_constraint_route(base_case.scenario, PERSONAS[persona_key])
            self.assertIsNotNone(route, persona_key)
            self.assertEqual(route.route[0], base_case.scenario["start_station"])
            self.assertEqual(route.route[-1], base_case.scenario["destination_station"])


if __name__ == "__main__":
    unittest.main()
