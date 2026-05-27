import unittest
from collections import Counter, defaultdict

from minillama.model.config import STATION_CLASS_ACCESS_MODES, STATION_CLASS_RATIOS
from minillama.model.metro_data import ADJACENCY, LINES, STATION_CLASSES, STATIONS, TRAVEL_TIMES, line_segment_key
from minillama.model.route_planner import optimal_time_route


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


if __name__ == "__main__":
    unittest.main()
