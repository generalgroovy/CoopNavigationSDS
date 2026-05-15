import unittest

from minillama.model.metro_data import LINES
from minillama.model.route_planner import line_direction_sequences


class RingLineTests(unittest.TestCase):
    def test_ring_line_uses_one_canonical_sequence(self):
        ring_name = next((name for name, data in LINES.items() if data.get("kind") == "Ring"), None)
        self.assertIsNotNone(ring_name)

        sequences = line_direction_sequences(ring_name)
        self.assertEqual(len(sequences), 1)
        self.assertGreaterEqual(len(sequences[0]), 3)
        self.assertEqual(sequences[0][0], sequences[0][-1])
