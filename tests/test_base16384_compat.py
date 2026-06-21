import unittest

from coop_navigation_sds.TextToSpeech import base16384_compat


class Base16384CompatibilityTests(unittest.TestCase):
    def test_upstream_decode_vector(self):
        encoded = "嵞喇濡虸氞喇濡虸氞喇濡虸氞咶箭祫棚薇濡蘀㴆"
        self.assertEqual(
            base16384_compat.decode_from_string(encoded),
            b"=xxxxxxxxxxxxxxxxxxxxxxkkkkkkkxxxx",
        )

    def test_round_trip_all_remainder_lengths(self):
        for length in range(1, 64):
            with self.subTest(length=length):
                payload = bytes((index * 37) % 256 for index in range(length))
                encoded = base16384_compat.encode_to_string(payload)
                self.assertEqual(base16384_compat.decode_from_string(encoded), payload)


if __name__ == "__main__":
    unittest.main()
