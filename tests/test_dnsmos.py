import unittest

from coop_navigation_sds.EvaluationMetrics.catalog import CORE_METRIC_KEYS, metric_metadata
from coop_navigation_sds.EvaluationMetrics.dnsmos import DNSMOSEvaluator


class DNSMOSTests(unittest.TestCase):
    def test_dnsmos_is_a_core_learned_mos_metric(self):
        self.assertIn("tts_dnsmos", CORE_METRIC_KEYS)
        metadata = metric_metadata("tts_dnsmos", "tts")
        self.assertEqual(metadata["class"], "L")
        self.assertEqual(metadata["tier"], "core")
        self.assertEqual(metadata["unit"], "mean_opinion_score_1_to_5")
        self.assertEqual(metadata["estimator"]["range"], [1.0, 5.0])
        self.assertFalse(metadata["estimator"]["personalized"])
        self.assertEqual(len(metadata["estimator"]["dimensions"]), 4)

    def test_evaluator_defaults_to_standard_non_personalized_mode(self):
        evaluator = DNSMOSEvaluator()
        report = evaluator.evaluate([
            {"path": "missing.wav", "speaker": "Agent A", "turn_index": 1}
        ])
        self.assertFalse(evaluator.personalized)
        self.assertEqual(report["requested_file_count"], 1)
        self.assertFalse(report["personalized"])
        self.assertIsNone(report["onnxruntime_version"])
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["reason"], "no_readable_wav_artifacts")
        self.assertIsNone(report["score"])
        self.assertEqual(report["by_agent"], {})


if __name__ == "__main__":
    unittest.main()
