import struct
import tempfile
from pathlib import Path
import unittest
import wave

from coop_navigation_sds.EvaluationMetrics.catalog import CORE_METRIC_KEYS, metric_metadata
from coop_navigation_sds.EvaluationMetrics.nisqa import NISQAEvaluator, read_mono_pcm_wave


class NISQATests(unittest.TestCase):
    def test_nisqa_is_a_core_learned_mos_metric(self):
        self.assertIn("tts_nisqa", CORE_METRIC_KEYS)
        metadata = metric_metadata("tts_nisqa", "tts")
        self.assertEqual(metadata["class"], "L")
        self.assertEqual(metadata["tier"], "core")
        self.assertEqual(metadata["unit"], "mean_opinion_score_1_to_5")
        self.assertEqual(metadata["estimator"]["range"], [1.0, 5.0])

    def test_pcm_wave_reader_downmixes_stereo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stereo.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(struct.pack("<hhhh", 16384, -16384, 8192, 8192))

            samples, sample_rate = read_mono_pcm_wave(path)

            self.assertEqual(sample_rate, 16000)
            self.assertAlmostEqual(samples[0], 0.0)
            self.assertAlmostEqual(samples[1], 0.25)

    def test_evaluator_reports_missing_audio_without_loading_dependencies(self):
        report = NISQAEvaluator().evaluate(["missing.wav"])
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["reason"], "no_readable_wav_artifacts")
        self.assertIsNone(report["score"])

    def test_evaluator_accepts_context_records_for_agent_analysis(self):
        report = NISQAEvaluator().evaluate([
            {"path": "missing.wav", "speaker": "Agent A", "turn_index": 1}
        ])
        self.assertEqual(report["requested_file_count"], 1)
        self.assertEqual(report["by_agent"], {})


if __name__ == "__main__":
    unittest.main()
