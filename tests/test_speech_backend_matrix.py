import json
from pathlib import Path
import tempfile
import unittest

from coop_navigation_sds.ResultsAndArtifacts.speech_matrix import _pipeline_config, run_speech_backend_matrix
from coop_navigation_sds.Configuration.component_catalog import speech_engine_profile
from coop_navigation_sds.DialogManagement.speech_pipeline import ASR_ENGINE_SPECS, TTS_ENGINE_SPECS


class SpeechBackendMatrixTests(unittest.TestCase):
    def test_chattts_profile_allows_slow_cpu_synthesis(self):
        self.assertGreaterEqual(speech_engine_profile("tts", "chattts")["tts_timeout_sec"], 180)

    def test_engine_switch_replaces_incompatible_model_paths(self):
        config = _pipeline_config(
            {
                "tts_engine": "sapi",
                "asr_engine": "sapi",
                "tts_model": "stale-tts-model",
                "asr_model": "stale-asr-model",
            },
            "piper",
            "vosk",
            ".",
        )

        self.assertTrue(config.tts_model.endswith("en_US-lessac-medium.onnx"))
        self.assertTrue(config.asr_model.endswith("vosk-model-small-en-us-0.15"))

    def test_engine_switch_preserves_unchanged_backend_override(self):
        config = _pipeline_config(
            {
                "tts_engine": "piper",
                "asr_engine": "sapi",
                "tts_model": "custom-piper.onnx",
            },
            "piper",
            "vosk",
            ".",
        )

        self.assertEqual(config.tts_model, "custom-piper.onnx")
        self.assertTrue(config.asr_model.endswith("vosk-model-small-en-us-0.15"))

    def test_every_registered_tts_asr_combination_executes_and_is_protocolled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol, paths = run_speech_backend_matrix(tmpdir)

            expected_count = len(TTS_ENGINE_SPECS) * len(ASR_ENGINE_SPECS)
            self.assertEqual(protocol["summary"]["combination_count"], expected_count)
            self.assertEqual(protocol["summary"]["contract_passed"], expected_count)
            self.assertEqual(protocol["summary"]["contract_failed"], 0)
            self.assertEqual(
                len({(case["tts_engine"], case["asr_engine"]) for case in protocol["cases"]}),
                expected_count,
            )
            self.assertTrue(all(case["contract_turns"] == 2 for case in protocol["cases"]))
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))

            stored = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(stored["summary"], protocol["summary"])


if __name__ == "__main__":
    unittest.main()
