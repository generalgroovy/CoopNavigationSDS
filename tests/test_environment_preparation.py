import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_test_environment


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentPreparationTests(unittest.TestCase):
    def test_manifest_covers_every_selectable_model_backend(self):
        manifest = json.loads(
            (ROOT / "coop_navigation_sds" / "Configuration" / "model_assets.json").read_text(
                encoding="utf-8"
            )
        )
        models = manifest["models"]
        self.assertEqual(
            set(models),
            {
                "tinyllama", "chattts", "piper", "coqui",
                "faster_whisper", "vosk", "whisper_cpp",
                "qwen3_asr", "sherpa_onnx",
            },
        )

    def test_check_mode_never_invokes_a_downloader(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            prepare_test_environment, "ROOT", Path(tmpdir)
        ), patch.object(
            prepare_test_environment, "READINESS_FILE", Path(tmpdir) / "readiness.json"
        ), patch.object(
            prepare_test_environment, "_prepare_asset"
        ) as downloader:
            report = prepare_test_environment.prepare(download=False)

        downloader.assert_not_called()
        self.assertFalse(report["ready"])

    def test_platform_entry_points_prepare_then_test(self):
        for name in ("prepare_windows_tests.ps1", "prepare_linux_tests.sh"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("prepare_test_environment.py", text)
            self.assertIn("run_speech_backend_matrix.py --live", text)


if __name__ == "__main__":
    unittest.main()
