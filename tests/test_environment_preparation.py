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
                "tinyllama", "chattts", "piper",
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

    def test_fail_fast_records_partial_readiness_and_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            prepare_test_environment, "ROOT", Path(tmpdir)
        ), patch.object(
            prepare_test_environment, "READINESS_FILE", Path(tmpdir) / "readiness.json"
        ), patch.object(
            prepare_test_environment.subprocess, "run", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                prepare_test_environment.prepare(download=True, fail_fast=True, show_progress=False)
            report = json.loads((Path(tmpdir) / "readiness.json").read_text(encoding="utf-8"))
            self.assertFalse(report["ready"])
            first = next(iter(report["models"].values()))
            self.assertIn("boom", first["error"])

    def test_platform_entry_points_prepare_then_test(self):
        for name in ("prepare_windows_tests.ps1", "prepare_linux_tests.sh"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("prepare_test_environment.py", text)
            self.assertIn("run_speech_backend_matrix.py --live", text)

    def test_cluster_userlm_script_is_fail_fast_and_sorted_by_tier(self):
        text = (ROOT / "scripts" / "cluster_userlm_agent_b_full_coverage.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("set -Eeuo pipefail", text)
        self.assertIn("prepare_test_environment.py", text)
        self.assertIn("--fail-fast", text)
        self.assertIn("userlm_transformers_speech_grid", text)
        self.assertIn('SELECTED_TIERS="${SELECTED_TIERS:-small medium large}"', text)
        self.assertIn("preview-small", text)
        self.assertIn("submit-large", text)
        self.assertIn("--provider transformers", text)
        self.assertIn("submit_agent_b_model_jobs.py", text)


if __name__ == "__main__":
    unittest.main()
