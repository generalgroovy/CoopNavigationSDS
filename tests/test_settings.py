import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from coop_navigation_sds.app import (
    default_run_config,
    normalize_run_config,
    select_run_config,
    validate_run_config_for_start,
)
from coop_navigation_sds.Configuration.settings import load_run_settings, save_run_settings
from coop_navigation_sds.Configuration.schema import CONFIG_SCHEMA_VERSION


class RunSettingsTests(unittest.TestCase):
    def test_loading_legacy_ollama_settings_migrates_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "config": {
                    "model_provider": "ollama",
                    "model_name": "llama3.2:3b",
                    "model_timeout_sec": 5.0,
                },
            }), encoding="utf-8")
            loaded = load_run_settings({}, path)

        self.assertEqual(loaded["model_name"], "llama3.2:latest")
        self.assertEqual(loaded["model_timeout_sec"], 180.0)

    def test_legacy_ollama_model_name_is_migrated(self):
        normalized = normalize_run_config({
            "model_provider": "ollama",
            "model_name": "llama3.2:3b",
        })
        self.assertEqual(normalized["model_name"], "llama3.2:latest")
        self.assertTrue(normalized["model_service_autostart"])

    def test_normalization_repairs_blank_chattts_and_whisper_default_for_vosk(self):
        config = default_run_config()
        config.update({
            "agent_b_plugin": "simple",
            "tts_engine": "chattts",
            "asr_engine": "vosk",
            "tts_model": "",
            "asr_model": "small.en",
        })

        normalized = normalize_run_config(config)

        self.assertTrue(Path(normalized["tts_model"]).is_dir())
        self.assertTrue(Path(normalized["asr_model"]).is_dir())
        self.assertTrue(normalized["tts_model"].endswith("models\\chattts"))
        self.assertTrue(normalized["asr_model"].endswith("vosk-model-small-en-us-0.15"))

    def test_prepared_speech_paths_do_not_depend_on_working_directory(self):
        config = default_run_config()
        config.update({
            "agent_b_plugin": "simple",
            "tts_engine": "chattts",
            "asr_engine": "vosk",
            "tts_model": ".speech-providers/models/chattts",
            "asr_model": ".speech-providers/models/vosk/vosk-model-small-en-us-0.15",
        })
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                normalized = normalize_run_config(config)
            finally:
                os.chdir(previous)

        self.assertTrue(Path(normalized["tts_model"]).is_dir())
        self.assertTrue(Path(normalized["asr_model"]).is_dir())

    def test_runtime_model_downloads_are_explicitly_configurable(self):
        self.assertIn("allow_model_download", default_run_config())
        self.assertFalse(default_run_config()["allow_model_download"])
        self.assertNotIn("allow_tts_model_download", default_run_config())

    def test_save_and_load_wrapped_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            saved_path = save_run_settings(
                {
                    "num_turns": 9,
                    "tts_engine": "file",
                    "execution_run_dir": "temporary-run",
                },
                path,
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            loaded = load_run_settings({"num_turns": 5, "asr_engine": "sapi"}, path)

        self.assertEqual(saved_path, path)
        self.assertEqual(document["schema_version"], CONFIG_SCHEMA_VERSION)
        self.assertNotIn("execution_run_dir", document["config"])
        self.assertEqual(loaded["num_turns"], 9)
        self.assertEqual(loaded["asr_engine"], "sapi")

    def test_speech_repair_settings_are_scriptable_and_persistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            save_run_settings(
                {
                    "clarification_max_attempts": 3,
                    "asr_domain_normalization_enabled": False,
                    "asr_domain_similarity_threshold": 0.91,
                },
                path,
            )
            loaded = load_run_settings({}, path)

        self.assertEqual(loaded["clarification_max_attempts"], 3)
        self.assertFalse(loaded["asr_domain_normalization_enabled"])
        self.assertEqual(loaded["asr_domain_similarity_threshold"], 0.91)

    def test_speech_endpoint_settings_are_scriptable_and_persistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            save_run_settings(
                {
                    "asr_end_silence_ms": 2600,
                    "asr_ambiguous_end_silence_ms": 4800,
                    "max_utterance_sec": 22.0,
                },
                path,
            )
            loaded = load_run_settings({}, path)

        self.assertEqual(loaded["asr_end_silence_ms"], 2600)
        self.assertEqual(loaded["asr_ambiguous_end_silence_ms"], 4800)
        self.assertEqual(loaded["max_utterance_sec"], 22.0)

    def test_plain_json_object_is_valid_for_scripts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text(
                json.dumps({"persona_key": "distracted_multitasker", "num_turns": 4}),
                encoding="utf-8",
            )
            loaded = load_run_settings({}, path)

        self.assertEqual(loaded["persona_key"], "distracted_multitasker")
        self.assertEqual(loaded["num_turns"], 4)

    def test_interactive_config_loads_last_values_and_saves_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "last.json"
            save_run_settings({
                "num_turns": 11,
                "tts_engine": "file",
                "asr_engine": "file",
                "agent_b_plugin": "simple",
            }, path)
            captured = {}

            class FakeDialog:
                def __init__(self, _choices, defaults, validator=None):
                    captured["defaults"] = defaults
                    captured["validator"] = validator

                def show(self):
                    selected = dict(captured["defaults"])
                    selected["num_turns"] = 12
                    return captured["validator"](selected)

            with patch.dict(os.environ, {"MINILLAMA_SETTINGS_FILE": str(path)}):
                with patch("coop_navigation_sds.Configuration.gui.StartupConfigDialog", FakeDialog):
                    selected = select_run_config()

            reloaded = load_run_settings(default_run_config(), path)

        self.assertEqual(captured["defaults"]["num_turns"], 11)
        self.assertEqual(selected["num_turns"], 12)
        self.assertEqual(reloaded["num_turns"], 12)

    def test_invalid_json_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            path.write_text("{invalid", encoding="utf-8")
            loaded = load_run_settings({"num_turns": 7}, path)

        self.assertEqual(loaded, {"num_turns": 7})

    def test_saved_settings_exclude_legacy_audio_bloat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            save_run_settings(
                {
                    "tts_engine": "chattts",
                    "agent_a_audio_persona": "neutral_caller",
                    "agent_a_seed": 17,
                    "agent_a_laugh_level": 2,
                    "agent_a_reference_audio": "reference.wav",
                    "agent_a_custom_audio": True,
                },
                path,
            )
            saved = json.loads(path.read_text(encoding="utf-8"))["config"]

        self.assertEqual(saved["agent_a_seed"], 17)
        self.assertNotIn("agent_a_laugh_level", saved)
        self.assertNotIn("agent_a_reference_audio", saved)
        self.assertNotIn("agent_a_custom_audio", saved)

    def test_legacy_metric_settings_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            save_run_settings(
                {
                    "metric_config": {
                        "asr_wer": True,
                        "audio_missing_rate": False,
                    },
                },
                path,
            )
            saved = json.loads(path.read_text(encoding="utf-8"))["config"]

        self.assertNotIn("metric_config", saved)
        self.assertNotIn("metric_tiers", saved)

    def test_normalization_replaces_whisper_model_default_for_vosk(self):
        config = default_run_config()
        config.update({
            "agent_b_plugin": "simple",
            "tts_engine": "file",
            "asr_engine": "vosk",
            "asr_model": "small.en",
        })
        normalized = normalize_run_config(config)
        self.assertTrue(Path(normalized["asr_model"]).is_dir())
        self.assertTrue(normalized["asr_model"].endswith("vosk-model-small-en-us-0.15"))

    def test_preflight_rejects_provider_interpreter_without_vosk(self):
        config = default_run_config()
        config.update({
            "agent_b_plugin": "simple",
            "tts_engine": "file",
            "asr_engine": "vosk",
            "asr_model": "",
            "asr_python_executable": sys.executable,
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            config["protocol_log_dir"] = tmpdir
            with patch("importlib.util.find_spec", return_value=None), patch(
                "coop_navigation_sds.app.subprocess.run",
                return_value=Mock(returncode=1),
            ):
                with self.assertRaisesRegex(ValueError, "provider (cannot initialize|process cannot start)"):
                    validate_run_config_for_start(config)

    def test_preflight_rejects_empty_assets_and_incompatible_provider_runtime(self):
        for engine in ("faster_whisper", "qwen3_asr"):
            with self.subTest(engine=engine):
                with tempfile.TemporaryDirectory() as tmpdir:
                    model = Path(tmpdir) / engine
                    model.mkdir()
                    config = default_run_config()
                    config.update({
                        "agent_b_plugin": "simple",
                        "tts_engine": "file",
                        "asr_engine": engine,
                        "asr_model": str(model),
                        "asr_python_executable": sys.executable,
                    })
                    config["protocol_log_dir"] = tmpdir
                    with patch("importlib.util.find_spec", return_value=None), patch(
                        "coop_navigation_sds.app.subprocess.run",
                        return_value=Mock(returncode=1),
                    ):
                        with self.assertRaisesRegex(ValueError, "provider (cannot initialize|process cannot start)"):
                            validate_run_config_for_start(config)

    def test_preflight_accepts_manifest_registered_whisper_cpp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            executable = root / "whisper-cli.exe"
            executable.write_bytes(b"exe")
            model = root / "ggml-base.en.bin"
            model.write_bytes(b"model")
            (root / "providers.json").write_text(
                json.dumps({
                    "providers": {
                        "whisper_cpp": {
                            "executable": executable.name,
                            "model": model.name,
                        }
                    }
                }),
                encoding="utf-8",
            )
            config = default_run_config()
            config.update({
                "agent_b_plugin": "simple",
                "tts_engine": "file",
                "asr_engine": "whisper_cpp",
                "asr_model": "",
                "asr_executable": "",
                "provider_environment_dir": str(root),
                "protocol_log_dir": tmpdir,
            })

            normalized = validate_run_config_for_start(config)

        self.assertEqual(normalized["asr_engine"], "whisper_cpp")

    def test_preflight_accepts_deterministic_pipeline(self):
        config = default_run_config()
        config.update({
            "agent_b_plugin": "simple",
            "tts_engine": "file",
            "asr_engine": "file",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            config["protocol_log_dir"] = tmpdir
            normalized = validate_run_config_for_start(config)

        self.assertEqual(normalized["tts_engine"], "file")
        self.assertEqual(normalized["asr_engine"], "file")

    def test_chattts_preflight_ignores_obsolete_download_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "models" / "chattts"
            config = default_run_config()
            config.update({
                "agent_b_plugin": "simple",
                "tts_engine": "chattts",
                "tts_model": str(model_path),
                "tts_python_executable": sys.executable,
                "allow_model_download": True,
                "asr_engine": "file",
                "protocol_log_dir": tmpdir,
            })
            with self.assertRaisesRegex(ValueError, "ChatTTS assets are missing"):
                validate_run_config_for_start(config)

    def test_chattts_preflight_rejects_missing_assets_when_download_is_off(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = default_run_config()
            config.update({
                "agent_b_plugin": "simple",
                "tts_engine": "chattts",
                "tts_model": str(Path(tmpdir) / "missing-chattts"),
                "tts_python_executable": sys.executable,
                "allow_model_download": False,
                "asr_engine": "file",
                "protocol_log_dir": tmpdir,
            })
            with self.assertRaisesRegex(ValueError, "ChatTTS assets are missing"):
                validate_run_config_for_start(config)

    def test_dialog_validation_failure_keeps_window_open(self):
        from coop_navigation_sds.Configuration.gui import StartupConfigDialog

        dialog = StartupConfigDialog.__new__(StartupConfigDialog)
        dialog.vars = {"value": Mock(get=lambda: "invalid")}
        dialog.metric_vars = {}
        dialog.validator = Mock(side_effect=ValueError("invalid provider"))
        dialog.root = Mock()
        dialog.result = "old"

        with patch("coop_navigation_sds.Configuration.gui.messagebox.showerror") as error:
            dialog.start()

        dialog.root.destroy.assert_not_called()
        self.assertIsNone(dialog.result)
        error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
