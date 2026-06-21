import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from coop_navigation_sds.DialogManagement.speech_pipeline import SpeechPipelineConfig, SpeechTransport
from coop_navigation_sds.DialogManagement.provider_runtime import (
    IsolatedSpeechToText,
    IsolatedTextToSpeech,
    _worker_config,
    resolve_provider_python,
)
from coop_navigation_sds.DialogManagement.whisper_cpp_runtime import whisper_cpp_ready
from coop_navigation_sds.TextToSpeech.setup import PROVIDER_PROFILES


class ProviderRuntimeTests(unittest.TestCase):
    def test_tts_worker_never_owns_playback_or_realtime_waiting(self):
        config = SpeechPipelineConfig(playback_enabled=True, realtime_enabled=True)
        worker = _worker_config(config, "tts", "chattts")
        self.assertFalse(worker["playback_enabled"])
        self.assertFalse(worker["realtime_enabled"])

    def test_isolated_tts_plays_worker_audio_in_application_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav = Path(tmpdir) / "speech.wav"
            wav.write_bytes(b"RIFF" + b"\0" * 64)
            config = SpeechPipelineConfig(
                audio_dir=tmpdir,
                playback_enabled=True,
                realtime_enabled=True,
            )
            tts = IsolatedTextToSpeech("chattts", sys.executable, config)
            tts.client.request = lambda _payload: {
                "text": "Take metro line M1 to Harbor.",
                "audio": {"path": str(wav), "duration_sec": 1.0},
                "diagnostics": {},
            }
            with patch(
                "coop_navigation_sds.DialogManagement.speech_pipeline.WaveFileTextToSpeech._play_wave",
                return_value=True,
            ) as playback:
                signal = tts.synthesize("Agent B", "Take metro line M1 to Harbor.")

        playback.assert_called_once()
        self.assertTrue(signal.audio["played"])
        self.assertTrue(signal.audio["waited"])

    def test_manifest_resolves_shared_profile_interpreter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            environment = root / "standard"
            python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            (root / "providers.json").write_text(
                json.dumps({
                    "providers": {
                        "chattts": {
                            "python": str(python.relative_to(root)),
                            "profile": "standard",
                        }
                    }
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_provider_python("chattts", environment_dir=root),
                python.resolve(),
            )

    def test_explicit_provider_python_selects_isolated_adapter(self):
        config = SpeechPipelineConfig(
            tts_engine="chattts",
            asr_engine="file",
            tts_python_executable=sys.executable,
            playback_enabled=False,
            realtime_enabled=False,
        )
        transport = SpeechTransport(config=config)
        self.assertIsInstance(transport.tts_engine, IsolatedTextToSpeech)

    def test_manifest_provider_python_selects_isolated_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "providers.json").write_text(
                json.dumps({"providers": {"qwen3_asr": {"python": sys.executable}}}),
                encoding="utf-8",
            )
            config = SpeechPipelineConfig(
                tts_engine="file",
                asr_engine="qwen3_asr",
                provider_environment_dir=str(root),
                playback_enabled=False,
                realtime_enabled=False,
            )
            transport = SpeechTransport(config=config)

        self.assertIsInstance(transport.asr_engine, IsolatedSpeechToText)

    def test_persistent_worker_round_trip_uses_real_wave_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpeechPipelineConfig(
                tts_engine="file",
                asr_engine="file",
                audio_dir=tmpdir,
                playback_enabled=False,
                realtime_enabled=False,
                tts_timeout_sec=30,
                asr_timeout_sec=30,
            )
            tts = IsolatedTextToSpeech("file", sys.executable, config)
            asr = IsolatedSpeechToText("file", sys.executable, config)
            try:
                signal = tts.synthesize("Agent A", "Take metro line M1 to Harbor.")
                transcript = asr.transcribe(signal)
            finally:
                tts.client.close()
                asr.client.close()
            self.assertEqual(transcript, "Take metro line M1 to Harbor.")
            self.assertTrue(Path(signal.audio["path"]).is_file())
            self.assertTrue((Path(tmpdir) / "provider_tts_file.log").is_file())
            self.assertTrue((Path(tmpdir) / "provider_asr_file.log").is_file())

    def test_supported_providers_share_the_python314_profile(self):
        self.assertEqual(tuple(PROVIDER_PROFILES), ("python314",))
        configured = {
            engine
            for profile in PROVIDER_PROFILES.values()
            for engine in profile.engines
        }
        self.assertEqual(
            configured,
            {"chattts", "piper", "faster_whisper", "vosk", "qwen3_asr"},
        )

    def test_whisper_cpp_readiness_uses_provider_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            executable = root / "whisper-cli.exe"
            model = root / "ggml-base.en.bin"
            executable.write_bytes(b"exe")
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

            ready, message, resolved = whisper_cpp_ready(environment_dir=root)

        self.assertTrue(ready)
        self.assertIn("resolved", message)
        self.assertEqual(resolved["executable"], str(executable.resolve()))
        self.assertEqual(resolved["model"], str(model.resolve()))

    def test_whisper_cpp_readiness_rejects_unrelated_main_path_lookup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unrelated = root / "main.cpl"
            unrelated.write_bytes(b"not whisper")
            model = root / "ggml-base.en.bin"
            model.write_bytes(b"model")

            def fake_which(name):
                return str(unrelated) if name == "main" else None

            with patch("coop_navigation_sds.DialogManagement.whisper_cpp_runtime.shutil.which", side_effect=fake_which):
                ready, _message, resolved = whisper_cpp_ready(
                    model=str(model),
                    environment_dir=root / "missing",
                )

        self.assertFalse(ready)
        self.assertEqual(resolved["executable"], "")


if __name__ == "__main__":
    unittest.main()
