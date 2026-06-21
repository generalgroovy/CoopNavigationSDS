import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from coop_navigation_sds.Configuration.speech import speech_pattern_keys
from coop_navigation_sds.DialogManagement.speech_pipeline import (
    ChatTTSTextToSpeech,
    CoquiTextToSpeech,
    EspeakNgTextToSpeech,
    F5TextToSpeech,
    FasterWhisperSpeechToText,
    KokoroTextToSpeech,
    MeloTextToSpeech,
    ParakeetSpeechToText,
    PatternedSpeechToText,
    PiperTextToSpeech,
    Qwen3SpeechToText,
    Qwen3TextToSpeech,
    SherpaOnnxSpeechToText,
    SpeechPipelineConfig,
    SpeechPipelineError,
    SpeechSignal,
    SpeechTransport,
    VoskSpeechToText,
    WaveFileTextToSpeech,
    WhisperCppSpeechToText,
    ASR_ENGINE_SPECS,
    TTS_ENGINE_SPECS,
    WindowsSapiSpeechToText,
    WindowsSapiTextToSpeech,
    available_asr_engine_keys,
    available_tts_engine_keys,
    normalize_text_for_speech,
    platform_default_asr_engine,
    platform_default_tts_engine,
    synthesis_controls,
)


class SpeechPipelineTests(unittest.TestCase):
    def file_transport(self, tmpdir, **overrides):
        values = {
            "tts_engine": "file",
            "asr_engine": "file",
            "audio_dir": tmpdir,
            "playback_enabled": False,
            "realtime_enabled": False,
        }
        values.update(overrides)
        return SpeechTransport(config=SpeechPipelineConfig(**values))

    def test_selectable_engines_are_explicit_and_exclude_text_bypasses(self):
        self.assertEqual(
            available_tts_engine_keys(),
            (
                "sapi",
                "chattts",
                "piper",
                "espeak_ng",
                "coqui",
                "file",
            ),
        )
        self.assertEqual(
            available_asr_engine_keys(),
            (
                "sapi",
                "faster_whisper",
                "vosk",
                "whisper_cpp",
                "qwen3_asr",
                "sherpa_onnx",
                "file",
            ),
        )
        self.assertNotIn("loopback", available_tts_engine_keys())

    def test_speech_engine_registry_is_the_single_source_of_selectable_backends(self):
        self.assertEqual(available_tts_engine_keys(), tuple(TTS_ENGINE_SPECS))
        self.assertEqual(available_asr_engine_keys(), tuple(ASR_ENGINE_SPECS))
        self.assertTrue(all(spec.kind == "tts" for spec in TTS_ENGINE_SPECS.values()))
        self.assertTrue(all(spec.kind == "asr" for spec in ASR_ENGINE_SPECS.values()))
        self.assertTrue(all(spec.label and spec.description for spec in TTS_ENGINE_SPECS.values()))
        self.assertTrue(all(spec.label and spec.description for spec in ASR_ENGINE_SPECS.values()))

    def test_optional_engines_are_selected_without_eager_model_loading(self):
        cases = (
            ("chattts", "file", ChatTTSTextToSpeech),
            ("piper", "file", PiperTextToSpeech),
            ("espeak_ng", "file", EspeakNgTextToSpeech),
            ("coqui", "file", CoquiTextToSpeech),
            ("file", "faster_whisper", FasterWhisperSpeechToText),
            ("file", "vosk", VoskSpeechToText),
            ("file", "whisper_cpp", WhisperCppSpeechToText),
            ("file", "qwen3_asr", Qwen3SpeechToText),
            ("file", "sherpa_onnx", SherpaOnnxSpeechToText),
        )
        for tts_key, asr_key, expected_type in cases:
            with self.subTest(tts=tts_key, asr=asr_key):
                transport = SpeechTransport(
                    config=SpeechPipelineConfig(
                        tts_engine=tts_key,
                        asr_engine=asr_key,
                        provider_environment_dir="__no_provider_runtime_for_test__",
                        playback_enabled=False,
                        realtime_enabled=False,
                    )
                )
                stage = transport.asr_engine if tts_key == "file" else transport.tts_engine
                self.assertIsInstance(stage, expected_type)

    def test_espeak_ng_builds_a_bounded_cross_platform_command(self):
        engine = EspeakNgTextToSpeech(SpeechPipelineConfig(tts_executable="espeak-ng"))
        prosody = {
            "words_per_minute": 145,
            "volume": 80,
            "pitch_semitones": 2,
            "voice": "en-us",
            "language": "en-US",
        }
        with patch("coop_navigation_sds.DialogManagement.speech_pipeline.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0, stderr="")
            metadata = engine._synthesize_wave(Path("out.wav"), "Take the metro.", prosody)

        command = run.call_args.args[0]
        self.assertEqual(command[0], "espeak-ng")
        self.assertIn("-w", command)
        self.assertIn("145", command)
        self.assertEqual(metadata["voice"], "en-us")

    def test_coqui_loads_lazily_and_writes_through_common_api(self):
        calls = []

        class FakeCoqui:
            speakers = ["speaker-one"]

            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def tts_to_file(self, **kwargs):
                calls.append(("write", kwargs))

        with patch.dict(sys.modules, {
            "TTS": SimpleNamespace(),
            "TTS.api": SimpleNamespace(TTS=FakeCoqui),
        }), tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").touch()
            (Path(tmpdir) / "model_file.pth").touch()
            engine = CoquiTextToSpeech(SpeechPipelineConfig(tts_model=tmpdir, tts_device="cpu"))
            metadata = engine._synthesize_wave(
                Path("out.wav"),
                "Take the tram.",
                {"voice": "", "language": "en-US"},
            )

        self.assertEqual(calls[0][1]["model_path"], str(Path(tmpdir) / "model_file.pth"))
        self.assertEqual(calls[0][1]["config_path"], str(Path(tmpdir) / "config.json"))
        self.assertEqual(calls[1][1]["speaker"], "speaker-one")
        self.assertEqual(metadata["model"], tmpdir)

    def test_sherpa_onnx_detects_transducer_model_layout(self):
        recognizer = object()

        class FakeOfflineRecognizer:
            @classmethod
            def from_transducer(cls, **kwargs):
                self.assertTrue(kwargs["encoder"].endswith("encoder.onnx"))
                self.assertTrue(kwargs["tokens"].endswith("tokens.txt"))
                return recognizer

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"):
                (root / name).touch()
            with patch.dict(sys.modules, {
                "sherpa_onnx": SimpleNamespace(OfflineRecognizer=FakeOfflineRecognizer),
            }):
                engine = SherpaOnnxSpeechToText(str(root))
                loaded = engine._load_model()

        self.assertIs(loaded, recognizer)
        self.assertEqual(engine._model_type, "transducer")

    def test_chattts_loads_prepared_local_weights_without_compilation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "chattts"
            model_path.mkdir()
            config = SpeechPipelineConfig(
                tts_engine="chattts",
                asr_engine="file",
                tts_model=str(model_path),
                playback_enabled=False,
                realtime_enabled=False,
            )
            engine = ChatTTSTextToSpeech(config)
            fake_chat = SimpleNamespace()
            fake_chat.load = unittest.mock.Mock(return_value=True)
            fake_module = SimpleNamespace(Chat=unittest.mock.Mock(return_value=fake_chat))
            with patch.dict(sys.modules, {"ChatTTS": fake_module}):
                self.assertIs(engine._model(), fake_chat)
        fake_chat.load.assert_called_once_with(
            source="custom",
            custom_path=str(model_path),
            compile=False,
        )

    def test_chattts_never_downloads_when_download_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "missing"
            engine = ChatTTSTextToSpeech(SpeechPipelineConfig(
                tts_engine="chattts",
                asr_engine="file",
                tts_model=str(model_path),
            ))
            fake_chat = SimpleNamespace(load=unittest.mock.Mock(return_value=False))
            fake_module = SimpleNamespace(Chat=unittest.mock.Mock(return_value=fake_chat))
            with patch.dict(sys.modules, {"ChatTTS": fake_module}):
                with self.assertRaisesRegex(SpeechPipelineError, "model loading failed"):
                    engine._model()
            fake_chat.load.assert_not_called()

    def test_chattts_reports_nested_missing_dependency(self):
        engine = ChatTTSTextToSpeech(SpeechPipelineConfig(
            tts_engine="chattts",
            asr_engine="file",
        ))
        missing = ModuleNotFoundError(
            "No module named 'pybase16384.backends.cython._core'",
            name="pybase16384.backends.cython._core",
        )
        with patch.object(engine, "_ensure_base16384", side_effect=missing):
            with self.assertRaisesRegex(
                SpeechPipelineError,
                "cannot import required module",
            ) as raised:
                engine._import_chattts()
        self.assertEqual(
            raised.exception.diagnostics["missing_module"],
            "pybase16384.backends.cython._core",
        )
        self.assertIn("Python 3.14", raised.exception.diagnostics["troubleshooting"])

    def test_piper_requires_explicit_voice_model(self):
        engine = PiperTextToSpeech(
            SpeechPipelineConfig(tts_engine="piper", asr_engine="file", tts_model="")
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            engine.audio_dir = Path(tmpdir)
            with self.assertRaisesRegex(SpeechPipelineError, "readable ONNX voice"):
                engine.synthesize("Agent A", "Piper test.")

    def test_piper_calls_official_synthesis_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "voice.onnx"
            model_path.write_bytes(b"model")
            config = SpeechPipelineConfig(
                tts_engine="piper",
                asr_engine="file",
                audio_dir=tmpdir,
                tts_model=str(model_path),
                agent_a_custom_audio=True,
                agent_a_speed=0.8,
            )
            engine = PiperTextToSpeech(config)
            fake_voice = SimpleNamespace()

            def synthesize_wav(_text, wav_file, syn_config):
                fake_voice.syn_config = syn_config
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\0\0" * 1600)

            fake_voice.synthesize_wav = synthesize_wav
            fake_piper = SimpleNamespace(
                SynthesisConfig=lambda **values: SimpleNamespace(**values)
            )
            with (
                patch.object(engine, "_model", return_value=fake_voice),
                patch.dict(sys.modules, {"piper": fake_piper}),
            ):
                signal = engine.synthesize("Agent A", "Piper test.")
            self.assertEqual(signal.audio["engine"], "piper")
            self.assertEqual(fake_voice.syn_config.length_scale, 1.25)

    def test_kokoro_combines_pipeline_audio_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            config = SpeechPipelineConfig(
                tts_engine="kokoro",
                asr_engine="file",
                audio_dir=tmpdir,
            )
            engine = KokoroTextToSpeech(config)

            def pipeline(_text, voice, speed):
                self.assertEqual(voice, "af_heart")
                self.assertGreater(speed, 0)
                yield "one", "one", np.ones(1200, dtype=np.float32) * 0.1
                yield "two", "two", np.ones(800, dtype=np.float32) * 0.1

            with patch.object(engine, "_pipeline", return_value=pipeline):
                signal = engine.synthesize("Agent A", "Kokoro test.")
            self.assertEqual(signal.audio["engine"], "kokoro")
            self.assertEqual(signal.diagnostics["sample_rate"], 24000)

    def test_qwen3_tts_selects_checkpoint_for_speaker_or_clone_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            reference = Path(tmpdir) / "reference.wav"
            reference.write_bytes(b"reference")
            models = []

            class FakeModel:
                def generate_custom_voice(self, **_kwargs):
                    return [np.ones(1000, dtype=np.float32) * 0.1], 24000

                def generate_voice_clone(self, **_kwargs):
                    return [np.ones(1000, dtype=np.float32) * 0.1], 24000

            speaker_engine = Qwen3TextToSpeech(SpeechPipelineConfig(
                tts_engine="qwen3_tts",
                asr_engine="file",
                audio_dir=tmpdir,
            ))
            with patch.object(
                speaker_engine,
                "_model",
                side_effect=lambda name: models.append(name) or FakeModel(),
            ), patch.object(speaker_engine, "_worker_request", return_value=None):
                speaker_engine.synthesize("Agent B", "Named voice.")
            self.assertTrue(models[-1].endswith("-CustomVoice"))

            clone_engine = Qwen3TextToSpeech(SpeechPipelineConfig(
                tts_engine="qwen3_tts",
                asr_engine="file",
                audio_dir=tmpdir,
                agent_a_custom_audio=True,
                agent_a_reference_audio=str(reference),
                agent_a_reference_text="Reference.",
            ))
            with patch.object(
                clone_engine,
                "_model",
                side_effect=lambda name: models.append(name) or FakeModel(),
            ), patch.object(clone_engine, "_worker_request", return_value=None):
                clone_engine.synthesize("Agent A", "Cloned voice.")
            self.assertTrue(models[-1].endswith("-Base"))

    def test_f5_tts_uses_reference_voice_and_configured_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.wav"
            reference.write_bytes(b"reference")
            config = SpeechPipelineConfig(
                tts_engine="f5_tts",
                asr_engine="file",
                audio_dir=tmpdir,
                tts_executable="f5-custom",
                tts_model="F5TTS_v1_Base",
                agent_a_custom_audio=True,
                agent_a_reference_audio=str(reference),
                agent_a_reference_text="Reference speech.",
            )
            engine = F5TextToSpeech(config)

            def create_wave(command, **_kwargs):
                output_dir = Path(command[command.index("--output_dir") + 1])
                output_file = command[command.index("--output_file") + 1]
                WaveFileTextToSpeech(output_dir)._write_wave(
                    output_dir / output_file,
                    "Agent A",
                    "Generated speech.",
                )
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with patch("coop_navigation_sds.DialogManagement.speech_pipeline.subprocess.run", side_effect=create_wave) as run:
                signal = engine.synthesize("Agent A", "Generated speech.")
            command = run.call_args.args[0]
            self.assertEqual(command[0], "f5-custom")
            self.assertIn(str(reference), command)
            self.assertEqual(signal.audio["engine"], "f5_tts")

    def test_whisper_cpp_reads_command_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "speech.wav"
            WaveFileTextToSpeech(tmpdir)._write_wave(wav_path, "Agent A", "Need Bravo to Harbor.")
            model_path = Path(tmpdir) / "ggml-base.en.bin"
            model_path.write_bytes(b"model")
            executable = Path(tmpdir) / "whisper-custom.exe"
            executable.write_bytes(b"exe")
            engine = WhisperCppSpeechToText(
                model_name=str(model_path),
                executable=str(executable),
                language="en-US",
            )
            signal = SpeechSignal("Agent A", "", audio={"path": str(wav_path)}, diagnostics={})

            def create_transcript(command, **_kwargs):
                output_base = Path(command[command.index("--output-file") + 1])
                Path(f"{output_base}.txt").write_text("Need Bravo to Harbor.", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("coop_navigation_sds.DialogManagement.speech_pipeline.subprocess.run", side_effect=create_transcript) as run:
                transcript = engine.transcribe(signal)
            self.assertEqual(transcript, "Need Bravo to Harbor.")
            self.assertEqual(Path(run.call_args.args[0][0]), executable.resolve())

    def test_whisper_cpp_resolves_manifest_registered_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            executable = root / "whisper-cli.exe"
            executable.write_bytes(b"exe")
            model_path = root / "ggml-base.en.bin"
            model_path.write_bytes(b"model")
            (root / "providers.json").write_text(
                '{"providers":{"whisper_cpp":{"executable":"whisper-cli.exe","model":"ggml-base.en.bin"}}}',
                encoding="utf-8",
            )
            wav_path = root / "speech.wav"
            WaveFileTextToSpeech(tmpdir)._write_wave(wav_path, "Agent B", "Take Ring.")
            engine = WhisperCppSpeechToText(provider_environment_dir=str(root))
            signal = SpeechSignal("Agent B", "", audio={"path": str(wav_path)}, diagnostics={})

            def create_transcript(command, **_kwargs):
                output_base = Path(command[command.index("--output-file") + 1])
                Path(f"{output_base}.txt").write_text("Take Ring.", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("coop_navigation_sds.DialogManagement.speech_pipeline.subprocess.run", side_effect=create_transcript) as run:
                transcript = engine.transcribe(signal)

            command = run.call_args.args[0]
            self.assertEqual(transcript, "Take Ring.")
            self.assertEqual(Path(command[0]), executable.resolve())
            self.assertEqual(Path(command[command.index("-m") + 1]), model_path.resolve())
            self.assertEqual(signal.diagnostics["resolved_model"], str(model_path.resolve()))

    def test_vosk_decodes_pcm_wave_with_final_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "speech.wav"
            WaveFileTextToSpeech(tmpdir)._write_wave(wav_path, "Agent A", "Need Bravo.")

            class FakeRecognizer:
                def __init__(self, _model, _sample_rate):
                    pass

                def SetWords(self, _enabled):
                    pass

                def AcceptWaveform(self, _data):
                    return False

                def FinalResult(self):
                    return '{"text": "need bravo"}'

            engine = VoskSpeechToText(model_name="test-model")
            signal = SpeechSignal("Agent A", "", audio={"path": str(wav_path)}, diagnostics={})
            with (
                patch.object(engine, "_load_model", return_value=object()),
                patch.dict(sys.modules, {"vosk": SimpleNamespace(KaldiRecognizer=FakeRecognizer)}),
            ):
                transcript = engine.transcribe(signal)
            self.assertEqual(transcript, "need bravo")
            self.assertEqual(signal.diagnostics["asr_engine"], "vosk-asr")

    def test_parakeet_and_qwen3_asr_record_model_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "speech.wav"
            WaveFileTextToSpeech(tmpdir)._write_wave(wav_path, "Agent B", "Take Ring.")

            parakeet = ParakeetSpeechToText()
            parakeet_signal = SpeechSignal("Agent B", "", audio={"path": str(wav_path)}, diagnostics={})
            parakeet_model = SimpleNamespace(
                transcribe=lambda _paths: [SimpleNamespace(text="take ring")]
            )
            with patch.object(parakeet, "_load_model", return_value=parakeet_model):
                self.assertEqual(parakeet.transcribe(parakeet_signal), "take ring")
            self.assertIn("parakeet", parakeet_signal.diagnostics["asr_model"])

            qwen = Qwen3SpeechToText()
            qwen_signal = SpeechSignal("Agent B", "", audio={"path": str(wav_path)}, diagnostics={})
            qwen_model = SimpleNamespace(
                transcribe=lambda **_kwargs: [SimpleNamespace(text="take ring", language="English")]
            )
            with patch.object(qwen, "_load_model", return_value=qwen_model):
                self.assertEqual(qwen.transcribe(qwen_signal), "take ring")
            self.assertEqual(qwen_signal.diagnostics["asr_language"], "English")

    def test_clock_notation_is_normalized_for_clean_speech(self):
        self.assertEqual(
            normalize_text_for_speech("Leave at 08:07 and return at 18:30."),
            "Leave at eight seven and return at eighteen thirty.",
        )

    def test_both_agents_always_pass_through_tts_and_asr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = self.file_transport(tmpdir)
            traces = [
                transport.transmit_trace("Agent A", "Need Alpha to Echo."),
                transport.transmit_trace("Agent B", "Take Metro from Alpha to Echo."),
            ]
            for trace in traces:
                self.assertEqual(trace.mode, "speech")
                self.assertTrue(trace.outgoing_enabled)
                self.assertTrue(trace.incoming_enabled)
                self.assertEqual(trace.tts_engine, "wavefile-tts")
                self.assertEqual(trace.asr_engine, "wavefile-asr")
                self.assertTrue(Path(trace.audio["path"]).exists())
                self.assertEqual(trace.incoming_transcript, trace.outgoing_text)

    def test_agent_specific_prosody_is_applied_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = self.file_transport(
                tmpdir,
                agent_a_volume=40,
                agent_a_pitch_semitones=-3,
                agent_a_pause_ms=60,
                agent_a_emphasis="reduced",
                agent_b_volume=90,
                agent_b_pitch_semitones=4,
                agent_b_pause_ms=180,
                agent_b_emphasis="strong",
            )
            agent_a = transport.transmit_trace("Agent A", "Alpha, then Echo.")
            agent_b = transport.transmit_trace("Agent B", "Take Metro to Echo.")

            self.assertEqual(agent_a.audio["prosody"]["volume"], 40)
            self.assertEqual(agent_a.audio["prosody"]["pitch_semitones"], -3)
            self.assertEqual(agent_b.audio["prosody"]["volume"], 90)
            self.assertEqual(agent_b.diagnostics["prosody"]["emphasis"], "strong")

    def test_sapi_command_contains_precise_prosody_controls(self):
        command = WindowsSapiTextToSpeech._powershell_command(
            Path("speech.wav"),
            voice="Microsoft David Desktop",
            speech_rate=3,
            volume=75,
            pitch_semitones=2,
            pause_ms=140,
            emphasis="moderate",
        )
        script = command[-1]
        self.assertIn("SelectVoice('Microsoft David Desktop')", script)
        self.assertIn("$speaker.Rate = 3", script)
        self.assertIn("$speaker.Volume = 75", script)
        self.assertIn("pitch='+2st'", script)
        self.assertIn('140ms', script)
        self.assertIn("emphasis level='moderate'", script)

    def test_sapi_named_audio_persona_does_not_pass_audit_metadata_to_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpeechPipelineConfig(
                agent_a_audio_persona="hurried_caller",
                agent_a_custom_audio=False,
            )
            engine = WindowsSapiTextToSpeech(tmpdir, config=config)

            def create_audio(command, **_kwargs):
                Path(command[0]).write_bytes(b"x" * 45)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch.object(
                    engine,
                    "_powershell_command",
                    side_effect=lambda wav_path, **_kwargs: [str(wav_path)],
                ) as command,
                patch("coop_navigation_sds.DialogManagement.speech_pipeline.subprocess.run", side_effect=create_audio),
                patch.object(engine, "_wave_duration", return_value=1.0),
            ):
                signal = engine.synthesize("Agent A", "Speech pipeline check.")

            command_kwargs = command.call_args.kwargs
            self.assertNotIn("audio_persona", command_kwargs)
            self.assertNotIn("custom_audio", command_kwargs)
            self.assertEqual(command_kwargs["speech_rate"], 7)
            self.assertEqual(signal.audio["prosody"]["audio_persona"], "hurried_caller")

    def test_synthesis_control_filter_excludes_research_metadata(self):
        controls = synthesis_controls({
            "audio_persona": "neutral_caller",
            "custom_audio": False,
            "voice": "",
            "speech_rate": 3,
        })
        self.assertEqual(controls, {"voice": "", "speech_rate": 3})

    def test_file_pipeline_applies_configured_speech_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = self.file_transport(tmpdir, pattern_key="stutter_heavy")
            trace = transport.transmit_trace("Agent A", "Please take Bravo to Harbor now.")
            self.assertNotEqual(trace.outgoing_text, trace.generated_text)
            self.assertEqual(trace.incoming_transcript, trace.outgoing_text)

    def test_health_check_requires_audio_and_transcript_for_both_agents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            health = self.file_transport(tmpdir).health_check()
            self.assertEqual(health["mode"], "speech")
            self.assertTrue(health["ok"])
            self.assertEqual({check["speaker"] for check in health["checks"]}, {"Agent A", "Agent B"})
            self.assertTrue(all(check["audio_ok"] for check in health["checks"]))
            self.assertTrue(all(check["transcript_ok"] for check in health["checks"]))

    def test_health_check_records_asr_quality_without_blocking_operational_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "probe.wav"
            audio_path.write_bytes(b"RIFF")
            transport = SpeechTransport.__new__(SpeechTransport)
            transport.config = SimpleNamespace(pattern_key="clean")
            transport.transmit_trace = lambda _speaker, _text: SimpleNamespace(
                audio={"path": str(audio_path)},
                incoming_transcript="recognizable speech without route entities",
                pipeline_ok=True,
                tts_engine="test-tts",
                asr_engine="test-asr",
                diagnostics={},
            )

            health = transport.health_check()

        self.assertTrue(health["ok"])
        self.assertFalse(health["quality_ok"])
        self.assertTrue(all(check["pipeline_ok"] for check in health["checks"]))
        self.assertFalse(any(check["quality_ok"] for check in health["checks"]))

    def test_realtime_without_playback_waits_before_recognition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = self.file_transport(
                tmpdir,
                realtime_enabled=True,
                max_utterance_sec=1.0,
            )
            with patch("coop_navigation_sds.DialogManagement.speech_pipeline.time.sleep") as sleep:
                trace = transport.transmit_trace("Agent A", "Wait before recognition.")
            self.assertTrue(trace.audio["waited"])
            sleep.assert_called_once()

    def test_pipeline_fails_when_tts_does_not_create_audio(self):
        class BrokenTextToSpeech:
            name = "broken-tts"

            def synthesize(self, speaker, text):
                return SpeechSignal(speaker=speaker, text=text, audio=None)

        transport = SpeechTransport(
            tts_engine=BrokenTextToSpeech(),
            config=SpeechPipelineConfig(asr_engine="file", realtime_enabled=False),
        )
        with self.assertRaises(SpeechPipelineError):
            transport.transmit_trace("Agent A", "This should fail.")

    def test_windows_asr_never_replaces_recognized_text_with_reference_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "turn.wav"
            audio_path.write_bytes(b"RIFFxxxxWAVEfmt ")
            signal = SpeechSignal(
                speaker="Agent A",
                text="Need Alpha to Echo.",
                audio={"path": str(audio_path)},
                diagnostics={},
            )
            completed = SimpleNamespace(returncode=0, stdout="meet apple two ego", stderr="")
            with patch("coop_navigation_sds.DialogManagement.speech_pipeline.subprocess.run", return_value=completed):
                transcript = WindowsSapiSpeechToText().transcribe(signal)
            self.assertEqual(transcript, "meet apple two ego")
            self.assertEqual(signal.diagnostics["raw_asr_transcript"], transcript)
            self.assertFalse(signal.diagnostics["asr_repair_used"])

    def test_windows_asr_repairs_common_route_mishearing(self):
        recognizer = WindowsSapiSpeechToText(phrase_hints=("route", "Bravo", "Harbor"))

        self.assertEqual(
            recognizer._normalize_domain_terms("Which rude should I take to hamper?"),
            "Which route should I take to Harbor?",
        )

    def test_shared_domain_repair_applies_after_any_recognizer(self):
        class ArtifactTextToSpeech:
            name = "artifact-tts"

            def __init__(self, path):
                self.path = path

            def synthesize(self, speaker, text):
                self.path.write_bytes(b"RIFFxxxxWAVEfmt ")
                return SpeechSignal(speaker, text, {"path": str(self.path)}, {})

        class GarbledRecognizer:
            name = "garbled-asr"

            def transcribe(self, _signal):
                return "Which rude goes to harder?"

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = SpeechTransport(
                tts_engine=ArtifactTextToSpeech(Path(tmpdir) / "turn.wav"),
                asr_engine=GarbledRecognizer(),
                config=SpeechPipelineConfig(
                    tts_engine="file",
                    asr_engine="file",
                    realtime_enabled=False,
                    asr_domain_normalization_enabled=True,
                ),
            )
            trace = transport.transmit_trace("Agent A", "Which route goes to Harbor?")

        self.assertEqual(trace.incoming_transcript, "Which route goes to Harbor?")
        self.assertEqual(trace.diagnostics["raw_asr_transcript"], "Which rude goes to harder?")
        self.assertTrue(trace.diagnostics["asr_domain_normalization_used"])
        self.assertEqual(trace.diagnostics["agent_input_transcript"], trace.incoming_transcript)
        self.assertEqual(
            [change["source_tokens"] for change in trace.diagnostics["transcript_corrections"]],
            [["rude"], ["harder"]],
        )
        self.assertEqual(
            [change["target_tokens"] for change in trace.diagnostics["transcript_corrections"]],
            [["route"], ["Harbor"]],
        )

    def test_shared_domain_repair_can_be_disabled(self):
        class ArtifactTextToSpeech:
            name = "artifact-tts"

            def __init__(self, path):
                self.path = path

            def synthesize(self, speaker, text):
                self.path.write_bytes(b"RIFFxxxxWAVEfmt ")
                return SpeechSignal(speaker, text, {"path": str(self.path)}, {})

        class GarbledRecognizer:
            name = "garbled-asr"

            def transcribe(self, _signal):
                return "Which rude goes to harder?"

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = SpeechTransport(
                tts_engine=ArtifactTextToSpeech(Path(tmpdir) / "turn.wav"),
                asr_engine=GarbledRecognizer(),
                config=SpeechPipelineConfig(
                    tts_engine="file",
                    asr_engine="file",
                    realtime_enabled=False,
                    asr_domain_normalization_enabled=False,
                ),
            )
            trace = transport.transmit_trace("Agent A", "Which route goes to Harbor?")

        self.assertEqual(trace.incoming_transcript, "Which rude goes to harder?")
        self.assertFalse(trace.diagnostics["asr_domain_normalization_used"])

    def test_domain_repair_normalizes_spoken_mode_line_codes(self):
        from coop_navigation_sds.NaturalLanguageUnderstanding.transcript_normalization import normalize_transit_transcript

        transcript = normalize_transit_transcript(
            "Take metro line em one and tram line tee two to Harbor."
        )

        self.assertIn("metro line M1", transcript)
        self.assertIn("tram line T2", transcript)

    def test_windows_asr_chooses_domain_grounded_alternative_and_normalizes_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "turn.wav"
            audio_path.write_bytes(b"RIFFxxxxWAVEfmt ")
            signal = SpeechSignal(
                speaker="Agent A",
                text="I am at Bravo, going to Harbor.",
                audio={"path": str(audio_path)},
                diagnostics={},
            )
            payload = (
                '{"text":"I am at brother going harder","confidence":0.42,'
                '"alternatives":[{"text":"I am at Bravo going to Harbour","confidence":0.38}]}'
            )
            completed = SimpleNamespace(returncode=0, stdout=payload, stderr="")
            with patch("coop_navigation_sds.DialogManagement.speech_pipeline.subprocess.run", return_value=completed):
                transcript = WindowsSapiSpeechToText(
                    phrase_hints=("Bravo", "Harbor", "route"),
                ).transcribe(signal)
            self.assertEqual(transcript, "I am at Bravo going to Harbor")
            self.assertEqual(
                signal.diagnostics["raw_asr_transcript"],
                "I am at brother going harder",
            )
            self.assertEqual(
                signal.diagnostics["asr_selected_transcript"],
                transcript,
            )
            self.assertEqual(signal.diagnostics["asr_confidence"], 0.42)

    def test_windows_asr_uses_configurable_long_pause_windows(self):
        command = WindowsSapiSpeechToText._powershell_command(
            Path("speech.wav"),
            initial_silence_sec=5.0,
            babble_timeout_sec=6.0,
            end_silence_ms=1700,
            ambiguous_end_silence_ms=2400,
        )
        script = command[-1]
        self.assertIn("InitialSilenceTimeout = [TimeSpan]::FromSeconds(5.0)", script)
        self.assertIn("BabbleTimeout = [TimeSpan]::FromSeconds(6.0)", script)
        self.assertIn("EndSilenceTimeout = [TimeSpan]::FromMilliseconds(1700)", script)
        self.assertIn("EndSilenceTimeoutAmbiguous = [TimeSpan]::FromMilliseconds(2400)", script)

    def test_platform_defaults_use_portable_linux_speech_engines(self):
        with patch("coop_navigation_sds.DialogManagement.speech_pipeline.platform.system", return_value="Linux"), \
             patch("coop_navigation_sds.DialogManagement.speech_pipeline.shutil.which", return_value=None), \
             patch("coop_navigation_sds.DialogManagement.speech_pipeline.importlib.util.find_spec", return_value=None):
            self.assertEqual(platform_default_tts_engine(), "file")
            self.assertEqual(platform_default_asr_engine(), "file")
            with self.assertRaisesRegex(SpeechPipelineError, "only available on Windows"):
                SpeechTransport(config=SpeechPipelineConfig(tts_engine="sapi", asr_engine="file"))

        with patch("coop_navigation_sds.DialogManagement.speech_pipeline.platform.system", return_value="Linux"), \
             patch("coop_navigation_sds.DialogManagement.speech_pipeline.shutil.which", return_value="/usr/bin/espeak-ng"), \
             patch("coop_navigation_sds.DialogManagement.speech_pipeline.importlib.util.find_spec", return_value=object()):
            self.assertEqual(platform_default_tts_engine(), "espeak_ng")
            self.assertEqual(platform_default_asr_engine(), "faster_whisper")

    def test_speech_patterns_include_natural_variants(self):
        keys = speech_pattern_keys()
        self.assertIn("mostly_clean", keys)
        self.assertIn("long_pauses", keys)
        self.assertIn("stutter_light", keys)
        text = PatternedSpeechToText("stutter_heavy", seed=2).transcribe(
            SpeechSignal("Agent A", "Please take Bravo to Harbor.")
        )
        self.assertNotEqual(text, "Please take Bravo to Harbor.")


if __name__ == "__main__":
    unittest.main()
