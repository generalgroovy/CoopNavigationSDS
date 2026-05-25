import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from minillama.agent_b.speech_io import SpeechPipelineConfig, SpeechPipelineError, SpeechSignal, SpeechTransport


class SpeechPipelineTests(unittest.TestCase):
    def test_text_only_pipeline_keeps_generated_text_unchanged(self):
        transport = SpeechTransport(
            config=SpeechPipelineConfig(
                incoming_enabled=False,
                outgoing_enabled=False,
                scope="none",
                pattern_key="hesitant",
            )
        )

        trace = transport.transmit_trace("Agent A", "Please get me from Alpha to Beta.")

        self.assertEqual(trace.generated_text, "Please get me from Alpha to Beta.")
        self.assertEqual(trace.outgoing_text, trace.generated_text)
        self.assertEqual(trace.incoming_transcript, trace.generated_text)
        self.assertFalse(trace.outgoing_enabled)
        self.assertFalse(trace.incoming_enabled)
        self.assertGreaterEqual(trace.simulated_duration_sec, 0.8)
        self.assertLessEqual(trace.simulated_duration_sec, 8.0)
        self.assertEqual(trace.mode, "pure_text")
        self.assertIn("pure_text", transport.description)

    def test_speech_scope_applies_only_to_selected_agent(self):
        transport = SpeechTransport(
            config=SpeechPipelineConfig(
                incoming_enabled=True,
                outgoing_enabled=True,
                mode="speech",
                scope="agent_b",
                engine="file",
                realtime_enabled=False,
            )
        )

        agent_b_trace = transport.transmit_trace(
            "Agent B",
            "Please I would compare the fastest route and the fewest switches.",
        )
        agent_a_trace = transport.transmit_trace(
            "Agent A",
            "Please I would like to avoid full trains.",
        )

        self.assertEqual(agent_b_trace.outgoing_text, agent_b_trace.generated_text)
        self.assertEqual(agent_b_trace.incoming_transcript, agent_b_trace.outgoing_text)
        self.assertTrue(agent_b_trace.outgoing_enabled)
        self.assertTrue(agent_b_trace.incoming_enabled)
        self.assertIsInstance(agent_b_trace.audio, dict)
        self.assertGreater(agent_b_trace.simulated_duration_sec, 1.0)

        self.assertEqual(agent_a_trace.outgoing_text, agent_a_trace.generated_text)
        self.assertEqual(agent_a_trace.incoming_transcript, agent_a_trace.generated_text)
        self.assertFalse(agent_a_trace.outgoing_enabled)
        self.assertFalse(agent_a_trace.incoming_enabled)

    def test_tts_and_asr_engines_are_independently_configurable(self):
        transport = SpeechTransport(
            config=SpeechPipelineConfig(
                incoming_enabled=True,
                outgoing_enabled=True,
                mode="speech",
                scope="both",
                pattern_key="compressed",
                tts_engine="file",
                asr_engine="patterned",
                realtime_enabled=False,
            )
        )

        trace = transport.transmit_trace(
            "Agent B",
            "Please I would compare the fastest route and the fewest switches.",
        )

        self.assertEqual(trace.tts_engine, "wavefile-tts")
        self.assertEqual(trace.asr_engine, "patterned-asr:compressed")
        self.assertEqual(trace.outgoing_text, trace.generated_text)
        self.assertEqual(
            trace.incoming_transcript,
            "I'd compare the fastest route and the fewest switches.",
        )
        self.assertIn("tts=file:asr=patterned", transport.description)

    def test_speech_mode_rejects_loopback_tts_or_asr(self):
        with self.assertRaises(SpeechPipelineError) as error:
            SpeechTransport(
                config=SpeechPipelineConfig(
                    mode="speech",
                    incoming_enabled=True,
                    outgoing_enabled=True,
                    scope="both",
                    tts_engine="loopback",
                    asr_engine="loopback",
                    realtime_enabled=False,
                )
            )

        self.assertIn("complete text-to-speech", str(error.exception))

    def test_file_speech_engine_creates_audio_and_transcribes_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = SpeechTransport(
                config=SpeechPipelineConfig(
                    incoming_enabled=True,
                    outgoing_enabled=True,
                    mode="speech",
                    scope="both",
                    engine="file",
                    audio_dir=tmpdir,
                    playback_enabled=False,
                    realtime_enabled=False,
                )
            )

            trace = transport.transmit_trace("Agent B", "Take Ring from Alpha to Bravo.")

            self.assertEqual(trace.incoming_transcript, trace.generated_text)
            self.assertTrue(trace.outgoing_enabled)
            self.assertTrue(trace.incoming_enabled)
            self.assertEqual(trace.tts_engine, "wavefile-tts")
            self.assertEqual(trace.asr_engine, "wavefile-asr")
            self.assertIsInstance(trace.audio, dict)
            self.assertTrue(Path(trace.audio["path"]).exists())
            self.assertGreater(Path(trace.audio["path"]).stat().st_size, 44)
            self.assertTrue(Path(trace.audio["transcript_path"]).exists())
            self.assertIn("duration_sec", trace.audio)
            self.assertFalse(trace.audio["played"])
            self.assertFalse(trace.audio["realtime"])
            self.assertFalse(trace.audio["waited"])

    def test_file_speech_engine_supports_playback_flag_without_breaking_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = SpeechTransport(
                config=SpeechPipelineConfig(
                    incoming_enabled=True,
                    outgoing_enabled=True,
                    mode="speech",
                    scope="both",
                    engine="file",
                    audio_dir=tmpdir,
                    playback_enabled=True,
                    max_utterance_sec=2.5,
                    realtime_enabled=False,
                )
            )

            with patch("minillama.agent_b.speech_io.WaveFileTextToSpeech._play_wave", return_value=True):
                trace = transport.transmit_trace("Agent A", "Short audible turn.")

            self.assertEqual(trace.incoming_transcript, "Short audible turn.")
            self.assertLessEqual(trace.simulated_duration_sec, 2.5)
            self.assertTrue(trace.audio["played"])

    def test_realtime_file_speech_waits_before_transcript_delivery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = SpeechTransport(
                config=SpeechPipelineConfig(
                    incoming_enabled=True,
                    outgoing_enabled=True,
                    mode="speech",
                    scope="both",
                    engine="file",
                    audio_dir=tmpdir,
                    playback_enabled=True,
                    realtime_enabled=True,
                    max_utterance_sec=2.5,
                )
            )

            with patch("minillama.agent_b.speech_io.WaveFileTextToSpeech._play_wave", return_value=True) as play_wave:
                trace = transport.transmit_trace("Agent B", "I can hear this after playback.")

            self.assertEqual(trace.incoming_transcript, "I can hear this after playback.")
            self.assertTrue(trace.audio["played"])
            self.assertTrue(trace.audio["realtime"])
            self.assertTrue(trace.audio["waited"])
            self.assertTrue(play_wave.call_args.kwargs["realtime"])
            self.assertGreater(play_wave.call_args.kwargs["fallback_duration"], 0)

    def test_realtime_without_playback_still_waits_before_listening(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = SpeechTransport(
                config=SpeechPipelineConfig(
                    incoming_enabled=True,
                    outgoing_enabled=True,
                    mode="speech",
                    scope="both",
                    engine="file",
                    audio_dir=tmpdir,
                    playback_enabled=False,
                    realtime_enabled=True,
                    max_utterance_sec=1.0,
                )
            )

            with patch("minillama.agent_b.speech_io.time.sleep") as sleep:
                trace = transport.transmit_trace("Agent A", "Wait before the other agent hears this.")

            self.assertEqual(trace.incoming_transcript, "Wait before the other agent hears this.")
            self.assertFalse(trace.audio["played"])
            self.assertTrue(trace.audio["waited"])
            sleep.assert_called_once()

    def test_speech_mode_fails_when_tts_does_not_create_audio(self):
        class BrokenTextToSpeech:
            name = "broken-tts"

            def synthesize(self, speaker, text):
                return SpeechSignal(speaker=speaker, text=text, audio=None)

        transport = SpeechTransport(
            tts_engine=BrokenTextToSpeech(),
            config=SpeechPipelineConfig(
                mode="speech",
                incoming_enabled=True,
                outgoing_enabled=True,
                scope="both",
                asr_engine="file",
                realtime_enabled=False,
            )
        )

        with self.assertRaises(SpeechPipelineError) as error:
            transport.transmit_trace("Agent A", "This should fail.")

        self.assertIn("audio artifact", str(error.exception))


if __name__ == "__main__":
    unittest.main()
