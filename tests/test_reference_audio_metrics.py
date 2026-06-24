import tempfile
import unittest
import wave
from pathlib import Path

from coop_navigation_sds.EvaluationMetrics.catalog import metric_metadata, metric_scale_percentage
from coop_navigation_sds.EvaluationMetrics.reference_audio import ReferenceAudioQualityEvaluator
from coop_navigation_sds.NaturalLanguageUnderstanding.clarification import (
    clarification_question,
    clarification_confirmation,
    last_substantive_agent_b_utterance,
    transcript_repair_question,
)


def write_wave(path):
    samples = [int(8000 * ((index % 40) / 40.0 - 0.5)) for index in range(4000)]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))


class ReferenceAudioMetricTests(unittest.TestCase):
    def test_word_clarification_does_not_append_route_instructions(self):
        question = clarification_question(
            "Take the rude to Harbor.",
            "Restate the complete route with every station and line.",
        )

        self.assertIn("route", question.lower())
        self.assertNotIn("Restate the complete route", question)
        self.assertLessEqual(len(question.split()), 10)

    def test_missing_reference_pairs_are_explicitly_unavailable(self):
        report = ReferenceAudioQualityEvaluator().evaluate([{"path": "missing.wav"}])

        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["reason"], "no_aligned_reference_audio_pairs")
        self.assertIsNone(report["scores"]["pesq"])
        self.assertIn("licensed", report["polqa_policy"].lower())

    def test_aligned_pair_always_calculates_si_sdr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "speech.wav"
            write_wave(path)
            report = ReferenceAudioQualityEvaluator().evaluate([
                {"path": str(path), "reference_path": str(path), "speaker": "Agent A"}
            ])

        self.assertEqual(report["status"], "available")
        self.assertEqual(report["evaluated_pair_count"], 1)
        self.assertGreaterEqual(report["scores"]["si_sdr_db"], 50.0)

    def test_audio_metric_metadata_explains_range_and_scale(self):
        metadata = metric_metadata("tts_nisqa", "tts")

        self.assertEqual(metadata["range"], [1.0, 5.0])
        self.assertIn("speech quality", metadata["meaning"])
        self.assertAlmostEqual(metric_scale_percentage("tts_nisqa", 4.3376), 86.752)

    def test_transcript_correction_becomes_dialogue_repair_and_confirmation(self):
        question = transcript_repair_question([{
            "operation": "replace",
            "source_tokens": ["tee", "won"],
            "target_tokens": ["T1"],
        }])
        confirmation = clarification_confirmation([
            ("Agent B", "Take tram line T1 from Bravo to Delta."),
            ("Agent A", question),
        ])

        self.assertIn("Did you mean 'T1'", question)
        self.assertIn("Yes, I meant T1", confirmation)
        self.assertNotIn("Take tram line T1", confirmation)

    def test_confirmation_uses_prior_spoken_term_and_does_not_repeat_route(self):
        conversation = [
            ("Agent B", "Take tram line T2 from Bravo to Juliett."),
            ("Agent A", "I heard Juliett. Did you mean like julie?"),
        ]

        confirmation = clarification_confirmation(
            conversation,
            "Take tram line T2 from Bravo to Juliett.",
        )

        self.assertEqual(confirmation, "I meant Juliett.")
        self.assertEqual(
            last_substantive_agent_b_utterance([*conversation, ("Agent B", confirmation)]),
            "Take tram line T2 from Bravo to Juliett.",
        )


if __name__ == "__main__":
    unittest.main()
