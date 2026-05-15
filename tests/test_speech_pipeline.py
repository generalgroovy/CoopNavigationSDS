import unittest

from minillama.agent_b.speech_io import SpeechPipelineConfig, SpeechTransport


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
        self.assertIn("text-only:hesitant:none", transport.description)

    def test_speech_scope_applies_only_to_selected_agent(self):
        transport = SpeechTransport(
            config=SpeechPipelineConfig(
                incoming_enabled=True,
                outgoing_enabled=True,
                scope="agent_b",
                pattern_key="compressed",
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

        self.assertEqual(
            agent_b_trace.outgoing_text,
            "I'd compare the fastest route and the fewest switches.",
        )
        self.assertEqual(agent_b_trace.incoming_transcript, agent_b_trace.outgoing_text)
        self.assertTrue(agent_b_trace.outgoing_enabled)
        self.assertTrue(agent_b_trace.incoming_enabled)

        self.assertEqual(agent_a_trace.outgoing_text, agent_a_trace.generated_text)
        self.assertEqual(agent_a_trace.incoming_transcript, agent_a_trace.generated_text)
        self.assertFalse(agent_a_trace.outgoing_enabled)
        self.assertFalse(agent_a_trace.incoming_enabled)


if __name__ == "__main__":
    unittest.main()
