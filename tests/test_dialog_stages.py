import unittest

from coop_navigation_sds.DialogManagement.stages import ConversationStage, dialog_context


class DialogStageTests(unittest.TestCase):
    def test_stage_progression_uses_conversation_content(self):
        self.assertEqual(dialog_context([]).stage, ConversationStage.DISCOVERY)

        opening = [("Agent A", "I am at Alpha at eight ten, going to Echo.")]
        self.assertEqual(dialog_context(opening).stage, ConversationStage.PROPOSAL)

        comparison = [
            *opening,
            ("Agent B", "Take Metro from Alpha to Echo."),
            ("Agent A", "Can you compare a faster alternative?"),
        ]
        context = dialog_context(comparison)
        self.assertEqual(context.stage, ConversationStage.COMPARISON)
        self.assertEqual(context.response_focus, "alternative")

        refinement = [
            *comparison,
            ("Agent B", "Take Tram from Alpha to Echo."),
            ("Agent A", "I need the option with lower delay risk."),
        ]
        context = dialog_context(refinement)
        self.assertEqual(context.stage, ConversationStage.REFINEMENT)
        self.assertEqual(context.response_focus, "reliability")

        confirmation = [
            *refinement,
            ("Agent B", "Take metro line M1 from Alpha to Echo."),
            ("Agent A", "That works. Please confirm the final route."),
        ]
        self.assertEqual(dialog_context(confirmation).stage, ConversationStage.CONFIRMATION)

    def test_context_remembers_latest_utterances_and_prior_turns(self):
        conversation = [
            ("Agent A", "First request."),
            ("Agent B", "First answer."),
            ("Agent A", "Latest request."),
        ]
        context = dialog_context(conversation)

        self.assertEqual(context.latest_agent_a, "Latest request.")
        self.assertEqual(context.latest_agent_b, "First answer.")
        self.assertEqual(context.agent_b_turn_count, 1)


if __name__ == "__main__":
    unittest.main()
