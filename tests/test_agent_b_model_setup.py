import unittest
from unittest.mock import patch

from coop_navigation_sds.Configuration.model_matrix import (
    AGENT_B_OLLAMA_BASE_URL,
    agent_b_model_proposals,
    model_catalog_folder,
    resolve_agent_b_model_store,
)
from scripts.setup_agent_b_models import (
    main,
    readiness_rows,
    selected_model_names,
)
from scripts.setup_transformers_agent_b_models import (
    TRANSFORMERS_AGENT_B_PROFILES,
    selected_profiles as selected_transformers_profiles,
)


class AgentBModelSetupTests(unittest.TestCase):
    def setUp(self):
        """Keep setup CLI tests from mutating the developer's local model catalog."""
        folder_patch = patch("scripts.setup_agent_b_models.initialize_platform_folders")
        catalog_patch = patch("scripts.setup_agent_b_models.write_local_catalog")
        folder_patch.start()
        catalog_patch.start()
        self.addCleanup(folder_patch.stop)
        self.addCleanup(catalog_patch.stop)

    def test_default_selection_contains_two_models_per_size(self):
        models = selected_model_names()

        self.assertEqual(len(models), 6)
        rows = readiness_rows(models, models)
        self.assertEqual(
            {
                tier: sum(row["size_tier"] == tier for row in rows)
                for tier in ("small", "medium", "large")
            },
            {"small": 2, "medium": 2, "large": 2},
        )

    def test_explicit_model_does_not_expand_unselected_tiers(self):
        self.assertEqual(selected_model_names(models=("qwen2.5:7b",)), ("qwen2.5:7b",))

    def test_status_mode_reports_missing_without_downloading(self):
        with patch(
            "scripts.setup_agent_b_models.ollama_model_inventory",
            return_value={
                "base_url": "http://127.0.0.1:11434/api",
                "available_models": ("llama3.2:1b",),
            },
        ), patch("scripts.setup_agent_b_models.pull_models") as pull:
            exit_code = main(["--tier", "small", "--json"])

        self.assertEqual(exit_code, 2)
        pull.assert_not_called()

    def test_pull_mode_downloads_only_missing_models(self):
        with patch(
            "scripts.setup_agent_b_models.ollama_model_inventory",
            return_value={
                "base_url": "http://127.0.0.1:11434/api",
                "available_models": ("llama3.2:1b",),
            },
        ), patch("scripts.setup_agent_b_models.pull_models") as pull, patch(
            "scripts.setup_agent_b_models.ensure_ollama_models_ready",
            return_value={
                "available_models": ("llama3.2:1b", "qwen2.5:1.5b"),
            },
        ):
            exit_code = main(["--tier", "small", "--pull", "--json"])

        self.assertEqual(exit_code, 0)
        pull.assert_called_once_with(
            ("qwen2.5:1.5b",),
            base_url=AGENT_B_OLLAMA_BASE_URL,
            models_dir=resolve_agent_b_model_store(),
        )

    def test_catalog_folders_sort_by_size_then_model(self):
        self.assertEqual(
            model_catalog_folder("llama3.2:1b").as_posix(),
            "01-small/01-llama3-2-1b",
        )
        self.assertEqual(
            model_catalog_folder("llama3.1:8b").as_posix(),
            "03-large/02-llama3-1-8b",
        )

    def test_platform_model_stores_are_isolated(self):
        windows = resolve_agent_b_model_store(system_name="Windows")
        linux = resolve_agent_b_model_store(system_name="Linux")
        self.assertNotEqual(windows, linux)
        self.assertEqual(windows.parts[-3:], ("agent_b", "windows", "ollama"))
        self.assertEqual(linux.parts[-3:], ("agent_b", "linux", "ollama"))

    def test_model_proposals_contain_at_least_three_unique_models_per_size(self):
        proposals = agent_b_model_proposals()

        self.assertEqual(set(proposals), {"small", "medium", "large"})
        for tier, rows in proposals.items():
            self.assertGreaterEqual(len(rows), 3, tier)
            self.assertEqual(len({row.model for row in rows}), len(rows))
            self.assertTrue(all(row.unique_aspect and row.use_case for row in rows))
        self.assertNotIn(
            "microsoft/UserLM-8b",
            {row.model for row in proposals["large"]},
        )
        self.assertIn(
            "tiiuae/Falcon3-7B-Instruct",
            {row.model for row in proposals["large"]},
        )

    def test_transformers_agent_b_setup_profiles_cover_four_per_size_without_ollama(self):
        self.assertEqual(
            {tier: len(profiles) for tier, profiles in TRANSFORMERS_AGENT_B_PROFILES.items()},
            {"small": 4, "medium": 4, "large": 4},
        )
        selected = selected_transformers_profiles(("small", "large"), ())
        self.assertEqual(len(selected), 8)
        self.assertTrue(all("ollama" not in profile for profile in selected))

    def test_unavailable_ollama_returns_actionable_status_without_traceback(self):
        with patch(
            "scripts.setup_agent_b_models.ollama_model_inventory",
            side_effect=RuntimeError("Ollama is not reachable"),
        ):
            exit_code = main(["--tier", "small", "--json"])

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
