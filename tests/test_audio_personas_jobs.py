import inspect
import tempfile
from pathlib import Path
import unittest

from coop_navigation_sds.app import default_run_config, normalize_run_config
from coop_navigation_sds.Configuration.gui import StartupConfigDialog, ToolTip
from coop_navigation_sds.Configuration.jobs import load_experiment_job
from coop_navigation_sds.experiments import build_condition_grid
from coop_navigation_sds.TextToSpeech.personas import audio_persona_keys, synthesis_values
from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import normalize_agent_a_type


class AudioPersonaAndJobTests(unittest.TestCase):
    def test_role_specific_audio_personas_are_available(self):
        callers = audio_persona_keys("caller")
        assistants = audio_persona_keys("assistant")
        self.assertIn("high_clarity_caller", callers)
        self.assertIn("high_clarity_operator", assistants)
        self.assertIn("hurried_caller", callers)
        self.assertIn("hesitant_caller", callers)
        self.assertIn("brisk_operator", assistants)
        self.assertIn("deliberate_operator", assistants)
        self.assertTrue(set(callers).isdisjoint(assistants))

    def test_named_audio_persona_controls_resolved_synthesis(self):
        config = default_run_config()
        config.update({
            "agent_a_audio_persona": "hurried_caller",
            "agent_a_custom_audio": False,
            "agent_a_words_per_minute": 80,
        })
        normalized = normalize_run_config(config)
        profile = synthesis_values("hurried_caller")
        self.assertEqual(normalized["agent_a_words_per_minute"], profile["words_per_minute"])
        self.assertEqual(normalized["agent_a_pause_ms"], profile["pause_ms"])
        self.assertEqual(normalized["agent_a_speed"], profile["speed"])
        self.assertEqual(normalized["agent_a_temperature"], profile["temperature"])
        self.assertEqual(normalized["agent_a_seed"], profile["seed"])
        self.assertEqual(normalized["agent_a_oral_level"], profile["oral_level"])

    def test_legacy_custom_audio_controls_do_not_override_profile(self):
        config = default_run_config()
        config.update({
            "agent_b_audio_persona": "brisk_operator",
            "agent_b_custom_audio": True,
            "agent_b_words_per_minute": 188,
            "agent_b_volume": 73,
        })
        normalized = normalize_run_config(config)
        profile = synthesis_values("brisk_operator")
        self.assertFalse(normalized["agent_b_custom_audio"])
        self.assertEqual(normalized["agent_b_words_per_minute"], profile["words_per_minute"])
        self.assertEqual(normalized["agent_b_volume"], profile["volume"])

    def test_job_file_builds_audio_persona_factorial_grid(self):
        source = Path(__file__).resolve().parents[1] / "jobs" / "audio_persona_matrix.job"
        job = load_experiment_job(source)
        grid = job["grid"]
        conditions = list(build_condition_grid(
            test_case_keys=grid["test_cases"][:1],
            persona_keys=grid["personas"][:1],
            speech_pattern_keys=grid["speech_patterns"][:1],
            model_param_keys=grid["model_params"],
            objective_modes=grid["objective_modes"],
            agent_a_audio_persona_keys=grid["agent_a_audio_personas"][:2],
            agent_b_audio_persona_keys=grid["agent_b_audio_personas"][:2],
            iterations=1,
        ))
        self.assertEqual(len(conditions), 4)
        self.assertEqual(
            {(condition.agent_a_audio_persona, condition.agent_b_audio_persona) for condition in conditions},
            {
                ("neutral_caller", "clear_operator"),
                ("neutral_caller", "brisk_operator"),
                ("hurried_caller", "clear_operator"),
                ("hurried_caller", "brisk_operator"),
            },
        )

    def test_invalid_job_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.job"
            path.write_text('{"schema_version": 99}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_job(path)

    def test_platform_speech_llm_matrix_jobs_cover_both_agent_a_modes(self):
        root = Path(__file__).resolve().parents[1] / "jobs"
        paths = sorted(root.glob("*_agent_a_*_speech_llm_matrix.job"))

        self.assertEqual(len(paths), 4)
        jobs = [load_experiment_job(path) for path in paths]
        self.assertEqual(
            {normalize_agent_a_type(job["config"]["agent_a_type"]) for job in jobs},
            {"staged", "userlm"},
        )
        for job in jobs:
            self.assertTrue(job["config"]["paired_audio_text_runs"])
            self.assertGreaterEqual(len(job["grid"]["tts_engines"]), 3)
            self.assertGreaterEqual(len(job["grid"]["asr_engines"]), 3)
            self.assertGreaterEqual(len(job["grid"]["agent_b_models"]), 3)

    def test_configuration_window_has_tooltips_and_dynamic_persona_details(self):
        source = inspect.getsource(StartupConfigDialog)
        self.assertIn("ToolTip", source)
        self.assertIn("_refresh_persona_detail", source)
        self.assertIn("_select_audio_persona", source)
        self.assertIn("_refresh_conditional_sections", source)
        self.assertNotIn("reference_audio", source)
        self.assertNotIn("laugh_level", source)
        self.assertTrue(callable(ToolTip))


if __name__ == "__main__":
    unittest.main()
