import inspect
import tempfile
from pathlib import Path
import unittest

from coop_navigation_sds.app import default_run_config, normalize_run_config
from coop_navigation_sds.Configuration.gui import StartupConfigDialog, ToolTip
from coop_navigation_sds.Configuration.jobs import (
    job_parameter_profiles,
    load_experiment_job,
)
from coop_navigation_sds.Configuration.assets import (
    faster_whisper_model_ready,
    resolve_faster_whisper_model,
)
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
            {"tinyllama", "userlm"},
        )
        for job in jobs:
            self.assertTrue(job["config"]["paired_audio_text_runs"])
            self.assertGreaterEqual(len(job["grid"]["tts_engines"]), 3)
            self.assertGreaterEqual(len(job["grid"]["asr_engines"]), 3)
            self.assertGreaterEqual(len(job["grid"]["agent_b_models"]), 3)

    def test_linux_tinyllama_speech_llm_matrix_is_balanced_for_provider_analysis(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "jobs"
            / "linux_agent_a_tinyllama_speech_llm_matrix.job"
        )
        job = load_experiment_job(source)
        grid = job["grid"]
        conditions = list(build_condition_grid(
            test_case_keys=grid["test_cases"],
            persona_keys=grid["personas"],
            speech_pattern_keys=grid["speech_patterns"],
            model_param_keys=grid["model_params"],
            objective_modes=grid["objective_modes"],
            agent_a_audio_persona_keys=grid["agent_a_audio_personas"],
            agent_b_audio_persona_keys=grid["agent_b_audio_personas"],
            tts_engine_keys=grid["tts_engines"],
            asr_engine_keys=grid["asr_engines"],
            agent_b_model_keys=grid["agent_b_models"],
            iterations=job["iterations"],
            parameter_profiles=job_parameter_profiles(job),
            pair_audio_with_text=job["config"]["paired_audio_text_runs"],
        ))

        self.assertEqual(job["config"]["agent_a_type"], "tinyllama")
        self.assertEqual(job["config"]["model_provider"], "ollama")
        self.assertEqual(len(grid["agent_b_models"]), 6)
        self.assertEqual(len(grid["tts_engines"]), 4)
        self.assertEqual(len(grid["asr_engines"]), 4)
        self.assertEqual(len(grid["personas"]), 3)
        self.assertEqual(set(grid["agent_a_audio_personas"]), {"high_clarity_caller", "degraded_caller"})
        self.assertEqual(set(grid["agent_b_audio_personas"]), {"high_clarity_operator", "degraded_operator"})
        self.assertEqual(
            {condition.parameter_values["profile_key"] for condition in conditions},
            {"baseline", "fast_compact_turns", "wide_asr_search"},
        )
        self.assertEqual({condition.run_type for condition in conditions}, {"text_only", "audio_variant"})
        self.assertEqual(
            sum(condition.run_type == "text_only" for condition in conditions),
            sum(condition.run_type == "audio_variant" for condition in conditions),
        )
        self.assertLessEqual(len(conditions), 15000)

    def test_userlm_agent_a_jobs_clone_platform_tinyllama_matrices(self):
        root = Path(__file__).resolve().parents[1] / "jobs"
        for platform_name in ("linux", "windows"):
            source = load_experiment_job(
                root / f"{platform_name}_agent_a_tinyllama_speech_llm_matrix.job"
            )
            clone = load_experiment_job(
                root / f"{platform_name}_agent_a_userlm_speech_llm_matrix.job"
            )

            self.assertEqual(clone["config"]["agent_a_type"], "userlm")
            self.assertEqual(clone["config"]["agent_b_plugin"], "llm")
            self.assertEqual(clone["grid"], source["grid"])
            self.assertEqual(clone["parameter_profiles"], source["parameter_profiles"])
            self.assertEqual(clone["iterations"], source["iterations"])
            source_config = dict(source["config"])
            clone_config = dict(clone["config"])
            source_config.pop("agent_a_type")
            clone_config.pop("agent_a_type")
            self.assertEqual(clone_config, source_config)

    def test_agent_b_size_jobs_are_complete_and_explicitly_parameterized(self):
        root = Path(__file__).resolve().parents[1] / "jobs"
        expected_models = {
            "small": ["llama3.2:1b", "qwen2.5:1.5b"],
            "medium": ["llama3.2:3b", "phi3:mini"],
            "large": ["qwen2.5:7b", "llama3.1:8b"],
        }

        paths = sorted(root.glob("*_agent_a_*_agent_b_*_llm_matrix.job"))
        self.assertEqual(len(paths), 12)
        for platform_name in ("linux", "windows"):
            for agent_a_type in ("tinyllama", "userlm"):
                for size, models in expected_models.items():
                    path = root / (
                        f"{platform_name}_agent_a_{agent_a_type}_"
                        f"agent_b_{size}_llm_matrix.job"
                    )
                    job = load_experiment_job(path)
                    self.assertEqual(job["config"]["agent_a_type"], agent_a_type)
                    self.assertEqual(job["grid"]["agent_b_models"], models)
                    self.assertEqual(job["grid"]["agent_b_model_tiers"], [size])
                    self.assertTrue(job["config"]["paired_audio_text_runs"])

    def test_agent_b_size_parameter_is_encoded_in_condition_name(self):
        condition = next(build_condition_grid(
            test_case_keys=["morning_peak_cross_city"],
            persona_keys=["focused_commuter"],
            speech_pattern_keys=["clean"],
            model_param_keys=["greedy"],
            objective_modes=["shortest_valid_route_with_constraints"],
            agent_a_audio_persona_keys=["high_clarity_caller"],
            agent_b_audio_persona_keys=["high_clarity_operator"],
            tts_engine_keys=["piper"],
            asr_engine_keys=["faster_whisper"],
            agent_b_model_keys=["llama3.2:1b"],
            parameter_grid={"agent_b_llm_size": ["small"]},
            pair_audio_with_text=False,
        ))

        self.assertIn("-SML-", condition.condition_id)
        self.assertEqual(condition.parameter_values["agent_b_llm_size"], "small")

    def test_tinyllama_piper_whisper_job_has_linked_comparable_profiles(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "jobs"
            / "tinyllama_piper_faster_whisper_comparison.job"
        )
        job = load_experiment_job(source)
        grid = job["grid"]
        conditions = list(build_condition_grid(
            test_case_keys=grid["test_cases"],
            persona_keys=grid["personas"],
            speech_pattern_keys=grid["speech_patterns"],
            model_param_keys=grid["model_params"],
            objective_modes=grid["objective_modes"],
            agent_a_audio_persona_keys=grid["agent_a_audio_personas"],
            agent_b_audio_persona_keys=grid["agent_b_audio_personas"],
            tts_engine_keys=grid["tts_engines"],
            asr_engine_keys=grid["asr_engines"],
            agent_b_model_keys=grid["agent_b_models"],
            iterations=job["iterations"],
            parameter_profiles=job_parameter_profiles(job),
            pair_audio_with_text=True,
        ))

        self.assertEqual(job["config"]["agent_a_type"], "tinyllama")
        self.assertEqual(job["grid"]["agent_b_models"], ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
        self.assertEqual(len(conditions), 32)
        self.assertEqual(
            {condition.parameter_values["profile_key"] for condition in conditions},
            {"baseline", "fast_speech", "acoustic_variation", "wide_asr_search"},
        )
        self.assertEqual({condition.run_type for condition in conditions}, {"text_only", "audio_variant"})

    def test_parallel_job_shards_inherit_one_profile_each(self):
        root = Path(__file__).resolve().parents[1] / "jobs"
        paths = sorted(root.glob("tinyllama_piper_faster_whisper_parallel_*.job"))
        self.assertEqual(len(paths), 4)
        jobs = [load_experiment_job(path) for path in paths]
        self.assertEqual(
            {job["parameter_profiles"][0]["profile_key"] for job in jobs},
            {"baseline", "fast_speech", "acoustic_variation", "wide_asr_search"},
        )
        self.assertTrue(all(len(job["parameter_profiles"]) == 1 for job in jobs))
        self.assertTrue(all(len(job["grid"]["test_cases"]) == 2 for job in jobs))

    def test_faster_whisper_cache_parent_resolves_to_ctranslate_snapshot(self):
        root = Path(__file__).resolve().parents[1]
        cache = root / ".speech-providers" / "models" / "faster-whisper"
        if not cache.exists():
            self.skipTest("Prepared Faster-Whisper cache is not present.")

        resolved = Path(resolve_faster_whisper_model(cache))
        ready, ready_path = faster_whisper_model_ready(cache)

        self.assertTrue(ready)
        self.assertEqual(Path(ready_path), resolved)
        self.assertTrue((resolved / "model.bin").is_file())
        self.assertTrue((resolved / "config.json").is_file())

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
