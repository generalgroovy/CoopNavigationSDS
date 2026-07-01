import inspect
import json
import tempfile
import tkinter as tk
from pathlib import Path
import unittest
from unittest.mock import patch

from coop_navigation_sds.app import default_run_config, normalize_run_config
from coop_navigation_sds.Configuration.gui import StartupConfigDialog, ToolTip
from coop_navigation_sds.Configuration.jobs import (
    job_linked_profiles,
    job_parameter_grid,
    job_parameter_profiles,
    load_experiment_job,
)
from coop_navigation_sds.Configuration.assets import (
    faster_whisper_model_ready,
    resolve_faster_whisper_model,
)
from coop_navigation_sds.experiments import (
    build_condition_grid,
    condition_coverage_report,
    pairwise_factor_rows,
)
from coop_navigation_sds.batch import preflight_agent_b_model_grid
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import AgentBPluginConfig
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import PERSONAS
from coop_navigation_sds.TextToSpeech.personas import audio_persona_keys, synthesis_values
from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import normalize_agent_a_type
from coop_navigation_sds.TransportNetwork.test_cases import TEST_CASES
from coop_navigation_sds.TransportNetwork.constraints import stage_viability_report
from coop_navigation_sds.TransportNetwork.network import (
    LINES,
    STATION_TRANSFER_TIMES,
    capacity_status,
    line_fullness_percent,
)


class AudioPersonaAndJobTests(unittest.TestCase):
    @patch("coop_navigation_sds.NaturalLanguageGeneration.models.ensure_ollama_models_ready")
    def test_batch_preflight_checks_every_unique_agent_b_model(self, ensure_models):
        ensure_models.return_value = {"available_models": ()}
        conditions = [
            type("Condition", (), {"agent_b_model": model})()
            for model in ("phi3:mini", "llama3.2:3b", "phi3:mini")
        ]

        status = preflight_agent_b_model_grid(
            AgentBPluginConfig("llm"),
            "ollama",
            "http://127.0.0.1:11434/api",
            conditions,
            30.0,
        )

        self.assertEqual(status, {"available_models": ()})
        ensure_models.assert_called_once_with(
            "http://127.0.0.1:11434/api",
            ["llama3.2:3b", "phi3:mini"],
            timeout_sec=30.0,
            models_dir=None,
        )

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

    def test_platform_jobs_inherit_one_os_neutral_canonical_matrix(self):
        root = Path(__file__).resolve().parents[1] / "jobs"
        canonical = load_experiment_job(root / "speech_llm_coverage_matrix.job")
        self.assertEqual(canonical["parameter_values"]["matrix_family"], ["speech_llm_coverage_v1"])
        self.assertEqual(canonical["parameter_values"]["experiment_platform"], ["current"])

        for platform_name in ("linux", "windows"):
            path = root / f"{platform_name}_agent_a_tinyllama_speech_llm_matrix.job"
            raw = json.loads(path.read_text(encoding="utf-8"))
            resolved = load_experiment_job(path)
            self.assertEqual(raw["extends"], "speech_llm_coverage_matrix.job")
            self.assertEqual(resolved["grid"], canonical["grid"])
            self.assertEqual(resolved["linked_profiles"], canonical["linked_profiles"])
            self.assertEqual(resolved["parameter_values"]["matrix_family"], ["speech_llm_coverage_v1"])
            self.assertEqual(resolved["parameter_values"]["experiment_platform"], [platform_name])

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
            linked_profiles=job_linked_profiles(job),
            coverage_strategy=job["coverage_strategy"],
            pair_audio_with_text=job["config"]["paired_audio_text_runs"],
        ))
        coverage = condition_coverage_report(conditions)

        self.assertEqual(job["config"]["agent_a_type"], "tinyllama")
        self.assertEqual(job["config"]["model_provider"], "ollama")
        self.assertEqual(job["coverage_strategy"], "pairwise")
        self.assertEqual(len(grid["agent_b_models"]), 6)
        self.assertEqual(len(grid["tts_engines"]), 4)
        self.assertEqual(len(grid["asr_engines"]), 4)
        self.assertEqual(set(grid["agent_a_audio_personas"]), set(audio_persona_keys("caller")))
        self.assertEqual(set(grid["agent_b_audio_personas"]), set(audio_persona_keys("assistant")))
        self.assertEqual(len(job_linked_profiles(job)["task"]), len(PERSONAS))
        self.assertEqual(
            {profile["persona_key"] for profile in job_linked_profiles(job)["task"]},
            set(PERSONAS),
        )
        self.assertEqual(
            {profile["test_case_key"] for profile in job_linked_profiles(job)["task"]},
            set(TEST_CASES),
        )
        self.assertEqual(coverage["pair_coverage_ratio"], 1.0)
        self.assertEqual(coverage["missing_pairs"], [])
        self.assertEqual(len(conditions), 330)
        self.assertEqual({condition.run_type for condition in conditions}, {"text_only", "audio_variant"})
        self.assertEqual(
            sum(condition.run_type == "text_only" for condition in conditions),
            sum(condition.run_type == "audio_variant" for condition in conditions),
        )
        self.assertLessEqual(len(conditions), 15000)
        audio_conditions = [
            condition for condition in conditions if condition.run_type == "audio_variant"
        ]
        self.assertEqual(
            {(condition.tts_engine, condition.speech_pattern_key) for condition in audio_conditions},
            {
                (tts, pattern)
                for tts in grid["tts_engines"]
                for pattern in grid["speech_patterns"]
            },
        )
        self.assertEqual(
            {(condition.agent_b_model, condition.agent_b_audio_persona) for condition in audio_conditions},
            {
                (model, persona)
                for model in grid["agent_b_models"]
                for persona in grid["agent_b_audio_personas"]
            },
        )
        self.assertEqual(
            {
                (condition.asr_engine, dict(condition.parameter_values)["asr_beam_size"])
                for condition in audio_conditions
            },
            {
                (engine, width)
                for engine in grid["asr_engines"]
                for width in (1, 6, 11, 16)
            },
        )

    def test_pairwise_rows_cover_every_value_pair(self):
        factors = [
            ("scenario", ["a", "b", "c"]),
            ("persona", ["p1", "p2"]),
            ("model", ["m1", "m2", "m3"]),
            ("audio", ["clear", "degraded"]),
        ]
        rows = pairwise_factor_rows(factors)

        for left_index, (left, left_values) in enumerate(factors):
            for right, right_values in factors[left_index + 1:]:
                observed = {(row[left], row[right]) for row in rows}
                expected = {(a, b) for a in left_values for b in right_values}
                self.assertEqual(observed, expected)
        self.assertLess(len(rows), 3 * 2 * 3 * 2)

    def test_comprehensive_task_profiles_support_three_constraint_stages(self):
        root = Path(__file__).resolve().parents[1]
        job = load_experiment_job(
            root / "jobs" / "linux_agent_a_tinyllama_speech_llm_matrix.job"
        )

        failures = []
        for profile in job_linked_profiles(job)["task"]:
            case = TEST_CASES[profile["test_case_key"]].with_persona(
                profile["persona_key"]
            )
            report = stage_viability_report(
                case.scenario,
                case.persona,
                max_constraints=3,
            )
            if not report["all_stage_requirements_satisfied"]:
                failures.append(profile["task_profile_key"])

        self.assertEqual(failures, [])

    def test_comprehensive_scenarios_cover_network_condition_classes(self):
        root = Path(__file__).resolve().parents[1]
        job = load_experiment_job(
            root / "jobs" / "linux_agent_a_tinyllama_speech_llm_matrix.job"
        )
        scenario_keys = {
            profile["test_case_key"] for profile in job_linked_profiles(job)["task"]
        }
        fullness_states = {
            capacity_status(line_fullness_percent(line_name, case.scenario["start_time_min"]))
            for case_key, case in TEST_CASES.items()
            if case_key in scenario_keys
            for line_name, line in LINES.items()
            if line.get("kind") != "walking"
        }
        delay_classes = {
            line.get("delay_probability_class")
            for line in LINES.values()
            if line.get("kind") != "walking"
        }

        self.assertEqual(fullness_states, {"near capacity", "not near capacity"})
        self.assertEqual(delay_classes, {"low", "moderate", "high"})
        self.assertEqual(set(STATION_TRANSFER_TIMES.values()), {2, 3, 4, 5, 6, 7})

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
            self.assertEqual(clone["linked_profiles"], source["linked_profiles"])
            self.assertEqual(clone["coverage_strategy"], source["coverage_strategy"])
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
                    self.assertEqual(job["parameter_values"]["experiment_platform"], [platform_name])
                    self.assertEqual(job["parameter_values"]["matrix_family"], ["speech_llm_coverage_v1"])

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
            parameter_grid=job_parameter_grid(job),
            parameter_profiles=job_parameter_profiles(job),
            coverage_strategy=job["coverage_strategy"],
            pair_audio_with_text=True,
        ))

        self.assertEqual(job["config"]["agent_a_type"], "tinyllama")
        self.assertEqual(job["grid"]["agent_b_models"], ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
        self.assertEqual(len(conditions), 308)
        self.assertEqual(
            {condition.parameter_values["profile_key"] for condition in conditions},
            {"baseline", "fast_speech", "acoustic_variation", "slow_clear_speech"},
        )
        self.assertEqual(
            {dict(condition.parameter_values)["asr_beam_size"] for condition in conditions},
            {1, 6, 11, 16},
        )
        self.assertEqual(len({condition.speech_pattern_key for condition in conditions}), 11)
        self.assertEqual(len({condition.agent_b_audio_persona for condition in conditions}), 7)
        self.assertEqual(condition_coverage_report(conditions)["missing_pairs"], [])
        self.assertEqual({condition.run_type for condition in conditions}, {"text_only", "audio_variant"})

    def test_tinyllama_piper_vosk_job_is_a_matched_recognizer_sibling(self):
        root = Path(__file__).resolve().parents[1] / "jobs"
        whisper = load_experiment_job(root / "tinyllama_piper_faster_whisper_comparison.job")
        vosk = load_experiment_job(root / "tinyllama_piper_vosk_comparison.job")

        self.assertEqual(vosk["config"]["agent_a_type"], "tinyllama")
        self.assertEqual(vosk["config"]["tts_engine"], "piper")
        self.assertEqual(vosk["config"]["asr_engine"], "vosk")
        self.assertEqual(vosk["grid"]["asr_engines"], ["vosk"])
        self.assertEqual(vosk["grid"]["test_cases"], whisper["grid"]["test_cases"])
        self.assertEqual(vosk["parameter_profiles"], whisper["parameter_profiles"])

    def test_requested_recognizer_jobs_are_bounded_and_paired(self):
        root = Path(__file__).resolve().parents[1] / "jobs"
        for filename, engine in (
            ("requested_tinyllama_piper_faster_whisper.job", "faster_whisper"),
            ("requested_tinyllama_piper_vosk.job", "vosk"),
        ):
            job = load_experiment_job(root / filename)
            self.assertEqual(job["iterations"], 1)
            self.assertEqual(job["config"]["num_turns"], 20)
            self.assertEqual(job["config"]["log_profile"], "full")
            self.assertTrue(job["config"]["paired_audio_text_runs"])
            self.assertEqual(job["grid"]["asr_engines"], [engine])
            self.assertEqual(len(job["grid"]["test_cases"]), 1)

    def test_parallel_job_shards_inherit_one_profile_each(self):
        root = Path(__file__).resolve().parents[1] / "jobs"
        paths = sorted(root.glob("tinyllama_piper_faster_whisper_parallel_*.job"))
        self.assertEqual(len(paths), 4)
        jobs = [load_experiment_job(path) for path in paths]
        self.assertEqual(
            {job["parameter_profiles"][0]["profile_key"] for job in jobs},
            {"baseline", "fast_speech", "acoustic_variation", "slow_clear_speech"},
        )
        self.assertTrue(all(len(job["parameter_profiles"]) == 1 for job in jobs))
        self.assertTrue(all(len(job["grid"]["test_cases"]) == 2 for job in jobs))

    def test_all_batch_jobs_use_twenty_turns_and_full_logging(self):
        root = Path(__file__).resolve().parents[1]
        paths = list((root / "jobs").glob("*.job"))
        paths.extend((root / "coop_navigation_sds" / "Configuration" / "presets").glob("*.job"))
        for path in paths:
            job = load_experiment_job(path)
            self.assertEqual(job["config"].get("num_turns"), 20, path.name)
            self.assertEqual(job["config"].get("log_profile"), "full", path.name)

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
        self.assertIn("WAYLAND_DISPLAY", source)
        self.assertIn("_scroll_wheel", source)
        self.assertTrue(callable(ToolTip))

    def test_configuration_window_reports_linux_headless_prerequisites(self):
        with patch("coop_navigation_sds.Configuration.gui.tk.Tk", side_effect=tk.TclError("display unavailable")):
            with self.assertRaisesRegex(RuntimeError, "python3-tk.*DISPLAY.*WAYLAND_DISPLAY"):
                StartupConfigDialog({}, {})


if __name__ == "__main__":
    unittest.main()
