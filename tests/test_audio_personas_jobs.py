import inspect
import csv
import json
import tempfile
import tkinter as tk
from pathlib import Path
import unittest
from unittest.mock import patch

from coop_navigation_sds.app import default_run_config, normalize_run_config
from coop_navigation_sds.batch import preflight_agent_b_model_grid, write_condition_configuration_breakdown
from coop_navigation_sds.Configuration.assets import (
    faster_whisper_model_ready,
    resolve_faster_whisper_model,
)
from coop_navigation_sds.Configuration.gui import StartupConfigDialog, ToolTip
from coop_navigation_sds.Configuration.jobs import (
    job_linked_profiles,
    job_parameter_grid,
    job_parameter_profiles,
    load_experiment_job,
)
from coop_navigation_sds.experiments import (
    build_condition_grid,
    condition_coverage_report,
    pairwise_factor_rows,
)
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import AgentBPluginConfig
from coop_navigation_sds.TextToSpeech.personas import audio_persona_keys, synthesis_values
from coop_navigation_sds.TransportNetwork.constraints import stage_viability_report
from coop_navigation_sds.TransportNetwork.network import (
    LINES,
    STATION_TRANSFER_TIMES,
    capacity_status,
    line_fullness_percent,
)
from coop_navigation_sds.TransportNetwork.test_cases import TEST_CASES


ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = ROOT / "jobs" / "agent_b_llm"


def _conditions(path):
    job = load_experiment_job(path)
    grid = job["grid"]
    return job, list(build_condition_grid(
        test_case_keys=grid.get("test_cases"),
        persona_keys=grid.get("personas"),
        speech_pattern_keys=grid.get("speech_patterns"),
        model_param_keys=grid.get("model_params"),
        objective_modes=grid.get("objective_modes"),
        agent_a_audio_persona_keys=grid.get("agent_a_audio_personas"),
        agent_b_audio_persona_keys=grid.get("agent_b_audio_personas"),
        tts_engine_keys=grid.get("tts_engines"),
        asr_engine_keys=grid.get("asr_engines"),
        agent_b_model_keys=grid.get("agent_b_models"),
        iterations=job["iterations"],
        parameter_grid=job_parameter_grid(job),
        parameter_profiles=job_parameter_profiles(job),
        linked_profiles=job_linked_profiles(job),
        coverage_strategy=job["coverage_strategy"],
        pair_audio_with_text=job["config"]["paired_audio_text_runs"],
    ))


class AudioPersonaAndJobTests(unittest.TestCase):
    def test_condition_configuration_breakdown_explains_every_paired_run(self):
        job, conditions = _conditions(ROOT / "jobs" / "support" / "small_agent_b_speech_grid.job")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_condition_configuration_breakdown(
                conditions,
                tmpdir,
                agent_a_type=job["config"]["agent_a_type"],
                agent_b_plugin=job["config"]["agent_b_plugin"],
                coverage_report=condition_coverage_report(conditions),
            )
            rows = list(csv.DictReader(paths["csv"].open(encoding="utf-8")))

        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row["run_type"] == "audio_variant" for row in rows), 4)
        self.assertEqual(sum(row["run_type"] == "text_only" for row in rows), 4)
        self.assertTrue(all(row["paired_condition_sequence"] for row in rows))
        self.assertTrue(all("run_type" in row["paired_differences"] for row in rows))
        self.assertEqual(set(paths), {"csv"})

    @patch("coop_navigation_sds.NaturalLanguageGeneration.models.ensure_ollama_models_ready")
    def test_batch_preflight_checks_every_unique_agent_b_model(self, ensure_models):
        ensure_models.return_value = {"available_models": ()}
        conditions = [
            type("Condition", (), {"agent_b_model": model})()
            for model in ("phi3:mini", "llama3.2:3b", "phi3:mini")
        ]
        status = preflight_agent_b_model_grid(
            AgentBPluginConfig("llm"), "ollama", "http://127.0.0.1:11434/api", conditions, 30.0
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
        self.assertTrue(set(callers).isdisjoint(assistants))

    def test_audio_persona_controls_resolved_synthesis(self):
        config = default_run_config()
        config.update({"agent_a_audio_persona": "hurried_caller", "agent_a_custom_audio": True})
        normalized = normalize_run_config(config)
        profile = synthesis_values("hurried_caller")
        self.assertFalse(normalized["agent_a_custom_audio"])
        self.assertEqual(normalized["agent_a_words_per_minute"], profile["words_per_minute"])
        self.assertEqual(normalized["agent_a_pause_ms"], profile["pause_ms"])

    def test_invalid_job_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.job"
            path.write_text('{"schema_version": 99}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_experiment_job(path)

    def test_agent_b_matrix_has_one_primary_and_one_comparison_per_size_and_caller(self):
        paths = sorted((JOB_ROOT / "userlm").rglob("*.job"))
        paths.extend(sorted((JOB_ROOT / "tinyllama_comparison").rglob("*.job")))
        self.assertEqual(len(paths), 12)
        observed = set()
        for path in paths:
            job = load_experiment_job(path)
            size = job["parameter_values"]["agent_b_llm_size"][0]
            role = job["parameter_values"]["agent_b_model_role"][0]
            caller = job["config"]["agent_a_type"]
            model = job["config"]["model_name"]
            observed.add((size, role, caller, model))
        expected_models = {
            ("small", "primary"): "llama3.2:1b",
            ("small", "model_comparison"): "qwen2.5:1.5b",
            ("medium", "primary"): "llama3.2:3b",
            ("medium", "model_comparison"): "phi3:mini",
            ("large", "primary"): "llama3.1:8b",
            ("large", "model_comparison"): "qwen2.5:7b",
        }
        self.assertEqual(observed, {
            (size, role, caller, model)
            for (size, role), model in expected_models.items()
            for caller in ("userlm", "tinyllama")
        })

    def test_caller_comparison_jobs_are_matched_except_caller_and_result_group(self):
        for user_path in sorted((JOB_ROOT / "userlm").rglob("*.job")):
            relative = user_path.relative_to(JOB_ROOT / "userlm")
            tiny_path = JOB_ROOT / "tinyllama_comparison" / relative
            user = load_experiment_job(user_path)
            tiny = load_experiment_job(tiny_path)
            self.assertEqual(tiny["config"]["agent_a_type"], "tinyllama")
            self.assertEqual(user["config"]["agent_a_type"], "userlm")
            self.assertEqual(tiny["grid"], user["grid"])
            self.assertEqual(tiny["linked_profiles"], user["linked_profiles"])
            self.assertEqual(tiny["parameter_values"], user["parameter_values"])
            comparable_user = dict(user["config"])
            comparable_tiny = dict(tiny["config"])
            for key in ("agent_a_type", "result_group"):
                comparable_user.pop(key)
                comparable_tiny.pop(key)
            self.assertEqual(comparable_tiny, comparable_user)

    def test_userlm_model_is_fixed_independently_of_agent_b(self):
        for path in (JOB_ROOT / "userlm").rglob("*.job"):
            job = load_experiment_job(path)
            config = job["config"]
            self.assertEqual(config["agent_a_model_name"], "microsoft/UserLM-8b")
            self.assertEqual(config["agent_a_model_provider"], "transformers")
            self.assertEqual(job["grid"]["agent_b_models"], [config["model_name"]])

    def test_every_agent_b_job_has_balanced_pairwise_conditions(self):
        groups = set()
        paths = list((JOB_ROOT / "userlm").rglob("*.job"))
        paths.extend((JOB_ROOT / "tinyllama_comparison").rglob("*.job"))
        for path in paths:
            job, conditions = _conditions(path)
            self.assertEqual(len(conditions), 26, path)
            self.assertEqual(condition_coverage_report(conditions)["missing_pairs"], [])
            self.assertEqual(sum(row.run_type == "text_only" for row in conditions), 13)
            self.assertEqual(sum(row.run_type == "audio_variant" for row in conditions), 13)
            self.assertEqual(job["config"]["num_turns"], 20)
            self.assertEqual(job["config"]["log_profile"], "full")
            group = job["config"]["result_group"]
            self.assertTrue(group.startswith("agent_b/"))
            self.assertNotIn(group, groups)
            groups.add(group)

    def test_userlm8b_expanded_speech_jobs_cover_two_models_per_size(self):
        paths = sorted((JOB_ROOT / "userlm_speech_grid").rglob("*.job"))
        self.assertEqual(len(paths), 6)
        observed = set()
        for path in paths:
            job, conditions = _conditions(path)
            self.assertEqual(job["config"]["agent_a_model_name"], "microsoft/UserLM-8b")
            self.assertEqual(job["config"]["agent_a_model_provider"], "transformers")
            self.assertEqual(len(conditions), 8)
            self.assertTrue(
                condition_coverage_report(conditions)["speech_performance_coverage"]["complete"]
            )
            observed.add((
                job["parameter_values"]["agent_b_llm_size"][0],
                job["parameter_values"]["agent_b_model_role"][0],
                job["config"]["model_name"],
            ))
        self.assertEqual(observed, {
            ("small", "primary", "llama3.2:1b"),
            ("small", "model_comparison", "qwen2.5:1.5b"),
            ("medium", "primary", "llama3.2:3b"),
            ("medium", "model_comparison", "phi3:mini"),
            ("large", "primary", "llama3.1:8b"),
            ("large", "model_comparison", "qwen2.5:7b"),
        })

    def test_expanded_small_agent_b_speech_grid_is_balanced_and_complete(self):
        path = ROOT / "jobs" / "support" / "small_agent_b_speech_grid.job"
        job, conditions = _conditions(path)
        coverage = condition_coverage_report(conditions)
        self.assertEqual(job["coverage_strategy"], "full_factorial")
        self.assertEqual(len(conditions), 8)
        self.assertEqual(sum(row.run_type == "audio_variant" for row in conditions), 4)
        self.assertEqual(sum(row.run_type == "text_only" for row in conditions), 4)
        self.assertEqual(job["grid"]["tts_engines"], ["piper"])
        self.assertEqual(job["grid"]["asr_engines"], ["vosk"])
        performance = coverage["speech_performance_coverage"]
        self.assertTrue(performance["complete"])
        self.assertEqual(performance["required_bands"], [
            "ceiling", "nominal", "challenging", "floor",
        ])
        self.assertEqual(len(job_linked_profiles(job)["speech_performance"]), 4)

    def test_userlm_speech_grid_is_an_exact_caller_comparison(self):
        source, source_conditions = _conditions(
            ROOT / "jobs" / "support" / "small_agent_b_speech_grid.job"
        )
        userlm, userlm_conditions = _conditions(
            ROOT / "jobs" / "support" / "small_agent_b_speech_grid_userlm.job"
        )
        self.assertEqual(userlm["config"]["agent_a_type"], "userlm")
        self.assertEqual(userlm["config"]["agent_a_model_name"], "microsoft/UserLM-8b")
        self.assertEqual(userlm["grid"], source["grid"])
        self.assertEqual(userlm["parameter_values"], source["parameter_values"])
        self.assertEqual(userlm["linked_profiles"], source["linked_profiles"])
        self.assertEqual(userlm["coverage_strategy"], source["coverage_strategy"])
        self.assertEqual(len(userlm_conditions), len(source_conditions), 8)

    def test_all_task_profiles_are_viable_for_three_constraint_stages(self):
        job = load_experiment_job(JOB_ROOT / "userlm" / "primary" / "01-small-llama3.2-1b.job")
        failures = []
        for profile in job_linked_profiles(job)["task"]:
            case = TEST_CASES[profile["test_case_key"]].with_persona(profile["persona_key"])
            report = stage_viability_report(case.scenario, case.persona, max_constraints=3)
            if not report["all_stage_requirements_satisfied"]:
                failures.append(profile["task_profile_key"])
        self.assertEqual(failures, [])

    def test_selected_scenarios_cover_network_condition_classes(self):
        job = load_experiment_job(JOB_ROOT / "userlm" / "primary" / "01-small-llama3.2-1b.job")
        scenario_keys = {row["test_case_key"] for row in job_linked_profiles(job)["task"]}
        fullness = {
            capacity_status(line_fullness_percent(line_name, case.scenario["start_time_min"]))
            for key, case in TEST_CASES.items() if key in scenario_keys
            for line_name, line in LINES.items() if line.get("kind") != "walking"
        }
        delays = {line.get("delay_probability_class") for line in LINES.values() if line.get("kind") != "walking"}
        self.assertEqual(fullness, {"near capacity", "not near capacity"})
        self.assertEqual(delays, {"low", "moderate", "high"})
        self.assertEqual(set(STATION_TRANSFER_TIMES.values()), {2, 3, 4, 5, 6, 7})

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
                self.assertEqual(
                    {(row[left], row[right]) for row in rows},
                    {(a, b) for a in left_values for b in right_values},
                )

    def test_all_job_files_use_twenty_turns_and_full_logging(self):
        paths = list((ROOT / "jobs").rglob("*.job"))
        paths.extend((ROOT / "coop_navigation_sds" / "Configuration" / "presets").rglob("*.job"))
        for path in paths:
            job = load_experiment_job(path)
            self.assertEqual(job["config"].get("num_turns"), 20, path)
            self.assertEqual(job["config"].get("log_profile"), "full", path)

    def test_faster_whisper_cache_resolves_to_ready_snapshot(self):
        cache = ROOT / ".speech-providers" / "models" / "faster-whisper"
        if not cache.exists():
            self.skipTest("Prepared Faster-Whisper cache is not present.")
        resolved = Path(resolve_faster_whisper_model(cache))
        ready, ready_path = faster_whisper_model_ready(cache)
        self.assertTrue(ready)
        self.assertEqual(Path(ready_path), resolved)
        self.assertTrue((resolved / "model.bin").is_file())

    def test_configuration_window_has_tooltips_and_linux_handling(self):
        source = inspect.getsource(StartupConfigDialog)
        self.assertIn("ToolTip", source)
        self.assertIn("WAYLAND_DISPLAY", source)
        self.assertIn("_scroll_wheel", source)
        self.assertTrue(callable(ToolTip))

    def test_configuration_window_reports_linux_headless_prerequisites(self):
        with patch("coop_navigation_sds.Configuration.gui.tk.Tk", side_effect=tk.TclError("display unavailable")):
            with self.assertRaisesRegex(RuntimeError, "python3-tk.*DISPLAY.*WAYLAND_DISPLAY"):
                StartupConfigDialog({}, {})


if __name__ == "__main__":
    unittest.main()
