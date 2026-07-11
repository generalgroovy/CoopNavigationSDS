import json
from pathlib import Path
import re
import tempfile

import pytest

from scripts.run_agent_b_llm_batch import (
    job_condition_count,
    job_overview,
    load_batch_manifest,
)
from scripts.submit_agent_b_model_jobs import (
    condition_chunks,
    condition_chunks_by_size,
    discover_jobs,
    resources_for,
)
from coop_navigation_sds.Configuration.jobs import (
    job_linked_profiles,
    job_parameter_grid,
    job_parameter_profiles,
    load_experiment_job,
)
from coop_navigation_sds.experiments import build_condition_grid


ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / "jobs" / "agent_b_llm" / "batches"
TRANSFORMERS_GRID_ROOT = ROOT / "jobs" / "agent_b_llm" / "transformers_speech_grid"
USERLM_LARGE2_JOB = ROOT / "jobs" / "agent_b_llm" / "userlm_speech_grid" / "large" / "02-qwen2.5-7b.job"
USERLM_SLURM_ARRAYS = {
    "userlm_small1_cpu_array.sbatch": ROOT / "jobs" / "agent_b_llm" / "userlm_speech_grid" / "small" / "01-llama3.2-1b.job",
    "userlm_small2_cpu_array.sbatch": ROOT / "jobs" / "agent_b_llm" / "userlm_speech_grid" / "small" / "02-qwen2.5-1.5b.job",
    "userlm_large2_cpu_array.sbatch": USERLM_LARGE2_JOB,
}
USERLM_SPEECH_ROOT = ROOT / "jobs" / "agent_b_llm" / "userlm_speech_grid"
USERLM_TRANSFORMERS_ROOT = ROOT / "jobs" / "agent_b_llm" / "userlm_transformers_speech_grid"


def test_complete_manifest_resolves_twelve_unique_jobs():
    manifest = load_batch_manifest(BATCH_ROOT / "all.json")
    assert len(manifest["jobs"]) == 12
    assert len(set(manifest["jobs"])) == 12
    assert sum(job_condition_count(path) for path in manifest["jobs"]) == 312


def test_userlm_manifest_covers_two_agent_b_models_per_size():
    manifest = load_batch_manifest(BATCH_ROOT / "05-userlm-all-models.json")
    rows = [job_overview(path) for path in manifest["jobs"]]
    assert len(rows) == 6
    assert sum(row["conditions"] for row in rows) == 156
    assert {row["agent_b_size"] for row in rows} == {"small", "medium", "large"}
    assert {row["model_role"] for row in rows} == {"primary", "model_comparison"}
    assert {row["agent_a"] for row in rows} == {"userlm"}
    assert {row["agent_a_model"] for row in rows} == {"microsoft/UserLM-8b"}


def test_userlm_expanded_speech_manifest_has_six_matched_jobs():
    manifest = load_batch_manifest(BATCH_ROOT / "06-userlm-speech-grid-all-models.json")
    rows = [job_overview(path) for path in manifest["jobs"]]
    assert len(rows) == 6
    assert sum(row["conditions"] for row in rows) == 504
    assert {row["agent_b_size"] for row in rows} == {"small", "medium", "large"}
    assert {row["agent_a_model"] for row in rows} == {"microsoft/UserLM-8b"}
    assert all(row["conditions"] == 84 for row in rows)


def _model_independent_condition_signature(condition):
    parameters = dict(condition.parameter_values)
    for key in (
        "agent_b_llm_size",
        "agent_b_model_role",
        "agent_b_model_slot",
        "matrix_family",
    ):
        parameters.pop(key, None)
    return (
        condition.test_case_key,
        condition.persona_key,
        condition.scenario_key,
        condition.speech_pattern_key,
        condition.model_param_key,
        condition.objective_mode,
        condition.iteration,
        condition.agent_a_audio_persona,
        condition.agent_b_audio_persona,
        condition.run_type,
        condition.tts_engine,
        condition.asr_engine,
        tuple(sorted(parameters.items())),
    )


def _condition_signatures(job_path):
    job = load_experiment_job(job_path)
    grid = job["grid"]
    return {
        _model_independent_condition_signature(condition)
        for condition in build_condition_grid(
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
            pair_audio_with_text=bool(job["config"].get("paired_audio_text_runs", False)),
        )
    }


def test_userlm_agent_b_slurm_jobs_share_identical_non_model_condition_coverage():
    job_paths = sorted(USERLM_SPEECH_ROOT.glob("*/*.job")) + sorted(USERLM_TRANSFORMERS_ROOT.glob("*/*.job"))
    assert len(job_paths) == 18
    reference = _condition_signatures(job_paths[0])
    assert job_condition_count(job_paths[0]) == 84
    assert len(reference) >= 60
    for job_path in job_paths[1:]:
        assert _condition_signatures(job_path) == reference


def test_transformers_agent_b_manifest_has_four_models_per_size_and_eighty_four_conditions_each():
    manifest = load_batch_manifest(BATCH_ROOT / "07-transformers-agent-b-all.json")
    rows = [job_overview(path) for path in manifest["jobs"]]

    assert len(rows) == 12
    assert sum(row["conditions"] for row in rows) == 1008
    assert {
        size: sum(row["agent_b_size"] == size for row in rows)
        for size in ("small", "medium", "large")
    } == {"small": 4, "medium": 4, "large": 4}
    assert {row["agent_a"] for row in rows} == {"tinyllama"}
    assert all(row["conditions"] == 84 for row in rows)


def test_userlm_transformers_submitter_filters_and_sorts_by_model_size():
    class Args:
        root = [str(USERLM_TRANSFORMERS_ROOT)]
        family = "all"
        tier = ("small",)
        provider = ("transformers",)
        cpus_per_task = 0
        memory = ""
        time_limit = ""

    jobs = discover_jobs(Args)

    assert [job.family for job in jobs] == ["userlm"] * 4
    assert [job.provider for job in jobs] == ["transformers"] * 4
    assert [job.agent_b_memory_gb for job in jobs] == sorted(job.agent_b_memory_gb for job in jobs)
    assert [resources_for(job, Args) for job in jobs] == [
        (6, "48G", "03:59:00"),
        (6, "48G", "03:59:00"),
        (6, "52G", "03:59:00"),
        (6, "52G", "03:59:00"),
    ]


def test_submitter_splits_eighty_four_conditions_into_four_balanced_arrays():
    assert condition_chunks(84, 4) == [
        (0, 20, 1, 4),
        (21, 41, 2, 4),
        (42, 62, 3, 4),
        (63, 83, 4, 4),
    ]


def test_submitter_splits_eighty_four_conditions_into_reliable_fourteen_task_arrays():
    assert condition_chunks_by_size(84, 14) == [
        (0, 13, 1, 6),
        (14, 27, 2, 6),
        (28, 41, 3, 6),
        (42, 55, 4, 6),
        (56, 69, 5, 6),
        (70, 83, 6, 6),
    ]


def test_transformers_slurm_arrays_are_single_condition_shards():
    scripts = {
        "small": ROOT / "slurm" / "transformers_agent_b_small_cpu_array.sbatch",
        "medium": ROOT / "slurm" / "transformers_agent_b_medium_cpu_array.sbatch",
        "large": ROOT / "slurm" / "transformers_agent_b_large_cpu_array.sbatch",
    }
    for tier, script_path in scripts.items():
        script = script_path.read_text(encoding="utf-8")
        assert "#SBATCH --array=0-83%1" in script
        assert "--condition-count 1" in script
        assert "--model-device cpu" in script
        assert "--no-update-coverage-registry" in script
        assert "--fail-fast" not in script
        assert "JOB_FILE" in script
        for job_path in (TRANSFORMERS_GRID_ROOT / tier).glob("*.job"):
            assert job_condition_count(job_path) == 84


@pytest.mark.parametrize(("script_name", "job_path"), sorted(USERLM_SLURM_ARRAYS.items()))
def test_userlm_slurm_arrays_match_job_condition_count(script_name, job_path):
    script = (ROOT / "slurm" / script_name).read_text(encoding="utf-8")
    match = re.search(r"#SBATCH\s+--array=0-(\d+)%1", script)
    assert match is not None
    assert int(match.group(1)) + 1 == job_condition_count(job_path)
    assert job_path.name in script
    assert "--condition-count 1" in script
    assert "--model-device cpu" in script
    assert "--no-update-coverage-registry" in script
    assert "--fail-fast" not in script
    assert "--model-base-url" in script
    assert "#SBATCH --output=slurm/logs/" in script


def test_generic_agent_b_model_array_disables_live_coverage_registry_updates():
    script = (ROOT / "slurm" / "agent_b_model_cpu_array.sbatch").read_text(encoding="utf-8")

    assert "--condition-count 1" in script
    assert "--model-device cpu" in script
    assert "--agent-a-model-device cpu" in script
    assert "--no-update-coverage-registry" in script
    assert "--no-require-complete-speech-performance-coverage" in script
    assert "--fail-fast" not in script


def test_slurm_logs_are_not_ignored_because_cluster_diagnostics_are_result_evidence():
    ignore_file = ROOT / ".gitignore"
    text = ignore_file.read_text(encoding="utf-8") if ignore_file.exists() else ""

    assert "slurm/logs/*.out" not in text
    assert "slurm/logs/*.err" not in text
    assert "slurm/logs/*.log" not in text


def test_overviews_have_unique_model_grouped_result_paths():
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        results = Path(temporary)
        manifest = load_batch_manifest(BATCH_ROOT / "all.json")
        rows = [job_overview(path, results) for path in manifest["jobs"]]
        paths = [Path(row["result_path"]) for row in rows]
        assert len(paths) == len(set(paths))
        assert all(results.resolve() in path.parents for path in paths)
        assert all("agent_b" not in path.relative_to(results).parts for path in paths)
        assert {row["agent_b_size"] for row in rows} == {"small", "medium", "large"}
        assert {row["model_role"] for row in rows} == {"primary", "model_comparison"}


def test_cyclic_batch_manifest_is_rejected():
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        first = Path(temporary) / "first.json"
        second = Path(temporary) / "second.json"
        first.write_text(json.dumps({"schema_version": 1, "includes": ["second.json"]}), encoding="utf-8")
        second.write_text(json.dumps({"schema_version": 1, "includes": ["first.json"]}), encoding="utf-8")
        with pytest.raises(ValueError, match="Cyclic"):
            load_batch_manifest(first)
