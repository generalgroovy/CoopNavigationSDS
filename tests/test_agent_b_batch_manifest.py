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


ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / "jobs" / "agent_b_llm" / "batches"
TRANSFORMERS_GRID_ROOT = ROOT / "jobs" / "agent_b_llm" / "transformers_speech_grid"
USERLM_LARGE2_JOB = ROOT / "jobs" / "agent_b_llm" / "userlm_speech_grid" / "large" / "02-qwen2.5-7b.job"
USERLM_SLURM_ARRAYS = {
    "userlm_small1_cpu_array.sbatch": ROOT / "jobs" / "agent_b_llm" / "userlm_speech_grid" / "small" / "01-llama3.2-1b.job",
    "userlm_small2_cpu_array.sbatch": ROOT / "jobs" / "agent_b_llm" / "userlm_speech_grid" / "small" / "02-qwen2.5-1.5b.job",
    "userlm_large2_cpu_array.sbatch": USERLM_LARGE2_JOB,
}


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
    assert "--model-base-url" in script
    assert "#SBATCH --output=slurm/logs/" in script


def test_overviews_have_unique_model_grouped_result_paths():
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        results = Path(temporary)
        manifest = load_batch_manifest(BATCH_ROOT / "all.json")
        rows = [job_overview(path, results) for path in manifest["jobs"]]
        paths = [Path(row["result_path"]) for row in rows]
        assert len(paths) == len(set(paths))
        assert all(results.resolve() in path.parents for path in paths)
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
