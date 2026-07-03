import json
from pathlib import Path
import tempfile

import pytest

from scripts.run_agent_b_llm_batch import (
    job_condition_count,
    job_overview,
    load_batch_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / "jobs" / "agent_b_llm" / "batches"


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
    assert sum(row["conditions"] for row in rows) == 312
    assert {row["agent_b_size"] for row in rows} == {"small", "medium", "large"}
    assert {row["agent_a_model"] for row in rows} == {"microsoft/UserLM-8b"}


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
