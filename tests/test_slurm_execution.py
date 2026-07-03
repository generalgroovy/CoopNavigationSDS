import json
from pathlib import Path
import tempfile

import pytest

from coop_navigation_sds.Configuration.slurm_grid import (
    SlurmConditionGrid,
    condition_directory_name,
    export_condition_json,
    reserve_condition_directory,
)
from minillama.orchestration.run_experiments import condition_job_document
from scripts.run_slurm_condition import condition_index
from coop_navigation_sds.TransportNetwork.constraints import stage_viability_report
from coop_navigation_sds.TransportNetwork.test_cases import get_test_case


def _document(run_modes=None):
    return {
        "schema_version": 1,
        "name": "test_grid",
        "agent_b": [
            "simple",
            "minillama",
            {
                "key": "custom_model",
                "plugin": "example.plugins:AgentB",
                "model_provider": "transformers",
                "model_name": "example/model",
            },
        ],
        "personas": ["focused_commuter", "hesitant_speaker"],
        "test_cases": ["morning_peak_cross_city"],
        "run_modes": run_modes or ["pure_text", "speech"],
        "speech_patterns": ["clean", "hesitant"],
        "seeds": [11, 29],
        "repetitions": 2,
        "agent_a_type": "staged",
        "speech": {"tts_engine": "piper", "asr_engine": "faster_whisper"},
        "base_config": {"num_turns": 8},
    }


def test_condition_index_mapping_is_deterministic():
    first = SlurmConditionGrid.from_document(_document())
    second = SlurmConditionGrid.from_document(_document())

    assert len(first.conditions) == 72
    assert [row.as_dict() for row in first.conditions] == [
        row.as_dict() for row in second.conditions
    ]
    assert first.condition(0).backend.key == "simple"
    assert first.condition(71).backend.key == "custom_model"
    assert condition_index(None, {"SLURM_ARRAY_TASK_ID": "17"}) == 17
    assert condition_index(3, {"SLURM_ARRAY_TASK_ID": "17"}) == 3


def test_result_directory_names_are_unique_and_retry_safe():
    grid = SlurmConditionGrid.from_document(_document(["pure_text"]))
    names = [condition_directory_name(condition) for condition in grid.conditions]
    assert len(names) == len(set(names))

    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=project_root) as temporary:
        first = reserve_condition_directory(temporary, grid.condition(0))
        retry = reserve_condition_directory(temporary, grid.condition(0))
        assert first != retry
        assert retry.name.endswith("-a02")


def test_speech_patterns_only_expand_speech_conditions():
    grid = SlurmConditionGrid.from_document(_document())
    text_rows = [row for row in grid.conditions if row.run_mode == "pure_text"]
    speech_rows = [row for row in grid.conditions if row.run_mode == "speech"]

    assert len(text_rows) == 24
    assert len(speech_rows) == 48
    assert all(row.speech_pattern_key is None for row in text_rows)
    assert all("speech_pattern_key" not in row.as_dict() for row in text_rows)
    assert {row.speech_pattern_key for row in speech_rows} == {"clean", "hesitant"}
    assert all(row.as_dict()["speech"]["tts_engine"] == "piper" for row in speech_rows)


def test_invalid_condition_index_and_slurm_index_are_explicit():
    grid = SlurmConditionGrid.from_document(_document(["pure_text"]))
    with pytest.raises(IndexError, match="outside"):
        grid.condition(-1)
    with pytest.raises(IndexError, match="outside"):
        grid.condition(len(grid.conditions))
    with pytest.raises(ValueError, match="SLURM_ARRAY_TASK_ID"):
        condition_index(None, {})
    with pytest.raises(ValueError, match="integer"):
        condition_index(None, {"SLURM_ARRAY_TASK_ID": "bad"})


def test_condition_json_export_is_exact_and_write_once():
    condition = SlurmConditionGrid.from_document(_document(["pure_text"])).condition(0)
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=project_root) as temporary:
        path = Path(temporary) / "condition.json"
        export_condition_json(condition, path, {"model_device": "cpu"})
        exported = json.loads(path.read_text(encoding="utf-8"))
        with pytest.raises(FileExistsError):
            export_condition_json(condition, path)

    assert exported["condition_id"] == condition.condition_id
    assert exported["runtime_overrides"] == {"model_device": "cpu"}
    assert "speech_pattern_key" not in exported
    job = condition_job_document(exported)
    assert job["config"]["tts_engine"] == "file"
    assert job["config"]["asr_engine"] == "file"
    assert job["config"]["agent_b_plugin"] == "simple"
    assert job["config"]["network_seed"] == 11
    assert job["config"]["agent_a_seed"] == 11
    assert job["config"]["agent_b_seed"] == 12
    assert job["parameter_values"]["experiment_seed"] == [11]
    assert job["parameter_values"]["run_mode"] == ["pure_text"]


def test_speech_grid_requires_both_transport_engines():
    document = _document(["speech"])
    document["speech"] = {"tts_engine": "piper"}
    with pytest.raises(ValueError, match="asr_engine"):
        SlurmConditionGrid.from_document(document)


def test_included_full_grids_fix_common_route_objective_without_private_constraints():
    root = Path(__file__).resolve().parents[1] / "slurm"
    for filename in ("minillama_grid.json", "minillama_speech_grid.json"):
        grid = SlurmConditionGrid.from_file(root / filename)
        assert len(grid.conditions) == 720
        assert {
            row.base_config["maximum_progressive_constraints"]
            for row in grid.conditions
        } == {0}


def test_every_full_grid_task_pair_passes_common_route_objective_viability():
    root = Path(__file__).resolve().parents[1] / "slurm"
    grid = SlurmConditionGrid.from_file(root / "minillama_grid.json")
    task_pairs = {
        (row.test_case_key, row.persona_key)
        for row in grid.conditions
    }
    failures = []
    for test_case_key, persona_key in sorted(task_pairs):
        case = get_test_case(test_case_key).with_persona(persona_key)
        report = stage_viability_report(
            case.scenario,
            case.persona,
            transfer_tolerance=1,
            max_constraints=0,
        )
        if not report["all_stage_requirements_satisfied"]:
            failures.append((test_case_key, persona_key))
    assert failures == []
