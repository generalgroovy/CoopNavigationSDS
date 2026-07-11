import csv
import json
from pathlib import Path

from coop_navigation_sds.ResultsAndArtifacts.condition_overview import (
    write_configuration_condition_overview,
)


ROOT = Path(__file__).resolve().parents[1]


def test_condition_overview_exports_exact_job_grid(tmp_path):
    output = tmp_path / "general"
    paths = write_configuration_condition_overview(
        [ROOT / "jobs" / "support" / "small_agent_b_speech_grid.job"],
        output,
        results_root=tmp_path,
    )

    expected_files = {
        "configuration_conditions",
        "configuration_model_overview",
        "configuration_factor_levels",
        "configuration_groups_exact",
        "configuration_condition_overview",
        "configuration_condition_manifest",
    }
    assert set(paths) == expected_files
    assert all(path.exists() for path in paths.values())

    with paths["configuration_condition_manifest"].open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["job_count"] == 1
    assert manifest["generated_condition_count"] > 0
    assert manifest["valid_condition_count"] > 0
    assert (
        manifest["generated_condition_count"]
        == manifest["valid_condition_count"] + manifest["invalid_condition_count"]
    )

    with paths["configuration_conditions"].open(newline="", encoding="utf-8") as handle:
        condition_rows = list(csv.DictReader(handle))
    assert len(condition_rows) == manifest["generated_condition_count"]
    assert {
        "generated_sequence",
        "valid_sequence",
        "stage_viable",
        "invalid_stages",
        "agent_a_type",
        "agent_b_model",
        "test_case_key",
        "persona_key",
        "run_type",
        "configured_tts_engine",
        "configured_asr_engine",
        "result_group",
    }.issubset(condition_rows[0])

    with paths["configuration_model_overview"].open(newline="", encoding="utf-8") as handle:
        model_rows = list(csv.DictReader(handle))
    assert model_rows == [
        {
            **model_rows[0],
            "agent_a_type": "tinyllama",
            "agent_b_llm_size": "small",
            "agent_b_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }
    ]

    html = paths["configuration_condition_overview"].read_text(encoding="utf-8")
    assert "Configuration condition overview" in html
    assert "Model coverage" in html
