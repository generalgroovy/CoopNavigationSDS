from types import SimpleNamespace
import re

from coop_navigation_sds.Configuration.pipeline import (
    PIPELINE_PHASES,
    metric_dependency_report,
    optimal_route_preview,
    route_layer_comparison,
)
from coop_navigation_sds.EvaluationMetrics.metrics import apply_paired_run_metrics
from coop_navigation_sds.app import default_run_config
from coop_navigation_sds.Configuration.gui import COMBINED_GUI_PHASES, GUI_METRIC_FAMILIES
from coop_navigation_sds.EvaluationMetrics.catalog import METRIC_FAMILY_SPECS
from coop_navigation_sds.experiments import build_condition_grid


def test_pipeline_order_and_optimal_preview():
    assert [key for key, _label in PIPELINE_PHASES] == [
        "network", "agent_a", "agent_b", "audio_tts", "asr",
        "metrics_logging", "batch_results",
    ]
    preview = optimal_route_preview(default_run_config())
    assert preview["available"]
    assert "1. Valid connected route:\n" in preview["summary"]
    assert " min, " in preview["summary"]
    assert " -> " in preview["summary"]
    assert preview["duration_min"] > 0
    lines = preview["summary"].splitlines()
    assert len(lines) == 15
    assert [lines[index][0] for index in range(0, 15, 3)] == ["1", "2", "3", "4", "5"]
    assert [layer["layer"] for layer in preview["layers"]] == [
        "validity", "time", "constraint_1", "constraint_2", "constraint_3",
    ]
    path_lines = lines[2::3]
    assert all(
        re.search(r"\((?:[MTB]\d+(?: : [^)]+)?|walk: \d+ min)\) ->", line)
        for line in path_lines
    )
    progressive_paths = [layer["path_text"] for layer in preview["layers"][1:]]
    assert all(
        previous != current
        for previous, current in zip(progressive_paths, progressive_paths[1:])
    )
    constraint_paths = [layer["path_text"] for layer in preview["layers"][2:]]
    assert len(constraint_paths) == len(set(constraint_paths))
    comparison = route_layer_comparison(preview["layers"], 2)
    assert comparison["selected"]["layer"] == "constraint_1"
    assert comparison["previous"]["layer"] == "time"
    assert comparison["added_edges"]
    assert comparison["removed_edges"]
    assert comparison["duration_delta_min"] == (
        preview["layers"][2]["duration_min"] - preview["layers"][1]["duration_min"]
    )


def test_gui_assigns_each_metric_family_to_exactly_one_program_phase():
    assigned = [family for key, _label in COMBINED_GUI_PHASES for family in GUI_METRIC_FAMILIES[key]]
    catalog = [family["key"] for family in METRIC_FAMILY_SPECS]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(catalog)


def test_metric_dependencies_mark_missing_learned_audio_evidence():
    report = metric_dependency_report(default_run_config())
    nisqa = report["metrics"]["tts_nisqa"]
    assert nisqa["obligatory"]
    assert not nisqa["enabled"]
    assert not nisqa["available"]
    assert "nisqa_model" in nisqa["missing_fields"]


def test_paired_grid_and_audio_minus_text_deltas():
    conditions = list(build_condition_grid(
        test_case_keys=["morning_peak_cross_city"],
        persona_keys=["focused_commuter"],
        speech_pattern_keys=["clean"],
        model_param_keys=["greedy"],
        tts_engine_keys=["sapi"],
        asr_engine_keys=["sapi"],
        agent_b_model_keys=["TinyLlama"],
        pair_audio_with_text=True,
    ))
    assert [condition.run_type for condition in conditions] == ["text_only", "audio_variant"]
    assert conditions[0].pair_id == conditions[1].pair_id

    base = dict(
        pair_id=conditions[0].pair_id,
        route_valid=True,
        unsatisfied_constraints="",
        metric_families={"whole_dialogue": {"repair_count": 0}},
    )
    text = SimpleNamespace(
        **base, run_type="text_only", success=True, message_count=5,
        asr_word_error_rate=0.0,
    )
    audio = SimpleNamespace(
        **base, run_type="audio_variant", success=False, message_count=7,
        asr_word_error_rate=0.25,
    )
    apply_paired_run_metrics([text, audio])
    assert audio.paired_task_success_delta == -1.0
    assert audio.paired_turn_count_delta == 2.0
    assert audio.paired_audio_error_effect == 0.25
