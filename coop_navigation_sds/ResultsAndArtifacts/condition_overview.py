"""Derived configuration-condition overview for planned experiment grids."""
from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from coop_navigation_sds.Configuration.jobs import (
    job_linked_profiles,
    job_parameter_grid,
    job_parameter_profiles,
    load_experiment_job,
)
from coop_navigation_sds.Configuration.runtime import AGENT_A_TRANSFER_TOLERANCE, NUM_TURNS
from coop_navigation_sds.Configuration.schema import resolve_result_group, resolve_results_root
from coop_navigation_sds.experiments import (
    build_condition_grid,
    condition_stage_viability,
)
from coop_navigation_sds.TransportNetwork.test_cases import get_test_case


DEFAULT_CONDITION_OVERVIEW_ROOTS = (
    Path("jobs/agent_b_llm/userlm_transformers_speech_grid"),
    Path("jobs/agent_b_llm/transformers_speech_grid"),
)

CONDITION_FACTOR_COLUMNS = (
    "agent_a_type",
    "agent_b_model",
    "agent_b_llm_size",
    "agent_b_model_role",
    "test_case_key",
    "persona_key",
    "agent_a_audio_persona",
    "agent_b_audio_persona",
    "speech_pattern_key",
    "run_type",
    "configured_tts_engine",
    "configured_asr_engine",
    "asr_beam_size",
    "network_seed",
    "speech_performance_band",
    "model_param_key",
    "objective_mode",
    "iteration",
)


def _write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _job_paths(job_roots):
    paths = []
    for root in job_roots or DEFAULT_CONDITION_OVERVIEW_ROOTS:
        root = Path(root)
        if root.is_file() and root.suffix == ".job":
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.glob("*/*.job")))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def _condition_grid(job):
    grid = job["grid"]
    return list(build_condition_grid(
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
    ))


def _scenario_overrides(job):
    config = job.get("config", {})
    return {
        key: value
        for key, value in {
            "maximum_progressive_constraints": config.get("maximum_progressive_constraints"),
            "minimum_compared_routes": config.get("minimum_compared_routes"),
            "require_constraint_retention": config.get("require_constraint_retention"),
            "acceptable_duration_ratio": config.get("acceptable_duration_ratio"),
            "min_stage_suboptimal_options": config.get("minimum_stage_suboptimal_options"),
            "require_stage_suboptimal_options": config.get("require_stage_suboptimal_options"),
        }.items()
        if value is not None
    }


def _configuration_condition_rows(job_paths, results_root):
    rows = []
    viability_cache = {}
    for job_path in job_paths:
        job = load_experiment_job(job_path)
        config = job["config"]
        parameters = job.get("parameter_values", {})
        result_path = Path(resolve_result_group(results_root, config.get("result_group")))
        result_group = result_path.relative_to(Path(resolve_results_root(results_root))).as_posix()
        conditions = _condition_grid(job)
        valid_sequence = 0
        scenario_overrides = _scenario_overrides(job)
        default_transfer_tolerance = int(
            config.get("agent_a_transfer_tolerance", AGENT_A_TRANSFER_TOLERANCE)
        )
        default_num_turns = int(config.get("num_turns", NUM_TURNS))
        for generated_sequence, condition in enumerate(conditions, start=1):
            test_case = get_test_case(condition.test_case_key).with_persona(condition.persona_key)
            scenario = dict(test_case.scenario)
            condition_parameters = dict(condition.parameter_values)
            viability_key = json.dumps(
                {
                    "test_case_key": condition.test_case_key,
                    "persona_key": condition.persona_key,
                    "network_seed": condition_parameters.get("network_seed"),
                    "transfer_tolerance": condition_parameters.get(
                        "transfer_tolerance", default_transfer_tolerance
                    ),
                    "default_num_turns": default_num_turns,
                    "scenario_overrides": scenario_overrides,
                },
                sort_keys=True,
                default=str,
            )
            if viability_key not in viability_cache:
                viability_cache[viability_key] = condition_stage_viability(
                    condition,
                    scenario_overrides=scenario_overrides,
                    default_transfer_tolerance=default_transfer_tolerance,
                    default_num_turns=default_num_turns,
                )
            viability = viability_cache[viability_key]
            stage_viable = bool(viability.get("all_stage_requirements_satisfied"))
            invalid_stages = [
                str(stage.get("stage"))
                for stage in viability.get("stages", ())
                if not stage.get("requirement_satisfied", True)
            ]
            valid_stage_count = sum(
                1 for stage in viability.get("stages", ())
                if stage.get("requirement_satisfied", True)
            )
            total_stage_count = len(viability.get("stages", ()))
            if stage_viable:
                valid_sequence += 1
            row = {
                "job_name": job["name"],
                "job_path": str(job_path),
                "job_group": job_path.parent.name,
                "coverage_strategy": job["coverage_strategy"],
                "result_group": result_group,
                "generated_sequence": generated_sequence,
                "valid_sequence": valid_sequence if stage_viable else "",
                "stage_viable": stage_viable,
                "invalid_stage_count": len(invalid_stages),
                "invalid_stages": ";".join(invalid_stages),
                "valid_stage_count": valid_stage_count,
                "total_stage_count": total_stage_count,
                "condition_id": condition.condition_id,
                "pair_id": condition.pair_id,
                "run_type": condition.run_type,
                "agent_a_type": config.get("agent_a_type"),
                "agent_b_plugin": config.get("agent_b_plugin"),
                "agent_b_model": condition.agent_b_model,
                "agent_b_llm_size": condition_parameters.get("agent_b_llm_size", ""),
                "agent_b_model_slot": condition_parameters.get("agent_b_model_slot", ""),
                "agent_b_model_role": condition_parameters.get("agent_b_model_role", ""),
                "matrix_family": condition_parameters.get("matrix_family", ""),
                "model_profile": config.get("model_profile", ""),
                "model_provider": config.get("model_provider", ""),
                "test_case_key": condition.test_case_key,
                "scenario_key": condition.scenario_key,
                "start_station": scenario.get("start_station", ""),
                "destination_station": scenario.get("destination_station", ""),
                "start_time_min": scenario.get("start_time_min", ""),
                "persona_key": condition.persona_key,
                "agent_a_audio_persona": condition.agent_a_audio_persona,
                "agent_b_audio_persona": condition.agent_b_audio_persona,
                "speech_pattern_key": condition.speech_pattern_key,
                "configured_tts_engine": condition.tts_engine,
                "effective_tts_engine": (
                    "file" if condition.run_type == "text_only" else condition.tts_engine
                ),
                "configured_asr_engine": condition.asr_engine,
                "effective_asr_engine": (
                    "file" if condition.run_type == "text_only" else condition.asr_engine
                ),
                "asr_beam_size": condition_parameters.get("asr_beam_size", ""),
                "network_seed": condition_parameters.get("network_seed", ""),
                "speech_performance_band": condition_parameters.get("speech_performance_band", ""),
                "speech_performance_rank": condition_parameters.get("speech_performance_rank", ""),
                "model_param_key": condition.model_param_key,
                "objective_mode": condition.objective_mode,
                "iteration": condition.iteration,
                "dialogue_stagnation_limit": condition_parameters.get("dialogue_stagnation_limit", ""),
                "transfer_tolerance": condition_parameters.get("transfer_tolerance", ""),
                "channel_noise_snr_db": condition_parameters.get("channel_noise_snr_db", ""),
                "channel_gain_db": condition_parameters.get("channel_gain_db", ""),
                "channel_clip_threshold": condition_parameters.get("channel_clip_threshold", ""),
                "channel_dropout_rate": condition_parameters.get("channel_dropout_rate", ""),
                "parameter_values_json": json.dumps(condition_parameters, sort_keys=True, default=str),
            }
            rows.append(row)
    return rows


def _model_summary_rows(condition_rows):
    grouped = defaultdict(list)
    for row in condition_rows:
        grouped[(row["agent_a_type"], row["agent_b_llm_size"], row["agent_b_model"])].append(row)
    rows = []
    for (agent_a_type, size, model), items in grouped.items():
        valid = [row for row in items if row["stage_viable"] is True]
        rows.append({
            "agent_a_type": agent_a_type,
            "agent_b_llm_size": size,
            "agent_b_model": model,
            "job_count": len({row["job_name"] for row in items}),
            "generated_conditions": len(items),
            "valid_conditions": len(valid),
            "invalid_conditions": len(items) - len(valid),
            "text_controls": sum(row["run_type"] == "text_only" for row in valid),
            "audio_variants": sum(row["run_type"] == "audio_variant" for row in valid),
            "scenarios": ";".join(sorted({row["test_case_key"] for row in valid})),
            "personas": ";".join(sorted({row["persona_key"] for row in valid})),
            "speech_bands": ";".join(sorted({row["speech_performance_band"] for row in valid})),
            "tts_engines": ";".join(sorted({row["configured_tts_engine"] for row in valid})),
            "asr_engines": ";".join(sorted({row["configured_asr_engine"] for row in valid})),
            "asr_beam_sizes": ";".join(sorted({str(row["asr_beam_size"]) for row in valid})),
        })
    order = {"small": 0, "medium": 1, "large": 2}
    return sorted(rows, key=lambda row: (
        row["agent_a_type"],
        order.get(str(row["agent_b_llm_size"]), 9),
        row["agent_b_model"],
    ))


def _factor_level_rows(condition_rows):
    rows = []
    for factor in CONDITION_FACTOR_COLUMNS:
        counts = Counter(
            str(row.get(factor, ""))
            for row in condition_rows
            if row.get(factor, "") not in (None, "")
        )
        valid_counts = Counter(
            str(row.get(factor, ""))
            for row in condition_rows
            if row.get(factor, "") not in (None, "") and row["stage_viable"] is True
        )
        for value in sorted(counts):
            rows.append({
                "factor": factor,
                "level": value,
                "generated_conditions": counts[value],
                "valid_conditions": valid_counts[value],
                "invalid_conditions": counts[value] - valid_counts[value],
            })
    return rows


def _configuration_group_rows(condition_rows):
    dimensions = (
        "agent_a_type",
        "agent_b_llm_size",
        "test_case_key",
        "persona_key",
        "speech_performance_band",
        "run_type",
        "configured_tts_engine",
        "configured_asr_engine",
        "asr_beam_size",
    )
    grouped = defaultdict(list)
    for row in condition_rows:
        grouped[tuple(row.get(key, "") for key in dimensions)].append(row)
    rows = []
    for identity, items in grouped.items():
        valid = [row for row in items if row["stage_viable"] is True]
        rows.append({
            **dict(zip(dimensions, identity)),
            "generated_conditions": len(items),
            "valid_conditions": len(valid),
            "invalid_conditions": len(items) - len(valid),
            "agent_b_models": ";".join(sorted({row["agent_b_model"] for row in items})),
        })
    return sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in dimensions))


def _write_html(path, *, condition_rows, model_rows, factor_rows, group_rows):
    def table(rows, columns):
        body = []
        for row in rows:
            body.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
                + "</tr>"
            )
        return (
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(column)}</th>" for column in columns)
            + "</tr></thead><tbody>"
            + "".join(body)
            + "</tbody></table>"
        )

    total = len(condition_rows)
    valid = sum(row["stage_viable"] is True for row in condition_rows)
    invalid = total - valid
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Configuration condition overview</title>
<style>body{{font:14px system-ui,sans-serif;background:#f5f7f9;color:#202a35;margin:0}}main{{padding:18px;max-width:1600px;margin:auto}}section{{background:#fff;border:1px solid #d7dee6;margin:12px 0;padding:12px;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dce2e8;padding:6px 8px;text-align:left;vertical-align:top;white-space:nowrap}}th{{background:#244b5a;color:white;position:sticky;top:0}}.summary{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:10px}}.summary div{{background:#eaf1ef;border:1px solid #cbd8d4;padding:10px}}strong{{font-size:1.45rem;display:block}}</style></head>
<body><main><h1>Configuration condition overview</h1>
<p>Derived from job definitions before dialogue execution. Raw job files remain authoritative; staged-route-valid marks whether the generated condition can support all progressive route-reasoning stages.</p>
<div class="summary"><div><strong>{total}</strong>generated conditions</div><div><strong>{valid}</strong>valid staged conditions</div><div><strong>{invalid}</strong>excluded invalid designs</div><div><strong>{len(model_rows)}</strong>Agent A/Agent B model cells</div></div>
<section><h2>Model coverage</h2>{table(model_rows, ("agent_a_type", "agent_b_llm_size", "agent_b_model", "generated_conditions", "valid_conditions", "invalid_conditions", "text_controls", "audio_variants", "scenarios", "personas", "speech_bands", "asr_beam_sizes"))}</section>
<section><h2>Factor levels</h2>{table(factor_rows, ("factor", "level", "generated_conditions", "valid_conditions", "invalid_conditions"))}</section>
<section><h2>Configuration groups</h2>{table(group_rows, ("agent_a_type", "agent_b_llm_size", "test_case_key", "persona_key", "speech_performance_band", "run_type", "configured_tts_engine", "configured_asr_engine", "asr_beam_size", "generated_conditions", "valid_conditions", "invalid_conditions", "agent_b_models"))}</section>
</main></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


def write_configuration_condition_overview(
    job_roots=None,
    output_directory="results/general",
    *,
    results_root="results",
):
    """Write exact generated and valid-condition tables for selected job roots."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    jobs = _job_paths(job_roots)
    condition_rows = _configuration_condition_rows(jobs, results_root)
    model_rows = _model_summary_rows(condition_rows)
    factor_rows = _factor_level_rows(condition_rows)
    group_rows = _configuration_group_rows(condition_rows)
    paths = {
        "configuration_conditions": output / "configuration_conditions.csv",
        "configuration_model_overview": output / "configuration_model_overview.csv",
        "configuration_factor_levels": output / "configuration_factor_levels.csv",
        "configuration_groups_exact": output / "configuration_groups_exact.csv",
        "configuration_condition_overview": output / "configuration_condition_overview.html",
    }
    _write_csv(paths["configuration_conditions"], condition_rows)
    _write_csv(paths["configuration_model_overview"], model_rows)
    _write_csv(paths["configuration_factor_levels"], factor_rows)
    _write_csv(paths["configuration_groups_exact"], group_rows)
    _write_html(
        paths["configuration_condition_overview"],
        condition_rows=condition_rows,
        model_rows=model_rows,
        factor_rows=factor_rows,
        group_rows=group_rows,
    )
    manifest = {
        "job_count": len(jobs),
        "jobs": [str(path) for path in jobs],
        "generated_condition_count": len(condition_rows),
        "valid_condition_count": sum(row["stage_viable"] is True for row in condition_rows),
        "invalid_condition_count": sum(row["stage_viable"] is not True for row in condition_rows),
        "generated_files": {key: str(path) for key, path in paths.items()},
    }
    manifest_path = output / "configuration_condition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["configuration_condition_manifest"] = manifest_path
    return paths
