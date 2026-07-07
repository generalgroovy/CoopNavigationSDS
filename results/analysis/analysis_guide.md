# Result Analysis Guide

This folder contains derived analysis tables only. The generator reads raw
run evidence, writes normalized comparison views, hashes the source evidence
before and after generation, and aborts if source evidence changes.

## Files

| File | Purpose |
| --- | --- |
| `run_inventory.csv` | One row per discovered run folder, including lifecycle state and completed/observed counts. |
| `configuration_groups.csv` | Planned and observed coverage grouped by controlled configuration factors. |
| `task_outcome_comparison.csv` | One row per planned condition; unfinished outcomes remain blank. |
| `task_success_by_configuration.csv` | Aggregated task success, route validity, duration gap, turns, and runtime by configuration. |
| `agent_b_model_summary.csv` | One row per Agent B model and Agent A pairing, sorted by Agent B size and model. |
| `phase_metric_comparison.csv` | One row per condition and dialogue-system phase with telemetry counts and timings. |
| `phase_summary_by_model.csv` | Phase-level aggregates by Agent A, Agent B, run type, TTS, ASR, and phase. |
| `dialogue_state_comparison.csv` | One row per condition with memory, route, candidate, and agreement evidence. |
| `dialogue_state_summary.csv` | Dialogue-state aggregates by run, scenario, persona, and run type. |
| `conversation_turns.csv` | Turn-level intended text, TTS text, raw ASR, understood text, and correction counts. |
| `analysis_manifest.json` | Source-evidence hash, generated file list, and read-only integrity check. |

## Interpretation Rules

- `completed` means the condition reached a canonical end event.
- `interrupted` means runtime evidence exists but no successful condition end was recorded.
- `not_started` means the condition was planned but has no observed session evidence.
- Blank numeric outcomes are missing evidence, not zero.
- HTML colors are exploratory visual markers only; statistical analysis should use the CSV files.

## Generated Files

- `comparison_overview.html`
- `analysis_guide.md`
- `run_inventory.csv`
- `configuration_groups.csv`
- `task_outcome_comparison.csv`
- `task_success_by_configuration.csv`
- `agent_b_model_summary.csv`
- `phase_metric_comparison.csv`
- `phase_summary_by_model.csv`
- `dialogue_state_comparison.csv`
- `dialogue_state_summary.csv`
- `conversation_turns.csv`
