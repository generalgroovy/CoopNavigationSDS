# Result Analysis Guide

This folder contains derived analysis tables only. The generator reads raw
run evidence, writes normalized comparison views, hashes the source evidence
before and after generation, and aborts if source evidence changes.

## Files

| File | Purpose |
| --- | --- |
| `run_inventory.csv` | One row per discovered run folder, including lifecycle state and completed/observed counts. |
| `program_execution_summary.csv` | Program execution completion by Agent B model, independent of task success or route satisfaction. |
| `configuration_groups.csv` | Planned and observed coverage grouped by controlled configuration factors. |
| `configuration_conditions.csv` | Exact generated job conditions with staged-validity audit. |
| `configuration_condition_overview.html` | Human-readable overview of planned configuration conditions. |
| `task_outcome_comparison.csv` | One row per planned condition; unfinished outcomes remain blank. |
| `task_success_by_configuration.csv` | Aggregated task success, route validity, duration gap, turns, and runtime by configuration. |
| `model_configuration_matrix.csv` | Wide matrix joining identical non-model conditions across Agent B models. |
| `model_configuration_matrix.html` | Color-coded inspection view of the same model-by-configuration matrix. |
| `agent_b_model_summary.csv` | One row per Agent B model and Agent A pairing, sorted by Agent B size and model. |
| `phase_metric_comparison.csv` | One row per condition and dialogue-system phase with telemetry counts and timings. |
| `phase_summary_by_model.csv` | Phase-level aggregates by Agent A, Agent B, run type, TTS, ASR, and phase. |
| `../comparison/metric_outcome_correlations.csv` | Descriptive Pearson correlations between pre-outcome metrics and task outcomes for finalized runs. |
| `../comparison/metric_outcome_correlations.html` | Phase-wise readable correlation overview. |
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
- `program_execution_summary.csv`
- `current_program_execution_summary.csv`
- `configuration_groups.csv`
- `task_outcome_comparison.csv`
- `current_task_outcome_comparison.csv`
- `task_success_by_configuration.csv`
- `current_task_success_by_configuration.csv`
- `model_configuration_matrix.csv`
- `current_model_configuration_matrix.csv`
- `model_configuration_matrix.html`
- `agent_b_model_summary.csv`
- `current_agent_b_model_summary.csv`
- `phase_metric_comparison.csv`
- `current_phase_metric_comparison.csv`
- `phase_summary_by_model.csv`
- `current_phase_summary_by_model.csv`
- `dialogue_state_comparison.csv`
- `current_dialogue_state_comparison.csv`
- `dialogue_state_summary.csv`
- `current_dialogue_state_summary.csv`
- `conversation_turns.csv`
- `current_conversation_turns.csv`
