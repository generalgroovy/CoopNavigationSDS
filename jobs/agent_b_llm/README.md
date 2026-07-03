# Agent B LLM Experiment

This directory defines the focused Agent B experiment. It varies Agent B model
scale and family while holding the dialogue task, speech stack, turn budget,
decoding, and coverage strategy constant.

## Experimental Design

| Size | Primary model | Family comparison | Approximate parameters |
| --- | --- | --- | ---: |
| Small | `llama3.2:1b` | `qwen2.5:1.5b` | 1.0B / 1.5B |
| Medium | `llama3.2:3b` | `phi3:mini` | 3.2B / 3.8B |
| Large | `llama3.1:8b` | `qwen2.5:7b` | 8.0B / 7.6B |

Every Agent B treatment is run with two callers:

- **UserLM:** `microsoft/UserLM-8b` through Transformers, fixed independently
  of Agent B. The official full-precision repository is approximately 32.1 GB
  and must be prepared on a machine with sufficient disk and accelerator/RAM.
- **TinyLlama comparison:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` through
  Transformers, with all non-caller settings inherited unchanged.

Each job expands to 13 pairwise-selected audio conditions and 13 matched
text-only controls. The four linked tasks cover distinct scenarios and caller
personas. Caller audio persona, operator audio persona, speech pattern,
recognition beam width, and network seed vary. Piper and Faster-Whisper remain
fixed so the primary contrast is language-model behavior.

## Layout

```text
agent_b_llm/
|-- base.template.json
|-- batches/
|   |-- 01-userlm-primary.json
|   |-- 02-userlm-model-comparison.json
|   |-- 03-tinyllama-primary-comparison.json
|   |-- 04-tinyllama-model-comparison.json
|   |-- 05-userlm-all-models.json
|   |-- 06-userlm-speech-grid-all-models.json
|   `-- all.json
|-- userlm/
|   |-- primary/
|   `-- model_comparison/
|-- userlm_speech_grid/
|   |-- small/
|   |-- medium/
|   `-- large/
`-- tinyllama_comparison/
    |-- primary/
    `-- model_comparison/
```

`base.template.json` owns every controlled setting. Child jobs state only the
Agent B model, size, role, caller override where applicable, and result group.

The `userlm_speech_grid/` family instead inherits the expanded 52-condition
speech design from `jobs/support/small_agent_b_speech_grid_userlm.job`. It has
six jobs: two Agent B models at each size. The complete manifest contains 312
conditions while keeping UserLM-8b, tasks, speech factors, repetitions, and
matched controls identical across Agent B models.

## Preview and Run

```bash
python scripts/run_agent_b_llm_batch.py \
  --batch jobs/agent_b_llm/batches/all.json \
  --results-dir results \
  --preview

python scripts/run_agent_b_llm_batch.py \
  --batch jobs/agent_b_llm/batches/05-userlm-all-models.json \
  --results-dir results

python scripts/run_agent_b_llm_batch.py \
  --batch jobs/agent_b_llm/batches/06-userlm-speech-grid-all-models.json \
  --results-dir results \
  --preview
```

The UserLM-only manifest contains 6 jobs and 156 conditions. The complete
manifest contains 12 jobs and 312 conditions. The runner executes
jobs sequentially and maintains `results/agent_b/experiment_run_table.csv`.
After completed jobs, it also rebuilds
`results/agent_b/comparison/comparison_report.html`, with color-coded run task
outcomes and robust pre-outcome metric outliers. The underlying
`run_outcomes.csv` and `metric_outliers.csv` retain the exact values and labels.

## Result Groups

```text
results/agent_b/
|-- experiment_run_table.csv
|-- primary/<size-model>/<userlm|tinyllama>/<run-id>/
`-- model_comparison/<size-model>/<userlm|tinyllama>/<run-id>/
```

Each completed run remains a standard batch result folder. The nested group is
an organizational factor only; all standard condition and metric schemas are
unchanged and can be concatenated directly.
