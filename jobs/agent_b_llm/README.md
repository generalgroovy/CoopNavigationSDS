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

Additional documented replacement or follow-up candidates:

| Size | Candidate | Distinct value |
| --- | --- | --- |
| Small | `HuggingFaceTB/SmolLM2-360M-Instruct` | sub-billion floor condition |
| Small | `Qwen/Qwen2.5-0.5B-Instruct` | tiny multilingual Transformers condition |
| Medium | `gemma2:2b` | Gemma-family architecture contrast |
| Medium | `qwen3:4b` | newer Qwen-family instruction contrast |
| Large | `mistral:7b` | non-Llama/Qwen large local baseline |
| Large | `microsoft/UserLM-8b` | user-simulator-trained ablation candidate |

Every Agent B treatment is run with two callers:

- **UserLM:** `microsoft/UserLM-8b` through Transformers, fixed independently
  of Agent B. The official full-precision repository is approximately 32.1 GB
  and must be prepared on a machine with sufficient disk and accelerator/RAM.
- **TinyLlama comparison:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` through
  Transformers, with all non-caller settings inherited unchanged.

The compact support speech-grid jobs expand to eight conditions: four
speech-performance bands and their matched text controls. Larger comparison
manifests may expand further through pairwise coverage. Caller audio persona,
operator audio persona, speech pattern, recognition beam width, and network
seed vary only through registered job factors, so every result can be traced
back to an explicit configuration row.

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

For schedulers where UserLM conditions are hard to place, use the CPU-first
array wrappers:

```bash
sbatch slurm/userlm_small1_cpu_array.sbatch
sbatch slurm/userlm_small2_cpu_array.sbatch
sbatch slurm/userlm_large2_cpu_array.sbatch
```

Each wrapper runs one UserLM speech-grid job as eight single-condition array
tasks with no GPU request. This improves backfill opportunities and prevents
one unavailable node class from blocking the full model comparison.

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
