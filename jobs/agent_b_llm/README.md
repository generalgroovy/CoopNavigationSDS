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
| Small | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | small/medium boundary compact instruction model |
| Medium | `Qwen/Qwen2.5-1.5B-Instruct` | non-Ollama Qwen medium-lite condition |
| Medium | `gemma2:2b` | Gemma-family architecture contrast |
| Medium | `qwen3:4b` | newer Qwen-family instruction contrast |
| Medium | `microsoft/Phi-3-mini-4k-instruct` | reasoning-focused Phi-family Transformers condition |
| Large | `mistral:7b` | non-Llama/Qwen large local baseline |
| Large | `tiiuae/Falcon3-7B-Instruct` | Falcon-family assistant model replacing UserLM as Agent B proposal |
| Large | `Qwen/Qwen2.5-7B-Instruct` | large multilingual non-Ollama condition |
| Large | `meta-llama/Llama-3.1-8B-Instruct` | large Llama-family Transformers condition |

Every Agent B treatment is run with two callers:

- **UserLM:** `microsoft/UserLM-8b` through Transformers, fixed independently
  of Agent B. The official full-precision repository is approximately 32.1 GB
  and must be prepared on a machine with sufficient disk and accelerator/RAM.
- **TinyLlama comparison:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` through
  Transformers, with all non-caller settings inherited unchanged.

The shared support speech-grid jobs use pairwise coverage and expand to 84
conditions per Agent A/Agent B model pairing. They vary scenario, task persona,
caller and operator audio persona, speech pattern, model decoding, ASR engine,
ASR beam width, network seed, transfer tolerance, and repair limits while
retaining paired audio/text controls. Caller audio persona, operator audio
persona, speech pattern, recognition beam width, and network seed vary only
through registered job factors, so every result can be traced back to an
explicit configuration row.

All Slurm Agent B jobs in this matrix are designed to be comparable by
construction. The non-model grid is identical for every UserLM Agent B job; the
only intended treatment difference is the Agent B model and its required
provider/profile. Derived result analysis writes `model_configuration_matrix`
CSV/HTML files that join those identical non-model conditions across models.

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
|-- userlm_transformers_speech_grid/
|   |-- small/
|   |-- medium/
|   `-- large/
|-- transformers_speech_grid/
|   |-- small/
|   |-- medium/
|   `-- large/
`-- tinyllama_comparison/
    |-- primary/
    `-- model_comparison/
```

`base.template.json` owns every controlled setting. Child jobs state only the
Agent B model, size, role, caller override where applicable, and result group.

The `userlm_speech_grid/` family covers the six Ollama-backed Agent B jobs
with UserLM-8B as Agent A. The `userlm_transformers_speech_grid/` family covers
twelve Transformers-backed Agent B jobs with the same UserLM caller and the
same condition design. Together they provide 18 independent UserLM Agent B jobs,
or 1512 planned condition tasks at 84 conditions per job. The TinyLlama caller
Transformers family provides 12 additional jobs, giving the full independent
submitter 30 jobs and 2520 planned condition tasks.

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

python scripts/run_agent_b_llm_batch.py \
  --batch jobs/agent_b_llm/batches/07-transformers-agent-b-all.json \
  --results-dir results \
  --preview

python scripts/run_agent_b_llm_batch.py \
  --batch jobs/agent_b_llm/batches/08-transformers-agent-b-small-medium.json \
  --results-dir results \
  --preview
```

Legacy JSON manifests remain available for sequential local runs. The current
Slurm submitter is preferred for thesis-scale coverage because it submits one
independent array per Agent B model:

```bash
python scripts/submit_agent_b_model_jobs.py --family userlm --tier small medium large --dry-run
python scripts/submit_agent_b_model_jobs.py --family userlm --tier small --array-concurrency 1
python scripts/submit_agent_b_model_jobs.py --family userlm --tier medium --array-concurrency 1
python scripts/submit_agent_b_model_jobs.py --family userlm --tier large --array-concurrency 1
```

The runner executes jobs sequentially when used directly and maintains
`results/agent_b/experiment_run_table.csv`.
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

Each wrapper runs one UserLM speech-grid job as 84 single-condition array tasks
with no GPU request. This improves backfill opportunities and prevents one
unavailable node class from blocking the full model comparison.

For non-Ollama Agent B runs, prepare selected Hugging Face assets and submit
the tiered Slurm arrays:

```bash
python scripts/setup_transformers_agent_b_models.py --tier small --download
python scripts/setup_transformers_agent_b_models.py --tier medium --download
scripts/submit_transformers_agent_b_arrays.sh --small-medium
```

Large Transformers jobs are intentionally separate because they request more
memory:

```bash
python scripts/setup_transformers_agent_b_models.py --tier large --download
JOB_FILE=jobs/agent_b_llm/transformers_speech_grid/large/04-falcon3-7b.job \
sbatch slurm/transformers_agent_b_large_cpu_array.sbatch
```

To submit every configured Agent B model as its own independent Slurm array,
use the model submitter. Each `.job` file becomes one scheduler job, and that
job's array tasks cover the condition rows for only that model. This prevents a
failure or queue delay for one model from blocking the rest of the experiment.

```bash
python scripts/submit_agent_b_model_jobs.py --dry-run
python scripts/submit_agent_b_model_jobs.py --family tinyllama --tier small medium
python scripts/submit_agent_b_model_jobs.py --family userlm --tier small medium
python scripts/submit_agent_b_model_jobs.py --family all --tier small medium large
```

The submitter computes the correct Slurm array range from each resolved job
definition and overrides the wrapper's default `#SBATCH --array`. It also
allocates memory from registered model estimates: caller model memory plus
Agent B model memory plus speech/runtime overhead, rounded upward for Slurm.
Ollama-backed jobs start a per-task local Ollama server on a task-specific
port; Transformers jobs run without Ollama.

The generated arrays use `slurm/agent_b_model_cpu_array.sbatch`, which is
CPU-first by design. The wrapper exports CPU-only CUDA guards and passes
`--model-device cpu`, `--agent-a-model-device cpu`, `--tts-device cpu`, and
`--asr-device cpu`. The Transformers runtime enforces the same policy before
loading tokenizer or model weights, so a CPU job can land on a node that exposes
an incompatible NVIDIA driver without failing during preflight. GPU execution
must be requested deliberately with a dedicated GPU wrapper and compatible
drivers.

Array tasks also pass `--no-update-coverage-registry`. Run coverage and
comparison refresh once after the scheduler jobs finish:

```bash
python3 scripts/update_experiment_coverage.py --results-dir results
python3 -m coop_navigation_sds.ResultsAndArtifacts.comparison results --output results/comparison
```

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
