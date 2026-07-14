# Current UserLM Thesis Result Overview

Source: condition-level aggregation over existing `conditions.jsonl` files. Agent A is restricted to `userlm`; archived large2/Mistral and TinyLlama-Agent-A controls are excluded from this thesis denominator.

| Slot | Agent B model | Size | Unique conditions observed | Completed | Task-success | Route-valid | Task-success rate of completed | Route-valid rate of completed | Mean turns |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small1 | TinyLlama 1.1B | small | 154 | 62 | 54 | 57 | 87.1% | 91.9% | 12.18 |
| small2 | Qwen2.5 0.5B | small | 154 | 62 | 54 | 57 | 87.1% | 91.9% | 12.27 |
| medium1 | Qwen2.5 1.5B | medium | 154 | 96 | 88 | 91 | 91.7% | 94.8% | 11.68 |
| medium2 | Phi-3 mini | medium | 154 | 60 | 52 | 54 | 86.7% | 90.0% | 12.17 |
| large1 | Qwen2.5 7B | large | 154 | 37 | 33 | 35 | 89.2% | 94.6% | 11.16 |

Interpretation notes:

- Completion is separated from task success because provider/runtime failures are experimental evidence, not failed conversations.
- A condition is counted once by condition ID for coverage; repeated failed and successful attempts remain in the raw run folders.
- Route-valid rate is calculated only among completed conditions.
- This file is derived; canonical evidence remains in each run folder.
