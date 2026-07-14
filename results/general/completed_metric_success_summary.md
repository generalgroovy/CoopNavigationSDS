# Completed Dialogue Metric-Success Analysis

Generated: 2026-07-14T22:29:12.784247+00:00

Scope: completed UserLM-Agent-A runs for the active selected Agent B models. Execution-failed and invalid-condition rows are excluded.

## Outcome counts

| Outcome | Runs |
| --- | ---: |
| successful | 409 |
| semi_successful | 19 |
| unsuccessful_dialogue | 24 |
| total_completed | 452 |

## Model summary

| Agent B size | Agent B model | Completed | Successful | Semi | Unsuccessful | Success rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| large | Qwen/Qwen2.5-7B-Instruct | 50 | 46 | 2 | 2 | 92.00% |
| medium | Qwen/Qwen2.5-1.5B-Instruct | 137 | 126 | 5 | 6 | 91.97% |
| medium | microsoft/Phi-3-mini-4k-instruct | 77 | 69 | 2 | 6 | 89.61% |
| small | Qwen/Qwen2.5-0.5B-Instruct | 94 | 84 | 5 | 5 | 89.36% |
| small | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 94 | 84 | 5 | 5 | 89.36% |

## Metric means by outcome

| Metric | Successful | Semi-successful | Unsuccessful dialogue |
| --- | ---: | ---: | ---: |
| `automatic_eval_score` | 0.954 | 0.216 | 0.069 |
| `quality_score` | 0.745 | 0.077 | 0.000 |
| `duration_score` | 0.977 | 0.666 | 0.000 |
| `agent_b_grounded_proposal_score` | 0.994 | 0.443 | 0.000 |
| `agent_b_actionability_score` | 1.000 | 0.776 | 0.000 |
| `nlg_faithfulness` | 0.994 | 0.443 | 0.000 |
| `whole_dialogue_goal_progress_auc` | 0.964 | 0.783 | 0.000 |
| `whole_dialogue_abandonment_rate` | 0.000 | 0.895 | 1.000 |
| `asr_word_error_rate` | 0.079 | 0.080 | 0.697 |
| `asr_station_f1` | 0.977 | 0.978 | 0.223 |
| `nlu_route_valid_rate` | 0.920 | 0.330 | 0.000 |
| `nlu_goal_reached_rate` | 0.878 | 0.039 | 0.000 |
| `tts_text_change_rate` | 0.050 | 0.121 | 0.286 |
| `candidate_route_count` | 3.702 | 1.579 | 0.000 |
| `station_mentions` | 26.252 | 15.684 | 0.833 |
| `dialogue_management_repair_success_rate` | 0.937 | 0.921 | 0.368 |

## Strongest diagnostic metric correlations

| Phase | Metric | n | corr(task success) | corr(outcome rank) | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| agent_b_response | `agent_b_grounded_proposal_score` | 452 | 0.956 | 0.995 | strong positive descriptive association; not causal |
| natural_language_generation | `nlg_faithfulness` | 452 | 0.956 | 0.995 | strong positive descriptive association; not causal |
| agent_b_response | `agent_b_actionability_score` | 452 | 0.849 | 0.971 | strong positive descriptive association; not causal |
| whole_dialogue | `duration_score` | 452 | 0.869 | 0.965 | strong positive descriptive association; not causal |
| whole_dialogue | `whole_dialogue_abandonment_rate` | 452 | -0.974 | -0.941 | strong negative descriptive association; not causal |
| whole_dialogue | `whole_dialogue_goal_progress_auc` | 452 | 0.799 | 0.926 | strong positive descriptive association; not causal |
| other | `peak_route_fullness` | 452 | 0.798 | 0.917 | strong positive descriptive association; not causal |
| dialogue_management | `dialogue_management_stagnation_rate` | 452 | 0.937 | 0.913 | strong positive descriptive association; not causal |
| other | `average_route_fullness` | 452 | 0.802 | 0.911 | strong positive descriptive association; not causal |
| agent_b_response | `agent_b_mode_permission_compliance` | 452 | 0.730 | 0.910 | strong positive descriptive association; not causal |
| dialogue_state | `dialogue_state_missing_trip_slot_rate` | 452 | -0.706 | -0.880 | strong negative descriptive association; not causal |
| dialogue_state | `dialogue_state_trip_fact_completeness` | 452 | 0.706 | 0.880 | strong positive descriptive association; not causal |

## Interpretation rule

Direct and construct-overlapping metrics confirm the task result. Diagnostic phase metrics are more useful for explaining where successful, semi-successful, and unsuccessful dialogues diverge. Correlations are descriptive associations and must not be presented as causal proof.
