# Thesis Result Configuration Effects

Scope: active, non-archived, thesis-relevant runs with Agent A in `{UserLM, TinyLlama}` and Agent B in the five selected Transformer backends. Raw run evidence is unchanged; this document is derived from `conditions.jsonl` files.

## Denominators

- Deduplicated active thesis rows: `1319`.
- Completed active thesis rows: `557`.
- Fully crossed text/audio subset: `11` condition groups, `220` runs.
- Fully crossed means every condition has both Agent A choices, all five Agent B choices, and both text/audio counterparts.

## Fully Crossed Text Versus Audio Table

| agent_a | agent_b | total | text_success_semi_fail | text_success_rate | audio_success_semi_fail | audio_success_rate | audio_text_delta_pp | text_mean_turns | audio_mean_turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tinyllama | Phi-3 Mini | 22 | 11/0/0 | 100.0% | 8/1/2 | 72.7% | -27.3 pp | 10.6 | 12.8 |
| tinyllama | Qwen2.5 0.5B | 22 | 11/0/0 | 100.0% | 8/0/3 | 72.7% | -27.3 pp | 10.6 | 14.0 |
| tinyllama | Qwen2.5 1.5B | 22 | 11/0/0 | 100.0% | 9/0/2 | 81.8% | -18.2 pp | 10.6 | 14.2 |
| tinyllama | Qwen2.5 7B | 22 | 11/0/0 | 100.0% | 9/0/2 | 81.8% | -18.2 pp | 10.6 | 12.6 |
| tinyllama | TinyLlama 1.1B | 22 | 11/0/0 | 100.0% | 9/0/2 | 81.8% | -18.2 pp | 10.6 | 14.5 |
| userlm | Phi-3 Mini | 22 | 11/0/0 | 100.0% | 8/0/3 | 72.7% | -27.3 pp | 10.5 | 14.1 |
| userlm | Qwen2.5 0.5B | 22 | 11/0/0 | 100.0% | 8/1/2 | 72.7% | -27.3 pp | 10.5 | 13.9 |
| userlm | Qwen2.5 1.5B | 22 | 11/0/0 | 100.0% | 9/0/2 | 81.8% | -18.2 pp | 10.6 | 14.5 |
| userlm | Qwen2.5 7B | 22 | 11/0/0 | 100.0% | 9/0/2 | 81.8% | -18.2 pp | 10.6 | 12.4 |
| userlm | TinyLlama 1.1B | 22 | 11/0/0 | 100.0% | 9/0/2 | 81.8% | -18.2 pp | 10.5 | 14.5 |

Safe inference:

- Text controls are near-ceiling in the fully crossed subset, so they mainly confirm that the route-dialogue policy and task validation can solve these conditions when speech degradation is removed.
- Audio variants reduce task success by roughly 18 to 27 percentage points in the fully crossed subset. This supports a speech-channel degradation claim, not a universal TTS/ASR benchmark claim.
- The matched subset is internally valid but small. Use it for direct Agent A/Agent B comparisons; use broader active rows only for descriptive coverage and association analysis.

## Factor Effects Across Active Completed Rows

The table below reports descriptive associations. It is not causal because factors are not always fully balanced.

### agent_a_type

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| userlm | 317 | 280 | 13 | 24 | 88.3% | 92.4% | 11.86 | 529.6 |
| tinyllama | 240 | 211 | 2 | 27 | 87.9% | 88.8% | 12.60 | 879.4 |

### agent_b_short

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 1.5B | 144 | 130 | 3 | 11 | 90.3% | 92.4% | 12.12 | 423.2 |
| Qwen2.5 7B | 85 | 76 | 2 | 7 | 89.4% | 91.8% | 11.75 | 1452.4 |
| TinyLlama 1.1B | 110 | 97 | 3 | 10 | 88.2% | 90.9% | 12.35 | 489.5 |
| Qwen2.5 0.5B | 110 | 95 | 4 | 11 | 86.4% | 90.0% | 12.40 | 339.3 |
| Phi-3 Mini | 108 | 93 | 3 | 12 | 86.1% | 88.9% | 12.18 | 957.2 |

### scenario_key

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hospital_appointment | 30 | 29 | 0 | 1 | 96.7% | 96.7% | 9.90 | 506.2 |
| morning_peak_cross_city | 281 | 251 | 2 | 28 | 89.3% | 90.0% | 12.69 | 697.9 |
| midday_transfer | 106 | 92 | 2 | 12 | 86.8% | 88.7% | 12.67 | 674.8 |
| crowded_event_exit | 68 | 57 | 1 | 10 | 83.8% | 85.3% | 11.63 | 621.0 |
| airport_connection | 46 | 46 | 0 | 0 | 100.0% | 100.0% | 10.80 | 906.3 |
| late_event | 16 | 16 | 0 | 0 | 100.0% | 100.0% | 10.50 | 492.6 |
| multi_destination_errands | 10 | 0 | 10 | 0 | 0.0% | 100.0% | 12.10 | 429.4 |

### persona_key

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| focused_commuter | 145 | 135 | 0 | 10 | 93.1% | 93.1% | 12.32 | 731.6 |
| crowd_averse_rider | 116 | 104 | 2 | 10 | 89.7% | 91.4% | 12.29 | 703.4 |
| delay_sensitive_traveler | 154 | 131 | 12 | 11 | 85.1% | 92.9% | 11.84 | 669.9 |
| risk_averse_novice | 112 | 92 | 0 | 20 | 82.1% | 82.1% | 12.56 | 652.0 |
| accessibility_rider | 4 | 3 | 1 | 0 | 75.0% | 100.0% | 12.00 | 374.2 |
| multi_stop_errand_runner | 26 | 26 | 0 | 0 | 100.0% | 100.0% | 11.15 | 522.5 |

### run_type

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text_only | 279 | 274 | 5 | 0 | 98.2% | 100.0% | 10.71 | 665.2 |
| audio_variant | 278 | 217 | 10 | 51 | 78.1% | 81.7% | 13.65 | 695.5 |

### speech_performance_band

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| challenging | 143 | 125 | 14 | 4 | 87.4% | 97.2% | 12.72 | 726.1 |
| floor | 102 | 54 | 1 | 47 | 52.9% | 53.9% | 11.86 | 461.3 |
| ceiling | 186 | 186 | 0 | 0 | 100.0% | 100.0% | 12.43 | 759.3 |
| nominal | 126 | 126 | 0 | 0 | 100.0% | 100.0% | 11.44 | 689.0 |

### speech_pattern_key

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hesitant | 143 | 125 | 14 | 4 | 87.4% | 97.2% | 12.72 | 726.1 |
| severe_channel | 102 | 54 | 1 | 47 | 52.9% | 53.9% | 11.86 | 461.3 |
| clean | 186 | 186 | 0 | 0 | 100.0% | 100.0% | 12.43 | 759.3 |
| mostly_clean | 126 | 126 | 0 | 0 | 100.0% | 100.0% | 11.44 | 689.0 |

### agent_a_audio_persona

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degraded_caller | 143 | 125 | 14 | 4 | 87.4% | 97.2% | 12.72 | 726.1 |
| barely_understandable_caller | 102 | 54 | 1 | 47 | 52.9% | 53.9% | 11.86 | 461.3 |
| high_clarity_caller | 186 | 186 | 0 | 0 | 100.0% | 100.0% | 12.43 | 759.3 |
| neutral_caller | 126 | 126 | 0 | 0 | 100.0% | 100.0% | 11.44 | 689.0 |

### agent_b_audio_persona

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| degraded_operator | 143 | 125 | 14 | 4 | 87.4% | 97.2% | 12.72 | 726.1 |
| barely_understandable_operator | 102 | 54 | 1 | 47 | 52.9% | 53.9% | 11.86 | 461.3 |
| high_clarity_operator | 186 | 186 | 0 | 0 | 100.0% | 100.0% | 12.43 | 759.3 |
| clear_operator | 126 | 126 | 0 | 0 | 100.0% | 100.0% | 11.44 | 689.0 |

### configured_asr_engine

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| faster_whisper | 168 | 156 | 11 | 1 | 92.9% | 99.4% | 10.73 | 661.2 |
| vosk | 389 | 335 | 4 | 50 | 86.1% | 87.1% | 12.80 | 688.6 |

### asr_search_width

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 143 | 125 | 14 | 4 | 87.4% | 97.2% | 12.72 | 726.1 |
| 1 | 102 | 54 | 1 | 47 | 52.9% | 53.9% | 11.86 | 461.3 |
| 16 | 186 | 186 | 0 | 0 | 100.0% | 100.0% | 12.43 | 759.3 |
| 11 | 126 | 126 | 0 | 0 | 100.0% | 100.0% | 11.44 | 689.0 |

### model_param_key

| level | completed | success | semi_success | unsuccessful | success_rate_completed | route_valid_rate_completed | mean_turns_completed | mean_runtime_sec_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nucleus0.9 | 136 | 126 | 0 | 10 | 92.6% | 92.6% | 12.29 | 768.1 |
| greedy | 351 | 309 | 13 | 29 | 88.0% | 91.7% | 12.22 | 669.3 |
| temp0.7 | 70 | 56 | 2 | 12 | 80.0% | 82.9% | 11.76 | 565.0 |

## Configuration-Level Inference Rules

- Agent A effects should be reported separately for UserLM and TinyLlama unless the subset is fully matched.
- Agent B effects are safest in the fully crossed subset; broader counts are useful for runtime feasibility and coverage.
- Scenario/persona/audio-persona effects are pressure-test effects: they indicate which conditions stress the pipeline, but they are not independent causal variables unless balanced.
- TTS is not varied in the fully crossed subset (`Piper` dominates), so current results do not support a comparative TTS-framework claim.
- ASR comparisons are descriptive because ASR engine and search width are partly tied to condition generation. Report as association unless a paired ASR-only subset is explicitly selected.
- Text/audio comparison is the strongest currently supported channel-level inference because paired counterparts exist.
