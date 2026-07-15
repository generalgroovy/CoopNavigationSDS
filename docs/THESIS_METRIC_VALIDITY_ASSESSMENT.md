# Thesis Metric Validity Assessment

Scope: metrics found in active thesis-relevant completed result rows. This document assesses what each metric can legitimately support in the thesis. It does not change metric values or raw evidence.

## Validity Principle

- A metric is valid only relative to a construct, required logged evidence, and interpretation boundary.
- Outcome-confirming metrics are valid for reporting task outcome but are not independent predictors of that outcome.
- Diagnostic phase metrics are useful when they capture earlier pipeline evidence that can explain later success or failure.
- Sparse, constant, or unavailable metrics should be reported as coverage/execution evidence, not as dialogue-quality evidence.
- Correlations are descriptive associations. They do not prove causality without controlled manipulation or held-out validation.

## Metric Role Counts

| role | metric_count |
| --- | --- |
| associated diagnostic metric | 59 |
| diagnostic phase metric | 24 |
| execution/constant metric | 36 |
| metric quality indicator | 5 |
| outcome-confirming | 14 |
| sparse metric | 1 |
| supplementary metric | 154 |

## Strongest Diagnostic Associations

| phase | metric | role | n_available | missingness_rate | corr_task_success | ceiling_rate | validity_assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dialogue state tracking | dialogue_state_joint_goal_accuracy | associated diagnostic metric | 557 | 0.0% | 1.000 | 88.2% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Natural language generation | nlg_constraint_satisfaction_rate | associated diagnostic metric | 557 | 0.0% | 0.998 | 88.2% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Agent B response / grounding | agent_b_active_constraint_compliance | diagnostic phase metric | 557 | 0.0% | 0.998 | 88.2% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Task outcome | task_outcome_acceptable_duration_completion | associated diagnostic metric | 557 | 0.0% | 0.974 | 88.7% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Whole dialogue | whole_dialogue_abandonment_rate | diagnostic phase metric | 557 | 0.0% | -0.974 | 11.3% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Whole dialogue | whole_dialogue_interaction_quality_trajectory | associated diagnostic metric | 557 | 0.0% | 0.974 | 0.2% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Whole dialogue | whole_dialogue_dialogue_success_score | associated diagnostic metric | 557 | 0.0% | 0.974 | 0.2% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Natural language generation | nlg_constraint_mention_precision | associated diagnostic metric | 493 | 11.5% | 0.970 | 87.8% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Natural language generation | nlg_constraint_mention_recall | associated diagnostic metric | 493 | 11.5% | 0.970 | 87.8% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Natural language generation | nlg_faithfulness | diagnostic phase metric | 557 | 0.0% | 0.967 | 69.1% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Agent B response / grounding | agent_b_grounded_proposal_score | diagnostic phase metric | 557 | 0.0% | 0.967 | 69.1% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Natural language generation | nlg_agent_a_guard_intervention_rate | associated diagnostic metric | 557 | 0.0% | 0.965 | 88.5% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Dialogue management | dialogue_management_stagnation_rate | diagnostic phase metric | 557 | 0.0% | 0.928 | 1.1% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Agent B response / grounding | agent_b_actionability_score | diagnostic phase metric | 557 | 0.0% | 0.915 | 88.7% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Task outcome | task_outcome_satisfied_constraint_count | associated diagnostic metric | 557 | 0.0% | 0.894 | 72.4% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Task outcome | task_outcome_stated_constraint_count | associated diagnostic metric | 557 | 0.0% | 0.887 | 72.4% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Other / task context | average_route_fullness | associated diagnostic metric | 557 | 0.0% | 0.882 | 0.4% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Other / task context | peak_route_fullness | associated diagnostic metric | 557 | 0.0% | 0.877 | 0.4% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Agent B response / grounding | agent_b_dominated_proposal_rate | associated diagnostic metric | 557 | 0.0% | 0.872 | 0.2% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Whole dialogue | whole_dialogue_goal_progress_auc | diagnostic phase metric | 557 | 0.0% | 0.869 | 50.3% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Agent B response / grounding | agent_b_mode_permission_compliance | associated diagnostic metric | 557 | 0.0% | 0.866 | 90.8% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Automatic speech recognition | asr_station_recall | associated diagnostic metric | 557 | 0.0% | 0.842 | 61.4% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Natural language generation | nlg_guard_intervention_rate | associated diagnostic metric | 557 | 0.0% | 0.829 | 87.3% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Natural language understanding | nlu_route_valid_rate | diagnostic phase metric | 557 | 0.0% | 0.825 | 62.5% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Natural language understanding | nlu_intent_accuracy | associated diagnostic metric | 557 | 0.0% | 0.825 | 62.5% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Agent B response / grounding | agent_b_route_validity_rate | associated diagnostic metric | 557 | 0.0% | 0.825 | 62.5% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Natural language understanding | nlu_station_mention_rate | associated diagnostic metric | 557 | 0.0% | 0.824 | 62.5% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Text-to-speech | tts_station_pronunciation_accuracy | associated diagnostic metric | 557 | 0.0% | 0.822 | 43.1% | usable as descriptive indicator; verify construct overlap and matched-condition stability |
| Automatic speech recognition | asr_station_f1 | diagnostic phase metric | 557 | 0.0% | 0.822 | 43.1% | high diagnostic value if evidence coverage is present; association is descriptive, not causal |
| Natural language generation | nlg_agent_b_guard_intervention_rate | associated diagnostic metric | 557 | 0.0% | 0.818 | 87.3% | usable as descriptive indicator; verify construct overlap and matched-condition stability |

## Phase-Level Interpretation

| phase | role | metric_count |
| --- | --- | --- |
| Agent A evaluation | associated diagnostic metric | 1 |
| Agent A evaluation | execution/constant metric | 4 |
| Agent A evaluation | outcome-confirming | 1 |
| Agent A evaluation | supplementary metric | 10 |
| Agent B response / grounding | associated diagnostic metric | 9 |
| Agent B response / grounding | diagnostic phase metric | 3 |
| Agent B response / grounding | execution/constant metric | 1 |
| Agent B response / grounding | supplementary metric | 9 |
| Audio / turn-taking | associated diagnostic metric | 1 |
| Audio / turn-taking | execution/constant metric | 2 |
| Audio / turn-taking | supplementary metric | 9 |
| Automatic speech recognition | associated diagnostic metric | 3 |
| Automatic speech recognition | diagnostic phase metric | 5 |
| Automatic speech recognition | execution/constant metric | 5 |
| Automatic speech recognition | supplementary metric | 12 |
| Dialogue management | associated diagnostic metric | 4 |
| Dialogue management | diagnostic phase metric | 3 |
| Dialogue management | execution/constant metric | 2 |
| Dialogue management | supplementary metric | 12 |
| Dialogue state tracking | associated diagnostic metric | 2 |
| Dialogue state tracking | diagnostic phase metric | 3 |
| Dialogue state tracking | execution/constant metric | 8 |
| Dialogue state tracking | supplementary metric | 5 |
| Metric validity | metric quality indicator | 5 |
| Natural language generation | associated diagnostic metric | 12 |
| Natural language generation | diagnostic phase metric | 2 |
| Natural language generation | execution/constant metric | 3 |
| Natural language generation | supplementary metric | 12 |
| Natural language understanding | associated diagnostic metric | 2 |
| Natural language understanding | diagnostic phase metric | 4 |
| Natural language understanding | execution/constant metric | 1 |
| Natural language understanding | supplementary metric | 11 |
| Other / task context | associated diagnostic metric | 9 |
| Other / task context | execution/constant metric | 5 |
| Other / task context | outcome-confirming | 6 |
| Other / task context | sparse metric | 1 |
| Other / task context | supplementary metric | 40 |
| Task outcome | associated diagnostic metric | 6 |
| Task outcome | outcome-confirming | 7 |
| Task outcome | supplementary metric | 6 |
| Text-to-speech | associated diagnostic metric | 4 |
| Text-to-speech | execution/constant metric | 4 |
| Text-to-speech | supplementary metric | 10 |
| Whole dialogue | associated diagnostic metric | 6 |
| Whole dialogue | diagnostic phase metric | 4 |
| Whole dialogue | execution/constant metric | 1 |
| Whole dialogue | supplementary metric | 18 |

## Metrics To Emphasize In The Thesis

- Task outcome: route validity, task success, constraint satisfaction, duration quality, stage completion.
- ASR: WER, entity error rate, station F1, critical slot accuracy, numeric/constraint preservation where available.
- NLU: slot F1, route-valid rate, goal-reached rate, semantic frame accuracy, origin/destination accuracy.
- Dialogue state/management: trip fact completeness, missing trip slot rate, constraint retention, repair success, stagnation, policy progress.
- Agent B/NLG: grounded proposal score, actionability, active constraint compliance, NLG faithfulness, executable utterance rate.
- Whole dialogue: goal progress AUC, abandonment rate, turn count, repair count, task focus, failure localization.

## Metrics To Treat Cautiously

- Generic lexical metrics such as BLEU/ROUGE/METEOR are supplementary because station/line correctness and route executability matter more than surface overlap.
- TTS quality metrics such as NISQA, DNSMOS, PESQ, POLQA, STOI, and SI-SDR are valid only when the required audio/reference evidence exists; unavailable values must not be imputed as bad quality.
- Runtime and latency metrics can indicate feasibility and cost but can be confounded by cluster node load and model-loading conditions.
- Metrics close to the task-success definition should be used to decompose the outcome, not to claim independent prediction.

Full per-metric table: `docs/THESIS_METRIC_VALIDITY_TABLE.csv`.
