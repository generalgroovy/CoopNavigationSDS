# Thesis Writing Aid: Automatic Evaluation of Spoken Dialogue Systems

Purpose: one compact reference for writing a research-grade bachelor thesis
around CoopNavigationSDS. It combines the current draft, the revised thesis
structures, and the implemented experiment design. It is not a replacement for
the thesis text. Use it as a checklist, argument map, citation guide, formula
sheet, and chapter scaffold.

## 0. Thesis Core

### One-sentence thesis claim

This thesis studies whether a controlled cooperative navigation task can be
used to automatically evaluate spoken task-oriented dialogue systems
phase-wise, so that success, failure, constraint satisfaction, and likely
failure origin can be explained from logged evidence instead of only final
task outcome.

### What the thesis is about

- Scientific object: automatic evaluation of spoken task-oriented dialogue
  systems.
- Experimental setting: cooperative route finding in a controlled transport
  network.
- Evaluated system role: Agent B, the route-information dialogue system.
- User role: Agent A/UserLM/TinyLlama as simulated caller with private task
  goal and progressively revealed constraints.
- Main method: run many controlled dialogue conditions, capture raw phase
  evidence, calculate metrics retrospectively, and compare model backends.
- Main outcome: determine which metrics and phases explain successful,
  semi-successful, and unsuccessful navigation dialogues.

### What the thesis is not about

- Not a general route-planning application.
- Not a claim that simulated users fully replace human users.
- Not a pure benchmark of model size alone.
- Not a universal evaluation of all spoken dialogue systems.
- Not a human-subject usability study.

### Central validity principle

Raw phase evidence is authoritative. Metrics, tables, and plots are derived
from stored evidence after the run. A missing provider, missing audio, invalid
route, or failed model call is not silently replaced; it is recorded as an
experimental outcome or unavailable evidence with reason.

### Current active experiment scope

- Agent A: UserLM is the thesis caller. TinyLlama remains a software control
  and debugging option, but it is not part of the main thesis denominator.
- Agent B: five selected Transformer models by size class.
- Active coverage target: `small1`, `small2`, `medium1`, `medium2`, and
  `large1`.
- Speech channel: configured TTS and ASR; text-only controls may be used where
  paired comparison is available.
- Task objective: shortest valid route under progressively revealed
  constraints.
- Observation unit: one complete condition/run, not one turn.

### Current empirical snapshot

Use this only as the current result basis, not as final thesis wording unless
the result set is frozen.

- Snapshot source: pulled result folders at commit `692b2756` on 2026-07-14.
- Finalized run folders found locally: 1764.
- Thesis denominator used here: UserLM as Agent A and five selected Agent B
  models. TinyLlama-Agent-A runs and archived models are excluded from the
  main denominator but remain useful for software control analysis.
- Calculation source: `conditions.jsonl` inside each run folder. Duplicate
  attempts are preserved as runtime evidence; coverage uses unique condition
  IDs and counts a condition as completed when any retained attempt completed.
- Archived large2/Mistral settings are removed from active setup and Slurm
  submission. Existing raw result folders are not altered.

Current UserLM-Agent-A unique-condition outcomes:

| Agent B slot | Model | Size | Unique conditions observed | Completed | Task-success | Route-valid | Task-success rate of completed | Route-valid rate of completed | Mean turns |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small1 | TinyLlama 1.1B | small | 154 | 62 | 54 | 57 | 87.1% | 91.9% | 12.18 |
| small2 | Qwen2.5 0.5B | small | 154 | 62 | 54 | 57 | 87.1% | 91.9% | 12.27 |
| medium1 | Qwen2.5 1.5B | medium | 154 | 96 | 88 | 91 | 91.7% | 94.8% | 11.68 |
| medium2 | Phi-3 mini | medium | 154 | 60 | 52 | 54 | 86.7% | 90.0% | 12.17 |
| large1 | Qwen2.5 7B | large | 154 | 37 | 33 | 35 | 89.2% | 94.6% | 11.16 |

Current UserLM-Agent-A execution-attempt evidence:

| Agent B model | Attempts | Completed attempts | Failed attempts | Task-successful completed attempts | Approx. summed runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B | 244 | 94 | 110 | 84 | 748.3 min |
| Qwen2.5 0.5B | 244 | 94 | 110 | 84 | 637.5 min |
| Qwen2.5 1.5B | 359 | 137 | 168 | 126 | 891.6 min |
| Phi-3 mini | 299 | 77 | 182 | 69 | 881.0 min |
| Qwen2.5 7B | 272 | 50 | 168 | 46 | 731.5 min |

Immediate interpretation:

- Completion and task success must be separated. Some models show high
  task-success rates among completed dialogues while also producing many
  failed execution attempts.
- Qwen2.5 1.5B currently provides the strongest combination of completion
  count and task success in the UserLM subset.
- Qwen2.5 7B has strong route validity among completed runs but substantially
  fewer completed unique conditions, so its lower coverage is a backend/runtime
  limitation before it is a dialogue-quality claim.
- TinyLlama 1.1B and Qwen2.5 0.5B form a useful small-model pair because their
  completed-run behavior is similar under the current grid.
- Failure analysis should inspect the earliest failing phase and provider
  failure messages before interpreting failed attempts as conversational
  inability.

## 1. Introduction

### Chapter function

Introduce the problem, motivate the research, define the scope, state research
questions, and summarize contributions. Keep implementation detail out of this
chapter.

### 1.1 Motivation and background

- Spoken dialogue systems support interaction through speech.
- Speech adds practical value:
  - hands-free interaction,
  - accessibility,
  - natural interaction in mobile or assistive contexts,
  - prosodic information such as pauses, hesitation, emphasis, and rhythm.
- Spoken systems are harder to evaluate than text systems because:
  - the acoustic channel can distort information,
  - turns unfold over time,
  - errors can propagate across phases,
  - final success may hide earlier degradation.
- Task-oriented spoken dialogue is suitable for automatic evaluation because
  goals and constraints can be checked against an external task environment.

### 1.2 Evaluation difficulty

- A failed spoken dialogue can originate from:
  - audio/TTS failure,
  - ASR misrecognition,
  - NLU slot or constraint error,
  - dialogue state drift,
  - wrong dialogue-management action,
  - hallucinated or invalid route,
  - incomplete natural-language answer,
  - unintelligible synthesized speech,
  - latency or turn-taking problems.
- Human evaluation is important but costly, slow, and subjective.
- Automatic evaluation is scalable and repeatable but only useful when each
  metric has a clear construct and evidence source.
- Final task success alone answers "what happened"; phase evidence helps
  answer "where did it go wrong".

### 1.3 Problem statement

Use this wording as the thesis backbone:

- Existing dialogue evaluation often focuses on final task success, human
  ratings, response similarity, or isolated component metrics.
- These views are useful but insufficient for diagnosing spoken
  task-oriented failures.
- In a route dialogue, failure may be caused by recognition, understanding,
  memory, policy, grounding, generation, or speech output.
- This thesis investigates whether phase-wise automatic evidence can explain
  success and failure in controlled spoken navigation dialogues.

### 1.4 Research questions

Recommended final set:

1. Can success and failure in a controlled spoken navigation dialogue be
   explained using automatically logged phase-wise evidence and metrics?
2. Which automatic metrics best correlate with task-level outcomes such as
   task completion, route validity, and constraint satisfaction?
3. Can phase-wise evidence identify the earliest likely failing phase in
   unsuccessful or degraded dialogues?
4. Which phase-level metrics remain useful across different Agent B language
   model backends and size classes?
5. If paired text/audio runs are available: how does the speech channel affect
   task success and phase-level errors compared with matched text controls?

### 1.5 Expected tendencies

- Task-grounded metrics should explain success better than generic text
  similarity metrics.
- Semantic ASR/entity metrics should be more informative than WER alone for
  navigation failures.
- Constraint extraction and retention should strongly influence progressive
  constraint success.
- Larger models may improve repair and route grounding but can increase
  latency and resource cost.
- Spoken audio should increase entity, number, line-name, and constraint
  errors compared with clean text controls.
- Earliest-failure labels should be treated as diagnostic candidates, not
  proven causal facts.

### 1.6 Contributions

- Phase-aware evaluation framing for spoken task-oriented dialogue.
- Controlled cooperative navigation task with externally checkable route
  validity.
- Implemented CoopNavigationSDS framework with configurable agents, speech,
  scenarios, model backends, logging, and retrospective metrics.
- Metric-outcome analysis across Agent B backends.
- Failure-localization method based on chronological phase evidence.

## 2. Theoretical Background

### Chapter function

Explain spoken dialogue systems and their phases. Do not describe the concrete
experiment yet except as brief motivation.

### 2.1 Dialogue systems and spoken dialogue systems

- Dialogue system: interactive language system that receives input, maintains
  context, selects an action, and responds.
- Spoken dialogue system: dialogue system with speech input and/or speech
  output.
- Spoken SDS adds:
  - acoustic uncertainty,
  - timing and turn-taking,
  - speech recognition,
  - speech synthesis,
  - intelligibility requirements.

### 2.2 Task-oriented dialogue

- Task-oriented systems help users complete defined goals.
- They contain:
  - user goal,
  - required slots,
  - optional constraints,
  - dialogue state,
  - system actions,
  - backend/task environment,
  - success criterion.
- In this thesis, task state includes:
  - start station,
  - destination station,
  - departure time,
  - active constraints,
  - candidate routes,
  - rejected routes,
  - accepted route.

### 2.3 Spoken versus text dialogue

- Text dialogue assumes symbolic input is already available.
- Spoken dialogue must recover text/meaning from audio.
- Spoken-specific errors:
  - station-name substitution,
  - line-name substitution,
  - number/time confusion,
  - negation loss,
  - clipping,
  - silence,
  - ASR hallucination,
  - TTS mispronunciation.

### 2.4 Pipeline as analysis model

Use the pipeline as an evaluation structure, not as a claim that every modern
system is implemented as separate modules.

```text
Audio / turn-taking
-> ASR
-> NLU / SLU
-> Dialogue state tracking
-> Dialogue management
-> Backend grounding
-> NLG
-> TTS
-> Whole-dialogue outcome
```

For each phase, define:

- input,
- output,
- responsibility,
- typical failure,
- logged evidence,
- relevant metrics.

### 2.5 LLM-based dialogue systems

- LLMs can absorb NLU, state tracking, policy, and NLG inside one model call.
- Advantages:
  - flexible language handling,
  - natural repair dialogue,
  - few-shot adaptation,
  - stronger reasoning for some tasks.
- Risks:
  - hallucination,
  - hidden state,
  - prompt sensitivity,
  - constraint omission,
  - resource cost,
  - latency.
- Therefore, phase-aware external evidence is still needed even when internal
  module boundaries are blurred.

### 2.6 Error propagation

Example chain:

```text
TTS makes "Juliett" unclear
-> ASR hears "juliet" or "julie"
-> NLU normalizes to wrong station
-> Agent memory stores wrong destination
-> route validator rejects proposal
-> dialogue loops in repair
-> task fails or becomes semi-successful
```

The thesis should emphasize propagation, not isolated component blame.

## 3. Evaluation Concepts

### Chapter function

Introduce evaluation concepts before the concrete metric catalog.

### 3.1 Human versus automatic evaluation

- Human evaluation captures perceived satisfaction, naturalness, usefulness,
  and trust.
- Automatic evaluation provides:
  - scalability,
  - repeatability,
  - parallelization,
  - precise traceability.
- Automatic metrics are not automatically valid; each metric must be tied to a
  construct.

### 3.2 Construct validity

For every metric, ask:

- What construct does it claim to measure?
- What logged data does it require?
- What formula is used?
- What range and direction does the value have?
- When is it unavailable?
- What can it not measure?

### 3.3 Why generic text metrics are insufficient

- BLEU/ROUGE-style overlap can miss task validity.
- A fluent route can be invalid.
- A terse route can be correct.
- Surface similarity does not guarantee constraint satisfaction.
- Use task-grounded metrics wherever possible.

### 3.4 Speech metrics and semantic metrics

- WER is useful but incomplete.
- A single station-name error can be more important than several harmless word
  errors.
- Navigation needs semantic ASR metrics:
  - station preservation,
  - line preservation,
  - time preservation,
  - constraint preservation,
  - negation preservation.

## 4. Related Work

### Chapter function

Organize papers by research strand. Do not list papers without explaining how
they relate to the thesis.

### 4.1 Classical SDS evaluation

- PARADISE connects task success, dialogue cost, and user satisfaction.
- Use it to justify combining outcome and cost metrics.
- Difference to this thesis:
  - this thesis focuses more on phase-wise retrospective evidence and model
    backend comparison.

### 4.2 Task-oriented dialogue benchmarks

- MultiWOZ and SGD motivate task-oriented dialogue, slot/state tracking, and
  multi-domain evaluation.
- Difference to this thesis:
  - many benchmarks are text-based and do not capture the full speech channel.

### 4.3 Spoken TOD and SLU

- SpokenWOZ and SLUE motivate spoken input, speech-text mismatch, and
  speech-aware evaluation.
- Difference to this thesis:
  - CoopNavigationSDS uses a controlled route task with route validation and
    staged constraints.

### 4.4 User simulation

- User simulation allows repeatable evaluation.
- Limitation:
  - simulated behavior is not real human behavior.
- In this thesis:
  - Agent A is controlled and has private knowledge boundaries.

### 4.5 Navigation and grounded dialogue

- Navigation dialogue is useful because it requires grounding, clarification,
  route communication, and shared task progress.
- CoopNavigationSDS uses transport-network grounding rather than open-ended
  map navigation.

### 4.6 LLM-agent evaluation

- Modern agent benchmarks emphasize tool use and final-state verification.
- This thesis adds spoken pipeline evidence and phase-wise metrics.

## 5. Limitations and Validity Threats

### Chapter function

Prepare careful interpretation. Do not wait until the discussion to define
validity limits.

### Main limitations to state

- Synthetic route network limits ecological validity.
- Simulated users do not fully represent real users.
- TTS/ASR settings may not represent real microphones, speakers, or accents.
- Model comparisons confound size, family, training data, quantization,
  provider, prompt, and hardware.
- Metrics are diagnostic indicators, not direct proof of causality.
- Turns are nested inside runs and should not be treated as independent
  observations.
- Missing metric evidence must be reported as missing, not zero.
- Provider failure is an experimental/runtime outcome, not task failure.

## 6. Methodology and Research Design

Starting here, focus on the actual experiment and metric selection.

### 6.1 Experimental unit

- Unit of analysis: one complete condition/run.
- A condition fixes:
  - scenario,
  - start/destination/departure time,
  - constraint set and reveal order,
  - Agent A implementation/persona/audio persona,
  - Agent B model backend,
  - TTS backend and audio pattern,
  - ASR backend and recognition settings,
  - seed,
  - repetition,
  - run type.
- Turns and phases are nested evidence inside a condition.
- Do not treat turns as independent samples.

### 6.2 Agent roles and knowledge boundaries

Agent A:

- simulated caller,
- knows start station, destination, departure time, station names, line names,
  persona priorities, and private constraints,
- does not know network topology or optimal routes,
- reveals constraints progressively after the prior goal is satisfied,
- is the only agent allowed to accept/end the call.

Agent B:

- evaluated dialogue system,
- receives network/task data available to a route-information assistant,
- does not know Agent A's hidden future constraints,
- must infer current user goal from what it heard,
- proposes line-identified routes,
- clarifies critical ambiguity,
- keeps its own memory based on its received transcripts and prior outputs.

Integrity rule:

- Agents do not share hidden memory.
- Listener consumes ASR/NLU output, not hidden intended text.

### 6.3 Dialogue stages

Recommended stage logic:

1. Establish valid route from start to destination.
2. Check whether duration is acceptable relative to the current optimal route.
3. Reveal constraint 1 if the previous stage is satisfied.
4. Ask for a better compliant route if the current proposal violates the new
   constraint.
5. Reveal constraint 2 only if the previous layer is satisfied.
6. Select best understood viable route or close dissatisfied after limits.

Stage status:

- success: route satisfies current stage and active constraints.
- semi-success: route is valid but some revealed constraint or optimality
  criterion remains unsatisfied.
- unsuccessful: no valid route, repeated unresolved misunderstanding, provider
  failure, or turn limit without usable route.

### 6.4 Route task and network

The network exists to make language objectively checkable.

Route representation should contain:

- transport type,
- line identifier,
- boarding station,
- alighting station,
- intermediate stations where useful,
- transfer points,
- duration,
- active constraints.

Complete route example:

```text
Metro M2 from Bravo to Delta via Charlie;
Tram T6 from Delta to Harbor via Juliett and Victor.
Total: 31 min, 1 change.
```

Validation checks:

- all stations exist,
- all line names exist,
- every segment is connected,
- transfers occur only when line changes,
- destination is reached,
- route duration is within configured threshold,
- active constraints are satisfied,
- route is compared against the current constraint-layer optimum.

### 6.5 Constraint layers and optimal route calculation

Compute optimal route separately for:

```text
Layer 0: valid route only
Layer 1: valid route + acceptable time
Layer 2: previous + constraint 1
Layer 3: previous + constraint 2
Layer 4: previous + optional constraint 3, if used
```

Purpose:

- makes staged progress measurable,
- prevents judging early dialogue against unrevealed constraints,
- lets the thesis show how added constraints change the optimal route,
- supports semi-success classification.

### 6.6 Speech channel

Speech pipeline:

```text
NLG intended text
-> TTS waveform
-> optional channel/audio pattern
-> ASR raw transcript
-> normalization
-> NLU frame
-> listener memory
```

Evidence to store:

- intended text,
- delivered TTS text,
- waveform path/metadata,
- ASR raw transcript,
- normalized transcript,
- corrections and substitutions,
- understood semantic frame,
- phase timings,
- unavailable reasons.

Key integrity rule:

- Corrected/normalized transcript must be transparent. If normalization repairs
  "juliet" to "Juliett", both raw and normalized forms must be visible.

### 6.7 Batch design

Use balanced coverage where possible:

- same non-model conditions for each Agent B model,
- one or more models per size class,
- exact condition IDs for joinable comparison,
- no silent reruns counted as new conditions unless repetition is explicit,
- archived/obsolete model settings are excluded from the active denominator.

Suggested active denominator:

```text
selected_userlm_models = small1 + small2 + medium1 + medium2 + large1
planned_conditions = sum(valid planned conditions for those five slots)
coverage = completed_planned_conditions / planned_conditions
```

Report two coverage levels separately:

```text
execution_coverage =
    observed_or_completed_conditions / planned_conditions

task_completion_coverage =
    completed_dialogues_with_terminal_outcome / planned_conditions
```

Reason:

- execution coverage shows whether the cluster/batch pipeline produced
  evidence;
- task completion coverage shows whether the dialogue reached a valid terminal
  state;
- task success is only interpretable after both coverage values are reported.

Current active denominator for the pulled result set:

```text
selected_thesis_conditions = 744
completed_selected_conditions = 310
coverage = 41.67%
```

For the current thesis denominator, UserLM-Agent-A comparisons are analyzed
separately from TinyLlama-Agent-A software controls. Do not mix both callers in
one unstratified success rate.

### 6.8 Data captured per run

Minimum reproducibility metadata:

- timestamp,
- git commit,
- random seed,
- job file,
- condition ID,
- model backend/profile,
- ASR/TTS backend and settings,
- scenario and persona,
- network seed,
- route task facts,
- full dialogue transcript,
- phase-level event log,
- route candidates and validation,
- optimal route by constraint layer,
- metric input rows,
- metric outputs,
- errors/warnings.

## 7. Metric Selection

### Metric selection principle

Prefer metrics that are:

- automatically calculable from logged evidence,
- interpretable in task terms,
- tied to a phase,
- comparable across runs,
- robust to missing optional data,
- useful for diagnosing success/failure.

Avoid metrics that:

- require unavailable human labels,
- duplicate another metric without new information,
- measure surface text overlap when task validity is available,
- cannot be computed consistently across conditions.

### 7.1 Core formulas

Word Error Rate:

```text
WER = (S + D + I) / N
```

- S: substitutions
- D: deletions
- I: insertions
- N: reference word count
- Lower is better.

Precision, recall, F1:

```text
precision = true_positive / (true_positive + false_positive)
recall    = true_positive / (true_positive + false_negative)
F1        = 2 * precision * recall / (precision + recall)
```

Use for slots, constraints, stations, lines, and route entities.

Task success:

```text
task_success = route_valid
               and destination_reached
               and active_constraints_satisfied
               and accepted_by_agent_a
```

Constraint satisfaction rate:

```text
constraint_satisfaction_rate =
    satisfied_active_constraints / active_constraints
```

Route optimality ratio:

```text
optimality_ratio = optimal_duration / selected_route_duration
```

- 1.0 means equal to optimum.
- Below 1.0 means slower than the optimum.

Duration regret:

```text
duration_regret = selected_route_duration - optimal_duration
```

Repair success rate:

```text
repair_success_rate =
    successful_repairs / repair_attempts
```

Failure-localization candidate:

```text
earliest_failing_phase =
    first chronological phase where critical evidence is missing,
    corrupted, contradicted, or invalid before downstream failure.
```

### 7.2 Phase metrics

Use this table to decide which metrics belong in Chapter 6/7 and why.

| Phase | Metrics to emphasize | Why it matters |
| --- | --- | --- |
| Audio/turn-taking | capture success, missing audio, turn latency, utterance duration, clipping/silence | proves whether the speech signal was usable |
| ASR | WER, entity error rate, station/line preservation, time preservation, semantic ASR error | explains whether Agent B heard task-critical content |
| NLU/SLU | slot F1, critical slot accuracy, route parse success, constraint extraction F1 | explains whether transcript became correct task meaning |
| Dialogue state | trip-fact completeness, constraint retention, state drift, shared-state agreement | explains whether agents remembered the task correctly |
| Dialogue management | clarification precision, repair success, premature answer rate, policy progress, route repetition | explains whether the system acted appropriately |
| Backend grounding | route validity, destination reach, grounded proposal score, hallucinated station/line rate, optimality gap | proves whether proposals are real task solutions |
| NLG | route mention completeness, faithfulness, executable utterance rate, conciseness | explains whether the route was communicated clearly |
| TTS | synthesis success, TTS latency, NISQA, DNSMOS, pronunciation/entity preservation | explains whether correct text remained intelligible |
| Whole dialogue | task success, constraint satisfaction, turn count, dialogue cost, failure-localization score | summarizes outcome and efficiency |

### 7.3 Metrics most likely to answer the research questions

For RQ1 and RQ2:

- task success,
- route validity,
- destination reach,
- constraint satisfaction rate,
- duration regret,
- grounded proposal score,
- route mention completeness,
- critical slot accuracy,
- constraint retention rate,
- repair success rate.

For RQ3:

- ASR entity error rate,
- NLU critical slot accuracy,
- state drift rate,
- invalid proposal rate,
- hallucinated station/line rate,
- failure-localization phase,
- first critical error turn.

For RQ4:

- task success by model,
- constraint satisfaction by model,
- mean duration regret by model,
- repair success by model,
- mean latency by model,
- success per 100 output tokens,
- route validity per model size.

### 7.4 Current metric evidence to emphasize

From the current pulled result set, the most thesis-relevant metric families
are the ones that are both calculable and directly linked to the task:

- backend task execution:
  - route validity,
  - destination reached,
  - active-constraint compliance,
  - grounded proposal score,
  - actionability score.
- dialogue state tracking:
  - joint goal accuracy,
  - shared state agreement,
  - route agreement,
  - constraint retention.
- dialogue management:
  - stagnation rate,
  - repair success or repair failure,
  - premature answer rate.
- NLG:
  - constraint mention precision and recall,
  - faithfulness,
  - executable utterance rate.
- ASR/TTS and audio:
  - station/line/time preservation,
  - misinterpreted tokens,
  - transcript corrections,
  - real-time factor,
  - pronunciation/entity preservation,
  - no-reference audio quality where available.

Current correlation tables show near-perfect association between task success
and several task-derived metrics such as constraint satisfaction, active
constraint compliance, joint goal accuracy, grounded proposal score, and
faithfulness. Treat this carefully: these metrics are construct-valid for task
success, but some are partly definitional. They are excellent for explaining
the task outcome, but they do not alone prove which upstream phase caused the
outcome.

Audio and speech metrics are useful mainly as diagnostic indicators. In the
current result set, outliers in station pronunciation accuracy, no-reference
speech quality, real-time interaction factor, ASR real-time factor, speech
rate, and silence ratio align with success/failure patterns often enough to
justify analysis, but they require cautious interpretation and should be
validated against dialogue evidence.

### 7.5 Missing-data rule

Never encode missing evidence as zero.

Use:

```text
value_numeric = null
available = false
unavailable_reason = "missing_audio" | "provider_failed" | "not_applicable" | ...
```

Interpretation:

- zero means the metric was calculated and the result is zero.
- null means the metric could not legitimately be calculated.

### 7.6 Semi-success classification

Recommended classification:

- successful:
  - final route valid,
  - destination reached,
  - accepted by Agent A,
  - all revealed constraints satisfied,
  - duration within configured threshold.
- semi-successful:
  - route valid and destination reached,
  - but one or more revealed constraints, optimality thresholds, or repair
    expectations remain unsatisfied.
- unsuccessful:
  - no valid final route,
  - critical trip facts unresolved,
  - repeated repair loop,
  - provider failure prevents dialogue,
  - turn limit reached without usable route.

## 8. Results Chapter Guide

### 8.1 Report order

1. Run inventory and coverage.
2. Exclusions and unavailable evidence.
3. Overall task outcomes.
4. Success/semi-success/unsuccess distribution.
5. Phase metric distributions.
6. Metric-outcome correlations.
7. Model-backend comparison.
8. Failure localization.
9. Representative success and failure cases.

### 8.2 Tables to include

- Condition coverage by Agent B model.
- Outcome distribution by Agent B model.
- Mean/median duration regret and turn count.
- Top metrics associated with task success.
- Failure phase distribution.
- ASR/TTS degradation versus outcome.
- Model runtime and memory summary.

### 8.3 Figures to include

- Pipeline diagram with evidence boundaries.
- Condition grid overview.
- Success distribution by model.
- Metric heatmap by phase.
- Failure-localization bar chart.
- Paired text/audio delta plot, if available.

### 8.4 Interpretation caution

Write:

- "associated with", "indicates", "diagnostic evidence suggests".

Avoid:

- "proves that phase X caused failure" unless manually validated.

### 8.5 Current result interpretation notes

Use these as analysis prompts for the current result set:

- First separate batch execution from dialogue quality:
  - not-started and interrupted conditions explain coverage loss;
  - completed dialogues explain task success and constraint satisfaction.
- Then report completed-dialogue quality:
  - route validity is high for completed UserLM-Agent-A dialogues;
  - task satisfaction differs more strongly by constraint handling and
    dialogue completion than by route validity alone.
- Then compare models:
  - Qwen2.5 1.5B currently has the strongest completed-dialogue task
    satisfaction in the pulled results;
  - TinyLlama 1.1B and Qwen2.5 0.5B are close small-model baselines;
  - Phi-3 mini has reasonable task satisfaction among completed dialogues but
    lower completion coverage in the pulled result set;
  - Qwen2.5 7B has valid routes when completed, but weaker execution coverage
    and lower selected-slot success under current CPU batch conditions.
- Finally discuss limitations:
  - TinyLlama-Agent-A control coverage is still missing locally;
  - archived model settings are excluded;
  - CPU cluster runtime, provider availability, and interruption patterns are
    part of the experimental condition and must not be hidden.

Suggested wording:

```text
Among completed UserLM-agent dialogues, most final routes were valid, which
suggests that route grounding itself is often successful once a terminal
dialogue state is reached. The main experimental variation therefore appears
in whether the dialogue completes, whether constraints remain satisfied, and
whether upstream speech and state evidence stays coherent enough for Agent A
to accept the final route.
```

## 9. Discussion and Conclusion

### Discussion structure

- Answer each research question directly.
- Explain which metrics were useful and why.
- Explain which metrics were weak or redundant.
- Discuss model-backend differences cautiously.
- Discuss speech-channel effects.
- Discuss limitations and threats to validity.
- Explain what a supervisor/examiner can learn from the framework.

### Conclusion structure

- Restate problem.
- Summarize method.
- Summarize main findings.
- State contributions.
- State limitations.
- Future work:
  - human validation,
  - real transit network,
  - real microphone input,
  - more languages/accents,
  - more speech-native models,
  - larger balanced condition grid,
  - additional task domains.

## 10. Citation Map

Use citations only when they support a specific claim.

| Claim | Recommended source |
| --- | --- |
| SDS evaluation can combine task success, dialogue cost, and satisfaction | Walker et al. 1997, PARADISE |
| Surface text metrics can correlate weakly with dialogue quality | Liu et al. 2016, How NOT To Evaluate Your Dialogue System |
| Task-oriented dialogue benchmarks rely on goal, slot, and state tracking | MultiWOZ; Schema-Guided Dialogue |
| Spoken task-oriented dialogue differs from written TOD | SpokenWOZ; SLUE |
| Automatic metrics need construct validity | Jacobs and Wallach 2021 |
| Speech quality can be predicted with learned no-reference metrics | NISQA; DNSMOS |
| LLM/backend comparisons require caution and transparency | HELM; model cards; reproducibility papers |
| Simulated users enable repeatable experiments but limit external validity | user simulation and agenda-based simulation work |
| Navigation dialogue is suitable for grounding and cooperation | Map Task Corpus; grounded navigation dialogue work |

### Verified starting references

- Walker, M. A., Litman, D. J., Kamm, C. A., and Abella, A. 1997.
  "PARADISE: A Framework for Evaluating Spoken Dialogue Agents." ACL/EACL.
  https://aclanthology.org/P97-1035/
- Liu, C.-W., Lowe, R., Serban, I., Noseworthy, M., Charlin, L., and Pineau, J.
  2016. "How NOT To Evaluate Your Dialogue System." EMNLP.
  https://aclanthology.org/D16-1230/
- Budzianowski, P. et al. 2018. "MultiWOZ: A Large-Scale Multi-Domain
  Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling."
  https://aclanthology.org/D18-1547/
- Rastogi, A., Zang, X., Sunkara, S., Gupta, R., and Khaitan, P. 2020.
  "Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided
  Dialogue Dataset." AAAI. https://doi.org/10.1609/aaai.v34i05.6394
- Si, S. et al. 2023. "SpokenWOZ: A Large-Scale Speech-Text Benchmark for
  Spoken Task-Oriented Dialogue Agents." https://arxiv.org/abs/2305.13040
- Mittag, G., Naderi, B., Chehadi, A., and Moeller, S. 2021. "NISQA: A Deep
  CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with
  Crowdsourced Datasets." Interspeech. https://arxiv.org/abs/2104.09494
- Reddy, C. K. A. et al. 2021. "DNSMOS: A Non-Intrusive Perceptual Objective
  Speech Quality Metric to Evaluate Noise Suppressors."
  https://arxiv.org/abs/2010.15258

## 11. Writing Rules

### Keep

- short paragraphs,
- clear claim -> evidence -> implication structure,
- phase-wise terminology,
- exact metric definitions,
- explicit limitations,
- condition-level analysis,
- cautious causal language.

### Remove

- repeated lists of every possible metric,
- implementation details before Chapter 6,
- broad claims about all SDS,
- claims about model size when model family/provider also differs,
- unexplained acronyms,
- metric names without formulas or evidence sources.

### Suggested wording pattern

Use this pattern often:

```text
This metric measures [construct] using [logged evidence].
It is calculated as [formula].
High/low values indicate [interpretation].
It is unavailable when [missing-data rule].
Its limitation is [scope boundary].
```

## 12. Final Thesis Checklist

- Research questions match available data.
- Chapter 6 defines the exact experiment before results are shown.
- Every metric in the results has a formula and evidence source.
- Coverage and missing data are reported before success rates.
- Large2 exclusion or other model exclusions are declared before analysis.
- Text/audio pairing is analyzed only for matched pairs.
- Model-size claims are framed as model-backend effects unless controlled.
- Failure localization is diagnostic, not absolute causality.
- Appendix contains schemas, prompts, metric catalog, and reproducibility notes.
- References are checked and cited only where used.
