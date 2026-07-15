# Thesis Writing Aid: Automatic Evaluation of Spoken Dialogue Systems

Purpose: one compact reference for writing a research-grade bachelor thesis
around CoopNavigationSDS. It combines the current draft, the revised thesis
structures, and the implemented experiment design. It is not a replacement for
the thesis text. Use it as a checklist, argument map, citation guide, formula
sheet, and chapter scaffold.

## 0. Thesis Core

Use this chapter as the control sheet for the whole thesis. If a paragraph in
the thesis does not support one of these points, it probably belongs in an
appendix or should be removed.

### One-sentence thesis claim

This thesis studies whether a controlled cooperative navigation task can be
used to automatically evaluate spoken task-oriented dialogue systems
phase-wise, so that success, failure, constraint satisfaction, and likely
failure origin can be explained from logged evidence instead of only final
task outcome.

### What the thesis is about

- Scientific object: automatic evaluation of spoken task-oriented dialogue
  systems.
  - Keep the wording centered on evaluation, not on building a chatbot.
  - The software is the research instrument; the object of study is how SDS
    performance can be measured automatically.
- Experimental setting: cooperative route finding in a controlled transport
  network.
  - Use route finding because it has externally checkable correctness.
  - Emphasize that route validity, duration, constraints, and destination
    reachability can be computed without subjective judgment.
- Evaluated system role: Agent B, the route-information dialogue system.
  - Agent B is the model/backend being compared.
  - Agent B performance is evaluated through both task outcome and phase
    evidence.
- User role: Agent A/UserLM as the main simulated caller with private task
  goal and progressively revealed constraints. TinyLlama-Agent-A remains a
  software-control option, not the primary thesis denominator.
  - Explain that Agent A is controlled to make repeated experiments possible.
  - Do not claim Agent A fully represents human behavior.
- Main method: run many controlled dialogue conditions, capture raw phase
  evidence, calculate metrics retrospectively, and compare model backends.
  - Retrospective calculation matters because the raw logs remain available
    for recalculation and audit.
  - Avoid calculating only final scores during runtime.
- Main outcome: determine which metrics and phases explain successful,
  semi-successful, and unsuccessful navigation dialogues.
  - The thesis should explain why a run succeeds or fails, not only whether it
    succeeds.

### What the thesis is not about

- Not a general route-planning application.
  - Route planning is the controlled task environment, not the product goal.
- Not a claim that simulated users fully replace human users.
  - Simulated callers support reproducibility; human validation remains future
    work.
- Not a pure benchmark of model size alone.
  - Model family, prompt behavior, runtime feasibility, and coverage also
    matter.
- Not a universal evaluation of all spoken dialogue systems.
  - Claims are bounded to the implemented route-dialogue framework and tested
    configurations.
- Not a human-subject usability study.
  - No subjective satisfaction claims should be made without human ratings.

### Central validity principle

Raw phase evidence is authoritative. Metrics, tables, and plots are derived
from stored evidence after the run. A missing provider, missing audio, invalid
route, or failed model call is not silently replaced; it is recorded as an
experimental outcome or unavailable evidence with reason.

### Current active experiment scope

- Agent A: UserLM is the primary thesis caller. TinyLlama is retained as a
  caller-control stratum for direct comparison, not as a row to merge into the
  UserLM headline denominator.
- Agent B: five selected Transformer models by size class.
- Active coverage target: `small1`, `small2`, `medium1`, `medium2`, and
  `large1`; Large1 jobs may still change the final coverage until all cluster
  results are pushed.
- Speech channel: configured TTS and ASR; text-only controls may be used where
  paired comparison is available.
- Task objective: shortest valid route under progressively revealed
  constraints.
- Observation unit: one complete condition/run, not one turn.

Current comparison axes:

- Agent A comparison:
  - UserLM versus TinyLlama as caller implementations;
  - compare only inside matched Agent B/scenario/speech-condition subsets;
  - report separately from the main UserLM headline denominator.
- Agent B comparison:
  - five active Agent B backends:
    - `small1` TinyLlama 1.1B,
    - `small2` Qwen2.5 0.5B,
    - `medium1` Qwen2.5 1.5B,
    - `medium2` Phi-3 mini,
    - `large1` Qwen2.5 7B;
  - compare direct backend effects only on conditions covered by all active
    models.
- Text versus speech comparison:
  - use `pair_id` to join `text_only` and `audio_variant`;
  - keep all non-audio factors identical;
  - interpret deltas as speech-channel effects only within paired evidence.
- Success-status comparison:
  - first separate successful, semi-successful, unsuccessful, and execution
    incomplete outcomes;
  - identify strongest phase metrics for distinguishing status;
  - when status is equal, compare efficiency and quality factors such as turns,
    repair turns, duration regret, task focus, route revisions, candidate
    routes, ASR WER, ASR station F1, and latency.

### Current empirical snapshot

Use this only as the current result basis, not as final thesis wording unless
the result set is frozen.

- Snapshot source: pulled result folders through commit `ed5f8cdb` on
  2026-07-15.
- Finalized run folders found locally: 1819.
- Thesis denominator used here: UserLM as Agent A and five selected Agent B
  models. TinyLlama-Agent-A runs and archived models are excluded from the
  main denominator but remain useful for software control analysis.
- Calculation source: `conditions.jsonl` inside each run folder. Duplicate
  attempts are preserved as runtime evidence; coverage uses unique condition
  IDs and counts a condition as completed when any retained attempt completed.
- Archived large2/Mistral settings are removed from active setup and Slurm
  submission. Existing raw result folders are not altered.
- New result commits after the completed-dialogue metric workflow added
  TinyLlama-Agent-A control runs for Phi-3 mini and Qwen2.5 7B. They do not
  change the UserLM thesis denominator or the completed-dialogue metric
  correlation tables below.

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

TinyLlama-Agent-A software-control evidence currently available for the same
five selected Agent B models:

| Agent B model | Completed control runs | Successful | Semi-successful | Unsuccessful dialogue | Invalid/skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B | 70 | 65 | 0 | 5 | 14 |
| Qwen2.5 0.5B | 70 | 63 | 1 | 6 | 14 |
| Qwen2.5 1.5B | 70 | 65 | 0 | 5 | 14 |
| Phi-3 mini | 70 | 63 | 1 | 6 | 14 |
| Qwen2.5 7B | 53 | 48 | 0 | 5 | 12 |
| **Total** | **333** | **304** | **2** | **27** | **68** |

Use these rows as a software-control comparison only. They are useful for
checking whether observed behavior depends on Agent A implementation, but they
should not be merged into the main UserLM-Agent-A thesis denominator.

### Current result conclusions that are defensible

These statements are safe if the result set remains close to the current
snapshot:

- The framework can distinguish at least three outcome layers:
  - execution/provider completion,
  - route-valid dialogue completion,
  - task-satisfied dialogue completion under revealed constraints.
- Completed dialogues often produce valid routes. Across the selected UserLM
  subset, 294 of 317 completed unique conditions have valid routes, so many
  observed losses occur before or around dialogue completion rather than after
  route grounding alone.
- Qwen2.5 1.5B currently has the best combined evidence profile in the active
  subset: highest completed unique-condition count, highest task-success count,
  and high route validity among completed conditions.
- Larger model size alone is not supported as a simple predictor of better
  experiment outcome. Qwen2.5 7B has high route validity once completed but
  lower completion coverage in the current CPU cluster setting.
- Small models are not merely failure baselines. TinyLlama 1.1B and Qwen2.5
  0.5B achieve similar task-success rates among completed runs, which makes
  them useful controls for separating dialogue competence from runtime cost.
- Phase-wise metrics are necessary because task success alone hides whether a
  run failed through provider/runtime interruption, ASR/NLU corruption, state
  drift, route invalidity, constraint violation, or premature closure.

### Claims that are not supported without more evidence

Avoid these claims unless additional controlled evidence is added:

- "Large models are worse" or "small models are better." Current evidence
  confounds model size, model family, runtime cost, memory pressure, and
  cluster completion.
- "The earliest failing phase was always ASR/TTS." The framework can propose
  failure-localization candidates, but causal phase attribution requires
  inspection or additional validation.
- "Automatic evaluation replaces human evaluation." The thesis can argue that
  automatic evaluation is scalable and useful for diagnosis, while human
  evaluation remains important for perceived satisfaction and naturalness.
- "Speech-channel effects are fully explained." Speech metrics are diagnostic
  only when matched text/audio pairs and available audio evidence are present.
- "Task-derived metrics validate themselves." Metrics such as route validity,
  constraint satisfaction, and faithfulness are construct-valid for task
  outcome, but some are close to definitional and should not be presented as
  independent predictors without correlation/ablation discussion.

### Recommended final thesis claim after current results

Use wording close to this:

```text
The experiment shows that a controlled cooperative route-finding task can
produce analyzable phase-wise evidence for automatic SDS evaluation. In the
current UserLM-Agent-A subset, completed dialogues usually reached valid
routes, while many losses arose from incomplete executions, unresolved
dialogue progress, or constraint/task satisfaction failures. Therefore, final
task success is useful but insufficient: separating execution completion,
speech/language evidence, dialogue state, route grounding, and constraint
satisfaction provides a clearer basis for diagnosing successful and failed
spoken dialogue runs.
```

### Chapter-level argument map

Use this map while drafting to keep the thesis coherent.

| Chapter | Main claim | Evidence type | Common mistake to avoid |
| --- | --- | --- | --- |
| 1 Introduction | SDS evaluation needs more than final outcome because failures are multi-phase. | Motivation, problem statement, research questions. | Explaining repository internals too early. |
| 2 Background | Spoken task-oriented dialogue can be described through phases, state, grounding, and repair. | SDS/TOD concepts and pipeline definitions. | Treating the pipeline as the only possible architecture. |
| 3 Evaluation | Automatic metrics are useful only when tied to constructs and logged evidence. | Metric theory, construct validity, human vs automatic evaluation. | Listing metrics without formula or interpretation. |
| 4 Related work | Existing work covers important pieces but rarely combines speech-channel evidence, LLM backends, staged constraints, and route validation. | Grouped literature strands. | Claiming prior work does not exist. |
| 5 Validity threats | The experiment is controlled and useful but not a full substitute for human evaluation or real transit interaction. | Construct/internal/external/statistical validity discussion. | Hiding limitations until after results. |
| 6 Methodology | CoopNavigationSDS operationalizes phase-wise SDS evaluation in a controlled route task. | Configuration, network, agents, logging, metrics, batch design. | Mixing methodology with result interpretation. |
| 7 Results | Outcomes must be interpreted through coverage, execution completion, phase evidence, and task metrics. | Tables, charts, condition rows, phase metrics. | Ranking models before reporting missing evidence. |
| 8 Discussion | Phase-wise evidence explains why final task outcomes differ and which metrics are diagnostically useful. | RQ-by-RQ interpretation. | Turning associations into causal claims. |
| 9 Conclusion | The framework is a valid bachelor-level research artifact for automatic SDS evaluation, with clear scope limits. | Summary of method, findings, limitations, future work. | Overclaiming generality beyond the controlled task. |

## 1. Introduction

### Chapter function

Introduce the problem, motivate the research, define the scope, state research
questions, and summarize contributions. Keep implementation detail out of this
chapter.

Suggested chapter flow:

- Paragraph 1: why spoken dialogue systems matter.
  - Mention accessibility, hands-free interaction, and practical deployment.
  - Keep it general; do not start with CoopNavigationSDS.
- Paragraph 2: why evaluation is difficult.
  - Explain multi-phase failure: audio, ASR, NLU, state, policy, NLG, TTS,
    task validation.
  - State that final success alone hides where errors arise.
- Paragraph 3: why a controlled navigation task is useful.
  - Route tasks have objective correctness.
  - Constraints make partial success visible.
  - The controlled network makes repeated experiments possible.
- Paragraph 4: what the thesis investigates.
  - State Agent A/UserLM and Agent B roles.
  - State phase-wise logging and retrospective metrics.
- Paragraph 5: contributions.
  - Keep contributions at research level:
    - framework,
    - task design,
    - metric pipeline,
    - matched model comparison,
    - result analysis.

### 1.1 Motivation and background

- Spoken dialogue systems support interaction through speech.
  - Use examples only briefly:
    - hotline systems,
    - navigation assistants,
    - customer support,
    - accessibility tools.
  - Then return to the evaluation problem.
- Speech adds practical value:
  - hands-free interaction,
  - accessibility,
  - natural interaction in mobile or assistive contexts,
  - prosodic information such as pauses, hesitation, emphasis, and rhythm.
  - These benefits also introduce evaluation challenges because speech output
    and speech input are variable over time.
- Spoken systems are harder to evaluate than text systems because:
  - the acoustic channel can distort information,
  - turns unfold over time,
  - errors can propagate across phases,
  - final success may hide earlier degradation.
  - Use one short example:
    - station name misheard,
    - wrong destination stored,
    - valid-looking but wrong route proposed.
- Task-oriented spoken dialogue is suitable for automatic evaluation because
  goals and constraints can be checked against an external task environment.
  - This justifies using navigation instead of open-ended conversation.

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
  - The same final failure can therefore have different causes.
  - The thesis should argue that automatic evaluation must retain enough phase
    evidence to distinguish these causes.
- Human evaluation is important but costly, slow, and subjective.
  - It captures perceived quality and satisfaction.
  - It is difficult to scale across hundreds of configurations.
- Automatic evaluation is scalable and repeatable but only useful when each
  metric has a clear construct and evidence source.
  - Metrics must be tied to logged data.
  - Missing data must not be treated as zero.
- Final task success alone answers "what happened"; phase evidence helps
  answer "where did it go wrong".
  - This sentence can be reused in the introduction and discussion.

### 1.3 Problem statement

Use this wording as the thesis backbone:

- Existing dialogue evaluation often focuses on final task success, human
  ratings, response similarity, or isolated component metrics.
  - Each view is useful but incomplete:
    - final success is too coarse,
    - human ratings are costly,
    - response similarity can ignore task correctness,
    - component metrics can miss downstream impact.
- These views are useful but insufficient for diagnosing spoken
  task-oriented failures.
  - Diagnosis requires chronological evidence from multiple phases.
- In a route dialogue, failure may be caused by recognition, understanding,
  memory, policy, grounding, generation, or speech output.
  - This motivates the phase-based evaluation framework.
- This thesis investigates whether phase-wise automatic evidence can explain
  success and failure in controlled spoken navigation dialogues.
  - Use "explain" carefully:
    - the framework provides diagnostic evidence,
    - it does not prove absolute causality without additional validation.

### 1.4 Research questions

Recommended final set:

1. Can success and failure in a controlled spoken navigation dialogue be
   explained using automatically logged phase-wise evidence and metrics?
   - This is the main feasibility question.
   - It is answered by showing the pipeline, logs, metric tables, and outcome
     bands.
2. Which automatic metrics best correlate with task-level outcomes such as
   task completion, route validity, and constraint satisfaction?
   - This is answered by metric means, correlations, and indicator patterns.
   - Separate outcome-confirming metrics from diagnostic metrics.
3. Can phase-wise evidence identify the earliest likely failing phase in
   unsuccessful or degraded dialogues?
   - Use "likely" because failure localization is diagnostic, not causal proof.
4. Which phase-level metrics remain useful across different Agent B language
   model backends and size classes?
   - This connects model comparison to metric robustness.
   - Use matched cases when discussing direct backend differences.
5. If paired text/audio runs are available: how does the speech channel affect
   task success and phase-level errors compared with matched text controls?
   - Only answer this where paired evidence exists.
   - Otherwise frame it as future work or exploratory.

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
  - The contribution is conceptual and methodological.
- Controlled cooperative navigation task with externally checkable route
  validity.
  - The task makes route validity, duration, transfers, and constraints
    computable.
- Implemented CoopNavigationSDS framework with configurable agents, speech,
  scenarios, model backends, logging, and retrospective metrics.
  - Mention implementation only at this level in the introduction.
- Metric-outcome analysis across Agent B backends.
  - Include completed-only and matched-condition analysis.
- Failure-localization method based on chronological phase evidence.
  - Present as a diagnostic method, not as definitive causal attribution.

## 2. Theoretical Background

### Chapter function

Explain spoken dialogue systems and their phases. Do not describe the concrete
experiment yet except as brief motivation.

Suggested chapter flow:

- Define dialogue systems and SDS.
- Define task-oriented dialogue.
- Explain why spoken dialogue differs from text dialogue.
- Introduce the pipeline as an analysis model.
- Explain LLM-based systems and why phase evidence is still needed.
- Explain error propagation.
- End by linking theory to the need for phase-wise automatic evaluation.

### 2.1 Dialogue systems and spoken dialogue systems

- Dialogue system: interactive language system that receives input, maintains
  context, selects an action, and responds.
  - Keep the definition broad enough to include both modular systems and LLM
    agents.
- Spoken dialogue system: dialogue system with speech input and/or speech
  output.
  - In this thesis the complete pipeline includes both input and output speech.
- Spoken SDS adds:
  - acoustic uncertainty,
  - timing and turn-taking,
  - speech recognition,
  - speech synthesis,
  - intelligibility requirements.
  - These additions create extra measurable phases and extra failure points.

### 2.2 Task-oriented dialogue

- Task-oriented systems help users complete defined goals.
  - The goal must be externally checkable for automatic evaluation.
- They contain:
  - user goal,
  - required slots,
  - optional constraints,
  - dialogue state,
  - system actions,
  - backend/task environment,
  - success criterion.
  - These concepts map directly onto the navigation task:
    - slots become start, destination, time, and constraints;
    - system actions become route proposals, clarifications, and repairs;
    - backend state becomes the transport network and route validator.
- In this thesis, task state includes:
  - start station,
  - destination station,
  - departure time,
  - active constraints,
  - candidate routes,
  - rejected routes,
  - accepted route.
  - Explain that state is dynamic:
    - Agent A reveals constraints progressively,
    - Agent B updates proposals,
    - the route validator checks current-stage validity.

### 2.3 Spoken versus text dialogue

- Text dialogue assumes symbolic input is already available.
  - Text-only runs can isolate dialogue/model behavior from speech-channel
    effects.
- Spoken dialogue must recover text/meaning from audio.
  - Speech errors can change entities while leaving the sentence fluent.
- Spoken-specific errors:
  - station-name substitution,
  - line-name substitution,
  - number/time confusion,
  - negation loss,
  - clipping,
  - silence,
  - ASR hallucination,
  - TTS mispronunciation.
  - In navigation, entity errors are especially dangerous because one station
    or line substitution can invalidate the whole route.

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
  - This creates the bridge from theory to the experiment design in Chapter 6.
  - Use the same phase names consistently in methods, results, and discussion.

### 2.5 LLM-based dialogue systems

- LLMs can absorb NLU, state tracking, policy, and NLG inside one model call.
  - This makes internal phase boundaries less visible.
  - External logs and validators become more important.
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
  - These risks are relevant to both task quality and runtime feasibility.
- Therefore, phase-aware external evidence is still needed even when internal
  module boundaries are blurred.
  - The thesis can evaluate the observable behavior of an LLM backend even if
    its internal reasoning is opaque.

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
  - A phase metric is most useful when connected to downstream consequences.
  - Example: ASR WER is less meaningful than whether ASR errors destroy route
    entities or constraints.

## 3. Evaluation Concepts

### Chapter function

Introduce evaluation concepts before the concrete metric catalog.

Suggested chapter flow:

- Distinguish human and automatic evaluation.
- Explain construct validity.
- Explain why generic text overlap is insufficient.
- Introduce speech-aware and semantic metrics.
- Classify metrics by role:
  - outcome-confirming,
  - diagnostic,
  - efficiency/cost,
  - matched comparison.

### 3.1 Human versus automatic evaluation

- Human evaluation captures perceived satisfaction, naturalness, usefulness,
  and trust.
  - It is necessary for claims about user experience.
  - It is not available in the current experiment.
- Automatic evaluation provides:
  - scalability,
  - repeatability,
  - parallelization,
  - precise traceability.
  - It allows systematic comparison across many configurations.
  - It makes retrospective analysis possible.
- Automatic metrics are not automatically valid; each metric must be tied to a
  construct.
  - This is the key theoretical caution for the whole thesis.

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

### 3.5 Metric roles used in this thesis

Separate metrics by what they are allowed to claim.

Outcome-confirming metrics:

- confirm whether the navigation task succeeded;
- include route validity, destination reached, constraint satisfaction,
  accepted route, and duration within threshold;
- are construct-valid for task success but partly overlap with the success
  definition;
- should not be presented as independent predictors of success.

Diagnostic phase metrics:

- indicate where successful, semi-successful, and unsuccessful completed
  dialogues diverge;
- include ASR station F1, ASR WER, NLU route-valid rate, NLU goal-reached
  rate, grounded proposal score, executable utterance rate, candidate-route
  count, station mentions, repair success, and abandonment rate;
- support failure localization, but correlations remain descriptive.

Efficiency and cost metrics:

- describe how expensive the dialogue was;
- include turn count, mean turn time, latency, word count, route-revision
  count, and runtime;
- are useful when two models both solve the task but differ in interaction
  cost.

Matched-comparison metrics:

- compare Agent B models only on cases where every non-model condition is
  equivalent;
- use a model-independent key over Agent A, scenario, persona, audio persona,
  speech pattern, TTS/ASR settings, ASR search width, seeds, objective mode,
  transfer tolerance, stagnation limit, run type, platform, and decoding
  profile;
- exclude Agent B model name, model size, model slot, and model-specific
  condition ID;
- answer a different question from coverage metrics: matched comparison asks
  "given the same condition, how did models differ?", while coverage asks
  "which conditions produced evidence at all?"

## 4. Related Work

### Chapter function

Organize papers by research strand. Do not list papers without explaining how
they relate to the thesis.

Suggested chapter flow:

1. Classical SDS evaluation.
   - Establish that SDS quality combines task outcome, interaction cost, and
     user-facing quality.
   - Use this strand to justify why the thesis does not rely on final route
     success alone.
2. Task-oriented dialogue benchmarks.
   - Introduce goals, slots, dialogue state, and constraint tracking.
   - Explain why these concepts are necessary for route dialogue evaluation.
3. Spoken dialogue and spoken language understanding.
   - Motivate transcript errors, entity preservation, and speech-aware metrics.
   - Connect ASR/TTS evidence to downstream dialogue behavior.
4. User simulation and agent-based evaluation.
   - Explain repeatability and controllability.
   - State the external-validity cost of replacing human callers with Agent A.
5. Grounded navigation dialogue.
   - Show why route tasks are objectively checkable.
   - Explain why station, line, and constraint grounding are central.
6. LLM-agent evaluation.
   - Position Agent B as a dialogue-system backend, not merely a text
     generator.
   - Motivate matched-condition comparison across model backends.

### 4.1 Classical SDS evaluation

- PARADISE connects task success, dialogue cost, and user satisfaction.
- Use it to justify combining outcome and cost metrics.
- What to write:
  - task success alone can hide excessive repair, repetition, latency, or poor
    interaction quality;
  - fluent responses can still fail the task if they are ungrounded;
  - a useful SDS evaluation therefore needs both outcome and process evidence.
- How it maps to this project:
  - task success is represented by valid route, destination reach, constraint
    satisfaction, and Agent A acceptance;
  - dialogue cost is represented by turn count, repair count, latency,
    repetition, and duration;
  - satisfaction is not measured with human ratings, so Agent A acceptance is
    only a task-internal proxy.
- Difference to this thesis:
  - this thesis focuses more on phase-wise retrospective evidence and model
    backend comparison.
- Boundary:
  - do not claim that this thesis reproduces a full PARADISE regression model.

### 4.2 Task-oriented dialogue benchmarks

- MultiWOZ and SGD motivate task-oriented dialogue, slot/state tracking, and
  multi-domain evaluation.
- What to cite these for:
  - structured user goals;
  - slot and dialogue-state tracking;
  - automatic evaluation of task-oriented systems.
- How this thesis differs:
  - the task domain is deliberately narrow;
  - routes can be validated against a known transport network;
  - the spoken channel is preserved rather than reduced to clean text.
- Difference to this thesis:
  - many benchmarks are text-based and do not capture the full speech channel.
- Bridge sentence:
  - "Text-based TOD benchmarks motivate state and constraint evaluation, but
    they do not identify whether a failure originated in TTS, ASR, NLU,
    dialogue management, or grounding."

### 4.3 Spoken TOD and SLU

- SpokenWOZ and SLUE motivate spoken input, speech-text mismatch, and
  speech-aware evaluation.
- What to emphasize:
  - speech introduces substitutions, insertions, deletions, pauses, and entity
    distortions;
  - station, line, and time errors can be more harmful than function-word
    errors;
  - the same intended utterance can lead to different outcomes under different
    TTS/ASR conditions.
- How this project uses the idea:
  - every turn stores intended text, TTS text, ASR raw transcript, normalized
    understanding, corrections, and phase timings;
  - these fields allow retrospective phase-wise metric calculation.
- Difference to this thesis:
  - CoopNavigationSDS uses a controlled route task with route validation and
    staged constraints.
- Boundary:
  - the experiment does not claim to cover all real microphone, accent, or
    noise conditions.

### 4.4 User simulation

- User simulation allows repeatable evaluation.
- Limitation:
  - simulated behavior is not real human behavior.
- In this thesis:
  - Agent A is controlled and has private knowledge boundaries.
- What to explain:
  - Agent A is a simulated caller, not a generic chatbot;
  - it knows start, destination, departure time, station names, line names,
    persona priorities, and private constraints;
  - it reveals constraints only after earlier dialogue goals are satisfied.
- Why this matters:
  - repeatable simulated callers make batch experiments possible;
  - private constraints allow staged evaluation of route refinement;
  - controlled behavior improves internal validity while limiting ecological
    validity.

### 4.5 Navigation and grounded dialogue

- Navigation dialogue is useful because it requires grounding, clarification,
  route communication, and shared task progress.
- CoopNavigationSDS uses transport-network grounding rather than open-ended
  map navigation.
- Why the task is suitable:
  - station and line names are discrete, checkable entities;
  - route validity can be evaluated automatically;
  - constraints such as duration, transfers, fullness, delay, or walking
    distance produce meaningful trade-offs;
  - misunderstandings have visible task consequences.
- How to frame the synthetic network:
  - it is an experimental instrument, not a complete public-transport model;
  - it is designed to create valid, suboptimal, and constraint-changing
    alternatives.

### 4.6 LLM-agent evaluation

- Modern agent benchmarks emphasize tool use and final-state verification.
- This thesis adds spoken pipeline evidence and phase-wise metrics.
- What to write:
  - Agent B backends can differ in fluency, grounding, repair behavior,
    memory, latency, and constraint following;
  - final answer correctness alone does not reveal where a dialogue failed;
  - matched-condition comparison is required because scenario, persona, audio
    persona, TTS, ASR, and seed all affect difficulty.
- Model-size caution:
  - model size is not isolated from model family, tuning, tokenizer, provider,
    quantization, or runtime environment.
- Strong thesis positioning:
  - the thesis evaluates Agent B as part of a complete SDS pipeline, not as an
    isolated LLM answering clean text prompts.

## 5. Limitations and Validity Threats

### Chapter function

Prepare careful interpretation. Do not wait until the discussion to define
validity limits.

Use this chapter to pre-register how strong the later claims may be. The
reader should know before the results chapter that the thesis is strongest as a
controlled automatic-evaluation study and weaker as a claim about real public
transport users.

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

### Construct validity

- Question:
  - do the metrics measure the constructs they claim to measure?
- Threats:
  - route validity and constraint satisfaction partly overlap with the task
    success definition;
  - transcript similarity does not guarantee semantic understanding;
  - audio quality metrics may not predict navigation success directly.
- Mitigation:
  - separate outcome metrics from diagnostic phase metrics;
  - explain which metrics are partly definitional;
  - use ASR/TTS metrics to localize possible failure points, not as standalone
    proof of task quality.

### Internal validity

- Question:
  - can differences be attributed to the manipulated configuration?
- Threats:
  - model comparisons confound size, family, backend, prompt behavior, and
    runtime environment;
  - execution failures can originate in assets, Slurm, provider setup, or
    timeouts rather than dialogue ability;
  - duplicate attempts can bias counts if treated as separate conditions.
- Mitigation:
  - compare models on matched non-model conditions;
  - separate execution failure from completed-dialogue outcome;
  - deduplicate by model and comparable condition before success analysis.

### External validity

- Question:
  - how far do findings generalize?
- Threats:
  - the network is synthetic;
  - Agent A is simulated;
  - speech conditions are generated rather than recorded from real users.
- Mitigation:
  - frame claims as evidence for controlled SDS evaluation;
  - avoid deployment claims;
  - propose real-user and real-audio validation as future work.

### Statistical conclusion validity

- Question:
  - are quantitative comparisons supported by enough comparable evidence?
- Threats:
  - uneven coverage by model;
  - missing phase evidence;
  - nested turn-level data;
  - repeated attempts for the same condition.
- Mitigation:
  - report denominators before rates;
  - use condition-level summaries for main claims;
  - use turn-level and phase-level observations as diagnostic evidence;
  - report matched-case counts separately from raw completed counts.

### Reliability and reproducibility

- Question:
  - can another researcher reproduce the run and metric calculation?
- Threats:
  - local model assets and optional providers differ by platform;
  - cluster nodes can interrupt or time out jobs;
  - large generated aggregates may not be tracked in Git.
- Mitigation:
  - store git commit, seed, job file, backend metadata, and condition
    configuration;
  - retain raw run evidence before derived metrics;
  - document excluded generated aggregates and regenerate them when needed.

## 6. Methodology and Research Design

Starting here, focus on the actual experiment and metric selection.

Chapter writing rule:

- explain the experiment before reporting any result;
- define each object once and reuse the same term consistently;
- show which logged values are needed for later metrics;
- make clear which variables are manipulated and which are held constant.

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
- Suggested thesis wording:
  - "A run is the smallest independent experimental unit."
  - "Turns and phases are observations inside a run."
  - "Model comparisons therefore use condition-level aggregation, while
    phase-wise metrics explain the run outcome."
- Why this matters:
  - prevents inflated sample sizes;
  - keeps repeated turns from being counted like independent participants;
  - makes seeds, repetitions, and reruns interpretable.

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
- Agent A memory contains:
  - own start, destination, departure time;
  - revealed and unrevealed constraints;
  - what Agent B appeared to propose;
  - whether the current proposal satisfies the current stage.
- Agent B memory contains:
  - what Agent B heard through ASR/NLU;
  - current understood trip facts;
  - route proposals already offered;
  - clarification and repair history.
- Thesis emphasis:
  - Agent B is evaluated on what it receives, not on hidden ground truth;
  - corrections are allowed only when logged transparently;
  - misunderstandings remain part of the evidence.

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
- Why stages are used:
  - they model a caller who first asks for a route and later adds preferences;
  - they prevent judging Agent B against constraints Agent A has not revealed;
  - they make dialogue-management progress measurable.
- What to report per stage:
  - current objective;
  - first turn in which the stage was satisfied;
  - route candidate active at that point;
  - current optimal route layer;
  - clarification or repair events before success/failure.
- Suggested wording:
  - "The dialogue progresses only when the current stage objective is
    satisfied. This makes stage failure observable rather than inferred only
    from the final outcome."

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
- Network design rationale:
  - station and line identifiers make speech entity preservation measurable;
  - multiple feasible routes make optimization and constraint negotiation
    meaningful;
  - constraint-changing alternatives force Agent B to revise rather than repeat
    a previously valid route;
  - a controlled network avoids relying on changing external timetable data.
- Methodology content:
  - include a compact network overview;
  - define the route grammar;
  - explain transfers, duration, delay/fullness classes, and walking
    constraints;
  - state how invalid or unreachable generated scenarios are excluded.
- Appendix content:
  - full network data;
  - route-validation schema;
  - example valid and invalid proposals.

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
- Methodological rule:
  - evaluate each stage only against constraints known at that stage;
  - after a new constraint is revealed, use the corresponding optimum layer;
  - do not penalize an early route for failing a later private constraint.
- Why this strengthens validity:
  - separates basic route validity from preference satisfaction;
  - makes semi-success meaningful;
  - reveals whether Agent B can improve a route after new user information.
- Reporting rule:
  - show the optimal route for each layer;
  - report duration regret relative to the active layer;
  - mark when an added constraint changes the optimal route.

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
- TTS contribution:
  - transforms intended text into a speech signal;
  - can affect pronunciation, pacing, pauses, clipping, and intelligibility;
  - is evaluated through synthesis success, latency, audio-quality proxies, and
    entity preservation when available.
- ASR contribution:
  - transforms speech back into text;
  - can distort stations, lines, departure times, constraints, and repair
    turns;
  - is evaluated through WER, entity error rate, critical slot preservation,
    and semantic ASR error.
- Normalization contribution:
  - maps recognized variants to known station/line names when justified;
  - must not silently inject ground truth;
  - every correction should be visible in transcript and metric evidence.
- Why this is central:
  - the experiment is a speech dialogue system evaluation, so downstream
    behavior must be based on what was heard and understood.

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

Matched Agent B comparison:

- Do not use raw `condition_id` alone for cross-model comparison because the
  condition ID contains model-specific fragments.
- Build a model-independent comparable-condition key from:
  - Agent A type,
  - scenario and test case,
  - persona,
  - Agent A and Agent B audio personas,
  - speech pattern and speech performance band,
  - run mode and run type,
  - TTS and ASR implementation,
  - ASR search width,
  - network and experiment seed,
  - objective mode,
  - transfer tolerance,
  - dialogue stagnation limit,
  - decoding/profile key,
  - repetition/iteration,
  - matrix family and execution platform.
- Exclude from the key:
  - Agent B model name,
  - Agent B model size,
  - Agent B slot label,
  - model-specific condition ID.
- A completed case is directly comparable across Agent B models when this key
  matches and the run completed for the models being compared.
- Report two matched counts:
  - completed in all selected models,
  - completed in at least one other selected model.
- If duplicate completed attempts exist for the same model and comparable key,
  count the case once. For outcome summary, use the best retained completed
  outcome in the order successful > semi-successful > unsuccessful completed.

Current observed UserLM thesis subset for the pulled result set:

```text
selected_userlm_condition_ids = 770
completed_selected_userlm_conditions = 317
task_successful_selected_userlm_conditions = 281
route_valid_selected_userlm_conditions = 294
completion_coverage = 317 / 770 = 41.17%
task_success_among_completed = 281 / 317 = 88.64%
route_validity_among_completed = 294 / 317 = 92.74%
```

For the current thesis denominator, UserLM-Agent-A comparisons are analyzed
separately from TinyLlama-Agent-A software controls. Do not mix both callers in
one unstratified success rate.
- Batch-design paragraph to include:
  - "Batch execution generates controlled conditions by varying Agent B,
    scenario, persona, audio persona, speech condition, and seed while keeping
    structured run metadata."
- What to emphasize:
  - equal planned coverage is the goal;
  - completed coverage can differ due to runtime failures or provider
    availability;
  - execution coverage must be reported before model performance.
- What to avoid:
  - ranking models by raw run counts;
  - treating failed setup as dialogue failure;
  - mixing UserLM-Agent-A thesis runs with TinyLlama-Agent-A control runs.

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
- Evidence hierarchy:
  - raw evidence:
    - transcript events,
    - ASR raw text,
    - TTS metadata,
    - route proposals,
    - validation decisions,
    - timings,
    - errors.
  - derived evidence:
    - metric rows,
    - phase summaries,
    - success labels,
    - comparison tables.
  - presentation evidence:
    - HTML scorecards,
    - coverage tables,
    - thesis figures.
- Integrity rule:
  - raw run folders should not be edited to improve results;
  - derived analysis may be regenerated from raw evidence;
  - if derived analysis changes, the generating script and commit should be
    traceable.

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

Chapter writing rule:

- introduce metrics by phase, not alphabetically;
- give each metric a construct, formula, input fields, range, and limitation;
- separate outcome metrics from diagnostic metrics;
- explicitly state when a metric is unavailable instead of converting missing
  values to zero.

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

In this thesis:

- slot F1 is appropriate for start station, destination, departure time,
  station names, line names, and constraints;
- critical slot accuracy is stricter and should focus on variables required
  for route solving;
- entity-level scores are often more interpretable than full transcript
  overlap for this task.

Task success:

```text
task_success = route_valid
               and destination_reached
               and active_constraints_satisfied
               and accepted_by_agent_a
```

Interpretation:

- main final outcome for completed dialogues;
- report only after coverage and completed-dialogue counts;
- do not mix with setup, provider, or Slurm failures.

Constraint satisfaction rate:

```text
constraint_satisfaction_rate =
    satisfied_active_constraints / active_constraints
```

Interpretation:

- distinguishes successful from semi-successful route dialogues;
- uses active/revealed constraints only;
- should be interpreted together with route validity.

Route optimality ratio:

```text
optimality_ratio = optimal_duration / selected_route_duration
```

- 1.0 means equal to optimum.
- Below 1.0 means slower than the optimum.

Interpretation:

- near 1.0 means close to the active optimum;
- use duration regret alongside it because ratios can hide small absolute
  differences.

Duration regret:

```text
duration_regret = selected_route_duration - optimal_duration
```

Interpretation:

- 0 minutes means the selected route matches the active optimum;
- positive values quantify lost travel time.

Repair success rate:

```text
repair_success_rate =
    successful_repairs / repair_attempts
```

Interpretation:

- high values indicate productive clarification;
- low values indicate loops, unresolved ambiguity, or poorly targeted repair.

Failure-localization candidate:

```text
earliest_failing_phase =
    first chronological phase where critical evidence is missing,
    corrupted, contradicted, or invalid before downstream failure.
```

Interpretation:

- diagnostic candidate, not causal proof;
- useful for grouping failures by likely origin.

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

For each phase, write the same pattern:

- what enters the phase;
- what the phase produces;
- which logged fields prove that the phase ran;
- which metric values indicate degradation;
- how degradation can propagate to the next phase.

Example:

```text
ASR receives a TTS waveform and produces a raw transcript. Its main
task-relevant risk is entity corruption. Therefore, WER is reported as a
general transcript metric, while station, line, and time preservation are
reported as critical entity metrics.
```

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

Suggested presentation:

- one compact table mapping each research question to primary metrics;
- one paragraph explaining which metrics are diagnostic rather than central;
- avoid long undifferentiated metric lists in the main text.

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

Safe current-evidence wording:

- "Task-derived metrics show the clearest relationship with outcome."
- "Speech metrics are most useful for explaining why a completed dialogue
  failed or required repair."
- "Metrics that directly reuse the success definition are decomposition tools,
  not independent predictors."

Avoid:

- claiming that high audio quality guarantees task success;
- ranking models without matched-condition coverage;
- inferring human satisfaction from Agent A acceptance alone.

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

In tables:

- show availability counts next to means;
- do not average null values as zero;
- distinguish not applicable, provider failed, and missing evidence.

In prose:

- state when a metric is excluded from comparison because required fields are
  unavailable.

### 7.6 Completed-dialogue outcome classification

Classify navigation outcome only after the dialogue runtime completed and
retrospective metrics were calculated. Provider failures, invalid test
conditions, missing model assets, Slurm interruption, and preflight failures
are execution outcomes. They belong in coverage and reliability tables, not in
the successful/semi-successful/unsuccessful navigation table.

Recommended completed-dialogue classification:

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
  - turn limit reached without usable route.
- execution failed or invalid:
  - provider failure prevents dialogue,
  - model/backend setup fails,
  - Slurm or process interruption prevents a terminal dialogue,
  - scenario is invalid because required staged alternatives are missing.
  - report separately from navigation outcome metrics.

Formula:

```text
completed_dialogue =
    execution_status == "completed"
    and metrics_wide.csv is present

successful =
    completed_dialogue
    and task_success == true

semi_successful =
    completed_dialogue
    and task_success == false
    and (route_valid == true or route_reaches_goal == true)

unsuccessful_dialogue =
    completed_dialogue
    and task_success == false
    and route_valid == false
    and route_reaches_goal == false
```

This distinction is methodologically important. It prevents a missing model,
bad cluster state, or invalid generated scenario from being misread as poor
conversational performance by Agent B.

Recommended result labels:

- execution-complete successful;
- execution-complete semi-successful;
- execution-complete unsuccessful;
- execution-incomplete/provider failed;
- invalid or excluded condition.

Thesis rule:

- only the first three labels belong in navigation-success analysis;
- all five labels belong in execution reliability and coverage analysis.

## 8. Results Chapter Guide

### 8.1 Report order

1. Run inventory and coverage.
2. Exclusions and unavailable evidence.
3. Execution completion before task success.
4. Overall task outcomes.
5. Success/semi-success/unsuccess distribution.
6. Phase metric distributions.
7. Metric-outcome correlations.
8. Model-backend comparison.
9. Failure localization.
10. Representative success and failure cases.

Do not start the results chapter with model rankings. Start with evidence
availability. A model cannot be fairly compared on task success until the
reader knows how many conditions produced complete evidence.

Minimum result table sequence:

1. **Condition inventory:** observed/planned condition IDs, completed unique
   conditions, failed attempts, unavailable evidence.
2. **Execution evidence:** completed versus failed attempts by model, runtime
   categories, provider/runtime errors.
3. **Task outcome:** route validity, task success, constraint satisfaction,
   turn count.
4. **Phase evidence:** per-phase metric averages and missingness.
5. **Diagnostic evidence:** earliest failing phase candidate, error
   signatures, repair-loop or stagnation indicators.

Writing structure for each result subsection:

- state the denominator;
- state the main number;
- explain what the number means;
- state the most important limitation;
- connect the result to one research question.

Example:

- "Among completed UserLM-Agent-A runs, 88.64% were successful. This rate
  describes completed dialogue behavior, not total planned-condition coverage."

### 8.2 Tables to include

- Condition coverage by Agent B model.
- Outcome distribution by Agent B model.
- Mean/median duration regret and turn count.
- Top metrics associated with task success.
- Failure phase distribution.
- ASR/TTS degradation versus outcome.
- Model runtime and memory summary.

For each table:

- include N;
- include missing count where applicable;
- sort models consistently by size then model key;
- mark whether rows are all completed runs or matched conditions.

Avoid:

- mixing execution failures, completed failures, and task success without
  labels;
- model rankings without comparable-condition counts;
- tables whose denominator changes without being stated.

### 8.3 Figures to include

- Pipeline diagram with evidence boundaries.
- Condition grid overview.
- Success distribution by model.
- Metric heatmap by phase.
- Failure-localization bar chart.
- Paired text/audio delta plot, if available.

Figure guidance:

- use coverage figures before success figures;
- keep color semantics stable:
  - green for better or successful values;
  - red for worse or failed values;
  - neutral/light colors for mid-range or non-outlier values;
- label whether values are normalized within a metric or absolute.

Useful figure captions:

- state the unit of analysis;
- state whether only completed dialogues are included;
- state whether duplicate attempts were deduplicated.

### 8.4 Interpretation caution

Write:

- "associated with", "indicates", "diagnostic evidence suggests".
- "in the completed-run subset".
- "under the tested speech and batch conditions".
- "for matched conditions where available".

Avoid:

- "proves that phase X caused failure" unless manually validated.
- "model size caused the improvement" unless family/backend are controlled.
- "the system works in real use" unless real-user evidence is added.
- "ASR failed" when only downstream task failure is observed.

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
  - TinyLlama-Agent-A control runs are outside the main thesis denominator;
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

### 8.6 Current model-by-model reading

Use this as a concise basis for Chapter 8. Recheck the latest result tables
before final submission.

| Model | Current reading |
| --- | --- |
| TinyLlama 1.1B | Small baseline with task-success rate close to Qwen2.5 0.5B among completed runs. Useful as a low-resource control. |
| Qwen2.5 0.5B | Similar outcome profile to TinyLlama in the current subset; shows that very small models can complete many grounded route dialogues. |
| Qwen2.5 1.5B | Strongest current evidence profile: most completed unique conditions and highest task-success count. Good candidate for "best practical backend in this batch." |
| Phi-3 mini | Reasonable completed-run success, but lower completion coverage than Qwen2.5 1.5B. Interpret as mixed evidence rather than model failure. |
| Qwen2.5 7B | High route validity among completed runs, but fewer completed conditions. Interpret as high-capacity but resource-sensitive under current execution conditions. |

### 8.7 Recommended result claims by strength

Strong claims:

- The framework records enough evidence to separate runtime completion,
  dialogue completion, route validity, and task success.
- In completed UserLM-Agent-A conditions, route validity is generally high.
- The current result set supports model-backend comparison only after
  stratifying by Agent A and excluding archived settings.

Moderate claims:

- Qwen2.5 1.5B is currently the strongest practical Agent B backend in the
  selected UserLM subset.
- Phase-aware evidence is more useful than final task success alone for
  explaining failures.
- Small Agent B models can be viable in a constrained grounded navigation
  task.

Weak or exploratory claims:

- Specific metric thresholds predict future failures.
- Larger models are systematically better or worse.
- Speech-channel effects generalize beyond the configured TTS/ASR conditions.

### 8.8 Outcome-indicating metrics in the current result set

Use this subsection when writing the result interpretation. It is based on
fully completed, metric-available UserLM-Agent-A runs for the five active Agent
B models. Execution-failed and invalid-condition rows are excluded from this
metric comparison. They remain important execution evidence, but they are not
dialogue-outcome evidence.

This is not the same denominator as unique-condition coverage: coverage answers
"which planned conditions are represented?", while this scan answers "which
calculated metrics most often separate completed dialogue outcome classes?"

Current completed-dialogue attempt classes:

| Outcome class | Count | Meaning for interpretation |
| --- | ---: | --- |
| successful | 409 | completed dialogue, valid route, accepted task outcome, revealed constraints satisfied |
| semi-successful | 19 | route evidence exists, but final task/constraint/optimality acceptance failed |
| unsuccessful dialogue | 24 | completed runtime trace, but no usable route outcome |

Excluded rows are provider/runtime failures or invalid test conditions. They
should be reported in an execution-completion table, not in the dialogue metric
comparison. This keeps the metric interpretation focused on completed
conversations regardless of whether the navigation task succeeded.

Completed-dialogue outcome by Agent B model:

| Agent B model | Completed runs | Successful | Semi-successful | Unsuccessful dialogue | Success rate | Route-or-better rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyLlama 1.1B | 94 | 84 | 5 | 5 | 89.36% | 94.68% |
| Qwen2.5 0.5B | 94 | 84 | 5 | 5 | 89.36% | 94.68% |
| Qwen2.5 1.5B | 137 | 126 | 5 | 6 | 91.97% | 95.62% |
| Phi-3 mini | 77 | 69 | 2 | 6 | 89.61% | 92.21% |
| Qwen2.5 7B | 50 | 46 | 2 | 2 | 92.00% | 96.00% |

Most useful indicators by outcome:

| Outcome | Strongest indicating metrics | Current observed pattern | Interpretation |
| --- | --- | --- | --- |
| successful | `task_outcome_route_validity`, `task_outcome_constraint_satisfaction_rate`, `agent_b_active_constraint_compliance`, `dialogue_state_joint_goal_accuracy`, `agent_b_grounded_proposal_score`, `nlg_faithfulness`, `whole_dialogue_goal_progress_auc` | successful runs are near 1.0 on task, grounding, faithfulness, and goal-progress metrics | these metrics confirm that the final route is not only syntactically present but task-grounded and constraint-compatible |
| semi-successful | `task_outcome_route_validity`, `duration_score`, `agent_b_actionability_score`, `whole_dialogue_goal_progress_auc`, `constraint_duration_gap_min`, `constraint_line_change_gap`, low `agent_b_active_constraint_compliance` | route validity remains high, but constraint satisfaction is 0.0; duration and line-change gaps rise | the dialogue found a route-like answer but failed later-stage preference or constraint negotiation |
| unsuccessful dialogue | low `nlu_route_valid_rate`, low `nlu_goal_reached_rate`, low `station_mentions`, low `candidate_route_count`, high `whole_dialogue_abandonment_rate`, high `asr_word_error_rate` and `asr_sentence_error_rate` | no candidate route is established; station/entity evidence collapses; abandonment reaches 1.0 | failure occurs before stable route negotiation, often around ASR/NLU entity preservation or dialogue-state grounding |

Useful numeric anchors from the completed-dialogue metric scan:

| Metric | Successful mean | Semi-successful mean | Unsuccessful-dialogue mean | Reading |
| --- | ---: | ---: | ---: | --- |
| `automatic_eval_score` | 0.954 | 0.216 | 0.069 | good compact outcome proxy, but still derived from task evidence |
| `quality_score` | 0.745 | 0.077 | 0.000 | separates accepted route dialogues from incomplete outcomes |
| `duration_score` | 0.977 | 0.666 | 0.000 | useful for distinguishing semi-success from total failure |
| `agent_b_grounded_proposal_score` | 0.994 | 0.443 | 0.000 | strong indicator that Agent B proposed a usable, network-grounded route |
| `agent_b_actionability_score` | 1.000 | 0.776 | 0.000 | semi-success can remain actionable while still violating constraints |
| `nlg_executable_utterance_rate` | 0.802 | 0.261 | 0.000 | indicates whether generated language can be converted into route actions |
| `whole_dialogue_goal_progress_auc` | 0.964 | 0.783 | 0.000 | best phase-aware progression signal; semi-success still shows progress |
| `whole_dialogue_abandonment_rate` | 0.000 | 0.895 | 1.000 | strong warning signal for incomplete or rejected dialogues |
| `asr_word_error_rate` | 0.079 | 0.080 | 0.697 | high values indicate upstream speech recognition failure; low values do not guarantee success |
| `asr_station_f1` | 0.977 | 0.978 | 0.223 | station preservation is a strong prerequisite for route grounding |
| `nlu_route_valid_rate` | 0.920 | 0.330 | 0.000 | separates stable route understanding from non-route dialogue |
| `nlu_goal_reached_rate` | 0.878 | 0.039 | 0.000 | strong indicator for whether the understood route reaches the target |
| `tts_text_change_rate` | 0.050 | 0.121 | 0.286 | higher values indicate speech realization drift, but interpret with ASR metrics |
| `candidate_route_count` | 3.702 | 1.579 | 0.000 | successful dialogues compare several candidates; failures rarely establish any |
| `station_mentions` | 26.252 | 15.684 | 0.833 | strong low-level signal that the dialogue stays grounded in network entities |
| `dialogue_management_repair_success_rate` | 0.937 | 0.921 | 0.368 | repairs can support success and semi-success; very low values indicate failed recovery |

How to phrase the main metric finding:

```text
The most reliable indicators are not single final scores but a chain of
consistent evidence: ASR preserves station entities, NLU extracts a valid
route, Agent B produces a grounded and executable proposal, dialogue state
retains the goal and constraints, and the task metrics confirm route validity
and constraint satisfaction. Semi-successful runs are especially useful because
they show that route validity alone is insufficient: they often preserve an
actionable route while failing constraint satisfaction or duration/transfer
optimality.
```

Metric families by thesis value:

- Best outcome-confirming metrics:
  - route validity,
  - constraint satisfaction,
  - active constraint compliance,
  - grounded proposal score,
  - NLG faithfulness.
- Best explanatory metrics:
  - whole-dialogue goal progress,
  - candidate route count,
  - station mentions,
  - NLU route-valid and goal-reached rates,
  - repair success rate.
- Best early-warning metrics:
  - ASR station F1,
  - ASR word/sentence error,
  - TTS text-change rate,
  - abandonment rate,
  - missing candidate routes.
- Weak or redundant alone:
  - direct task completion metrics when used as predictors of task completion;
  - trace completeness, TTS success, and pipeline success inside completed
    runs, because they are near-constant once the run reaches retrospective
    metric calculation;
  - shared-state agreement without route outcome context, because a dialogue
    can agree on an incomplete or wrong state.

### 8.9 Matched Agent B comparison in the current result set

Matched cases are the strongest basis for comparing Agent B models, because
all non-model configuration fields are equivalent. This controls for scenario,
persona, speech condition, TTS/ASR setup, seeds, objective, and decoding
profile.

Current strict matched coverage:

- UserLM-Agent-A:
  - 173 comparable conditions are covered by all five active Agent B models.
  - Successful matched cases per Agent B model range from 32 to 33.
- TinyLlama-Agent-A:
  - 56 comparable conditions are covered by all five active Agent B models.
  - Successful matched cases per Agent B model range from 37 to 39.

Current matched successful-run diagnostic table:

| Agent A | Agent B model | Matched cases | Successful matched cases | Mean turns | Repair turns | Runs with repair | Candidate routes | Duration regret | ASR WER | ASR station F1 | Goal progress |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UserLM | TinyLlama 1.1B | 173 | 33 | 11.27 | 0.79 | 54.5% | 3.70 | 1.24 | 0.063 | 0.975 | 0.947 |
| UserLM | Qwen2.5 0.5B | 173 | 32 | 11.12 | 0.78 | 56.2% | 3.66 | 1.00 | 0.057 | 0.979 | 0.955 |
| UserLM | Qwen2.5 1.5B | 173 | 33 | 11.73 | 0.79 | 54.5% | 3.79 | 1.67 | 0.061 | 0.975 | 0.956 |
| UserLM | Phi-3 mini | 173 | 32 | 11.09 | 0.66 | 53.1% | 3.84 | 1.38 | 0.058 | 0.978 | 0.945 |
| UserLM | Qwen2.5 7B | 173 | 33 | 11.15 | 0.70 | 51.5% | 3.76 | 1.67 | 0.064 | 0.978 | 0.952 |
| TinyLlama | TinyLlama 1.1B | 56 | 39 | 12.92 | 1.10 | 59.0% | 3.82 | 0.69 | 0.099 | 0.974 | 0.953 |
| TinyLlama | Qwen2.5 0.5B | 56 | 37 | 12.73 | 1.03 | 56.8% | 3.81 | 0.73 | 0.087 | 0.979 | 0.953 |
| TinyLlama | Qwen2.5 1.5B | 56 | 39 | 12.90 | 1.13 | 56.4% | 3.74 | 0.72 | 0.099 | 0.974 | 0.950 |
| TinyLlama | Phi-3 mini | 56 | 37 | 12.57 | 0.97 | 54.1% | 3.76 | 0.73 | 0.093 | 0.979 | 0.953 |
| TinyLlama | Qwen2.5 7B | 56 | 39 | 12.77 | 1.08 | 56.4% | 3.82 | 0.69 | 0.098 | 0.977 | 0.950 |

Interpretation:

- The matched table is the strongest direct Agent B comparison because each
  row uses conditions covered by all active models and removes duplicate
  attempts.
- UserLM successful matched runs use fewer repair turns than TinyLlama
  successful matched runs, suggesting caller implementation affects repair
  burden and dialogue efficiency.
- Agent B differences inside the successful matched subset are modest. The
  stronger distinction is not only whether the task succeeds, but how much
  repair, route revision, duration regret, and ASR degradation remain in runs
  with equal success status.
- Model differences are still visible in coverage and unmatched completed
  cases, especially because Qwen2.5 1.5B has more completed UserLM cases than
  the other selected models. Those unmatched cases must be discussed as
  coverage/completion evidence, not as fully controlled direct comparison.

Useful matched UserLM metric means:

| Metric | TinyLlama 1.1B | Qwen2.5 0.5B | Qwen2.5 1.5B | Phi-3 mini | Qwen2.5 7B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `automatic_eval_score` | 0.862 | 0.844 | 0.857 | 0.840 | 0.860 |
| `agent_b_grounded_proposal_score` | 0.906 | 0.894 | 0.903 | 0.878 | 0.903 |
| `nlg_faithfulness` | 0.906 | 0.894 | 0.903 | 0.878 | 0.903 |
| `whole_dialogue_goal_progress_auc` | 0.885 | 0.892 | 0.892 | 0.858 | 0.889 |
| `asr_station_f1` | 0.924 | 0.923 | 0.932 | 0.915 | 0.935 |
| `nlu_route_valid_rate` | 0.836 | 0.834 | 0.822 | 0.825 | 0.840 |
| `candidate_route_count` | 3.324 | 3.243 | 3.432 | 3.378 | 3.405 |
| `station_mentions` | 23.378 | 22.784 | 24.189 | 23.811 | 24.243 |
| `repair_success_rate` | 0.901 | 0.899 | 0.910 | 0.905 | 0.946 |

Suggested thesis wording:

```text
Matched completed cases provide the cleanest model comparison because only the
Agent B backend differs. In the current UserLM matched subset, most cases were
either solved by all models or failed by all models, and only one case showed
mixed model outcomes. This suggests that the present result set is stronger for
evaluating phase-wise metrics and condition difficulty than for claiming large
performance differences between the selected Agent B models. Model-backend
differences should therefore be reported together with coverage and completion
evidence, not only with matched-case success rates.
```

### 8.10 Text versus speech comparison

Use this comparison only where `pair_id` links a `text_only` run and an
`audio_variant` run with the same non-audio factors.

Report:

- number of matched text/audio pairs;
- task-success delta;
- route-validity delta;
- constraint-satisfaction delta;
- turn-count delta;
- repair-turn delta;
- ASR and speech-channel error indicators.

Interpretation rules:

- If text succeeds and speech fails:
  - inspect ASR WER, ASR station F1, transcript corrections, and NLU slot
    accuracy first.
- If both succeed:
  - compare turns, repair burden, latency, duration regret, and task focus.
- If both fail:
  - inspect task difficulty and route/constraint feasibility before blaming
    the speech channel.
- If speech succeeds and text fails:
  - treat as possible stochastic/model variation unless the logs show a clear
    repair or normalization advantage.

Safe wording:

```text
Paired text/audio comparisons isolate observed speech-channel effects because
the non-audio condition factors are shared. The comparison remains descriptive:
audio degradation can explain a failure only when the transcript, NLU state,
and dialogue trace show propagation from speech error into task state or route
selection.
```

### 8.11 Success-status and equal-status analysis

Analyze outcome bands in this order:

1. successful,
2. semi-successful,
3. unsuccessful completed,
4. execution incomplete.

For distinguishing success status, prioritize:

- task outcome:
  - route validity,
  - constraint satisfaction,
  - duration regret,
  - correct route selection;
- grounding and NLG:
  - grounded proposal score,
  - executable utterance rate,
  - NLG faithfulness,
  - route mention completeness;
- ASR/NLU:
  - ASR station F1,
  - ASR WER,
  - critical slot accuracy,
  - NLU route-valid rate;
- dialogue management:
  - repair success rate,
  - repeated repair rate,
  - stagnation rate,
  - premature answer/closure rate.

If success status is equal, do not repeat the success label. Compare
secondary success factors:

- efficiency:
  - turn count,
  - turns to success,
  - runtime,
  - latency;
- cooperation quality:
  - task focus,
  - candidate-route count,
  - route revisions,
  - route repetition;
- repair burden:
  - repair turns,
  - clarification count,
  - correction-turn rate,
  - successful runs with repair;
- route quality:
  - duration regret,
  - optimality ratio,
  - constraint gap,
  - active constraint compliance.

Suggested wording:

```text
When two configurations reach the same success status, the analysis shifts from
whether the task was solved to how it was solved. Lower repair burden, fewer
turns, lower duration regret, stronger task focus, and fewer route revisions
indicate a more efficient and stable successful dialogue.
```

### 8.12 Current defensible inferences

Use this section as a conclusion scaffold. Keep claims tied to the evidence
level that supports them.

Strongly supported inferences:

- The framework can separate four concepts that are often blurred:
  - result availability,
  - completed dialogue,
  - valid route production,
  - full task and constraint satisfaction.
- Completed UserLM dialogues usually succeed once a usable terminal dialogue
  exists: 281 of 317 unique completed UserLM conditions are task-successful.
- The strongest metric evidence is phase-chain evidence, not a single final
  number:
  - station/entity preservation,
  - route-valid NLU,
  - grounded Agent B proposal,
  - executable and faithful NLG,
  - goal progress,
  - constraint satisfaction.
- Semi-successful cases are methodologically valuable because they show that
  route validity alone is insufficient. A route can be actionable yet still
  fail duration, transfer, or constraint expectations.
- In the fully matched all-model subset, most cases are solved by all models
  or fail for all models. This suggests that condition difficulty and pipeline
  conditions are currently stronger explanatory factors than small differences
  between the selected Agent B backends.

Moderately supported inferences:

- Qwen2.5 1.5B is the strongest practical UserLM-Agent-A backend in the current
  evidence set because it has the largest number of completed UserLM cases and
  high task success among completed cases.
- Very small Agent B models are viable for the controlled task. TinyLlama 1.1B
  and Qwen2.5 0.5B both solve many completed UserLM conditions and perform
  similarly in the matched subset.
- Larger model size alone is not a reliable explanation of success in this
  experiment. Qwen2.5 7B performs well on matched completed cases, but its
  completion coverage is lower, so runtime/resource feasibility must be part
  of the interpretation.
- ASR and NLU metrics are useful early-warning indicators: high ASR word error,
  low ASR station F1, low NLU route-valid rate, and low goal-reached rate align
  with unsuccessful completed dialogues.

Weak or exploratory inferences:

- Specific thresholds for predicting future failure are not yet validated.
  Current thresholds should be presented as descriptive observations, not as a
  trained predictor.
- The experiment does not prove that one model family is generally better than
  another. It supports claims about this controlled route-dialogue task under
  the tested speech and batch conditions.
- Human user satisfaction cannot be inferred directly. Automatic task success,
  dialogue cost, and phase evidence are proxies that need human validation for
  naturalness and perceived usefulness.

Potential explanation for the observed pattern:

```text
The task may currently contain many conditions that are either easy enough for
all selected backends or hard enough that every backend fails under the same
speech/dialogue constraints. This compresses model differences in the fully
matched subset. More model-discriminating evidence would require additional
conditions near the decision boundary: understandable but noisy speech,
constraints that require route revision, and scenarios where an initial valid
route exists but a later constraint changes the optimum.
```

Recommended final answer to the main research question:

```text
Automatic phase-wise evaluation is feasible and informative for this controlled
spoken route-dialogue task. Final task success alone is too coarse: the most
useful evaluation comes from combining task outcome metrics with diagnostic
phase metrics and matched-condition comparisons. The current evidence supports
reliable analysis of completed dialogues and failure modes, while model-ranking
claims must remain cautious because many matched cases are not
model-discriminating.
```

Recommended caveat:

```text
The results should be interpreted as evidence for the framework and for the
tested controlled conditions, not as a universal ranking of LLMs or speech
dialogue systems. The strongest contribution is the reproducible data pipeline
that makes success, semi-success, failure, and likely failure origins
inspectable after the run.
```

## 9. Discussion and Conclusion

### Discussion structure

Use the discussion to interpret the results, not to introduce new methods.
Every paragraph should connect a finding to a research question, a validity
condition, or a limitation.

- Answer each research question directly.
  - Start each answer with one clear sentence.
  - Then give the strongest supporting evidence:
    - completed-run counts,
    - matched-case counts,
    - metric patterns,
    - representative failure modes.
  - End with one implication for automatic SDS evaluation.
  - Avoid making the reader infer the answer from tables alone.

- Explain which metrics were useful and why.
  - Outcome-confirming metrics:
    - route validity,
    - constraint satisfaction,
    - active constraint compliance,
    - task success.
    - Use these to show whether the task was solved.
    - Explain that they are close to the success definition and therefore not
      independent predictors.
  - Diagnostic metrics:
    - ASR station F1,
    - ASR WER,
    - NLU route-valid rate,
    - NLU goal-reached rate,
    - grounded proposal score,
    - NLG faithfulness,
    - candidate-route count,
    - station mentions,
    - abandonment rate.
    - Use these to explain where successful and unsuccessful completed
      dialogues diverge.
  - Matched-comparison metrics:
    - use them for direct model comparison only when all non-model conditions
      match.
    - Emphasize that matched evidence is smaller but more internally valid.
  - Efficiency metrics:
    - turn count,
    - latency,
    - runtime,
    - dialogue cost.
    - Use them when two configurations both solve the task but differ in cost.

- Explain which metrics were weak, redundant, or only confirmatory.
  - Direct task metrics are necessary but can be tautological if treated as
    predictors.
  - Pipeline success and TTS success are often near-constant inside completed
    runs, so they are more useful for execution coverage than dialogue-quality
    comparison.
  - Shared-state agreement can be misleading if agents agree on an incomplete
    or wrong route.
  - Generic text similarity is weak for this task because a fluent answer can
    still violate network constraints.

- Discuss model-backend differences cautiously.
  - Separate three claims:
    - coverage/completion: did the backend finish enough runs?
    - task quality: did completed dialogues solve the route task?
    - matched comparison: did the backend differ when all non-model settings
      were equal?
  - Current strongest model-related observation:
    - Qwen2.5 1.5B has the strongest practical UserLM evidence profile because
      it completed the most UserLM conditions and retained high success among
      completed cases.
  - Current matched-case observation:
    - most all-model matched cases are solved by all models or fail for all
      models.
    - This limits broad model-ranking conclusions.
  - Avoid saying:
    - "large models are worse",
    - "small models are better",
    - "model size determines success".
  - Prefer saying:
    - "under these local/cluster conditions, completion coverage and matched
      task outcomes do not support a simple model-size ranking."

- Discuss speech-channel effects.
  - Explain how the speech pipeline can affect the route task:
    - TTS pronunciation affects station and line recognizability;
    - ASR errors alter station names, times, and constraints;
    - normalization can repair some errors but must remain transparent;
    - NLU converts transcripts into route/task state.
  - Interpret speech metrics as diagnostic:
    - high WER does not always imply task failure;
    - a single station substitution can matter more than many harmless words;
    - station F1 and goal-reached rate are more task-relevant than WER alone.
  - Discuss audio personas as controlled stressors:
    - clear speech tests the upper-performance range;
    - hesitant/degraded speech tests repair and robustness;
    - severe-channel conditions help produce failure cases for metric analysis.

- Discuss semi-successful cases explicitly.
  - Semi-success is not noise; it is an important outcome band.
  - It identifies cases where the dialogue produced route evidence but failed
    final acceptance, constraints, duration, or route optimality.
  - Use semi-success to argue that binary success/failure is too coarse for
    SDS evaluation.
  - Connect it to phase evidence:
    - route validity can be present while constraint satisfaction is absent;
    - grounded proposal score can be moderate while active constraint
      compliance is zero;
    - goal progress can be nonzero while final task success fails.

- Discuss condition difficulty.
  - In the matched all-model subset, many cases are uniformly solved or
    uniformly unsolved.
  - This suggests that some conditions primarily test task/audio difficulty
    rather than model differences.
  - Use this to justify future boundary-case design:
    - cases where a valid route exists but later constraints change the
      optimum;
    - cases with moderate ASR noise rather than extreme corruption;
    - cases requiring one repair step but not repeated repair loops.

- Discuss limitations and threats to validity.
  - Construct validity:
    - automatic metrics approximate task quality but do not measure human
      satisfaction directly.
    - Some metrics overlap with the success definition.
  - Internal validity:
    - model comparisons are strongest only for matched conditions.
    - execution failures and invalid conditions must not be mixed with
      dialogue failures.
  - External validity:
    - simulated callers are not real callers;
    - the network and scenarios are controlled abstractions;
    - TTS/ASR engines do not cover all speech conditions, accents, or devices.
  - Statistical validity:
    - matched all-model subsets are currently small;
    - descriptive correlations are not predictive models;
    - no human-rating ground truth is available for calibration.

- Explain what a supervisor or examiner can learn from the framework.
  - The project is not only a route-dialogue demo; it is an evaluation
    instrument.
  - Its main scientific value is traceability:
    - configuration,
    - speech pipeline,
    - dialogue state,
    - route validation,
    - metric input,
    - metric output.
  - It demonstrates how automatic SDS evaluation can move beyond final task
    success toward phase-wise diagnosis.

### Conclusion structure

The conclusion should be shorter than the discussion and should not introduce
new evidence. It should leave the examiner with a precise understanding of what
was built, what was shown, and what remains limited.

- Restate the problem.
  - Spoken dialogue systems can fail in many phases before final task outcome
    is known.
  - Final success/failure alone does not explain whether the error came from
    speech recognition, semantic interpretation, dialogue state, route
    grounding, response generation, or task constraints.
  - Automatic evaluation therefore needs phase-aware evidence.

- Summarize the method.
  - CoopNavigationSDS creates controlled route-finding dialogues between:
    - Agent A/UserLM as caller,
    - Agent B as the evaluated dialogue backend.
  - The task uses:
    - a controlled transport network,
    - staged route constraints,
    - TTS/ASR speech pipeline evidence,
    - route validation,
    - retrospective metric calculation.
  - Results are analyzed through:
    - completed-dialogue outcome bands,
    - phase metrics,
    - matched Agent B comparisons,
    - coverage and missing-evidence reports.

- Summarize main findings.
  - Completed UserLM dialogues usually succeed once terminal route evidence is
    available.
  - Semi-successful cases show that route validity is not enough; constraint
    and optimality evidence are necessary.
  - Matched all-model cases show limited model discrimination in the current
    subset because most cases are either solved by all selected models or fail
    for all selected models.
  - Diagnostic metrics such as ASR station F1, NLU route-valid rate, grounded
    proposal score, NLG faithfulness, candidate-route count, and abandonment
    rate help explain outcome differences.

- State contributions.
  - A reproducible experiment framework for automatic SDS evaluation.
  - A controlled cooperative navigation task with staged constraints.
  - A phase-aware logging and retrospective metric pipeline.
  - A matched-comparison method for Agent B backends.
  - A distinction between execution failure, unsuccessful dialogue,
    semi-success, and task success.
  - A thesis-ready result interpretation structure.

- State limitations.
  - The data supports controlled-task conclusions, not universal SDS rankings.
  - Human satisfaction and perceived naturalness are not directly measured.
  - Model comparisons are strongest only in matched subsets.
  - Some result differences reflect execution feasibility and cluster/runtime
    behavior rather than dialogue quality alone.
  - The current matched subset is useful but still small.

- Future work.
  - Human validation:
    - collect ratings for naturalness, helpfulness, and perceived task success;
    - compare human judgments with automatic metric scores.
  - Real transit network:
    - use real schedules, transfer constraints, delays, and station names;
    - test whether route-grounding metrics still behave similarly.
  - Real microphone input:
    - replace synthetic speech-only runs with live or recorded human speech;
    - evaluate microphone noise, accent, and channel variability.
  - More languages and accents:
    - test multilingual station names and non-native pronunciation;
    - analyze whether entity-preservation metrics remain reliable.
  - More speech-native models:
    - compare cascaded TTS/ASR pipelines with end-to-end speech-capable
      dialogue models.
  - Larger balanced condition grid:
    - increase matched all-model cases;
    - add boundary cases that are neither trivially easy nor impossible.
  - Additional task domains:
    - emergency hotline triage,
    - appointment scheduling,
    - customer-service troubleshooting,
    - indoor navigation.

### Final paragraph template

```text
This thesis showed that a controlled spoken route-dialogue task can produce
research-grade evidence for automatic SDS evaluation. The main value of the
framework is not a single model ranking, but the ability to trace how speech,
language understanding, dialogue state, route grounding, and constraint
satisfaction contribute to task outcome. The results support phase-wise
automatic evaluation as a useful complement to human evaluation, while also
showing that model comparisons require careful coverage, matched-condition, and
validity analysis.
```

## 10. Citation Map

Use citations only when they support a specific claim.

Citation use rules:

- cite a paper where the claim first becomes methodologically important;
- do not place a citation after a generic sentence that the paper does not
  specifically support;
- separate background citations from method justification citations;
- cite the same foundational paper consistently instead of scattering many
  weak references.

Recommended citation pattern:

```text
[Claim about evaluation problem]. [Citation]. In this thesis, that problem is
handled by [specific design decision].
```

Example:

```text
Prior SDS evaluation connects task success with dialogue cost and user-facing
quality. In this thesis, that idea motivates reporting route success together
with turn count, repair behavior, latency, and phase-wise evidence.
```

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

How to use the citation strands:

- PARADISE:
  - use for the broad idea that task outcome and dialogue cost both matter;
  - do not imply that the thesis estimates a full satisfaction model.
- "How NOT To Evaluate Your Dialogue System":
  - use to motivate caution with surface-level text metrics;
  - connect it to why route validation and task grounding matter.
- MultiWOZ and SGD:
  - use for task-oriented dialogue concepts such as goal, slot, state, and
    constraint;
  - contrast with this thesis's spoken and grounded route task.
- SpokenWOZ/SLUE:
  - use for the need to evaluate spoken inputs and ASR-induced differences;
  - connect to the preserved raw transcript and normalized understanding.
- NISQA/DNSMOS:
  - use for no-reference speech-quality metric background;
  - avoid claiming these metrics alone measure task success.
- HELM/model-card/reproducibility work:
  - use for transparent backend comparison, configuration reporting, and
    cautious model claims.
- Navigation/Map Task literature:
  - use to motivate cooperative route dialogue and grounding;
  - clarify that this project uses a synthetic transit network, not an
    open-ended map.

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

Good paragraph shape:

- claim:
  - one sentence saying what the subsection shows;
- evidence:
  - one or two numbers, table references, or metric definitions;
- interpretation:
  - what the evidence means for the research question;
- limitation:
  - what cannot be concluded from this evidence.

Example:

```text
Completed UserLM-Agent-A dialogues show high task success in the current
result set. This indicates that the framework can produce successful
cooperative route dialogues under controlled speech conditions. However, this
rate applies only to execution-complete runs and does not include unavailable
or interrupted conditions.
```

### Remove

- repeated lists of every possible metric,
- implementation details before Chapter 6,
- broad claims about all SDS,
- claims about model size when model family/provider also differs,
- unexplained acronyms,
- metric names without formulas or evidence sources.

Also remove or rewrite:

- phrases like "the model understands" unless supported by NLU/state evidence;
- phrases like "ASR caused the failure" unless the transcript and downstream
  state show the link;
- generic praise such as "robust" without specifying the tested conditions;
- implementation detail that does not affect experiment validity or metric
  calculation.

### Suggested wording pattern

Use this pattern often:

```text
This metric measures [construct] using [logged evidence].
It is calculated as [formula].
High/low values indicate [interpretation].
It is unavailable when [missing-data rule].
Its limitation is [scope boundary].
```

Outcome wording pattern:

```text
This result is calculated over [denominator]. It shows [finding]. It supports
[research question] because [reason]. It should be interpreted cautiously
because [validity boundary].
```

Model comparison wording pattern:

```text
For matched conditions, [model/backend] differed from [comparison] in
[metric/outcome]. This comparison controls [listed non-model factors] but does
not isolate [confounded model properties].
```

Failure wording pattern:

```text
The earliest observable failure signal occurred in [phase]. The logged evidence
was [evidence]. This suggests [diagnostic interpretation], but downstream
effects require manual case inspection for causal confirmation.
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

How to use the checklist:

- before writing Chapter 6:
  - verify that the experiment unit, stage logic, model set, and result
    denominator are final;
  - remove obsolete models, invalid conditions, and outdated settings from the
    main narrative.
- before writing Chapter 7:
  - ensure every metric has required logged fields;
  - mark unavailable metrics explicitly;
  - separate task, phase, and execution metrics.
- before writing Chapter 8:
  - regenerate analysis tables from raw results;
  - confirm deduplication rules;
  - report coverage before success.
- before writing Chapter 9:
  - collect the strongest and weakest claims;
  - map each claim to evidence and limitation;
  - separate conclusions from future work.
- before submission:
  - verify all cited papers are in the bibliography;
  - check every figure caption for denominator and unit of analysis;
  - ensure no paragraph claims more than the experiment can support.

Minimum appendix package:

- configuration schema;
- active model/backend table;
- TTS/ASR provider table;
- scenario and persona overview;
- transport-network data or generation rules;
- route-validation rules;
- metric catalog with formulas and input fields;
- result-folder schema;
- reproducibility commands;
- selected transcript examples for success, semi-success, and unsuccessful
  completed dialogues.
