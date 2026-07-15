# Chapter 1: Introduction - Outline

Purpose: in-depth bullet outline for drafting thesis prose. Use this file for structure and argument order.

## Chapter Function

- Motivate the research problem, define the scope, state research questions, and present contributions without implementation detail.

## Core Argument

- Spoken task-oriented systems are practically useful but difficult to evaluate.
- Final task success does not explain whether failure came from speech, understanding, dialogue management, route reasoning, or generation.
- A route task provides externally checkable correctness and therefore a useful testbed for automatic evaluation.
- The thesis asks how far phase-wise automatic metrics can explain successful and failed SDS interactions.

## Subchapter Writing Plan

### 1.1 Motivation and background

- Purpose: Open with the relevance of spoken task-oriented systems.
- Points to write:
  - Mention navigation, hotline-style assistance, accessibility, and hands-free use.
  - Explain that speech adds timing, audibility, recognition, and repair problems.
  - Use route finding as a concrete example where a caller needs correct actionable instructions.
- Avoid:
  - Do not start with implementation details or model names.

### 1.2 Evaluation difficulty

- Purpose: Explain why final success alone is insufficient.
- Points to write:
  - A dialogue can fail despite fluent text if a station is misrecognized.
  - A dialogue can semi-succeed if it finds a valid route but violates a revealed constraint.
  - Speech and ASR failures can be hidden if only normalized final transcripts are inspected.
  - Automatic metrics must therefore be phase-aware and task-grounded.

### 1.3 Problem statement

- Purpose: Narrow from general SDS evaluation to the thesis gap.
- Points to write:
  - Current automatic metrics often measure surface text, semantic similarity, or final success separately.
  - For spoken task-oriented dialogue, the evaluation needs to connect speech evidence, semantic state, dialogue behavior, and task outcome.
  - The problem is not only to score a dialogue but to make the score interpretable.

### 1.4 Research questions

- Purpose: State research questions so later chapters can answer them directly.
- Points to write:
  - RQ1: Can a controlled route-dialogue framework produce reliable evidence for automatic phase-wise SDS evaluation?
  - RQ2: Which metrics best distinguish successful, semi-successful, and unsuccessful completed dialogues?
  - RQ3: How do speech conditions affect task success compared with paired text-only controls?
  - RQ4: How do Agent A and Agent B backend choices affect outcome and diagnostic metrics under matched conditions?
  - RQ5: Which failures can be localized to speech, understanding, dialogue management, or route reasoning evidence?

### 1.5 Contributions

- Purpose: List concrete thesis contributions.
- Points to write:
  - A reproducible route-dialogue experiment framework.
  - A logging schema that preserves phase evidence for retrospective metrics.
  - A metric set separating outcome, phase diagnostics, and supplementary indicators.
  - A matched text/audio analysis showing the effect of speech-channel degradation.
  - A validity-aware interpretation guide for automatic SDS metrics.

## Minimum Quality Checklist

- Every claim has a clear denominator, source, or citation.
- The chapter distinguishes task outcome, phase evidence, and interpretation.
- The wording stays cautious where the evidence is descriptive rather than causal.
- No raw implementation detail appears unless it supports a methodological point.
