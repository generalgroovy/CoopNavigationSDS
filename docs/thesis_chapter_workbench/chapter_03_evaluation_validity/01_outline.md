# Chapter 3: Evaluation Concepts and Validity Threats - Outline

Purpose: in-depth bullet outline for drafting thesis prose. Use this file for structure and argument order.

## Chapter Function

- Define metric families, validity criteria, and why task-grounded phase metrics are needed before the methodology is described.

## Core Argument

- Automatic metrics are useful only when the construct and evidence source are explicit.
- Lexical and semantic text metrics are supplementary because they cannot verify route executability.
- Speech metrics diagnose audio or recognition quality but do not by themselves establish task success.
- Validity threats must be visible before results are interpreted.

## Subchapter Writing Plan

### 3.1 Human versus automatic evaluation

- Purpose: Set the scope of automatic evaluation.
- Points to write:
  - Human evaluation is valuable but expensive and hard to scale.
  - Automatic evaluation is repeatable but only valid for explicitly defined constructs.
  - The thesis uses automatic metrics as controlled evidence, not as a replacement for all human judgment.

### 3.2 Construct validity

- Purpose: Define what each metric is allowed to mean.
- Points to write:
  - Task success metrics measure whether the dialogue achieved the route goal.
  - Phase metrics measure intermediate evidence that may explain success or failure.
  - Outcome-confirming metrics overlap with the success definition and should not be sold as independent predictors.

### 3.3 Lexical and semantic text metrics

- Purpose: Use the passed XW-style depth here: explain usefulness and limits precisely.
- Points to write:
  - Lexical metrics compare surface overlap and can detect wording similarity or divergence.
  - Semantic text metrics compare meaning more flexibly than exact overlap.
  - Both families can miss route executability errors such as wrong station order, omitted line names, or violated constraints.
  - Therefore they are supplementary indicators in this thesis.

### 3.4 Speech and semantic metrics

- Purpose: Separate audio quality, ASR transcript quality, and task semantics.
- Points to write:
  - Audio quality metrics describe the signal or perceived quality.
  - WER describes word-level transcription errors.
  - Entity and semantic ASR metrics are closer to task success because they measure station, line, time, and constraint preservation.
  - The thesis should emphasize semantic ASR/entity metrics over WER when explaining route failures.

### 3.5 Validity threats

- Purpose: Make limitations part of the analysis rather than an afterthought.
- Points to write:
  - Internal validity: compare only matched conditions for direct model claims.
  - External validity: synthetic network and simulated callers limit generalization.
  - Reliability: metrics must be regenerable from raw logs.
  - Missing data: unavailable evidence must be reported as unavailable, not imputed silently.

## Minimum Quality Checklist

- Every claim has a clear denominator, source, or citation.
- The chapter distinguishes task outcome, phase evidence, and interpretation.
- The wording stays cautious where the evidence is descriptive rather than causal.
- No raw implementation detail appears unless it supports a methodological point.
