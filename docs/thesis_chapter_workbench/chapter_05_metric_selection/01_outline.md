# Chapter 5: Metric Selection - Outline

Purpose: in-depth bullet outline for drafting thesis prose. Use this file for structure and argument order.

## Chapter Function

- Define the selected metrics, formulas, evidence requirements, interpretation, and limitations.

## Core Argument

- Metrics are selected because they answer a research question and can be calculated from reliable evidence.
- Outcome metrics describe final status; diagnostic metrics explain phase behavior; supplementary metrics add context.
- Metrics without required evidence should be marked unavailable rather than computed from assumptions.
- Metric validity is assessed by role, evidence availability, and relation to outcome.

## Subchapter Writing Plan

### 5.1 Selection principle

- Purpose: Explain why each metric belongs.
- Points to write:
  - Include only metrics with a defined construct, required logged fields, calculation rule, and interpretation boundary.
  - Reject or disable metrics that cannot be calculated from captured evidence.
  - Separate direct outcome decompositions from independent diagnostic indicators.

### 5.2 Core formulas

- Purpose: Give formulas compactly.
- Points to write:
  - Task success rate = successful completed runs / completed runs.
  - Semi-success rate = semi-successful completed runs / completed runs.
  - Route-valid rate = runs with valid accepted/proposed route / completed runs.
  - Constraint satisfaction rate = satisfied revealed constraints / revealed constraints.
  - Audio-text delta = audio success rate minus paired text success rate.
  - Route optimality gap = proposed route duration minus constraint-layer optimal duration.

### 5.3 Phase metrics

- Purpose: Map metrics to pipeline phases.
- Points to write:
  - TTS/audio: audio availability, speech duration, intelligibility/quality where evidence exists.
  - ASR: WER, station/line/time entity preservation, semantic ASR error.
  - NLU: constraint extraction, route entity extraction, semantic frame correctness.
  - Dialogue state: constraint retention, state drift, shared-state consistency where observable.
  - Dialogue management: clarification rate, repair success, premature answer rate, stagnation.
  - Agent B response: grounded proposal score, hallucinated content, actionability.
  - NLG: faithfulness, semantic adequacy, executable utterance rate.
  - Whole dialogue: task success, semi-success, turns, runtime, failure-localization candidate.

### 5.4 Current metric validity evidence

- Purpose: Use the generated metric validity assessment.
- Points to write:
  - Current completed active rows used for metric validity assessment: 557.
  - Strong diagnostic associations include ASR station F1, NLU route-valid rate, grounded proposal score, actionability, stagnation, and faithfulness.
  - Outcome-overlapping metrics should be used to decompose success, not as independent predictors.
  - Generic lexical metrics are supplementary unless linked to task-specific evidence.
- Evidence hooks:
  - See docs/THESIS_METRIC_VALIDITY_ASSESSMENT.md.
  - Use docs/THESIS_METRIC_VALIDITY_TABLE.csv for exact metric roles and correlation values.

### 5.5 Missing evidence rule

- Purpose: Prevent invalid calculations.
- Points to write:
  - If audio evidence is missing, do not compute audio-quality metrics.
  - If a route is not parseable, route metrics should record parse failure rather than infer correctness.
  - If an ASR transcript is unavailable, ASR metrics are unavailable and the provider failure is part of the result.

## Minimum Quality Checklist

- Every claim has a clear denominator, source, or citation.
- The chapter distinguishes task outcome, phase evidence, and interpretation.
- The wording stays cautious where the evidence is descriptive rather than causal.
- No raw implementation detail appears unless it supports a methodological point.
