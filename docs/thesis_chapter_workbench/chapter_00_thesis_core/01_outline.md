# Chapter 0: Thesis Core - Outline

Purpose: in-depth bullet outline for drafting thesis prose. Use this file for structure and argument order.

## Chapter Function

- Keep the whole thesis aligned with the central object: automatic evaluation of spoken task-oriented dialogue systems using CoopNavigationSDS as the research instrument.

## Core Argument

- The software is not the research result by itself; it is the controlled instrument for collecting evidence.
- The evaluated object is Agent B as a route-information dialogue system under different backend and speech-channel conditions.
- Task outcome, constraint satisfaction, and phase-wise evidence must be interpreted together.
- Raw run evidence is authoritative; generated tables and conclusions are derived views.

## Subchapter Writing Plan

### Central claim

- Purpose: State the thesis in one precise sentence.
- Points to write:
  - This thesis studies whether a controlled cooperative navigation task can automatically evaluate spoken task-oriented dialogue systems phase-wise.
  - The key contribution is explainability of success, semi-success, and failure from logged evidence, not only final outcome labels.
  - Use the phrase 'likely failure origin' instead of causal attribution unless an ablation proves causality.
- Avoid:
  - Do not claim universal SDS evaluation, real-user validity, or model-size causality.

### Validity boundary

- Purpose: Define the claims that are allowed and the claims that are outside scope.
- Points to write:
  - Construct validity: every metric must map to a named construct and evidence field.
  - Internal validity: direct model comparisons require matched non-model conditions.
  - External validity: the transport network, caller personas, and speech channel are controlled abstractions.
  - Statistical conclusion validity: report denominators before percentages.

### Current empirical basis

- Purpose: Use current derived results without overclaiming.
- Points to write:
  - Active thesis-relevant deduplicated rows: 1319.
  - Completed active thesis rows: 557.
  - Fully crossed matched subset: 11 condition groups, 220 runs.
  - The fully crossed subset supports strongest direct text/audio and Agent A/Agent B comparisons.
- Evidence hooks:
  - See docs/THESIS_RESULT_CONFIGURATION_EFFECTS.md for current denominators.
  - See docs/THESIS_METRIC_VALIDITY_ASSESSMENT.md for metric-role classification.

## Minimum Quality Checklist

- Every claim has a clear denominator, source, or citation.
- The chapter distinguishes task outcome, phase evidence, and interpretation.
- The wording stays cautious where the evidence is descriptive rather than causal.
- No raw implementation detail appears unless it supports a methodological point.
