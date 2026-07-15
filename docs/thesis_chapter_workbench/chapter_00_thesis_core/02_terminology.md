# Chapter 0: Thesis Core - Terminology

Purpose: chapter-local definitions and explanations. Define terms before using them in thesis prose.

## Automatic evaluation

- Definition: Metric-based evaluation computed from stored evidence rather than direct human ratings.
- Explanation:
  - In this thesis, automatic evaluation is useful because route validity and constraint satisfaction can be checked objectively.

## Phase-wise evidence

- Definition: Intermediate outputs from pipeline stages such as TTS, ASR, NLU, dialogue state, route validation, NLG, and final outcome.
- Explanation:
  - Phase-wise evidence makes failure explanation possible.

## Outcome metric

- Definition: A metric that describes final task status.
- Explanation:
  - Examples: task success, semi-success, unsuccessful completed dialogue, execution failure.

## Diagnostic metric

- Definition: A metric used to identify where and how a run degraded.
- Explanation:
  - Examples: ASR station F1, repair success, grounded proposal score, state drift.

## Completed run

- Definition: A run that reaches the end of the dialogue pipeline and produces usable outcome evidence.
- Explanation:
  - Navigation failure inside a completed run is analytically useful and should not be removed.

## Term-Use Rules

- Use one term for one construct; avoid alternating synonyms for key variables.
- Do not use `success`, `route validity`, and `constraint satisfaction` interchangeably.
- Reserve `causal` for controlled comparisons or ablations; otherwise use `associated with` or `indicates`.
- Define abbreviations at first use and prefer full terms in headings.
