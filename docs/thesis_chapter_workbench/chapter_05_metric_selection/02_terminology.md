# Chapter 5: Metric Selection - Terminology

Purpose: chapter-local definitions and explanations. Define terms before using them in thesis prose.

## Task success rate

- Definition: The proportion of completed runs ending with satisfactory task completion.
- Explanation:
  - Use completed runs as denominator when analyzing dialogue performance; separately report execution failures.

## Route optimality gap

- Definition: Difference between proposed route duration and the constraint-layer optimal duration.
- Explanation:
  - Useful for route quality after basic validity is satisfied.

## Constraint extraction F1

- Definition: F1 score for extracted constraints against known/revealed constraints.
- Explanation:
  - Requires captured reference constraints and extracted semantic frames.

## Repair success rate

- Definition: Proportion of repair attempts that resolve a misunderstanding or missing slot.
- Explanation:
  - Useful for diagnosing whether dialogue can recover from speech or NLU errors.

## Failure-localization candidate

- Definition: The earliest phase whose evidence plausibly explains the final failure.
- Explanation:
  - Diagnostic, not causal proof.

## Term-Use Rules

- Use one term for one construct; avoid alternating synonyms for key variables.
- Do not use `success`, `route validity`, and `constraint satisfaction` interchangeably.
- Reserve `causal` for controlled comparisons or ablations; otherwise use `associated with` or `indicates`.
- Define abbreviations at first use and prefer full terms in headings.
