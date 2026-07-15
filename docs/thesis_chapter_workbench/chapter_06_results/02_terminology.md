# Chapter 6: Results - Terminology

Purpose: chapter-local definitions and explanations. Define terms before using them in thesis prose.

## Coverage

- Definition: The set of planned or observed conditions represented by completed evidence.
- Explanation:
  - Always state coverage before model or metric comparisons.

## Fully crossed subset

- Definition: A subset where every compared factor combination is present.
- Explanation:
  - This is the safest basis for direct comparisons.

## Audio-text delta

- Definition: Difference between audio and paired text success rates.
- Explanation:
  - Negative values show speech-channel degradation relative to clean text control.

## Ceiling condition

- Definition: A condition where nearly all runs succeed.
- Explanation:
  - Useful for validating the pipeline but weak for distinguishing metrics.

## Floor condition

- Definition: A condition where many runs fail.
- Explanation:
  - Useful for stress-testing whether metrics identify failure.

## Term-Use Rules

- Use one term for one construct; avoid alternating synonyms for key variables.
- Do not use `success`, `route validity`, and `constraint satisfaction` interchangeably.
- Reserve `causal` for controlled comparisons or ablations; otherwise use `associated with` or `indicates`.
- Define abbreviations at first use and prefer full terms in headings.
