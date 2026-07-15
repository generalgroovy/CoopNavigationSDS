# Chapter 4: Methodology and Research Design - Terminology

Purpose: chapter-local definitions and explanations. Define terms before using them in thesis prose.

## Agent A

- Definition: The simulated caller/user role with private travel goals and constraints.
- Explanation:
  - UserLM is the primary thesis caller; TinyLlama can be used as a control stratum.

## Agent B

- Definition: The evaluated route-information dialogue-system role.
- Explanation:
  - Agent B is instantiated by different LLM backends.

## Knowledge boundary

- Definition: A rule defining what each agent can know directly.
- Explanation:
  - This preserves the authenticity of dialogue-based cooperation.

## Matched condition

- Definition: A set of runs that differ only in the factor being compared.
- Explanation:
  - Matched conditions are required for direct model or channel claims.

## Retrospective metric calculation

- Definition: Metrics are calculated after the run from stored evidence.
- Explanation:
  - This improves auditability and reproducibility.

## Term-Use Rules

- Use one term for one construct; avoid alternating synonyms for key variables.
- Do not use `success`, `route validity`, and `constraint satisfaction` interchangeably.
- Reserve `causal` for controlled comparisons or ablations; otherwise use `associated with` or `indicates`.
- Define abbreviations at first use and prefer full terms in headings.
