# Chapter 2: Background and Related Work - Terminology

Purpose: chapter-local definitions and explanations. Define terms before using them in thesis prose.

## Dialogue state

- Definition: The system's current structured representation of the task and conversation.
- Explanation:
  - In this thesis it includes known trip details, revealed constraints, current route, and satisfaction status.

## Dialogue management

- Definition: The phase that decides the next system action.
- Explanation:
  - Examples: propose route, ask clarification, repair misunderstanding, accept final route.

## Large language model backend

- Definition: A concrete LLM/model provider used to generate Agent B behavior.
- Explanation:
  - Treat it as an implementation of the evaluated system role, not as the only explanation of performance.

## Error propagation

- Definition: The process by which an error in one phase affects later phases.
- Explanation:
  - For example, ASR losing a station name can produce invalid route reasoning.

## Term-Use Rules

- Use one term for one construct; avoid alternating synonyms for key variables.
- Do not use `success`, `route validity`, and `constraint satisfaction` interchangeably.
- Reserve `causal` for controlled comparisons or ablations; otherwise use `associated with` or `indicates`.
- Define abbreviations at first use and prefer full terms in headings.
