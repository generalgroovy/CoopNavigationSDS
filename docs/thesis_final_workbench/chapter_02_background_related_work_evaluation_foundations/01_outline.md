# Chapter 2: Background, Related Work, and Evaluation Foundations - Outline

Use this as the direct writing plan for the final compact thesis structure.

## Function

- Combine theory, related work, and general evaluation validity.
- Explain only the concepts needed to understand the experiment and metrics.
- Avoid a separate literature catalog chapter.

## Sections

### 2.1 Task-Oriented and Spoken Dialogue Systems

- define dialogue system, SDS, TOD, dialogue state, system action
- explain why navigation is evaluable

### 2.2 Phase Model

- ASR, NLU, dialogue state, dialogue management, backend grounding, NLG, TTS
- pipeline phases are also evidence boundaries

### 2.3 LLM-Based Dialogue Backends

- LLMs blur internal module boundaries
- external evidence remains necessary for evaluation

### 2.4 Error Propagation

- early speech errors can cause downstream task errors
- failure localization is diagnostic, not causal proof

### 2.5 Automatic Evaluation Foundations

- human versus automatic evaluation
- lexical and semantic text metrics
- speech metrics
- task-grounded metrics
- construct validity

### 2.6 Related Work Gap

- classical SDS evaluation
- TOD benchmarks
- spoken TOD
- user simulation
- navigation and grounded dialogue
- LLM-agent evaluation

## Keep Out

- exhaustive metric catalogs;
- raw per-run tables;
- long prompt templates;
- implementation details that do not support the chapter argument.
