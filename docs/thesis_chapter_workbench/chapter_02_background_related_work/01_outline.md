# Chapter 2: Background and Related Work - Outline

Purpose: in-depth bullet outline for drafting thesis prose. Use this file for structure and argument order.

## Chapter Function

- Define the technical and theoretical context: dialogue systems, spoken pipelines, task-oriented structure, LLM backends, and related evaluation traditions.

## Core Argument

- Task-oriented dialogue provides structured goals and state.
- Spoken dialogue adds a speech pipeline where errors can propagate.
- LLM-based systems change implementation details but do not remove the need for task grounding.
- Related work motivates the thesis design but does not already solve phase-wise automatic evaluation for this controlled route task.

## Subchapter Writing Plan

### 2.1 Dialogue systems and SDS

- Purpose: Introduce dialogue systems before narrowing to speech.
- Points to write:
  - Define user, system, turn, dialogue state, system action, and response.
  - Explain the classical SDS pipeline: user speech, ASR, NLU, dialogue state, dialogue management, NLG, TTS.
  - Use the pipeline as an analysis model, even when some components are implemented with LLMs.

### 2.2 Task-oriented dialogue

- Purpose: Show why this task can be evaluated automatically.
- Points to write:
  - Task-oriented systems have goals, constraints, and external task state.
  - For navigation, the task state is the route network and constraint set.
  - Slot-like variables include start station, destination, time, line names, route duration, transfer count, crowding, and delay risk.

### 2.3 Spoken versus text dialogue

- Purpose: Clarify why paired text/audio runs matter.
- Points to write:
  - Text runs test dialogue and route reasoning without speech degradation.
  - Audio runs test the additional TTS/ASR transmission channel.
  - A paired design is stronger than comparing unrelated text and audio conditions.

### 2.4 LLM-based dialogue systems

- Purpose: Position Agent B backends without turning the thesis into a model leaderboard.
- Points to write:
  - LLMs can produce flexible responses but may still hallucinate routes or ignore constraints.
  - Backend size is only one dimension; family, instruction tuning, provider, prompt, and decoding also matter.
  - The thesis compares selected feasible backends under controlled conditions.

### 2.5 Error propagation

- Purpose: Prepare the logic for phase-wise metrics.
- Points to write:
  - Speech errors can corrupt entity recognition.
  - Entity errors can corrupt dialogue state.
  - State errors can lead to invalid or constraint-violating routes.
  - Later metrics may detect a failure, but earlier evidence is needed to localize it.

## Minimum Quality Checklist

- Every claim has a clear denominator, source, or citation.
- The chapter distinguishes task outcome, phase evidence, and interpretation.
- The wording stays cautious where the evidence is descriptive rather than causal.
- No raw implementation detail appears unless it supports a methodological point.
