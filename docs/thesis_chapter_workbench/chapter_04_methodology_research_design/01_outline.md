# Chapter 4: Methodology and Research Design - Outline

Purpose: in-depth bullet outline for drafting thesis prose. Use this file for structure and argument order.

## Chapter Function

- Describe the experiment precisely enough that a reader can reproduce the logic without reading the code.

## Core Argument

- The experiment is a controlled route-dialogue task with two agents and explicit knowledge boundaries.
- Agent A has private goals and staged constraints; Agent B must cooperate by proposing and refining routes.
- Text-only and speech runs separate dialogue reasoning from speech-channel degradation.
- All metric-relevant evidence is captured during execution and evaluated retrospectively.

## Subchapter Writing Plan

### 4.1 Experimental unit and condition

- Purpose: Define what one row of analysis means.
- Points to write:
  - One condition/run combines Agent A, Agent B, scenario, persona, audio persona, TTS/ASR settings, seed/repetition, and run type.
  - The run, not the individual turn, is the primary unit for outcome rates.
  - Turns provide nested evidence for diagnosis.

### 4.2 Agent roles and knowledge boundaries

- Purpose: Protect the experiment from hidden shared knowledge.
- Points to write:
  - Agent A knows start, destination, time, station/line names, and private constraints as staged goals.
  - Agent B knows the network and must infer Agent A's needs from what it hears.
  - Each agent maintains its own memory based on its own intended speech and understood transcript.
  - Clarification and repair must occur through dialogue, not shared hidden state.

### 4.3 Dialogue stages

- Purpose: Explain the staged task logic.
- Points to write:
  - Stage 1: establish valid route from start to destination within acceptable time.
  - Stage 2: reveal first additional constraint if Stage 1 is satisfied.
  - Stage 3: reveal second constraint if Stage 2 is satisfied.
  - Final: Agent A accepts the best route if goals are satisfied or classifies the result as semi-success/unsuccessful.

### 4.4 Route task and network

- Purpose: Describe why the network is useful for evaluation.
- Points to write:
  - Routes must specify station sequence, line names, and transport segments.
  - Optimal route is recalculated per constraint layer.
  - Constraints are designed to change the optimal route where possible.
  - This makes cooperation and route revision observable.

### 4.5 Text and speech channel

- Purpose: Describe paired controls.
- Points to write:
  - Text-only runs preserve the dialogue policy without TTS/ASR degradation.
  - Audio-variant runs transmit utterances through TTS and ASR.
  - The paired design supports estimating the performance loss caused by the speech channel.

### 4.6 Evidence logging

- Purpose: Show that retrospective metrics are legitimate.
- Points to write:
  - Log intended utterance, TTS output, ASR raw transcript, normalized understanding, corrections, memory update, route proposal, validation, timing, and outcome.
  - Do not aggregate away raw evidence during runtime.
  - Derived metrics must be traceable back to fields in the run folder.

## Minimum Quality Checklist

- Every claim has a clear denominator, source, or citation.
- The chapter distinguishes task outcome, phase evidence, and interpretation.
- The wording stays cautious where the evidence is descriptive rather than causal.
- No raw implementation detail appears unless it supports a methodological point.
