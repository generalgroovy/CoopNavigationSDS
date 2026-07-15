# Chapter 3: Methodology and Experiment Design - Outline

Use this as the direct writing plan for the final compact thesis structure.

## Function

- Describe CoopNavigationSDS precisely enough to reproduce the experiment logic.
- State condition factors, knowledge boundaries, logging, and validity controls.
- Do not interpret results in this chapter.

## Sections

### 3.1 Research Design

- controlled cooperative route-dialogue task
- retrospective metric calculation
- matched text/audio comparisons

### 3.2 Experimental Unit

- one run/condition is the main analysis row
- turns are nested diagnostic evidence
- completed and execution-incomplete runs are separated

### 3.3 Agent Roles

- Agent A as simulated caller
- UserLM as primary caller
- TinyLlama as caller-control
- Agent B as evaluated route-information system

### 3.4 Route Task

- start, destination, time, station and line entities
- shortest valid route under progressively revealed constraints
- constraint-layer optimal routes

### 3.5 Speech and Text Conditions

- text-only control
- audio variant through TTS and ASR
- audio personas and speech patterns as pressure conditions

### 3.6 Evidence Logging

- intended utterance
- TTS speech
- ASR raw transcript
- normalized understanding
- memory update
- route proposal
- validation
- timing
- outcome

### 3.7 Validity Controls

- no hidden shared memory
- paired conditions
- deduplication
- large2 exclusion
- raw evidence preserved

## Keep Out

- exhaustive metric catalogs;
- raw per-run tables;
- long prompt templates;
- implementation details that do not support the chapter argument.
