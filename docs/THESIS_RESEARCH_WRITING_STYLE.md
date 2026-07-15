# Research Writing and Citation Style

Purpose: style guide for writing the bachelor thesis in a precise,
research-grade way. The main thesis aid should describe chapter structure,
technical/theoretical content, reasoning, and validity. This document describes
how to write that content and how to use citations.

## Core Writing Principle

- Write every paragraph as:
  - claim,
  - evidence,
  - interpretation,
  - limitation.
- Prefer short, direct claims.
- Avoid decorative prose.
- Make denominators explicit before presenting rates.
- Distinguish:
  - execution completion,
  - dialogue completion,
  - route validity,
  - constraint satisfaction,
  - final task success.

Good paragraph shape:

- claim:
  - one sentence saying what the subsection shows;
- evidence:
  - one or two numbers, table references, or metric definitions;
- interpretation:
  - what the evidence means for the research question;
- limitation:
  - what cannot be concluded from this evidence.

Example:

```text
Completed UserLM-Agent-A dialogues show high route validity in the current
result set. This indicates that the framework can produce grounded route
dialogues under controlled conditions. However, this rate applies only to
execution-complete runs and does not include unavailable or interrupted
conditions.
```

## Citation Style

Use citations only when they support a specific claim.

Citation use rules:

- cite a paper where the claim first becomes methodologically important;
- do not place a citation after a generic sentence that the paper does not
  specifically support;
- separate background citations from method-justification citations;
- cite the same foundational paper consistently instead of scattering many
  weak references;
- do not cite a paper only because it is famous;
- never use a citation to hide an unclear claim.

Recommended citation pattern:

```text
[Claim about evaluation problem]. [Citation]. In this thesis, that problem is
handled by [specific design decision].
```

Example:

```text
Prior SDS evaluation connects task success with dialogue cost and user-facing
quality. In this thesis, that idea motivates reporting route success together
with turn count, repair behavior, latency, and phase-wise evidence.
```

## Citation Placement by Function

- Background claim:
  - cite when defining a field concept.
  - Example:
    - task-oriented dialogue uses goals, slots, and state.
- Metric claim:
  - cite when introducing a metric family or formula.
  - Example:
    - BLEU as lexical overlap;
    - BERTScore as contextual semantic similarity.
- Method claim:
  - cite when a design choice follows from prior work.
  - Example:
    - phase-wise speech logging because spoken TOD differs from clean text.
- Result interpretation:
  - cite only when connecting observed results back to known evaluation theory.
  - Example:
    - task success alone is insufficient because interaction cost matters.

## Preferred Wording Patterns

Metric definition:

```text
This metric measures [construct] using [logged evidence].
It is calculated as [formula].
High/low values indicate [interpretation].
It is unavailable when [missing-data rule].
Its limitation is [scope boundary].
```

Outcome wording:

```text
This result is calculated over [denominator]. It shows [finding]. It supports
[research question] because [reason]. It should be interpreted cautiously
because [validity boundary].
```

Model comparison wording:

```text
For matched conditions, [model/backend] differed from [comparison] in
[metric/outcome]. This comparison controls [listed non-model factors] but does
not isolate [confounded model properties].
```

Failure wording:

```text
The earliest observable failure signal occurred in [phase]. The logged evidence
was [evidence]. This suggests [diagnostic interpretation], but downstream
effects require manual case inspection for causal confirmation.
```

Speech-channel wording:

```text
The paired audio run differs from the text-only run only in the speech channel.
Therefore, changes in outcome or repair burden are interpreted as speech-
pipeline effects, while exact causal attribution requires inspection of TTS,
ASR, transcript normalization, and NLU evidence.
```

Matched-condition wording:

```text
Matched-condition comparisons are the strongest model comparisons in this
thesis because scenario, persona, audio persona, TTS, ASR, seed, objective, and
run type are held constant. They are narrower than all-run summaries but more
internally valid.
```

## Cautious Language

Use:

- "is associated with"
- "indicates"
- "suggests"
- "is consistent with"
- "diagnostic evidence points to"
- "within the tested conditions"
- "in the current active result set"
- "for matched conditions"
- "among execution-complete runs"

Avoid unless manually verified:

- "proves"
- "caused"
- "always"
- "the model understands"
- "the ASR caused the failure"
- "large models are better"
- "small models are worse"
- "speech failed"
- "human-like"
- "robust" without specifying tested conditions

Rewrite examples:

- Weak:
  - "The model understands the route."
- Better:
  - "The normalized transcript and route validator show that the system
    preserved the required station and line entities for this route."

- Weak:
  - "ASR caused the failure."
- Better:
  - "The earliest observable failure signal was an ASR station substitution,
    followed by incorrect NLU route state; this suggests an ASR-linked failure
    path."

- Weak:
  - "Qwen2.5 1.5B is the best model."
- Better:
  - "Qwen2.5 1.5B has the strongest current UserLM active-scope success count,
    but this does not isolate model size from family, runtime, or provider
    effects."

## What to Keep

- short paragraphs;
- phase-wise terminology;
- exact metric definitions;
- explicit limitations;
- condition-level analysis;
- matched-condition comparisons;
- denominator-first reporting;
- clear distinction between task outcome and dialogue process.

## What to Remove

- repeated lists of every possible metric;
- implementation details before the methodology chapter;
- broad claims about all SDS;
- claims about model size when model family/provider also differs;
- unexplained acronyms;
- metric names without formulas or evidence sources;
- duplicated caveats;
- broad "AI" wording when "Agent B backend" is meant.

Also remove or rewrite:

- phrases like "the model understands" unless supported by NLU/state evidence;
- phrases like "ASR caused the failure" unless transcript and downstream state
  show the link;
- generic praise such as "robust" without specifying tested conditions;
- implementation detail that does not affect experiment validity or metric
  calculation.

## Table and Figure Writing

- Every table caption should state:
  - denominator,
  - unit of analysis,
  - whether archived runs are excluded,
  - whether rows are matched or unmatched,
  - whether values are means, ranges, or counts.
- Every figure caption should state:
  - what is compared,
  - which conditions are included,
  - what color or axis values mean,
  - whether missing values are excluded or shown.

Example table caption:

```text
Active paired run outcomes by Agent A and Agent B model. Rows exclude archived
duplicate/noncanonical runs and include only paired text/audio conditions.
```

Example figure caption:

```text
Phase-wise metric heatmap for execution-complete active runs. Darker color
indicates stronger deviation within the observed metric range; missing values
are shown separately and are not treated as zero.
```

## Citation Strands

- PARADISE:
  - use for the broad idea that task outcome and dialogue cost both matter;
  - do not imply that the thesis estimates a full satisfaction model.
- BLEU/ROUGE/METEOR:
  - use for lexical text metrics;
  - connect to why surface overlap is insufficient for route validity.
- BERTScore/BLEURT/MoverScore/MAUVE:
  - use for semantic or learned text metrics;
  - connect to why semantic text similarity still does not validate route
    executability.
- MultiWOZ and Schema-Guided Dialogue:
  - use for task-oriented dialogue concepts such as goal, slot, state, and
    constraint;
  - contrast with this thesis's spoken and grounded route task.
- SpokenWOZ and SLUE:
  - use for the need to evaluate spoken inputs and ASR-induced differences;
  - connect to raw transcript, normalized understanding, and entity
    preservation.
- NISQA/DNSMOS/STOI:
  - use for no-reference or objective speech-quality background;
  - avoid claiming these metrics alone measure task success.
- User-simulation work:
  - use for repeatable simulated callers;
  - state external-validity limits clearly.

## Final Style Checklist

- Does the paragraph make one claim?
- Is the denominator explicit?
- Is the evidence source named?
- Is the interpretation tied to a research question?
- Is the limitation stated?
- Are Agent A and Agent B roles clear?
- Are text-only, audio, matched, and active-scope rows distinguished?
- Are execution failures separated from task failures?
- Are citations attached to precise claims?
- Could the statement survive a supervisor asking "what exactly proves this?"
