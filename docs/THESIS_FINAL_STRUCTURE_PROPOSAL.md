# Final Thesis Structure Proposal

Purpose: adapt the passed `BA_Propose_2026` LaTeX project into a compact
writing outline for the bachelor thesis on automatic evaluation of spoken
dialogue systems. The goal is not to preserve every historical planning
chapter, but to create a structure that is easy to write, easy to supervise,
and faithful to the implemented CoopNavigationSDS experiment.

## Source Structure Analysis

The passed zip contains two useful layers:

- `bachelor_thesis_ms_structure_only/`:
  - clean `report`-style LaTeX master file,
  - front matter,
  - modular chapter files,
  - bibliography file,
  - appendix file.
- `rest/`:
  - broad planning notes,
  - expanded metric lists,
  - literature notes,
  - earlier high-detail chapter ideas.

Use the first layer as the structural target. Use the second layer only as
source material. Do not copy the full `rest/` structure into the thesis because
it creates too many chapters and repeats background, evaluation theory,
limitations, methods, metrics, results, and discussion.

## Main Restructuring Decision

The thesis should use a compact XW-style `report` structure with logically
combined chapters:

```text
Front matter
1 Introduction
2 Background, Related Work, and Evaluation Foundations
3 Methodology and Experiment Design
4 Evaluation, Results, and Metric Analysis
5 Discussion, Conclusion, and Future Work
Bibliography
Appendix
```

This is the recommended final structure.

Rationale:

- It keeps the thesis short enough for a bachelor thesis.
- It avoids a separate standalone "Evaluation of Dialogue Systems" chapter
  that would repeat background and metric sections.
- It avoids a separate "Limitations" chapter; limitations belong in method
  validity notes and final discussion.
- It avoids separating results and discussion unless the results chapter grows
  too large.
- It keeps all empirical analysis in one place, which makes the argument easier
  to follow.

Fallback if Chapter 4 becomes too long:

```text
1 Introduction
2 Background, Related Work, and Evaluation Foundations
3 Methodology and Experiment Design
4 Results and Metric Analysis
5 Discussion
6 Conclusion and Future Work
```

Use the fallback only if the results tables and interpretation exceed a
manageable chapter length.

## File Layout To Use

Recommended project layout:

```text
bachelor_thesis_ms.tex
bibliography.bib
frontmatter/
  00_title_page.tex
  01_abstract.tex
  02_german_summary.tex
chapters/
  01_introduction.tex
  02_background_related_work_evaluation_foundations.tex
  03_methodology_experiment_design.tex
  04_evaluation_results_metric_analysis.tex
  05_discussion_conclusion_future_work.tex
appendix/
  07_appendix.tex
```

This keeps the clean LaTeX structure from the passed zip, but removes
unnecessary chapter fragmentation.

## Mapping From Passed Zip To Final Structure

| Source material | Final destination | Treatment |
| --- | --- | --- |
| `01_introduction.tex` | Chapter 1 | Keep, tighten wording, add concise contribution paragraph. |
| `02_theoretical_background_and_related_work.tex` | Chapter 2 | Keep as base, merge evaluation foundations into it. |
| `rest/main.tex` background/evaluation/related-work sections | Chapter 2 | Use selectively; avoid duplicating paragraphs. |
| `rest/literature.tex` | Chapter 2 and references companion | Use for citation placement, not as thesis prose. |
| `rest/metrics.tex` | Chapters 3, 4, and appendix | Keep only central metric selection in thesis; full catalog goes to appendix. |
| `03_methodology.tex` | Chapter 3 | Expand with current CoopNavigationSDS design and evidence logging. |
| `04_evaluation_and_results.tex` | Chapter 4 | Combine metric definitions, coverage, results, metric-outcome relations, and failure localization. |
| Separate discussion/limitations notes | Chapter 5 | Integrate as interpretation, limitations, and RQ answers. |
| `05_conclusion.tex` and `06_future_work.tex` | Chapter 5 | Merge into one final chapter for a compact thesis. |
| `07_appendix.tex` | Appendix | Store schemas, prompts, full metric catalog, full condition tables, and reproducibility notes. |

## Chapter 1: Introduction

Function:

- Motivate the problem.
- State the research gap.
- Define the research questions.
- State the contribution.
- Give the thesis structure.

Keep:

- Motivation for spoken dialogue systems:
  - hands-free interaction,
  - accessibility,
  - natural interaction,
  - route information as practical task.
- Evaluation difficulty:
  - multi-phase errors,
  - ASR/NLU/state/policy/NLG/TTS dependencies,
  - final task success alone does not explain failure.
- Research questions:
  - phase-wise evidence,
  - metric-outcome relation,
  - earliest likely failing phase,
  - robustness across Agent B backends,
  - speech versus text channel effect.

Tighten:

- Replace broad AI-growth motivation with a concrete SDS evaluation problem.
- Avoid long metric lists.
- Avoid implementation details.
- Use "spoken dialogue system" consistently.

Suggested chapter structure:

```text
1.1 Motivation
1.2 Evaluation Problem
1.3 Research Gap
1.4 Research Questions
1.5 Contributions
1.6 Thesis Structure
```

Writing target:

- 4-6 pages.
- No tables except possibly a short RQ table.

## Chapter 2: Background, Related Work, and Evaluation Foundations

Function:

- Give the reader enough theory to understand the experiment.
- Combine the old background, related work, and evaluation-theory chapters.
- Establish why the selected metrics are needed without presenting all formulas.

Include:

- Dialogue systems and task-oriented dialogue.
- Spoken dialogue systems and the phase pipeline:
  - TTS,
  - ASR,
  - NLU,
  - dialogue state,
  - dialogue management,
  - backend grounding,
  - NLG,
  - whole dialogue outcome.
- LLM-based dialogue systems:
  - flexible generation,
  - hidden internal state,
  - need for external evidence.
- Error propagation:
  - speech errors can cause semantic and task errors downstream.
- Automatic evaluation:
  - task success,
  - dialogue cost,
  - text metrics,
  - speech metrics,
  - task-grounded metrics,
  - construct validity.
- Related work strands:
  - PARADISE and SDS evaluation,
  - task-oriented dialogue benchmarks,
  - spoken task-oriented dialogue and SLU,
  - user simulation,
  - navigation/grounded dialogue,
  - LLM-agent evaluation.

Combine:

- The old "Theoretical Background" chapter.
- The old "Evaluation of Dialogue Systems" chapter.
- The old "Related Work" chapter.
- The conceptual part of "Limitations" where it concerns metric validity.

Move to appendix:

- Long metric catalogs.
- Exhaustive lists of all possible metrics.
- Detailed provider/model tables.

Suggested chapter structure:

```text
2.1 Task-Oriented and Spoken Dialogue Systems
2.2 Phase Model of Spoken Dialogue Systems
2.3 LLM-Based Dialogue Backends
2.4 Error Propagation and Phase Evidence
2.5 Automatic Evaluation and Construct Validity
2.6 Related Work and Research Gap
```

Writing target:

- 8-12 pages.
- Citation-heavy but not a literature dump.
- Each subsection should end by explaining why it matters for this thesis.

## Chapter 3: Methodology and Experiment Design

Function:

- Explain exactly how CoopNavigationSDS operationalizes the research problem.
- Make the experiment reproducible and valid.
- Do not present results yet.

Include:

- Research design:
  - controlled route-dialogue experiment,
  - retrospective metric calculation,
  - matched text/audio comparisons.
- Experimental unit:
  - one run/condition,
  - nested turns as evidence,
  - completed versus execution-incomplete runs.
- Agent roles:
  - Agent A as simulated caller,
  - UserLM as primary caller,
  - TinyLlama as caller-control stratum,
  - Agent B as evaluated route-information system.
- Knowledge boundaries:
  - no shared hidden memory,
  - each agent acts on what it intended, said, heard, and understood.
- Task and network:
  - route validity,
  - optimal route under revealed constraints,
  - station/line entities,
  - shortest valid route objective.
- Conditions:
  - Agent A,
  - Agent B backend,
  - scenario,
  - persona,
  - audio persona,
  - run type,
  - TTS/ASR,
  - speech pattern,
  - seed/repetition.
- Evidence logging:
  - intended utterance,
  - TTS speech,
  - ASR raw transcript,
  - normalized understanding,
  - memory update,
  - route candidate,
  - validation result,
  - timing,
  - outcome.
- Validity controls:
  - paired text/audio conditions,
  - deduplication,
  - exclusion of unstable `large2`,
  - no silent fallback,
  - raw evidence preserved.

Combine:

- Methodology notes.
- Metric evidence-capture design.
- Missing-data handling.
- Reproducibility design.

Move to Chapter 4:

- Actual metric values.
- Success rates.
- model comparisons.

Move to appendix:

- Full prompt templates.
- Full configuration schemas.
- Full route schemas.
- Full batch-job listings.

Suggested chapter structure:

```text
3.1 Research Design
3.2 Experimental Unit and Condition Factors
3.3 Agent Roles and Knowledge Boundaries
3.4 Route Task, Network, and Constraint Layers
3.5 Text-Only and Spoken Conditions
3.6 Evidence Logging and Retrospective Metric Calculation
3.7 Validity Controls and Exclusion Rules
```

Writing target:

- 8-10 pages.
- Most important chapter for experiment integrity.

## Chapter 4: Evaluation, Results, and Metric Analysis

Function:

- Define the reported metrics.
- Present coverage and results.
- Analyze which metrics explain success, semi-success, and failure.

Include in this order:

1. Metric selection principle.
2. Metric groups and formulas needed for reported results.
3. Run inventory and coverage.
4. Completed-run outcomes:
   - success,
   - semi-success,
   - unsuccessful completed dialogue,
   - execution incomplete separately.
5. Text versus speech comparison.
6. Agent B model comparison.
7. Agent A comparison where matched.
8. Scenario/persona/audio-persona effects.
9. Phase-wise metric patterns.
10. Metric-outcome associations.
11. Failure-localization examples.

Use current result evidence:

- Active thesis-relevant deduplicated rows: 1319.
- Completed active thesis rows: 557.
- Fully crossed subset: 11 condition groups and 220 runs.
- Fully crossed text controls are at ceiling.
- Audio variants reduce task success by about 18.2 to 27.3 percentage points.
- Qwen2.5 1.5B has the strongest broad completed-row profile.
- Qwen2.5 7B performs well when completed but is more runtime-sensitive.
- Severe-channel/floor conditions provide useful failure pressure.
- Clean/nominal conditions are ceiling-like.

Combine:

- Metric selection.
- Results.
- Phase-wise analysis.
- Model-backend comparison.
- Failure localization.

Move to Chapter 5:

- Final interpretation and implications.
- Limitations.
- direct RQ answers.

Move to appendix:

- All per-run rows.
- Full metric table.
- Full generated CSV outputs.
- Long failure traces.

Suggested chapter structure:

```text
4.1 Metric Selection and Calculation Scope
4.2 Coverage and Completed-Run Inventory
4.3 Outcome Results
4.4 Text-Only Versus Speech Runs
4.5 Agent A and Agent B Comparisons
4.6 Phase-Wise Metric Results
4.7 Metric-Outcome Relations
4.8 Failure Localization and Representative Cases
```

Writing target:

- 10-14 pages.
- This is the empirical core.
- Every percentage must name its denominator.

## Chapter 5: Discussion, Conclusion, and Future Work

Function:

- Answer the research questions.
- Interpret the contribution.
- State limitations.
- Give concrete future work.

Combine:

- Discussion.
- Conclusion.
- Future work.
- Limitations that are not already handled in methodology.

Reason:

- A separate discussion, conclusion, and future-work sequence is useful in a
  long thesis, but for this bachelor thesis it risks repetition.
- The final chapter should be short, direct, and tied to the research
  questions.

Suggested chapter structure:

```text
5.1 Answers to the Research Questions
5.2 Interpretation of the Main Findings
5.3 Validity Boundaries and Limitations
5.4 Contribution
5.5 Future Work
5.6 Final Statement
```

Safe conclusions:

- Phase-wise automatic evaluation is feasible for the controlled route-dialogue
  setting.
- Final success is necessary but insufficient.
- Task-grounded and semantic phase metrics are stronger indicators than
  generic text metrics.
- Speech-channel degradation is visible in matched text/audio comparisons.
- Model ranking must remain cautious and matched-condition based.

Unsafe conclusions:

- Automatic metrics replace human evaluation.
- One Agent B model is universally best.
- Larger models always perform better.
- The simulated caller fully represents human callers.
- Metric correlations prove causal failure origin.

Writing target:

- 4-6 pages.
- No new results.
- No new literature review.

## Appendix

Use the appendix to keep the thesis readable.

Include:

- Full metric catalog.
- Metric dependency table.
- Prompt templates.
- Configuration schema.
- Route schema.
- Network overview.
- Example transcripts.
- Reproducibility commands.
- Exclusion and archive policy.
- Generated result artifact references.

Do not include:

- Material needed to understand the main argument.
- Repeated versions of tables already summarized in Chapter 4.

## Streamlined Writing Process

Write in this order, not in final chapter order:

1. Methodology:
   - easiest to write from the project,
   - fixes terminology and condition definitions.
2. Results:
   - write coverage and outcome tables first,
   - then metric interpretation.
3. Background:
   - add only what the methodology/results require.
4. Introduction:
   - write after the argument is clear.
5. Discussion and conclusion:
   - answer research questions using Chapter 4 evidence.
6. Abstract and German summary:
   - write last.

Practical rule:

- If a paragraph does not support a research question, a method choice, a
  metric interpretation, or a result claim, remove it or move it to the
  appendix.

## Minimal Thesis Argument Flow

Use this as the internal logic of the final thesis:

```text
Spoken task-oriented dialogue systems can fail at multiple phases.
Final task success alone does not explain these failures.
A controlled route task provides objective task grounding.
CoopNavigationSDS logs phase evidence for each run.
Metrics are calculated retrospectively from this evidence.
Completed runs can be classified as successful, semi-successful, or unsuccessful.
Matched text/audio runs reveal speech-channel effects.
Phase metrics help identify likely failure origins.
The framework is useful, but claims remain bounded to controlled conditions.
```

## Chapter Combination Summary

| Earlier planning chapter | Final location | Reason |
| --- | --- | --- |
| Theoretical Background | Chapter 2 | Core concepts and literature belong together. |
| Evaluation of Dialogue Systems | Chapter 2 and Chapter 4 | General theory in Chapter 2; reported metrics and values in Chapter 4. |
| Related Work | Chapter 2 | Related work should support concepts, not become a separate catalog. |
| Limitations | Chapter 3 and Chapter 5 | Methodological limits near design; interpretive limits in conclusion. |
| Methodology | Chapter 3 | Keep as independent reproducibility chapter. |
| Metric Selection | Chapter 4 | Present only metrics that are reported; full catalog in appendix. |
| Results | Chapter 4 | Empirical core. |
| Discussion | Chapter 5 | Interpret results by research question. |
| Conclusion | Chapter 5 | Direct final answers. |
| Future Work | Chapter 5 | Short and limitation-driven. |

## Immediate Edits To The LaTeX Project

Rename or create chapter files:

```text
chapters/01_introduction.tex
chapters/02_background_related_work_evaluation_foundations.tex
chapters/03_methodology_experiment_design.tex
chapters/04_evaluation_results_metric_analysis.tex
chapters/05_discussion_conclusion_future_work.tex
```

Update the master file:

```latex
\input{chapters/01_introduction}
\input{chapters/02_background_related_work_evaluation_foundations}
\input{chapters/03_methodology_experiment_design}
\input{chapters/04_evaluation_results_metric_analysis}
\input{chapters/05_discussion_conclusion_future_work}

\appendix
\input{appendix/07_appendix}
```

Keep the separate files from the passed zip as drafting references, but use the
merged files above for the final thesis.
