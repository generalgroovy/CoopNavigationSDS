# Chapter 7: Discussion and Conclusion - Outline

Purpose: in-depth bullet outline for drafting thesis prose. Use this file for structure and argument order.

## Chapter Function

- Answer research questions, state supported inferences, discuss limitations, and close with realistic future work.

## Core Argument

- The thesis can support a methodological claim: phase-wise automatic evaluation is feasible and informative in a controlled route-dialogue SDS.
- The evidence supports a speech-channel degradation claim in matched text/audio comparisons.
- The evidence supports metric usefulness claims, especially for task-grounded and semantic phase metrics.
- The evidence does not support universal model-size conclusions or replacement of human evaluation.

## Subchapter Writing Plan

### 7.1 Answer research questions

- Purpose: Answer each RQ directly.
- Points to write:
  - RQ1: Yes, the framework produces analyzable phase evidence for completed runs.
  - RQ2: Task-grounded and semantic phase metrics are strongest; generic lexical metrics are supplementary.
  - RQ3: Speech variants reduce success relative to paired text controls in the fully crossed subset.
  - RQ4: Backend effects are observable but must be interpreted within matched coverage and runtime feasibility.
  - RQ5: Failure-localization candidates can be derived, but remain diagnostic rather than causal.

### 7.2 Interpret successful, semi-successful, and unsuccessful runs

- Purpose: Use outcome categories as analytical structure.
- Points to write:
  - Successful runs show that the full pipeline can preserve enough task information for correct route completion.
  - Semi-successful runs are especially valuable because they reveal partial route competence with constraint failure.
  - Unsuccessful completed runs reveal where speech, understanding, state, or route reasoning broke down.

### 7.3 Discuss model and configuration effects

- Purpose: Avoid overclaiming while still drawing useful conclusions.
- Points to write:
  - Qwen2.5 1.5B appears practically strong in current broad rows, but model family and condition coverage also matter.
  - Very small models remain useful baselines because they show lower-resource behavior.
  - Large models increase runtime/resource cost and should be justified by matched-condition improvement.
  - Speech pressure settings are methodologically useful because they create non-ceiling cases.

### 7.4 Limitations

- Purpose: Separate methodological limits from implementation limits.
- Points to write:
  - Simulated Agent A is controlled but not a human caller.
  - Synthetic network supports validation but limits ecological realism.
  - Current TTS/ASR provider coverage is not a universal speech-technology benchmark.
  - Metric correlations are descriptive and should not be interpreted as causal predictors without additional ablation.

### 7.5 Final conclusion

- Purpose: Close with the strongest defensible thesis claim.
- Points to write:
  - A controlled navigation task can provide objective task grounding for SDS evaluation.
  - Phase-wise logging makes automatic metrics more interpretable than final success alone.
  - The most useful metrics are those tied to task entities, route validity, constraint satisfaction, repair, and grounded proposal behavior.
  - Future work should add human validation, real microphone input, broader TTS/ASR variation, and real transit data.

## Minimum Quality Checklist

- Every claim has a clear denominator, source, or citation.
- The chapter distinguishes task outcome, phase evidence, and interpretation.
- The wording stays cautious where the evidence is descriptive rather than causal.
- No raw implementation detail appears unless it supports a methodological point.
