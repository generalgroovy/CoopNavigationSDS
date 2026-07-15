# Thesis Result Inference Guide

This document translates the finalized result snapshot into thesis-safe claims. It separates direct evidence from cautious interpretation.

## Current Evidence Base

- Active thesis-relevant deduplicated rows: `1319`.
- Completed rows: `557`.
- Fully crossed matched rows: `220`.
- Fully crossed condition groups: `11`.
- Raw evidence remains in per-run folders; this document is derived.

## Strongest Defensible Claims

1. Phase-wise automatic evaluation is feasible in this controlled route-dialogue task.
   - The result folders contain turn evidence, task state, route validation, metric inputs, and phase metrics.
   - This supports retrospective analysis without rerunning dialogues.
2. Final task success alone is insufficient.
   - Semi-successful runs and unsuccessful completed runs need route-validity, constraint, repair, ASR/NLU, and grounding evidence to be interpreted.
3. Speech-channel degradation is visible in paired comparisons.
   - In the fully crossed subset, text controls are at ceiling while audio variants lose success rate.
4. Direct model ranking must be cautious.
   - Model comparisons are strongest only in matched subsets.
   - Broader completed counts mix coverage, runtime feasibility, scenario pressure, and backend behavior.
5. Metric-outcome correlations are useful diagnostic evidence, not causal proof.
   - Use the strongest phase metrics to explain likely failure origin and cooperation quality.

## Model A Interpretation

- UserLM is the main thesis caller because it is the intended user-simulation model.
- TinyLlama-Agent-A is useful as a control stratum.
- Do not merge UserLM and TinyLlama rows for headline claims unless explicitly using the fully crossed subset.
- If Agent A changes, caller behavior, constraint revelation, repair behavior, and closure behavior also change; those are not merely random variation.

## Model B Interpretation

- The five selected Agent B backends cover small, medium, and large Transformer-style local models.
- The current evidence does not justify a simple 'larger is always better' conclusion.
- Qwen2.5 1.5B often has a strong practical profile because it combines high completion volume with high completed-run success.
- Qwen2.5 7B is valuable for size comparison but runtime cost and coverage must be reported beside success.
- TinyLlama and Qwen2.5 0.5B are useful small-model baselines; similar outcomes can indicate task/pipeline limitations rather than model-specific behavior.

## Scenario, Persona, and Audio Persona Interpretation

- Scenarios represent controlled task pressure: route complexity, constraint pressure, and dialogue-stage difficulty.
- Personas change interaction behavior and constraint priorities; they are part of the experimental condition, not noise.
- Audio personas and speech performance bands act as channel stressors. They are expected to create a performance range from ceiling to failure.
- Analyze these factors as pressure conditions unless the subset is explicitly balanced for causal comparison.

## TTS and ASR Interpretation

- Text/audio pairing is currently the strongest channel-level evidence.
- TTS framework comparison is limited if Piper dominates matched conditions.
- ASR engine and search width can be discussed as descriptive associations; claim comparative ASR effects only for matched ASR-only subsets.
- ASR WER should be interpreted alongside station F1, slot accuracy, and route-state outcomes because route dialogue is entity-sensitive.

## Thesis Conclusion Skeleton

The experiment supports the thesis that a controlled spoken route-dialogue framework can produce useful automatic evaluation evidence across SDS phases. The most valid conclusions are about the evaluation method, phase-wise diagnosis, text/audio degradation, and metric usefulness. Claims about model superiority, TTS superiority, or ASR superiority require matched subsets and should be worded cautiously.
