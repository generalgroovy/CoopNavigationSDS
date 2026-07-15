# Chapter 5: Metric Selection - References

Purpose: citable papers, concrete claims they support, and boundaries for safe use.

## Walker et al. (1997), PARADISE: A Framework for Evaluating Spoken Dialogue Agents

- Cite for:
  - spoken dialogue evaluation should combine task success with dialogue behavior
  - dialogue quality cannot be reduced to a final success label
  - task requirements and interaction costs can be separated analytically
- Use in this chapter:
  - motivate the thesis split between task outcome metrics and process/phase metrics
  - justify reporting turns, repairs, latency, and route success together
- Do not use for:
  - do not claim this thesis estimates human user satisfaction unless human ratings are collected

## Walker et al. (2000), Problematic Dialogue Predictor

- Cite for:
  - early dialogue behavior can indicate problematic interactions
  - repair, misunderstanding, and interaction-cost evidence can support failure prediction
- Use in this chapter:
  - support failure-localization and early-warning metric interpretation
- Do not use for:
  - do not claim the thesis trains a deployed predictor unless that model is actually implemented

## Papineni et al. (2002), BLEU

- Cite for:
  - automatic text-generation evaluation can be repeatable and scalable
  - lexical overlap is a historically important but surface-level evaluation family
- Use in this chapter:
  - contrast text overlap with task-grounded route validation
- Do not use for:
  - BLEU does not verify whether a proposed route is executable

## Zhang et al. (2020), BERTScore

- Cite for:
  - contextual embeddings can compare candidate and reference text semantically
- Use in this chapter:
  - distinguish semantic text quality from task-grounded correctness
- Do not use for:
  - embedding similarity does not prove that station order, line names, or constraints are correct

## Shon et al. (2021), SLUE

- Cite for:
  - spoken language understanding should evaluate semantic content, not only word transcription
  - named entities are important evidence in speech understanding
- Use in this chapter:
  - justify station, line, time, and constraint entity metrics
- Do not use for:
  - do not treat generic WER as sufficient for route-dialogue understanding

## Mittag et al. (2021), NISQA

- Cite for:
  - neural non-intrusive speech quality assessment estimates perceived audio quality
- Use in this chapter:
  - explain why audio quality can be measured separately from ASR and task outcome
- Do not use for:
  - only calculate or interpret NISQA where the implementation has the required audio evidence

## Reddy et al. (2021), DNSMOS

- Cite for:
  - non-intrusive speech quality estimation can approximate perceived speech quality without a clean reference
- Use in this chapter:
  - classify audio quality as diagnostic evidence for TTS/channel problems
- Do not use for:
  - do not treat DNSMOS as a task-success metric

## Citation Practice

- Cite the paper at the first point where its concept is needed.
- Attach each citation to one specific claim, not to a whole paragraph of unrelated statements.
- Prefer one strong reference per claim; add a second only when it contributes a different angle.
- Pair project-specific result claims with generated result documents, not with external papers.
