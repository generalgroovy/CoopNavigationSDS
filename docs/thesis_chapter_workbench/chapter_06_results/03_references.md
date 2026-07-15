# Chapter 6: Results - References

Purpose: citable papers, concrete claims they support, and boundaries for safe use.

## Walker, Passonneau, and Boland (2001), Quantitative and Qualitative Evaluation of DARPA Communicator Spoken Dialogue Systems

- Cite for:
  - full SDS evaluation benefits from multiple measures rather than one final score
  - system-level evaluation can combine task completion, dialogue behavior, and error analysis
  - comparative SDS evaluation requires explicit conditions and documented evidence
- Use in this chapter:
  - position CoopNavigationSDS as a small controlled evaluation instrument inspired by full-system SDS evaluation
  - support the decision to report both success categories and diagnostic phase evidence
- Do not use for:
  - do not imply DARPA Communicator used the same LLM or route-navigation setup

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

## Si et al. (2023), SpokenWOZ

- Cite for:
  - spoken task-oriented dialogue differs from written dialogue because speech introduces recognition and interaction errors
  - speech evidence can reveal failures hidden by clean text transcripts
- Use in this chapter:
  - justify paired text-only and audio-variant conditions
  - support the need to inspect ASR output and normalized understanding separately
- Do not use for:
  - SpokenWOZ is a dataset; CoopNavigationSDS is an experimental framework

## Shon et al. (2021), SLUE

- Cite for:
  - spoken language understanding should evaluate semantic content, not only word transcription
  - named entities are important evidence in speech understanding
- Use in this chapter:
  - justify station, line, time, and constraint entity metrics
- Do not use for:
  - do not treat generic WER as sufficient for route-dialogue understanding

## Citation Practice

- Cite the paper at the first point where its concept is needed.
- Attach each citation to one specific claim, not to a whole paragraph of unrelated statements.
- Prefer one strong reference per claim; add a second only when it contributes a different angle.
- Pair project-specific result claims with generated result documents, not with external papers.
