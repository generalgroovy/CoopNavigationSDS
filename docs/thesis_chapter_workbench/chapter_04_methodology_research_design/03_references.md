# Chapter 4: Methodology and Research Design - References

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

## Budzianowski et al. (2018), MultiWOZ

- Cite for:
  - task-oriented dialogue can be represented using goals, domains, dialogue state, and system/user turns
  - structured goals and slots support automatic evaluation of task progress
- Use in this chapter:
  - justify explicit start station, destination, time, route, and constraint state
- Do not use for:
  - MultiWOZ is primarily text dialogue; use it for task-state structure, not speech-channel claims

## Rastogi et al. (2020), Schema-Guided Dialogue

- Cite for:
  - task-oriented dialogue state can be defined through explicit schemas and slots
  - structured slot descriptions make evaluation more reproducible across services
- Use in this chapter:
  - support the use of structured semantic frames and constraint slots
- Do not use for:
  - do not claim the thesis covers broad multi-service schemas; it uses one controlled navigation domain

## Si et al. (2023), SpokenWOZ

- Cite for:
  - spoken task-oriented dialogue differs from written dialogue because speech introduces recognition and interaction errors
  - speech evidence can reveal failures hidden by clean text transcripts
- Use in this chapter:
  - justify paired text-only and audio-variant conditions
  - support the need to inspect ASR output and normalized understanding separately
- Do not use for:
  - SpokenWOZ is a dataset; CoopNavigationSDS is an experimental framework

## Schatzmann et al. (2007), agenda-based user simulation for dialogue systems

- Cite for:
  - simulated users support repeatable dialogue-system evaluation
  - user simulation trades ecological realism for experimental control
- Use in this chapter:
  - justify Agent A as a controlled caller with private goals and staged constraints
- Do not use for:
  - do not claim simulated callers replace human callers

## Citation Practice

- Cite the paper at the first point where its concept is needed.
- Attach each citation to one specific claim, not to a whole paragraph of unrelated statements.
- Prefer one strong reference per claim; add a second only when it contributes a different angle.
- Pair project-specific result claims with generated result documents, not with external papers.
