# Chapter 3: Evaluation Concepts and Validity Threats - References

Purpose: citable papers, concrete claims they support, and boundaries for safe use.

## Papineni et al. (2002), BLEU

- Cite for:
  - automatic text-generation evaluation can be repeatable and scalable
  - lexical overlap is a historically important but surface-level evaluation family
- Use in this chapter:
  - contrast text overlap with task-grounded route validation
- Do not use for:
  - BLEU does not verify whether a proposed route is executable

## Lin (2004), ROUGE

- Cite for:
  - overlap-based metrics evaluate shared textual units
  - lexical metrics can be useful but do not by themselves establish task correctness
- Use in this chapter:
  - define lexical metrics as supplementary wording evidence
- Do not use for:
  - do not interpret high lexical similarity as route validity

## Banerjee and Lavie (2005), METEOR

- Cite for:
  - automatic text metrics can use alignment beyond exact n-gram overlap
- Use in this chapter:
  - show the range from strict lexical overlap toward more flexible text matching
- Do not use for:
  - semantic alignment is still not a substitute for network-grounded validation

## Zhang et al. (2020), BERTScore

- Cite for:
  - contextual embeddings can compare candidate and reference text semantically
- Use in this chapter:
  - distinguish semantic text quality from task-grounded correctness
- Do not use for:
  - embedding similarity does not prove that station order, line names, or constraints are correct

## Sellam et al. (2020), BLEURT

- Cite for:
  - learned text metrics can approximate human judgments in some generation settings
- Use in this chapter:
  - frame learned metrics as useful but not sufficient for route-task evaluation
- Do not use for:
  - do not use learned text similarity as the primary success criterion

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
