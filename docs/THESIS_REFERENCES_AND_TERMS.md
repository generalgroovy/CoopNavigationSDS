# Thesis References and Key Terms

Purpose: quick writing aid for citations and definitions in the bachelor
thesis on automatic evaluation of spoken dialogue systems. Use this document
to decide where a paper belongs, what claim it supports, and which terms need
clear definitions before they appear in the experiment chapters.

## How to Use This Document

- Cite a paper where the concept is first needed.
  - Do not save all citations for a separate related-work block.
  - The current thesis structure combines background with related work and
    evaluation with validity.
- Cite for one concrete claim at a time.
  - Good: "SDS evaluation should consider task success and dialogue cost."
  - Weak: "Many papers study dialogue systems."
- Keep the citation chain simple.
  - Prefer one strong primary reference over several loosely related papers.
  - Add a second reference only when it contributes a different idea.

## Referenceable Papers by Earliest Citable Chapter

### Chapter 1: Introduction and Problem Motivation

- Walker et al. (1997), PARADISE: A Framework for Evaluating Spoken Dialogue
  Agents.
  - Link: <https://aclanthology.org/P97-1035/>
  - Cite for:
    - spoken dialogue evaluation should combine task outcome and dialogue
      behavior;
    - dialogue quality cannot be reduced to final task completion;
    - task requirements and dialogue behavior can be separated analytically.
  - Use in thesis:
    - motivate why the project measures both route success and process
      evidence such as turns, repairs, and latency.
  - Do not use for:
    - claiming that this thesis estimates user satisfaction with human ratings.

- Papineni et al. (2002), BLEU.
  - Link: <https://aclanthology.org/P02-1040/>
  - Cite for:
    - automatic evaluation can be cheaper and more repeatable than human
      evaluation;
    - lexical overlap metrics became influential in language-generation
      evaluation.
  - Use in thesis:
    - introduce the broader motivation for automatic metrics.
  - Limitation to state:
    - BLEU is not task-grounded and is weak for route validity.

### Chapter 2: Background and Related Work

- Budzianowski et al. (2018), MultiWOZ.
  - Link: <https://aclanthology.org/D18-1547/>
  - Cite for:
    - task-oriented dialogue can be represented with goals, domains, dialogue
      state, and annotated system/user behavior;
    - multi-domain TOD benchmarks support automatic state and response
      evaluation.
  - Use in thesis:
    - justify slots, constraints, dialogue state, and task-oriented evaluation.
  - Difference to this thesis:
    - MultiWOZ is written text dialogue; CoopNavigationSDS keeps speech
      pipeline evidence.

- Rastogi et al. (2020), Schema-Guided Dialogue.
  - Link: <https://arxiv.org/abs/1909.05855>
  - Cite for:
    - scalable TOD evaluation across services and schemas;
    - explicit schema/slot descriptions for dialogue state.
  - Use in thesis:
    - support the idea that goals and constraints should be represented
      structurally.
  - Difference to this thesis:
    - this project evaluates a controlled navigation domain with speech
      evidence and route validation.

- Shon et al. (2021), SLUE.
  - Link: <https://arxiv.org/abs/2111.10367>
  - Cite for:
    - spoken language understanding needs benchmark tasks beyond raw ASR;
    - natural speech evaluation can include named entities and higher-level
      understanding tasks.
  - Use in thesis:
    - motivate ASR entity preservation and semantic speech metrics.
  - Difference to this thesis:
    - CoopNavigationSDS uses generated speech and route-task validation rather
      than a broad natural-speech benchmark.

- Si et al. (2023), SpokenWOZ.
  - Link: <https://arxiv.org/abs/2305.13040>
  - Cite for:
    - spoken TOD differs from written TOD because speech introduces additional
      errors and interaction characteristics;
    - speech-text datasets can expose difficulties hidden by clean text.
  - Use in thesis:
    - justify paired text/audio runs and speech-specific evidence.
  - Difference to this thesis:
    - SpokenWOZ is a dataset; CoopNavigationSDS is an experimental evaluation
      framework.

- Schatzmann et al. (2007), agenda-based user simulation.
  - Link: <https://aclanthology.org/W07-0304/>
  - Cite for:
    - simulated users can support repeatable dialogue-system evaluation;
    - user simulation trades ecological realism for experimental control.
  - Use in thesis:
    - justify Agent A as a controlled caller.
  - Boundary:
    - do not claim simulated Agent A fully replaces human callers.

### Chapter 3: Evaluation Concepts and Validity

- Papineni et al. (2002), BLEU.
  - Link: <https://aclanthology.org/P02-1040/>
  - Cite for:
    - lexical overlap metrics are historically important automatic text
      metrics.
  - Use in thesis:
    - contrast lexical metrics with task-grounded route validation.

- Lin (2004), ROUGE.
  - Link: <https://aclanthology.org/W04-1013/>
  - Cite for:
    - overlap-based summarization metrics evaluate textual similarity through
      shared units.
  - Use in thesis:
    - explain why lexical metrics are surface-level text metrics.

- Banerjee and Lavie (2005), METEOR.
  - Link: <https://aclanthology.org/W05-0909/>
  - Cite for:
    - automatic text metrics can use alignment beyond exact n-gram matching.
  - Use in thesis:
    - show a transition from strict lexical overlap toward more flexible text
      matching.

- Zhang et al. (2020), BERTScore.
  - Link: <https://dblp.org/rec/conf/iclr/ZhangKWWA20>
  - Cite for:
    - contextual embeddings can compare candidate and reference text by
      semantic token similarity.
  - Use in thesis:
    - distinguish semantic text metrics from lexical overlap metrics.
  - Limitation:
    - semantic similarity still does not verify route executability.

- Sellam et al. (2020), BLEURT.
  - Link: <https://aclanthology.org/2020.acl-main.704/>
  - Cite for:
    - learned metrics can model human judgments better than simple overlap in
      some text-generation settings.
  - Use in thesis:
    - explain why learned semantic metrics are useful but not sufficient for
      grounded task success.

- Zhao et al. (2019), MoverScore.
  - Link: <https://aclanthology.org/D19-1053/>
  - Cite for:
    - semantic text evaluation can combine contextual embeddings with a
      distance measure.
  - Use in thesis:
    - support the distinction between semantic text quality and task-grounded
      correctness.

- Pillutla et al. (2021), MAUVE.
  - Link: <https://openreview.net/forum?id=Tqx7nJp7PR>
  - Cite for:
    - distributional text metrics compare generated text distributions with
      human text distributions.
  - Use in thesis:
    - classify MAUVE as a broad generation-quality metric, not a route-task
      validator.

- Taal et al. (2011), STOI.
  - Link: <https://ieeexplore.ieee.org/document/5495701>
  - Cite for:
    - speech intelligibility can be estimated with objective measures.
  - Use in thesis:
    - motivate TTS/ASR intelligibility evidence where audio data is available.

- Reddy et al. (2021), DNSMOS.
  - Link: <https://arxiv.org/abs/2010.15258>
  - Cite for:
    - non-intrusive speech quality estimation can approximate perceptual
      speech quality without a clean reference.
  - Use in thesis:
    - classify DNSMOS as a TTS/audio quality diagnostic metric.

- Mittag et al. (2021), NISQA.
  - Link: <https://arxiv.org/abs/2104.09494>
  - Cite for:
    - neural non-intrusive speech quality assessment can estimate perceived
      speech quality.
  - Use in thesis:
    - motivate audio-quality metrics as diagnostic evidence, not task-success
      metrics.

### Chapter 4: Methodology and Research Design

- Walker et al. (1997), PARADISE.
  - Link: <https://aclanthology.org/P97-1035/>
  - Cite for:
    - separating task success from dialogue behavior.
  - Use in methodology:
    - justify logging route outcome and dialogue process separately.

- Budzianowski et al. (2018), MultiWOZ; Rastogi et al. (2020), SGD.
  - Links:
    - <https://aclanthology.org/D18-1547/>
    - <https://arxiv.org/abs/1909.05855>
  - Cite for:
    - structured task goals and dialogue state.
  - Use in methodology:
    - justify explicit start, destination, time, constraints, and route state.

- SLUE and SpokenWOZ.
  - Links:
    - <https://arxiv.org/abs/2111.10367>
    - <https://arxiv.org/abs/2305.13040>
  - Cite for:
    - spoken evidence and speech-text mismatch.
  - Use in methodology:
    - justify logging intended speech, ASR raw transcript, normalized
      transcript, and corrections.

### Chapter 5: Metric Selection

- PARADISE.
  - Use for:
    - task success plus dialogue cost.
  - Metrics supported:
    - task success,
    - turn count,
    - repair count,
    - dialogue cost.

- BLEU/ROUGE/METEOR.
  - Use for:
    - explaining why lexical metrics are not primary metrics here.
  - Metrics supported:
    - optional text similarity descriptors only.

- BERTScore/BLEURT/MoverScore/MAUVE.
  - Use for:
    - explaining semantic text metrics.
  - Metrics supported:
    - optional NLG adequacy descriptors,
    - not route validity.

- NISQA/DNSMOS/STOI.
  - Use for:
    - speech quality and intelligibility diagnostics.
  - Metrics supported:
    - TTS/audio quality,
    - not direct task completion.

### Chapter 6: Results

- PARADISE.
  - Cite when interpreting:
    - successful but expensive dialogues,
    - repair-heavy success,
    - runtime and turn count as cost.

- SpokenWOZ/SLUE.
  - Cite when interpreting:
    - text/audio differences,
    - ASR entity preservation,
    - why speech errors can change task outcome.

- Semantic and lexical metric papers.
  - Cite when explaining:
    - why the thesis does not rely on BLEU/BERTScore-style text similarity for
      route success.

### Chapter 7: Discussion and Conclusion

- Use citations sparingly.
  - Discussion should mostly interpret this experiment's evidence.
  - Cite earlier work only when returning to a general point:
    - SDS quality combines outcome and interaction cost;
    - task-oriented dialogue uses goals and state;
    - spoken dialogue adds speech-channel uncertainty;
    - simulated users improve repeatability but limit ecological validity.

## Key Terms by Area and Relevance

### Core Thesis Terms

- Spoken dialogue system.
  - A dialogue system that processes speech input, speech output, or both.
  - In this thesis:
    - includes NLG, TTS, ASR, NLU, dialogue state, dialogue management, and
      task validation.
  - Do not reduce it to:
    - a clean-text chatbot.

- Task-oriented dialogue.
  - Dialogue aimed at completing a defined user goal.
  - In this thesis:
    - the goal is route finding from start station to destination under
      progressively revealed constraints.

- Automatic evaluation.
  - Metric calculation without human rating for each run.
  - In this thesis:
    - metrics are derived retrospectively from logged evidence.
  - Boundary:
    - automatic evaluation does not replace human usability or satisfaction
      evaluation.

- Phase-wise evaluation.
  - Evaluation that attributes evidence to pipeline phases.
  - Relevant phases:
    - turn-taking/audio,
    - ASR,
    - NLU,
    - dialogue state,
    - dialogue management,
    - backend grounding,
    - NLG,
    - TTS,
    - whole-dialogue outcome.

### Dialogue and Task Terms

- User goal.
  - The target the caller wants to achieve.
  - In this thesis:
    - start station,
    - destination,
    - departure time,
    - private constraints.

- Dialogue state.
  - The system's current representation of task-relevant facts.
  - In this thesis:
    - known trip facts,
    - active constraints,
    - proposed routes,
    - rejected routes,
    - accepted route.

- Constraint.
  - A condition the route must satisfy.
  - Examples:
    - maximum duration,
    - maximum transfers,
    - acceptable fullness,
    - acceptable delay risk,
    - ticket availability,
    - walking limit.

- Route validity.
  - Whether the route is executable in the transport network.
  - Requires:
    - valid stations,
    - valid line names,
    - connected segments,
    - destination reached.

- Duration regret.
  - Difference between selected route duration and optimal route duration.
  - Formula:
    - `duration_regret = selected_route_duration - optimal_duration`.
  - Interpretation:
    - lower is better;
    - zero means no time loss relative to the relevant optimum.

### Speech Pipeline Terms

- TTS.
  - Text-to-speech.
  - Produces audio from intended text.
  - Important because:
    - station names, line names, and times can be mispronounced or unclear.

- ASR.
  - Automatic speech recognition.
  - Produces transcript text from audio.
  - Important because:
    - wrong transcript entities can corrupt route state.

- Raw ASR transcript.
  - Direct recognizer output before correction or normalization.
  - Must be logged separately from understood transcript.

- Normalized understood transcript.
  - Transcript after transparent correction or domain normalization.
  - Must not hide errors:
    - corrections need to be logged.

- Word Error Rate.
  - Measures word-level ASR difference from a reference.
  - Formula:
    - `WER = (substitutions + deletions + insertions) / reference_words`.
  - Limitation:
    - one critical station error can matter more than several harmless word
      errors.

- Entity error rate.
  - Measures errors in task-critical entities.
  - More relevant than WER for:
    - station names,
    - line names,
    - times,
    - constraints.

### Metric Terms

- Text metric.
  - Any metric evaluating text output.
  - Includes:
    - lexical metrics,
    - semantic similarity metrics,
    - fluency/adequacy metrics,
    - task-grounded text metrics.

- Lexical metric.
  - Text metric based on surface overlap or edit distance.
  - Examples:
    - BLEU,
    - ROUGE,
    - word overlap.
  - Relationship:
    - lexical metrics are a subset of text metrics.

- Semantic text-similarity metric.
  - Text metric based on meaning similarity.
  - Examples:
    - BERTScore,
    - BLEURT,
    - MoverScore.
  - Limitation:
    - semantic similarity does not prove route executability.

- Outcome-confirming metric.
  - Metric close to the task-success definition.
  - Examples:
    - route validity,
    - destination reached,
    - constraint satisfaction.
  - Interpretation:
    - strong for describing success;
    - not independent evidence of why success occurred.

- Diagnostic metric.
  - Metric that helps locate possible failure points.
  - Examples:
    - ASR station F1,
    - NLU route-valid parse rate,
    - grounded proposal score,
    - repair success rate.

- Efficiency metric.
  - Metric that measures cost of achieving an outcome.
  - Examples:
    - turn count,
    - runtime,
    - repair turns,
    - clarification count.

### Experiment Design Terms

- Agent A.
  - Simulated caller/user.
  - Has private goal and constraints.
  - In this thesis:
    - UserLM is the main caller;
    - TinyLlama is a control stratum.

- Agent B.
  - Route-information dialogue-system backend.
  - The main model/backend comparison target.

- Paired text/audio run.
  - Two runs sharing the same non-audio condition:
    - one text-only,
    - one audio variant.
  - Used to isolate speech-channel effects.

- Matched condition.
  - Condition where all non-model factors are equivalent across compared
    models.
  - Used to compare Agent B models without scenario/audio/persona confounds.

- Execution incomplete.
  - Run without usable completed dialogue evidence.
  - Not the same as task failure.
  - Can indicate:
    - provider failure,
    - timeout,
    - missing model asset,
    - cluster interruption.

- Semi-successful run.
  - Run with meaningful progress but incomplete task satisfaction.
  - Examples:
    - valid route but constraint violation,
    - destination reached but duration threshold exceeded,
    - route proposed but not accepted.

## Citation Priority Checklist

- Chapter 1:
  - PARADISE for SDS evaluation problem;
  - BLEU only if introducing why automatic metrics became attractive.
- Chapter 2:
  - MultiWOZ/SGD for TOD structure;
  - SLUE/SpokenWOZ for speech-aware TOD;
  - user simulation reference for Agent A.
- Chapter 3:
  - BLEU/ROUGE/METEOR for lexical metrics;
  - BERTScore/BLEURT/MoverScore/MAUVE for semantic text metrics;
  - NISQA/DNSMOS/STOI for speech quality/intelligibility;
  - validity terminology should be explained in thesis prose rather than
    overloaded with citations.
- Chapter 4:
  - cite only concepts that justify design choices.
- Chapter 5:
  - cite metrics where formulas or metric families are introduced.
- Chapter 6:
  - cite only when interpreting why a metric family matters.
- Chapter 7:
  - use citations to reconnect findings to prior work, not to introduce new
    theory.
