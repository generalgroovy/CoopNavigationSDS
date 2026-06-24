# Automatic SDS Metrics Catalog

## Implementation Status

The runtime catalog is maintained in
`coop_navigation_sds/EvaluationMetrics/catalog.py`. The catalog contains 220
single-run metrics with explicit retrospective calculations and 15 batch-only
validity metrics. Metrics without a calculation path have been removed.

Every implemented metric is obligatory and cannot be disabled per run.
Retrospective metric calculation keeps all catalog metrics visible even when a
value cannot be calculated. Missing or inapplicable values are represented as
`null` with calculation evidence explaining which evidence or estimator was
missing; they are never represented as fabricated zeroes.

The generated `metric_catalog.json` records each metric's evidence class,
scope, unit, required trace family, missing-data policy, and interpretation
metadata. Metric tiers and per-metric switches are not used. Every phase defines
at least seven metrics. Dependency preflight marks metrics as calculable or not
calculable for the selected configuration and shows missing fields before the
run. NISQA and DNSMOS are calculated only when their versioned local estimators
and readable audio are present. The console prints one line per catalog
metric; exact formulas, operands, substitutions, and unavailable reasons are
stored in result files for audit.

## Metric Selection Rationale

A metric is included only when it serves at least one of four experiment needs:

1. measure correctness or information preservation inside one pipeline phase;
2. expose the earliest trace-supported failure point;
3. quantify dialogue efficiency, cooperation, repair, or task completion;
4. test whether a metric remains stable and predictive across controlled
   models, scenarios, seeds, and speech conditions.

Every metric must name its required captured evidence and retrospective
calculation. Measures that require unavailable human labels, reference signals,
or licensed estimators remain null rather than being approximated. The
normative per-metric table, including the individual selection rationale for
every catalog entry, is [METRIC_REFERENCE.md](METRIC_REFERENCE.md).

The result folder contains the same contract in machine-readable form:
`metric_catalog.json` stores definitions, `retrospective_metrics.json` stores
detailed calculation evidence, and `metrics_long.csv` plus
`metrics_long.jsonl` provide one graphable row per condition and metric.

Single-run deterministic and reference metrics are calculated after the
dialogue. Batch runs first write a combined `metric_inputs.json`, then rebuild
all metric rows from that evidence file. Population metrics are populated by
`apply_cross_run_metrics` after a batch completes. A single run does not emit
cross-run values.

Current trace support includes:

- generated, outgoing, and recognized text;
- isolated TTS, ASR, pipeline, generation, NLU, speech, raw-turn, and capped
  turn timing;
- WAV path and synthesis/playback diagnostics;
- parsed route and route-validity flags;
- route candidates, duplicate decisions, durations, active constraints, and
  constraint status;
- ordered runtime events and constraint-state snapshots;
- heard trip-fact snapshots with recovered start station, destination,
  departure time, missing slots, and extraction evidence;
- per-turn Agent A and Agent B task-memory snapshots with route candidate,
  active constraints, focus, latest heard text, and memory additions;
- final route, outcome, timing, persona, provider, condition, and iteration
  identifiers.
- sanitized resolved configuration, random seed, runtime environment, model
  condition, and provider token usage when available;
- a versioned `metric_inputs.json` evidence document written before any
  derived metric calculation.

Additional phase-aware measures now captured from those traces include:

- stated and satisfied constraint counts;
- first deviation turn, phase, and elapsed time;
- Agent A and Agent B task-focus scores;
- dialogue distraction rate;
- correction-turn and corrected-token rates.
- dialogue-state trip-fact completeness and missing-trip-slot rate.
- memory trace coverage, memory update rate, and route-memory retention rate.

Batch exports include `failure_indicators.json`, an exploratory threshold
search over pre-outcome phase metrics. It excludes task outcome, whole-dialogue,
and metric-validity metrics to avoid label leakage and reports
balanced accuracy, failure sensitivity, success specificity, support, threshold,
and direction. Use this report for hypothesis generation unless evaluated on a
held-out batch.

Endpointing, overlap, interruption, barge-in, confidence calibration, human
ratings, learned language quality, and speaker embeddings are not in the
active catalog because the runtime does not currently capture trustworthy
evidence for them. The proposal table below defines the evidence contracts
required before selected measures can be implemented.

## Canonical Dialogue-System Phases

All runtime configuration, phase JSONL files, XLSX worksheets, catalog
metadata, and documentation use this order:

| Phase | Stable ID | Boundary |
| --- | --- | --- |
| 0 | `user_simulation` | Experiment caller state to caller utterance and evaluation. |
| 1 | `audio_input` | Incoming signal, endpointing, and turn-taking. |
| 2 | `asr` | Audio signal to final transcript. |
| 3 | `nlu` | Transcript to semantic frame and route interpretation. |
| 4 | `dialogue_state_tracking` | Semantic frame to committed persistent state. |
| 5 | `dialogue_management` | Committed state to selected system action. |
| 6 | `backend_task_execution` | Action to grounded route/tool result. |
| 7 | `nlg` | Grounded result to response text. |
| 8 | `tts` | Response text to audio output and playback. |
| 9 | `task_outcome` | Completed pipeline to final task result. |
| 10 | `whole_dialogue` | All turns to interaction-level quality and cost. |
| 11 | `metric_validity` | Multiple runs to uncertainty and robustness evidence. |

Metric keys retain their established prefixes for configuration and flat-file
compatibility. For example, `agent_a_*` metrics belong to Phase 0 and
`agent_b_*` metrics belong to Phase 6, while their global keys remain
unchanged.

## Measurement Policy

Goal completion and constraint satisfaction are the primary outcomes. Phase metrics explain why those outcomes succeeded or failed.

Automation classes:

- **D**: deterministic from network state, configuration, or captured traces.
- **R**: reference-based, using ground truth already known by the experiment.
- **L**: optional learned estimator. Report the estimator name and version.

Never replace missing measurements with zero. Store `null`; phase exports
report availability and coverage. A zero is valid only when the required
measurement was attempted and the observed numerator is zero.

### Classification Rules

The classes describe the evidence required to calculate a metric, not whether
the final formula is mathematically deterministic.

#### D: Deterministic Trace Or System Metric

A deterministic metric uses only authoritative runtime data:

- configuration;
- generated network data;
- route-planner output;
- timestamps and resource counters;
- pipeline status;
- parsed or stored application state.

The same recorded run always produces the same value without a separately
prepared target annotation and without a predictive model.

**Example: route validity rate**

For each proposed route, verify every adjacent station pair against the
network and divide valid proposals by all route proposals.

Why it is **D**:

- The network is the application's authoritative state.
- Route validity follows explicit graph rules.
- No expected natural-language answer or human annotation is required.
- No model estimates whether the route "seems" valid.

Why it is not **R**:

- It does not compare the proposal to a separately specified correct route.
- Several different routes may all be valid.
- The network validator supplies rules, not a target annotation.

Why it is not **L**:

- No trained model, embedding similarity, or learned quality predictor is
  involved.
- The result is exactly reproducible from the route and network.

#### R: Reference-Based Metric

A reference-based metric compares an observed pipeline output with an expected
target established independently of that output. References may come from:

- generated source text;
- scenario ground truth;
- known semantic frames;
- expected station, line, or constraint entities;
- deterministic perturbation labels;
- human-reviewed annotations.

Its scoring formula can still be deterministic. The defining property is that
the metric cannot be calculated meaningfully without the reference.

**Example: ASR word error rate**

Use the text supplied to TTS as the reference and the ASR transcript as the
hypothesis. Count substitutions, deletions, and insertions, then divide by the
number of reference words.

Why it is **R**:

- Error is defined relative to the words that should have been recognized.
- The transcript alone cannot reveal which words were omitted or substituted.
- The experiment provides an independent source-text reference.

Why it is not **D**:

- Edit-distance computation is deterministic, but the metric requires an
  external target beyond the observed ASR output.
- Without the source text, only properties such as transcript length or
  empty-transcript rate could be deterministic trace metrics.

Why it is not **L**:

- Word alignment and edit distance use fixed rules.
- No learned speech-quality or semantic-similarity model estimates the score.

#### L: Learned Estimator Metric

A learned metric uses a trained statistical or neural model to infer a quality
that is not directly available from authoritative state or exact references.
Examples include perceived naturalness, grammar acceptability, audio quality,
speaker consistency, and ambiguous-reference detection.

Every learned metric must record:

- estimator name;
- model and checkpoint version;
- input representation;
- inference settings;
- score range and direction;
- calibration dataset, when known;
- failure and unavailable-data behavior.

**Example: predicted conversational naturalness**

Pass each utterance or dialogue to a versioned naturalness estimator trained on
human ratings and report its normalized predicted score.

Why it is **L**:

- "Naturalness" is not an exact property stored in the run trace.
- The score is inferred from patterns learned from rated examples.
- Different estimator versions may assign different scores to identical text.

Why it is not **D**:

- Word count, pause count, and repetition rate are deterministic, but they are
  only observable features, not naturalness itself.
- No fixed project rule can establish a uniquely correct naturalness value.

Why it is not **R**:

- The evaluated run has no exact target naturalness score to compare against.
- If humans supplied ratings for the same utterances, error against those
  ratings would be an **R** metric, while the estimator's predicted score would
  remain an **L** metric.

### Decision Procedure

Classify each metric in this order:

1. Does calculation invoke a trained estimator or learned embedding model?
   - Yes: **L**.
2. Does correctness require an independently defined expected answer, label,
   transcript, semantic frame, or human annotation?
   - Yes: **R**.
3. Can it be derived exactly from authoritative configuration, network data,
   counters, timestamps, traces, and explicit rules?
   - Yes: **D**.
4. If none applies, the metric is not automatically measurable and must remain
   unavailable until a valid reference or estimator is introduced.

### Common Boundary Cases

| Measurement | Class | Reason |
| --- | --- | --- |
| Transcript is empty | D | Directly observable from the trace. |
| Transcript matches spoken source | R | Requires the source as the expected transcript. |
| Transcript meaning is similar according to an embedding model | L | Similarity is estimated by a learned representation. |
| Proposed route is connected | D | Evaluated against authoritative graph rules. |
| Parsed route matches the route expressed in annotated text | R | Requires an expected route annotation. |
| Response sounds actionable according to a trained classifier | L | Actionability is predicted rather than rule-verified. |
| Response contains origin, destination, and boarding order | D | Required fields can be checked explicitly. |
| NLU confidence calibration | R | Confidence is compared with known correctness labels. |
| Predicted speech quality | L | Quality is inferred by a trained acoustic estimator. |
| Audio file can be decoded | D | File validity is directly testable. |

### Mixed Metrics

Do not assign one class to a composite that silently mixes evidence types.
Store its components separately.

Example:

- route validity: **D**;
- semantic similarity to a reference answer: **L** if embedding-based;
- exact slot agreement with a reference frame: **R**.

A combined "response quality" score may aggregate these values for ranking, but
its manifest must list the component metrics, weights, classes, and missing-data
policy. The composite does not make an **L** component deterministic or remove
an **R** component's reference dependency.

### Experimental Condition Identifiers

Every metric row must include:

- scenario and test case;
- Agent A behavioral persona;
- Agent A audio persona;
- Agent B audio persona;
- fully resolved synthesis settings in the trace-level protocol;
- speech pattern;
- TTS and ASR implementation;
- Agent B implementation and model parameters;
- objective mode, iteration, and seed when available.

Behavioral and audio personas are independent variables. Do not aggregate them
into one label: hurried speech does not imply a hurried route preference, and
a risk-averse behavioral persona does not imply slow speech.

The following phase tables are a measurement-definition inventory, not an
activation list. The Python catalog is the authoritative implemented set.
[METRIC_PROPOSALS.md](METRIC_PROPOSALS.md) contains additions that remain
disabled and absent from outputs until their stated evidence contracts exist.

## Experimental Preflight: Scenario And Network

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Network connectivity | D | Fraction of stations reachable from every station. |
| Scenario reachability | D | Whether the requested destinations are reachable on the generated line graph. |
| Baseline availability | D | Whether an optimal route was calculated before dialogue. |
| Constraint-route availability | D | Whether at least one route satisfies all private constraints. |
| Stage alternative availability | D | Viable distinct alternatives available at each dialogue stage. |
| Alternative coverage | D | Available alternatives divided by configured required alternatives. |
| Route choice count | D | Number of viable routes within the search limit. |
| Scenario difficulty | D | Composite of optimal path length, transfers, constraints, and choice scarcity. |
| Constraint tightness | D | Viable routes after constraints divided by viable routes before constraints. |
| Optimality gap | D | Difference between best and second-best route utility. |
| Transfer burden baseline | D | Transfers on the optimal constrained route. |
| Risk burden baseline | D | Delay and transfer-miss classes on the optimal constrained route. |
| Capacity burden baseline | D | Near-capacity segments on the optimal constrained route. |
| Network generation validity | D | Station ratios, line coverage, unique identifiers, and segment integrity pass. |

## Phase 0: User Simulation And Input

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Invalid-route catch rate | D | Invalid proposals rejected by Agent A. |
| Constraint-violation catch rate | D | Violating proposals challenged after constraint activation. |
| Time-limit violation catch rate | D | Overlong routes challenged. |
| False acceptance rate | D | Invalid or noncompliant routes accepted. |
| False rejection rate | D | Fully compliant routes rejected without a new objective. |
| Correct acceptance rate | D | Compliant routes accepted. |
| Critique specificity | D | Critique identifies the actual failed objective or constraint. |
| Critique factuality | D | Critique claims agree with recomputed route properties. |
| Alternative elicitation success | D | Critique is followed by a distinct viable proposal. |
| Constraint revelation accuracy | R | Agent A reveals its assigned constraints and no unassigned constraints. |
| Constraint revelation order | D | Constraint one and two appear only in their intended stages. |
| Preference consistency | D | Evaluations remain consistent with the persona profile. |
| Best-candidate retention | D | Agent A does not lose a prior candidate that still satisfies current goals. |
| Selected-route utility | D | Utility of final selected route under persona preferences. |
| Selection regret | D | Optimal candidate utility minus selected-candidate utility. |
| Satisfaction calibration | D | Expressed satisfaction class matches computed outcome class. |
| Closure correctness | D | Closing utterance names or clearly references the selected viable route. |
| User effort | D | Agent A turns, words, clarifications, and repeated requests before completion. |
| Caller latency | D | State availability to generated caller response. |

## Phase 1: Audio Input And Turn-Taking

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Audio capture success rate | D | Turns with a readable audio signal divided by speech turns. |
| Missing-audio rate | D | Speech turns without an audio artifact or stream. |
| Turn latency | D | Previous end-of-utterance to next response start. |
| Response latency | D | Input transcript completion to response generation start. |
| End-of-utterance detection latency | R | Detected endpoint minus known signal endpoint. |
| Early endpoint rate | R | Endpoints detected before the known final speech frame. |
| Late endpoint rate | R | Endpoints exceeding configured endpoint tolerance. |
| Overlap rate | D | Overlapping speech duration divided by total speech duration. |
| Overlap event count | D | Number of simultaneous-speaker intervals. |
| Interruption rate | D | Turns beginning before the other speaker ends. |
| Inter-turn gap | D | Silence between consecutive speakers. |
| Excessive-gap rate | D | Gaps above the configured natural-dialogue threshold. |
| Barge-in success rate | D | Accepted interruptions divided by attempted interruptions. |
| Utterance duration | D | Audio duration per turn and speaker. |
| Speech rate | R | Reference word count divided by audio minutes. |
| Silence ratio | D | Non-speech frames divided by total audio frames. |
| Pause count and duration | D | Internal silence intervals above the configured pause threshold. |
| Long-pause rate | D | Pauses above the long-pause threshold per utterance. |
| Audio clipping rate | D | Samples at or near amplitude limits divided by samples. |
| Loudness stability | D | Variation of loudness across turns. |
| Signal-to-noise estimate | L | Optional acoustic signal-to-noise estimator. |
| Turn budget violation rate | D | Turns exceeding configured maximum elapsed seconds. |
| Real-time interaction factor | D | Total processing plus playback time divided by conversation audio duration. |

## Phase 2: Automatic Speech Recognition

The generated or spoken text is the automatic reference in controlled experiments.

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| ASR success rate | D | Non-empty transcripts divided by attempted recognition turns. |
| ASR failure rate | D | Recognition exceptions or empty transcripts divided by attempts. |
| Word error rate | R | Word substitutions, deletions, and insertions divided by reference words. |
| Character error rate | R | Character edit distance divided by reference characters. |
| Sentence error rate | R | Utterances containing any recognition error divided by utterances. |
| Semantic ASR error rate | R | Transcripts whose parsed semantic frame differs from the reference frame. |
| Entity error rate | R | Incorrect transport entities divided by reference entities. |
| Station precision, recall, F1 | R | Correctly recognized station names against reference station names. |
| Line precision, recall, F1 | R | Correctly recognized line names against references. |
| Critical-slot accuracy | R | Accuracy for origin, destination, time, line, and constraints. |
| Route sequence edit distance | R | Edit distance between reference and transcript station sequences. |
| Constraint preservation rate | R | Constraints preserved in transcript divided by spoken constraints. |
| Negation preservation rate | R | Correct preservation of negated preferences and restrictions. |
| Intent preservation rate | R | Transcript intent equals spoken-text intent. |
| Numeric preservation rate | R | Correct times, durations, and transfer counts. |
| Empty-transcript rate | D | Empty recognized strings divided by recognition attempts. |
| Hallucinated-token rate | R | Inserted words divided by transcript words. |
| Transcript correction count | D | Number of token edit groups applied between raw ASR output and listener input. |
| Domain correction yield | D | Transcript correction groups divided by detected raw token misinterpretations. |
| Uncorrected misinterpretation count | D | Speech-to-raw-ASR token edit groups that remain different in listener input. |
| Recognition latency | D | Audio availability to final transcript. |
| ASR real-time factor | D | Recognition processing time divided by audio duration. |
| Confidence calibration error | R | Difference between ASR confidence and observed token correctness, if confidence exists. |
| Repair-trigger rate | D | Turns causing clarification because of transcript uncertainty or error. |
| ASR repair success | R | Failed semantic frames corrected after clarification divided by repair attempts. |

Clock-time evidence is evaluated at two levels. Raw ASR metrics compare the
spoken form with the recognizer transcript exactly as captured. Semantic slot
metrics allow documented clock normalizations, for example `8-7`, `8, 7`,
`8 7`, `eight seven`, and `eight oh seven` as `08:07` when the utterance is a
departure-time expression or focused time-repair answer. This prevents a
successful time repair from being counted as a dialogue-state failure while
preserving the ASR error for word, character, and token-level metrics.

## Phase 3: Spoken-Language Understanding

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Intent accuracy | R | Predicted intent equals scenario-derived reference intent. |
| Intent macro F1 | R | Macro F1 across route request, critique, constraint, comparison, acceptance, and closure. |
| Slot precision, recall, F1 | R | Extracted slot values against known utterance frame. |
| Joint frame accuracy | R | Every required intent and slot is correct for a turn. |
| Semantic frame exact match | R | Predicted frame exactly equals reference frame. |
| Critical-slot accuracy | R | Origin, destination, time, route, and constraint slots. |
| Constraint extraction F1 | R | Extracted revealed constraints against scenario and caller state. |
| Constraint value accuracy | R | Correct threshold or preference value for each constraint. |
| Route parse success rate | D | Assistant turns yielding a route parse. |
| Valid-route parse rate | D | Parsed routes that are connected in the line graph. |
| Goal-reaching parse rate | D | Parsed routes reaching the requested destination. |
| Station sequence exact match | R | Parsed station sequence equals the route expressed in reference text. |
| Station sequence edit distance | R | Edit distance between parsed and reference station sequences. |
| Origin and destination accuracy | R | Correct route endpoints. |
| Line-change extraction accuracy | R | Extracted boarding and transfer lines match reference. |
| Temporal expression accuracy | R | Correct start time, duration, and arrival-time values. |
| Negation-scope accuracy | R | Correct interpretation of exclusions such as "not crowded." |
| Ambiguity detection rate | R | Ambiguous utterances correctly marked as requiring clarification. |
| False parse rate | R | A confident semantic frame produced from an unparseable utterance. |
| Unknown-entity detection | R | Unknown stations or lines rejected instead of grounded incorrectly. |
| NLU latency | D | Transcript availability to completed semantic frame. |
| NLU confidence calibration | R | Confidence error against frame correctness, if confidence exists. |

## Phase 4: Dialogue State Tracking

The tracked runtime stage vocabulary is `discovery`, `proposal`, `comparison`,
`refinement`, `confirmation`, and `closed`. Stage metrics compare this explicit
trace state with the state implied by the recognized conversation.

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Joint goal accuracy | R | Full tracked state equals ground-truth state after each turn. |
| Slot accuracy | R | Correct tracked state slots divided by tracked slots. |
| Trip-fact completeness | D | Mean share of required trip facts recovered from Agent B's heard memory before each Agent B action. |
| Missing trip-slot rate | D | Missing start, destination, and departure-time slots divided by all required trip slots across heard-state snapshots. |
| Memory trace coverage | D | Memory snapshots divided by dialogue messages. |
| Memory update rate | D | Snapshots with new remembered facts, routes, constraints, or focus changes divided by memory snapshots. |
| Route-memory retention rate | D | Post-route snapshots where both agents retain a route candidate divided by all post-route snapshots. |
| Stage accuracy | R | Tracked dialogue stage equals deterministic experiment stage. |
| Stage drift rate | R | Turns where state moves to an incorrect stage. |
| Constraint retention rate | R | Previously stated constraints still present in current state. |
| Constraint corruption rate | R | Retained constraint value changes without user correction. |
| Constraint omission rate | R | Active constraints missing from state. |
| Premature constraint activation | R | Private constraints activated before Agent A states them. |
| Shared-state agreement | R | Agent assumptions and controller state agree on goals and constraints. |
| Candidate-memory precision | D | Stored candidates correspond to actually proposed valid routes. |
| Candidate-memory recall | D | Proposed valid routes represented in candidate memory. |
| Candidate deduplication accuracy | R | Repeated routes detected and distinct routes retained. |
| Selected-route consistency | D | Final selected route exists in candidate memory. |
| Route-state consistency | D | Stored route duration and attributes match network recomputation. |
| State contradiction rate | R | Mutually incompatible slot values simultaneously active. |
| State update latency | D | Semantic frame availability to committed state update. |
| Recovery from state error | R | Incorrect state corrected within a configured number of turns. |

## Phase 5: Dialogue Management And Policy

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Correct next-action rate | R | Selected dialogue action belongs to the valid action set for current state. |
| Stage-transition precision | R | Triggered transitions that were valid. |
| Stage-transition recall | R | Required transitions that occurred. |
| Stage skip rate | R | Required stages bypassed. |
| Constraint-order adherence | D | Constraints revealed only after prior-stage success and in configured order. |
| Premature answer rate | R | Final-answer action occurs before sufficient valid route evidence. |
| Premature closure rate | R | Conversation closes before a viable route is selected. |
| Clarification precision | R | Clarifications issued when information was actually insufficient. |
| Clarification recall | R | Ambiguous or failed states that triggered clarification. |
| Unnecessary clarification rate | R | Clarifications despite a complete reliable frame. |
| Clarification calibration | R | Clarification probability aligned with estimated misunderstanding probability. |
| Repair attempt rate | D | Repair actions divided by eligible failures. |
| Repair success rate | R | Repairs producing a correct frame or valid proposal. |
| Repeated repair rate | D | Multiple repairs for the same unresolved failure. |
| Invalid-proposal handling accuracy | D | Invalid routes rejected and failure counters updated correctly. |
| Constraint-violation handling accuracy | D | Violating routes rejected or explicitly qualified. |
| Alternative-request appropriateness | R | Alternative requested only when current route misses an active objective. |
| Proposal comparison coverage | D | Candidate comparisons mentioning all currently relevant trade-offs. |
| Route repetition rate | D | Repeated normalized route sequences divided by proposals. |
| Distinct proposal rate | D | Unique route sequences divided by proposals. |
| Policy progress rate | D | Turns that advance route validity, constraints, comparison, or completion. |
| Stagnation rate | D | Consecutive turns without state or candidate improvement. |
| Turn efficiency | D | Successful stage transitions divided by turns. |
| Stop-decision accuracy | R | Stop reason matches deterministic outcome state. |
| Turn-limit utilization | D | Used turns divided by configured maximum. |
| Policy latency | D | State update to selected next action. |

## Phase 6: Backend Task Execution And Grounding

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Proposal parse rate | D | Assistant proposals yielding a station route. |
| Route validity rate | D | Connected proposals divided by route proposals. |
| Destination reach rate | D | Proposals reaching the requested destination. |
| Complete-route rate | D | Proposals containing enough information to execute the journey. |
| Grounded proposal score | D | Weighted correctness of stations, lines, timing, and constraints. |
| Hallucinated station rate | D | Mentioned stations absent from the network. |
| Hallucinated line rate | D | Mentioned lines absent from the network. |
| Hallucinated connection rate | D | Claimed adjacent stations not connected by the claimed line. |
| Unsupported attribute rate | D | Incorrect duration, fullness, risk, headway, or transfer claims. |
| Transfer-time factuality | D | Spoken or implied transfer time matches recomputed transfer time. |
| Duration factuality | D | Claimed duration within configured tolerance of recomputed duration. |
| Transfer-count factuality | D | Claimed transfers equal line changes. |
| Fullness factuality | D | Capacity claims match route data. |
| Delay-risk factuality | D | Delay-class claims match route data. |
| Active-constraint compliance | D | Proposal satisfies constraints revealed at that turn. |
| Unprompted-constraint mention rate | D | Secondary constraints mentioned before Agent A reveals them. |
| Actionability score | D | Route has origin, destination, boarding order, and line changes. |
| Route novelty | D | Sequence distance from all previous proposals. |
| Pareto improvement rate | D | New proposals improve at least one active objective without worsening all others. |
| Dominated proposal rate | D | Proposed route is dominated by an already known candidate. |
| Optimality ratio | D | Proposal utility divided by optimal active-constraint utility. |
| Duration regret | D | Proposal duration minus active-stage optimum. |
| Constraint penalty regret | D | Proposal penalty minus active-stage optimum penalty. |
| Viable alternative coverage | D | Distinct viable proposals divided by available viable alternatives. |
| Best-route discovery rate | D | Runs where Agent B proposes the active-stage optimum. |
| Best-route discovery turn | D | Turn where the best proposed route first appears. |
| Plugin execution success | D | Successful assistant calls divided by attempted calls. |
| Model generation latency | D | Prompt submission to raw response. |
| Valid proposals per 100 output tokens | D | `100 * valid proposals / output tokens`; `null` when the provider exposes no token count. |
| Repair generation success | D | Invalid or repeated drafts repaired into acceptable proposals. |

## Phase 7: Natural-Language Generation

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Semantic adequacy | R | Required semantic frame elements realized in output. |
| Faithfulness | D | Realized claims agree with route, state, and network data. |
| Slot error rate | R | Missing, incorrect, or extra semantic slots divided by required slots. |
| Executable utterance rate | D | Assistant utterance can be parsed into an executable route. |
| Route mention completeness | D | Necessary boarding points and line changes are verbalized. |
| Constraint mention precision | D | Mentioned constraints are active and relevant. |
| Constraint mention recall | D | Active constraints relevant to the decision are addressed. |
| Information-order accuracy | D | Route information appears in travel order. |
| Conciseness | D | Words, sentences, and clauses per turn. |
| Excess verbosity rate | D | Turns exceeding configured word or sentence limits. |
| Underspecification rate | D | Turns too short to execute or evaluate. |
| Repetition rate | D | Repeated n-grams, sentences, or prior-turn content. |
| Self-repetition rate | D | Duplicate content within one utterance. |
| Distinct-1 and Distinct-2 | D | Unique unigrams or bigrams divided by total unigrams or bigrams. |
| Lexical diversity | D | Unique normalized words divided by total words. |
| Filler rate | D | Configured filler tokens per word. |
| Hesitation rate | D | Hesitation markers per utterance. |
| Disfluency compliance | D | Generated pattern frequency matches configured speech pattern. |
| Formatting violation rate | D | Bullets, code, markup, or forbidden structured output in spoken responses. |
| Hidden-reasoning leakage rate | D | Responses containing configured reasoning markers. |
| Pronoun ambiguity rate | L | Optional coreference estimator flags unclear route references. |
| Grammatical acceptability | L | Optional versioned grammar or acceptability estimator. |
| Text naturalness | L | Optional versioned conversational naturalness estimator. |
| Estimated spoken duration | D | Word count and configured speech rate converted to seconds. |

## Phase 8: Text-To-Speech And Audio Output

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Synthesis success rate | D | Valid audio outputs divided by synthesis attempts. |
| Synthesis failure rate | D | Exceptions, empty files, or invalid signals divided by attempts. |
| Audio validity rate | D | Readable audio with supported format and nonzero frames. |
| Synthesis latency | D | Text submission to available audio. |
| TTS real-time factor | D | Synthesis processing time divided by generated audio duration. |
| Audio duration error | R | Actual duration minus duration expected from configured speech rate. |
| Speaking-rate accuracy | R | Realized words per minute against configured rate. |
| Pause-pattern adherence | R | Realized pause count and duration against configured pattern. |
| Stutter-pattern adherence | R | Realized repetitions against configured stutter settings. |
| Pronunciation accuracy | R | ASR round-trip word accuracy on synthesized audio. |
| Station pronunciation accuracy | R | ASR round-trip accuracy for station and line names. |
| Round-trip semantic intelligibility | R | Semantic frame preserved after TTS-to-ASR round trip. |
| Round-trip route accuracy | R | Parsed route after round trip equals source route. |
| Loudness compliance | D | Loudness remains within configured range. |
| Clipping rate | D | Saturated samples divided by samples. |
| Leading and trailing silence | D | Silence before first and after last voiced frame. |
| Speaker consistency | L | Optional speaker-embedding similarity across turns. |
| NISQA overall MOS | L | Core TorchMetrics NISQA v2.0 estimate from each turn WAV, averaged across readable turns. |
| DNSMOS overall MOS | L | Core non-personalized TorchMetrics DNSMOS estimate, with P.808, signal, background, and overall dimensions retained. |
| PESQ | R | ITU-T P.862 perceptual score when an aligned clean reference and the optional PESQ implementation are available. |
| POLQA | R | Licensed ITU-T P.863 score imported from a configured compliant implementation; never approximated. |
| STOI | R | Short-time objective intelligibility from an aligned clean-reference pair. |
| SI-SDR | R | Scale-invariant signal-to-distortion ratio from aligned reference and synthesized samples. |
| Playback success rate | D | Successful playback attempts divided by requested playbacks. |

NISQA v2 and DNSMOS are non-intrusive and need only synthesized speech. PESQ,
POLQA, STOI, and SI-SDR are intrusive metrics and remain `null` unless the
experiment logs aligned clean-reference audio. POLQA additionally requires a
licensed implementation; the framework accepts its trace score but does not
substitute an open metric for POLQA.

## Phase 9: End-To-End Task Outcome

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Task completion | D | Final selected route is valid and reaches every required destination. |
| Valid-route completion | D | Final route is connected, uses allowed modes, and reaches the current destination. |
| Acceptable-duration completion | D | Final duration is within the configured ratio or minute limit of the baseline. |
| All-constraint satisfaction | D | Every revealed constraint evaluates as satisfied on the final route. |
| Constraint satisfaction rate | D | Satisfied revealed constraints divided by revealed constraints. |
| Weighted constraint satisfaction | D | Configured importance-weighted fraction of satisfied constraints. |
| Stage completion rate | D | Successfully completed dialogue stages divided by configured stages. |
| Final outcome | D | `satisfied`, `semi_satisfied`, `unsatisfied`, or `pipeline_failure`. |
| Constrained optimality ratio | D | Final route utility divided by optimal constrained-route utility. |
| Duration ratio | D | Final duration divided by optimal allowed-route duration. |
| Duration regret | D | Final duration minus optimal allowed-route duration. |
| Constraint regret | D | Final route constraint penalty minus optimal constrained-route penalty. |
| Correct route selection | D | Agent A selects the best known candidate under its revealed preferences. |
| Turns to success | D | Number of turns before the first fully satisfactory proposal. |
| First-valid-route turn | D | First turn containing a valid route to the destination. |
| First-compliant-route turn | D | First turn satisfying time and all constraints revealed at that point. |
| Successful natural closure | D | Agent A closes only after selecting a viable route. |
| Constraint-induced route change rate | D | Changed adjacent staged optima divided by eligible stage transitions. |

Report all primary outcomes per run, scenario, persona, model, speech condition, and seed.

## Phase 10: Whole-Dialogue Interaction

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Dialogue success score | D | Weighted primary outcome score with separately reported components. |
| Interaction quality trajectory | D | Per-turn progress score and slope over the dialogue. |
| Goal-progress area under curve | D | Area under per-turn stage and constraint completion curve. |
| Dialogue cost | D | Weighted turns, words, latency, model calls, repairs, and audio duration. |
| Turn count | D | Total and per-speaker turns. |
| Word count | D | Total and per-speaker words. |
| Mean words per turn | D | Words divided by turns, overall and per speaker. |
| Total runtime | D | Ready-to-converse time to completion. |
| Mean and maximum turn time | D | Generation plus speech time per turn. |
| Candidate count | D | Distinct route candidates proposed. |
| Route revision count | D | Changes to the currently best candidate. |
| Clarification count | D | Clarification turns. |
| Repair count | D | Repair attempts across phases. |
| Warning count | D | Detected invalid, repeated, or noncompliant events. |
| Abandonment rate | D | Runs ending without a selected route. |
| Failure localization score | D | Fraction of failed runs assigned to the earliest trace-supported failing phase. |
| Pipeline dependency integrity | D | Downstream input equals the preceding phase output. |
| Trace completeness rate | D | Required raw evidence collections present divided by required collections. |
| Cooperative progress rate | D | Turns that add valid information or improve route quality. |
| Task-focus score | D | Task-relevant content divided by total content. |
| Comparison quality | D | Relevant route trade-offs correctly compared. |
| Conversation repetition rate | D | Repeated semantic content across turns. |
| Natural closure rate | D | Dialogues ending with a concise selection and acknowledgement. |
| Resource cost | D | Model tokens, CPU time, memory, audio seconds, and external API calls. |
| Estimated monetary cost | D | Provider usage multiplied by configured price table. |

## Phase 11: Cross-Run Validity And Robustness

These are computed across multiple runs.

| Metric | Class | Automatic calculation |
| --- | --- | --- |
| Success confidence interval | D | Bootstrap interval for task completion. |
| Mean metric confidence interval | D | Bootstrap interval for each scalar metric. |
| Seed variance | D | Variance across identical conditions with different seeds. |
| Test-retest agreement | D | Agreement across repeated identical deterministic conditions. |
| Metric-outcome correlation | D | Correlation between phase metrics and primary outcomes. |
| Partial metric-outcome correlation | D | Correlation controlling for scenario difficulty. |
| Discriminative power | D | Effect size between successful and failed dialogues. |
| Failure prediction accuracy | D | Cross-validated prediction of final failure from early phase metrics. |
| Failure localization agreement | D | Agreement between phase failures and deterministic root-cause labels. |
| Ceiling and floor rate | D | Fraction of scores at minimum or maximum. |
| Missingness rate | D | Unavailable values divided by expected measurements. |
| Metric redundancy | D | Pairwise correlation or mutual information among metrics. |
| Monotonicity | D | Whether worsening controlled perturbations worsen the metric. |
| Perturbation sensitivity | D | Score change under controlled ASR, delay, fullness, or disfluency changes. |
| Persona robustness | D | Outcome and metric variance across persona profiles. |
| Scenario robustness | D | Outcome and metric variance across route difficulty levels. |
| Speech-pattern robustness | D | Performance degradation by speech pattern. |
| Provider robustness | D | Outcome variance across assistant, TTS, and ASR providers. |
| Speech-configuration robustness | D | Outcome variance across matched TTS, ASR, voice, rate, and speech-pattern conditions. |
| Subgroup performance gap | D | Difference across persona, speech-pattern, provider, or scenario groups. |
| Configuration sensitivity | D | Outcome change under turn, latency, and constraint thresholds. |

## Recommended Core Set

Enable this smaller set by default:

- task completion;
- all-constraint satisfaction;
- acceptable-duration completion;
- duration regret;
- turns to success;
- turn latency;
- ASR word error rate;
- station F1;
- semantic ASR error rate;
- NLU joint frame accuracy;
- constraint extraction F1;
- dialogue-state constraint retention;
- correct next-action rate;
- clarification precision and recall;
- repair success rate;
- route validity and destination reach rates;
- grounded proposal score;
- route repetition rate;
- Agent A invalid-route catch rate;
- Agent A false acceptance rate;
- NLG faithfulness and executable utterance rate;
- TTS round-trip semantic intelligibility;
- dialogue cost;
- pipeline dependency integrity;
- failure localization score.

All other metrics should be individually configurable and disabled unless their required trace fields or estimators are available.
