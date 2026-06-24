# Implementable Metric Proposals

These metrics are proposals, not active output fields. A proposal enters the
runtime catalog only after its evidence collector, retrospective formula,
missing-data behavior, and tests exist.

## Recently Admitted

The following proposals are no longer merely proposed because their evidence,
calculation, and tests are implemented in the runtime catalog:

- first deviation turn, phase, and elapsed time;
- Agent A and Agent B task-focus score;
- whole-dialogue distraction rate;
- correction-turn and corrected-token rates;
- stated and satisfied constraint counts;
- exploratory batch failure-threshold indicators over pre-outcome metrics.

## Admission Rule

1. Add versioned raw fields to the protocol and `metric_inputs.json`.
2. Capture those fields during the run without deriving the final score.
3. Implement the retrospective calculation with explicit operands.
4. Declare dependencies in catalog and configuration preflight.
5. Test success, missing evidence, and inapplicable-run behavior.
6. Emit the value only when calculation succeeds.

## Proposals By Phase

| Phase | Proposed metric | Calculation and interpretation | Required data | Best runtime capture |
| --- | --- | --- | --- | --- |
| Network and scenario | Stage-alternative diversity | Mean normalized route-sequence distance among viable alternatives per constraint stage. Higher means choices differ meaningfully. | Viable stage routes, normalized signatures, active constraints | Store the bounded route-search result and route signature during preflight. |
| Network and scenario | Scenario discriminability | Utility difference between the best and median viable route. Higher values make quality easier to distinguish. | Utility components for every viable route | Persist duration, transfers, risk, fullness, walking, and utility per preflight route. |
| User simulation | Repair-focus rate | Focused correction turns divided by Agent A repair turns. | Agent A action, disputed spans, generated utterance frame | Log the expected repair frame before natural-language generation and align it with emitted text. |
| User simulation | Constraint-revelation latency | Turns between prior-objective satisfaction and the next assigned constraint. Negative values mean premature revelation. | Objective event, constraint activation event, turn index | Timestamp both caller-state transitions. |
| Audio input and turn-taking | Endpoint truncation duration | Reference speech milliseconds after the detected endpoint. Lower is better. | Speech interval, endpoint timestamp, sample rate | Record recognizer endpoint and run offline voice-activity detection on the full waveform. |
| Audio input and turn-taking | Overlap burden | Simultaneous-speaker milliseconds divided by voiced milliseconds. Lower is better. | Per-speaker playback and capture boundaries | Use a monotonic clock at playback and capture boundaries. |
| Automatic speech recognition | Domain-correction precision | Correct station/line corrections divided by all automatic corrections. | Raw/corrected transcript, spoken reference, token alignment | Store every correction with source, replacement, rule, and reference-match label. |
| Automatic speech recognition | Critical-error repair latency | Turns from a critical entity error to restoration of the intended entity. | Critical error, entity, repair-resolution event | Assign a stable repair identifier at detection and close it on recovery. |
| Natural-language understanding | Route-component F1 | Macro F1 over transport type, line, origin, destination, intermediate stops, and order. | Reference and predicted route frame | Generate the reference from source text/network state and persist both frames before state update. |
| Natural-language understanding | Ambiguity detection recall | Ambiguous reference frames marked uncertain divided by all ambiguous frames. | Ambiguity label, parser decision, reference alternatives | Generate labelled ambiguous cases or record deterministic multiple-parse cases. |
| Dialogue state tracking | Post-repair recovery turns | Turns until all corrupted slots match ground truth after repair. Lower is better. | Per-turn predicted/reference state, repair identifier | Snapshot full state and slot provenance after every committed update. |
| Dialogue state tracking | State-provenance completeness | Active slots with source turn and evidence span divided by active slots. | Slot, source turn/span, update reason | Require provenance whenever the state updater creates or changes a slot. |
| Dialogue management | Clarification-loop rate | Repair identifiers receiving repeated equivalent clarification divided by repairs. Lower is better. | Repair identifier, action, disputed entity, turn | Log structured actions and normalized clarification targets before generation. |
| Dialogue management | Action-to-progress precision | Policy actions improving state, route, or stage divided by all actions. | Action and state/candidate before and after | Persist before/after hashes and objective deltas for every policy decision. |
| Backend task execution | Constraint-preserving improvement rate | Alternatives improving a requested property without violating satisfied active constraints divided by alternatives. | Old/new candidates, active constraints, properties | Validate against a frozen constraint snapshot and store pairwise deltas. |
| Backend task execution | Valid-improvement yield per 100 tokens | `100 * distinct valid improving proposals / output tokens`. | Token usage, normalized candidates, validity/improvement | Link parsed proposals to the backend generation call identifier. |
| Natural-language generation | Repair-utterance focus | Repair responses realizing disputed entities without unrelated route restatement divided by repairs. | Repair frame, generated text, disputed spans | Store the NLG input frame and label generated semantic units. |
| Natural-language generation | Semantic density | Correct decision-relevant facts divided by spoken words. Interpret with route completeness. | Grounded facts, words, fact-to-span alignment | Persist exact NLG input frame and pre-TTS text. |
| Text-to-speech | Punctuation-pause deviation | Mean absolute configured-versus-realized pause difference at punctuation. Lower is better. | Punctuation offsets, expected pauses, word timestamps | Retain synthesis marks or run a versioned offline forced aligner. |
| Text-to-speech | Route-entity phoneme error rate | Phoneme edit distance for station/line names against canonical pronunciations. | Pronunciation lexicon, audio, aligned phonemes | Version a domain lexicon and use a fixed offline phoneme aligner. |
| Task outcome | Stage-wise optimality regret | Candidate utility minus the precomputed optimum at each constraint stage. Zero is optimal. | Stage optimum, selected candidate, utility | Freeze stage constraints/optimum at preflight and log stage selections. |
| Task outcome | Constraint-adaptation success | Revelations followed by a distinct compliant route before the next stage divided by revelations. | Reveal event, candidates, compliance, stage | Link each reveal to candidate validation using a stable stage identifier. |
| Whole dialogue | Critical-failure recovery latency | Seconds and turns from critical failure to restored valid shared state. | Failure/recovery events, timestamps, repair identifier | Emit typed events from the failing phase boundary. |
| Whole dialogue | Information gain per turn | Reduction in unresolved slots and active-objective regret per turn. | Per-turn unresolved slots, regret, state | Store a compact objective-state snapshot after every turn. |
| Metric validity | Early-failure area under receiver operating characteristic curve | Cross-validated final-failure discrimination using features available by a fixed early turn. | Multi-run early metrics, outcome, scenario groups | Export per-turn features and create scenario-grouped folds after the batch. |
| Metric validity | Controlled-degradation monotonicity | Rank correlation between perturbation severity and metric degradation. | Ordered severity, pair identifier, metric | Generate paired seeded conditions differing in one declared perturbation. |

## Priorities

The highest-value next additions are route-component F1,
constraint-preserving improvement rate, stage-wise optimality regret,
critical-error repair latency, and clarification-loop rate. They directly
explain task success and mostly require structured state near existing runtime
boundaries. Endpoint and prosody metrics should follow only after sample-level
timing and a versioned offline aligner are available.
