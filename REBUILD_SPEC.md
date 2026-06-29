# CoopNavigationSDS System Specification

## Objective

CoopNavigationSDS is a reproducible full-speech research system for measuring
cooperative route-finding dialogue. Agent A is a transit-hotline caller with
private preferences. Agent B proposes and revises grounded routes. The system
must support interactive configuration, scripted runs, factorial batches,
complete pipeline traces, retrospective metrics, and analysis-ready outputs.

## Component Boundaries

| Component | Responsibility |
| --- | --- |
| `Configuration` | Shared schemas, defaults, persistent settings, jobs, startup GUI |
| `NaturalLanguageGeneration` | Agent policies, prompts, utterance generation, LLM adapters |
| `TextToSpeech` | TTS engines, synthesis controls, audio personas |
| `AutomaticSpeechRecognition` | ASR engine API and provider selection |
| `NaturalLanguageUnderstanding` | Heard-text route and constraint interpretation |
| `DialogManagement` | Stages, state, candidate memory, turn orchestration |
| `TransportNetwork` | Network, scenarios, route search, timing, constraints |
| `EvaluationMetrics` | Metric catalog, computation, NISQA, DNSMOS |
| `ResultsAndArtifacts` | Protocols, logs, audio, SVG, JSON, XLSX |

Dependencies point toward component interfaces. Package initializers must not
eagerly import modules that create cycles. Optional provider libraries load
only after provider selection.

## Runtime Invariants

1. Every utterance passes through NLG, TTS, audio, ASR, and NLU.
2. The receiving agent reacts only to recognized text and committed state.
3. TTS or ASR failure stops the run and records diagnostics.
4. Agent A knows valid station and line names but not line stops, connectivity,
   schedules, route candidates, or service attributes.
5. Transfer time is added only when the transport line changes.
6. Agent B proposals must be connected and grounded in the network.
7. Duplicate route proposals are rejected and recorded.
8. Agent A reveals at most one new private constraint per turn.
9. A new constraint is revealed only after the current objective is met.
10. A replacement route is eligible only when it retains all stated
   constraints and remains within the acceptable duration threshold.
11. At the turn limit, Agent A chooses the best currently viable retained
    candidate.
12. Metrics are computed retrospectively from captured evidence.
13. Missing metric evidence is represented as `null`, never fabricated zero.
14. Raw `metric_inputs.json` evidence is persisted before derived metrics run.
15. Credentials are never persisted in settings or result metadata.
16. A deterministic smoke condition traverses orchestration and artifact
    generation without model downloads or external services.
17. `results_root` is the only output root; every artifact for an execution is
    written to its single flat run folder.
18. Each agent retains its own intended utterances and only the other agent's
    recognized pipeline transcript; source text is never substituted for ASR.
19. Clarification targets one missing slot or term, is not repeated after
    confirmation, and ends explicitly when its configured budget is exhausted.
20. Clock notation is synthesized in recognizer-friendly form, including
    explicit minute zeroes such as `08:07` -> `eight oh seven`.
21. Spoken or ASR-rendered compact clock variants such as `8-7`, `8, 7`,
    `8 7`, `eight seven`, and `eight oh seven` may update the departure-time
    semantic slot only when the utterance is a departure-time expression or a
    focused time-repair answer; the raw ASR transcript remains unchanged.
22. Batch execution writes all raw condition evidence before calculating
    metrics and cross-run validity reports.
23. First-deviation, task-focus, correction-burden, and constraint-count
    metrics must be trace-derived and reproducible from stored evidence.

## Conversation Phases

### 1. Route validity and time

Agent A states start, destination, and departure time. Agent B gives one
complete route. The route must reach the destination and satisfy:

```text
duration < optimal_duration * acceptable_duration_ratio
```

The exact whole-minute limit is calculated before dialogue startup.

### 2. Progressive constraints

Agent A reveals one configured constraint after the previous phase succeeds.
Supported constraints include:

- maximum line changes relative to the constrained baseline;
- no near-capacity vehicle;
- acceptable delay-risk class;
- acceptable missed-transfer-risk class;
- possession of exactly two of the three public transport tickets;
- maximum cumulative walking time;

Agent B may preserve an earlier route when it already satisfies the new
constraint, but it must still answer the latest heard request directly.

The pre-dialogue baseline is calculated independently for validity, fastest
time, constraint 1, constraints 1-2, and constraints 1-3. A private ticket or
walking constraint must not influence an earlier layer before it is revealed.

### 3. Comparison and selection

At least `minimum_compared_routes` distinct valid routes are discussed before
normal satisfactory closure. Candidate selection ranks:

1. reaches destination;
2. satisfies acceptable duration;
3. satisfies all stated constraints;
4. minimizes constraint misses;
5. minimizes duration.

The final spoken choice identifies transport mode, line code, origin station,
and destination station for each ride leg. Public lines are named `M1`-`M20`,
`T1`-`T25`, or `B1`-`B30`. Walking is expressed as minutes between named
stations and has no line code.

Structured edge records retain every network step. Human-readable paths
condense consecutive edges on one line while preserving intermediate stops:
`Foxtrot --tram T1 (Bravo, Charlie)--> Delta --metro M2--> Zulu`. Walking uses
`Foxtrot --walk 5 min--> Bravo`. A station-only array is explicitly a station
sequence and must not be presented as a complete route. The validity, time,
and each progressive constraint optimum are emitted on separate lines. Each
added constraint must differ from the preceding qualifying path; otherwise its
layer is explicitly unavailable in configuration and protocol output.

Standard experiment scenarios must pass preflight with a distinct optimum for
every progressive constraint stage and at least one viable comparison route.
The protocol records both optima and the route-change result for every stage.

## Configuration Contract

All experiment variables have defaults and can be overridden by:

1. startup GUI;
2. saved JSON settings;
3. `.job` base configuration and grids;
4. command-line arguments;
5. `COOP_NAVIGATION_SDS_*` environment variables.

Legacy `MINILLAMA_*` variables and keys remain read-compatible only.

The startup GUI is fullscreen and combines configuration plus relevant metrics
inside eight independently scrollable chronological phase cards arranged in a
two-by-four dashboard. Every card can be hidden or restored. Network and staged
route previews, metric readiness, and the result/export plan remain visible.
The horizontal sash between rows changes card heights; each row has three
independent vertical sashes for card widths. All eight areas are therefore
resizable in both dimensions by direct dragging. Text wraps automatically and
explicit newlines are reserved for ordered multi-line evidence such as numbered
route layers.
Provider-specific fields are created only for
the selected language-model, TTS, or ASR implementation. Named audio personas
are the primary reproducible speech condition; only controls that materially
affect the selected provider are shown. There is no runtime GUI.

Before the GUI closes, preflight validates provider packages, platform support,
model paths, local model cache availability, executables, required credentials,
and result storage. A validation failure leaves the GUI open and reports a
corrective action.

Saved JSON contains only fundamental experiment factors, selected
implementation settings, and output location. Legacy
custom-prosody, laugh, and reference-audio fields remain readable in historical
records but are not exposed or persisted by new runs.
Metrics are obligatory catalog entries rather than configuration settings.
Each is calculated retrospectively when its evidence exists, or recorded as
`null` with an explicit missing-evidence reason.

After resolution and preflight, all values are stored in one recursively
immutable experiment specification. It has a stable non-secret fingerprint,
schema version, source, and resolution timestamp. Runtime phases may read but
cannot mutate it. Batch condition parameter mappings are immutable as well.

## Provider Contract

### Language models

The common adapter supports provider-neutral message generation.

- Transformers: local PyTorch/Hugging Face inference.
- OpenAI-compatible: chat-completions HTTP API. Requires an API key from the
  startup GUI, `OPENAI_API_KEY`, or `--model-api-key`.
- Ollama: native local chat API.

Provider settings include model, endpoint, credentials, device, request
timeout, output tokens, input budget, and the explicit
`allow_model_download` switch for Transformers.

### Speech

TTS and ASR are independently selectable. Each provider must implement:

- construction from `SpeechPipelineConfig`;
- explicit availability and health diagnostics;
- deterministic trace fields;
- timeout handling;
- non-empty output validation;
- platform and dependency reporting.

The deterministic file engines are test controls, not substitutes for failed
neural or native engines.

## Transport Network

The network is deterministic for configured seeds and contains metro, tram,
bus, and walking services.

- Every station is served by exactly two distinct public modes from metro,
  tram, and bus.
- Walking is additional, available locally at every station, and does not
  count toward the two-public-mode invariant.
- Each caller owns exactly two public-mode tickets; the third mode is invalid
  after that private constraint has been stated.
- Cumulative walking must not exceed the caller's configured 5- or 10-minute
  persona limit unless the experiment explicitly overrides it.
- Travel-time scaling is fastest to slowest: metro, tram, bus, walking.
- Lines have headway, fullness class, delay class, and line-specific segment
  travel times.
- Stations have unique transfer times.
- Risk exposed in dialogue uses low, medium, or high classes.
- Fullness exposed in dialogue is near capacity or not near capacity.

Startup preflight proves journey reachability, a constraint-compatible route,
and the configured number of viable alternatives for each progressive stage.

## Metric Contract

The 12 canonical metric families are:

0. user simulation;
1. audio input and turn-taking;
2. automatic speech recognition;
3. spoken-language understanding;
4. dialogue state tracking;
5. dialogue management;
6. backend task execution;
7. natural-language generation;
8. text-to-speech;
9. task outcome;
10. whole dialogue;
11. metric validity.

Metrics have no enable/disable or tier controls. Every catalog metric is part
of every run's retrospective evaluation contract. Every metric declares:

- stable key and phase;
- deterministic, reference, or learned evidence class;
- unit;
- required trace fields;
- calculation method;
- missing-data policy.

Console output presents one non-duplicated phase report with numbered values
and calculation methods. JSON and XLSX exports retain stable keys.

## Result Contract

Every run receives one timestamped folder directly under the configured result
root. It contains:

- `run_summary.json` and `conditions.jsonl` with an identical schema for single
  and batch execution, suitable for concatenation across runs;
- a root-level `naming_scheme.json` whose keys are compact abbreviations used
  in result folder and condition names and whose values describe the
  corresponding configuration setting; this file is refreshed when the naming
  scheme changes and is also embedded in run and batch manifests;
- exact configuration snapshot;
- generated speech, raw ASR transcript, token misinterpretations, transcript
  corrections, and the exact heard-text input consumed by the listener;
- perspective-specific Agent A and Agent B memory histories;
- speech phase events and audio;
- candidate routes and selection evidence;
- network data and graph;
- runtime and failure diagnostics;
- retrospective metric values;
- metric catalog metadata;
- phase logs;
- analysis workbook.

Manifest artifact paths are relative to the run folder. Detailed calculation
evidence remains in retrospective and long-form exports; the summary and
condition table provide the concise analysis entry point.

Batch runs additionally include condition identifiers, iteration values, and
cross-run validity metrics.

Batch `.job` definitions may cross explicit `parameter_values` and inclusive
`parameter_ranges` using `{start, stop, step}`. Network seed, dialogue limits,
calculation budgets, speech-pipeline fields, and scenario constraints are
resolved independently for every generated condition.

## Acceptance Tests

A release is acceptable when:

- all unit and integration tests pass;
- the package compiles without import cycles;
- all provider registries expose their configured implementations;
- a deterministic full-speech run completes and writes artifacts;
- Agent B proposes distinct routes;
- Agent A reveals constraints sequentially;
- earlier constraints remain satisfied after revisions;
- final-turn selection produces a natural spoken closure;
- every metric family has at least seven obligatory metrics;
- the console prints calculation steps without duplicate summaries.
