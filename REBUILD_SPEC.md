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
| `Configuration` | Defaults, persistent settings, job files, startup GUI |
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
4. Transfer time is added only when the transport line changes.
5. Agent B proposals must be connected and grounded in the network.
6. Duplicate route proposals are rejected and recorded.
7. Agent A reveals at most one new private constraint per turn.
8. A new constraint is revealed only after the current objective is met.
9. A replacement route is eligible only when it retains all stated
   constraints and remains within the acceptable duration threshold.
10. At the turn limit, Agent A chooses the best currently viable retained
    candidate.
11. Metrics are computed retrospectively from captured evidence.
12. Missing metric evidence is represented as `null`, never fabricated zero.

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

Structured and displayed complete paths use one labelled edge per network
step: `Foxtrot --T1--> Bravo --M2--> Zulu`. Walking uses
`Foxtrot --walk 5 min--> Bravo`. A station-only array is explicitly a station
sequence and must not be presented as a complete route. The validity, time,
and each progressive constraint optimum are emitted on separate lines. Each
added constraint must differ from the preceding qualifying path; otherwise its
layer is explicitly unavailable in configuration and protocol output.

## Configuration Contract

All experiment variables have defaults and can be overridden by:

1. startup GUI;
2. saved JSON settings;
3. `.job` base configuration and grids;
4. command-line arguments;
5. `COOP_NAVIGATION_SDS_*` environment variables.

Legacy `MINILLAMA_*` variables and keys remain read-compatible only.

The startup GUI is maximized, contains separate Experiment and Metrics cards,
uses scrollable compact groups, and closes before execution. Provider-specific
fields are created only for the selected language-model, TTS, or ASR
implementation. Named audio personas are the primary reproducible speech
condition; only controls that materially affect the selected provider are
shown. There is no runtime GUI.

Before the GUI closes, preflight validates provider packages, platform support,
model paths, local model cache availability, executables, required credentials,
and result storage. A validation failure leaves the GUI open and reports a
corrective action.

Saved JSON contains only fundamental experiment factors, selected
implementation settings, output location, and metric selection. Legacy
custom-prosody, laugh, and reference-audio fields remain readable in historical
records but are not exposed or persisted by new runs.
Every metric has two persisted settings: `enabled` and `tier`. Core metrics
are forced on for the run; supplementary metrics are optional.

## Provider Contract

### Language models

The common adapter supports provider-neutral message generation.

- Transformers: local PyTorch/Hugging Face inference.
- OpenAI-compatible: chat-completions HTTP API. Requires an API key from the
  startup GUI, `OPENAI_API_KEY`, or `--model-api-key`.
- Ollama: native local chat API.

Provider settings include model, endpoint, credentials, device, request
timeout, output tokens, input budget, and model-download policy.

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

Each metric has two controls: enabled on/off, and core/supplementary tier.
Core metrics are forced on. Supplementary metrics are calculated only when
enabled. Every metric declares:

- stable key and phase;
- core or supplementary tier;
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

- exact configuration snapshot;
- generated speech, raw ASR transcript, token misinterpretations, transcript
  corrections, and the exact heard-text input consumed by the listener;
- speech phase events and audio;
- candidate routes and selection evidence;
- network data and graph;
- runtime and failure diagnostics;
- retrospective metric values;
- metric catalog metadata;
- phase logs;
- analysis workbook.

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
- every metric family has at least seven core metrics;
- the console prints calculation steps without duplicate summaries.
