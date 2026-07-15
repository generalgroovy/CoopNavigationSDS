"""Generate chapter-local thesis writing aids.

The generated files are derived writing support only. They do not alter raw
experiment results. The goal is to make each thesis chapter writable with three
small files open at once: outline, terminology, and references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "thesis_chapter_workbench"


@dataclass(frozen=True)
class Reference:
    paper: str
    cite_for: list[str]
    use_in_chapter: list[str]
    boundary: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Term:
    name: str
    definition: str
    explanation: list[str]


@dataclass(frozen=True)
class SectionPlan:
    title: str
    purpose: str
    write: list[str]
    avoid: list[str] = field(default_factory=list)
    evidence_hooks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chapter:
    number: str
    slug: str
    title: str
    function: str
    argument: list[str]
    sections: list[SectionPlan]
    terms: list[Term]
    references: list[Reference]


COMMON_REFERENCES = {
    "paradise": Reference(
        paper="Walker et al. (1997), PARADISE: A Framework for Evaluating Spoken Dialogue Agents",
        cite_for=[
            "spoken dialogue evaluation should combine task success with dialogue behavior",
            "dialogue quality cannot be reduced to a final success label",
            "task requirements and interaction costs can be separated analytically",
        ],
        use_in_chapter=[
            "motivate the thesis split between task outcome metrics and process/phase metrics",
            "justify reporting turns, repairs, latency, and route success together",
        ],
        boundary=[
            "do not claim this thesis estimates human user satisfaction unless human ratings are collected",
        ],
    ),
    "communicator": Reference(
        paper="Walker, Passonneau, and Boland (2001), Quantitative and Qualitative Evaluation of DARPA Communicator Spoken Dialogue Systems",
        cite_for=[
            "full SDS evaluation benefits from multiple measures rather than one final score",
            "system-level evaluation can combine task completion, dialogue behavior, and error analysis",
            "comparative SDS evaluation requires explicit conditions and documented evidence",
        ],
        use_in_chapter=[
            "position CoopNavigationSDS as a small controlled evaluation instrument inspired by full-system SDS evaluation",
            "support the decision to report both success categories and diagnostic phase evidence",
        ],
        boundary=[
            "do not imply DARPA Communicator used the same LLM or route-navigation setup",
        ],
    ),
    "multiwoz": Reference(
        paper="Budzianowski et al. (2018), MultiWOZ",
        cite_for=[
            "task-oriented dialogue can be represented using goals, domains, dialogue state, and system/user turns",
            "structured goals and slots support automatic evaluation of task progress",
        ],
        use_in_chapter=[
            "justify explicit start station, destination, time, route, and constraint state",
        ],
        boundary=[
            "MultiWOZ is primarily text dialogue; use it for task-state structure, not speech-channel claims",
        ],
    ),
    "sgd": Reference(
        paper="Rastogi et al. (2020), Schema-Guided Dialogue",
        cite_for=[
            "task-oriented dialogue state can be defined through explicit schemas and slots",
            "structured slot descriptions make evaluation more reproducible across services",
        ],
        use_in_chapter=[
            "support the use of structured semantic frames and constraint slots",
        ],
        boundary=[
            "do not claim the thesis covers broad multi-service schemas; it uses one controlled navigation domain",
        ],
    ),
    "spokenwoz": Reference(
        paper="Si et al. (2023), SpokenWOZ",
        cite_for=[
            "spoken task-oriented dialogue differs from written dialogue because speech introduces recognition and interaction errors",
            "speech evidence can reveal failures hidden by clean text transcripts",
        ],
        use_in_chapter=[
            "justify paired text-only and audio-variant conditions",
            "support the need to inspect ASR output and normalized understanding separately",
        ],
        boundary=[
            "SpokenWOZ is a dataset; CoopNavigationSDS is an experimental framework",
        ],
    ),
    "slue": Reference(
        paper="Shon et al. (2021), SLUE",
        cite_for=[
            "spoken language understanding should evaluate semantic content, not only word transcription",
            "named entities are important evidence in speech understanding",
        ],
        use_in_chapter=[
            "justify station, line, time, and constraint entity metrics",
        ],
        boundary=[
            "do not treat generic WER as sufficient for route-dialogue understanding",
        ],
    ),
    "user_sim": Reference(
        paper="Schatzmann et al. (2007), agenda-based user simulation for dialogue systems",
        cite_for=[
            "simulated users support repeatable dialogue-system evaluation",
            "user simulation trades ecological realism for experimental control",
        ],
        use_in_chapter=[
            "justify Agent A as a controlled caller with private goals and staged constraints",
        ],
        boundary=[
            "do not claim simulated callers replace human callers",
        ],
    ),
    "bleu": Reference(
        paper="Papineni et al. (2002), BLEU",
        cite_for=[
            "automatic text-generation evaluation can be repeatable and scalable",
            "lexical overlap is a historically important but surface-level evaluation family",
        ],
        use_in_chapter=[
            "contrast text overlap with task-grounded route validation",
        ],
        boundary=[
            "BLEU does not verify whether a proposed route is executable",
        ],
    ),
    "rouge": Reference(
        paper="Lin (2004), ROUGE",
        cite_for=[
            "overlap-based metrics evaluate shared textual units",
            "lexical metrics can be useful but do not by themselves establish task correctness",
        ],
        use_in_chapter=[
            "define lexical metrics as supplementary wording evidence",
        ],
        boundary=[
            "do not interpret high lexical similarity as route validity",
        ],
    ),
    "meteor": Reference(
        paper="Banerjee and Lavie (2005), METEOR",
        cite_for=[
            "automatic text metrics can use alignment beyond exact n-gram overlap",
        ],
        use_in_chapter=[
            "show the range from strict lexical overlap toward more flexible text matching",
        ],
        boundary=[
            "semantic alignment is still not a substitute for network-grounded validation",
        ],
    ),
    "bertscore": Reference(
        paper="Zhang et al. (2020), BERTScore",
        cite_for=[
            "contextual embeddings can compare candidate and reference text semantically",
        ],
        use_in_chapter=[
            "distinguish semantic text quality from task-grounded correctness",
        ],
        boundary=[
            "embedding similarity does not prove that station order, line names, or constraints are correct",
        ],
    ),
    "bleurt": Reference(
        paper="Sellam et al. (2020), BLEURT",
        cite_for=[
            "learned text metrics can approximate human judgments in some generation settings",
        ],
        use_in_chapter=[
            "frame learned metrics as useful but not sufficient for route-task evaluation",
        ],
        boundary=[
            "do not use learned text similarity as the primary success criterion",
        ],
    ),
    "stir": Reference(
        paper="Taal et al. (2011), STOI",
        cite_for=[
            "objective speech intelligibility can be estimated from audio signals",
        ],
        use_in_chapter=[
            "motivate audio intelligibility diagnostics when required signal evidence is available",
        ],
        boundary=[
            "speech-quality metrics diagnose audio, not route reasoning",
        ],
    ),
    "dnsmos": Reference(
        paper="Reddy et al. (2021), DNSMOS",
        cite_for=[
            "non-intrusive speech quality estimation can approximate perceived speech quality without a clean reference",
        ],
        use_in_chapter=[
            "classify audio quality as diagnostic evidence for TTS/channel problems",
        ],
        boundary=[
            "do not treat DNSMOS as a task-success metric",
        ],
    ),
    "nisqa": Reference(
        paper="Mittag et al. (2021), NISQA",
        cite_for=[
            "neural non-intrusive speech quality assessment estimates perceived audio quality",
        ],
        use_in_chapter=[
            "explain why audio quality can be measured separately from ASR and task outcome",
        ],
        boundary=[
            "only calculate or interpret NISQA where the implementation has the required audio evidence",
        ],
    ),
    "problematic": Reference(
        paper="Walker et al. (2000), Problematic Dialogue Predictor",
        cite_for=[
            "early dialogue behavior can indicate problematic interactions",
            "repair, misunderstanding, and interaction-cost evidence can support failure prediction",
        ],
        use_in_chapter=[
            "support failure-localization and early-warning metric interpretation",
        ],
        boundary=[
            "do not claim the thesis trains a deployed predictor unless that model is actually implemented",
        ],
    ),
}


def t(name: str, definition: str, *explanation: str) -> Term:
    return Term(name=name, definition=definition, explanation=list(explanation))


def s(title: str, purpose: str, write: list[str], avoid: list[str] | None = None, evidence: list[str] | None = None) -> SectionPlan:
    return SectionPlan(title=title, purpose=purpose, write=write, avoid=avoid or [], evidence_hooks=evidence or [])


CHAPTERS: list[Chapter] = [
    Chapter(
        number="0",
        slug="thesis_core",
        title="Thesis Core",
        function="Keep the whole thesis aligned with the central object: automatic evaluation of spoken task-oriented dialogue systems using CoopNavigationSDS as the research instrument.",
        argument=[
            "The software is not the research result by itself; it is the controlled instrument for collecting evidence.",
            "The evaluated object is Agent B as a route-information dialogue system under different backend and speech-channel conditions.",
            "Task outcome, constraint satisfaction, and phase-wise evidence must be interpreted together.",
            "Raw run evidence is authoritative; generated tables and conclusions are derived views.",
        ],
        sections=[
            s(
                "Central claim",
                "State the thesis in one precise sentence.",
                [
                    "This thesis studies whether a controlled cooperative navigation task can automatically evaluate spoken task-oriented dialogue systems phase-wise.",
                    "The key contribution is explainability of success, semi-success, and failure from logged evidence, not only final outcome labels.",
                    "Use the phrase 'likely failure origin' instead of causal attribution unless an ablation proves causality.",
                ],
                ["Do not claim universal SDS evaluation, real-user validity, or model-size causality."],
            ),
            s(
                "Validity boundary",
                "Define the claims that are allowed and the claims that are outside scope.",
                [
                    "Construct validity: every metric must map to a named construct and evidence field.",
                    "Internal validity: direct model comparisons require matched non-model conditions.",
                    "External validity: the transport network, caller personas, and speech channel are controlled abstractions.",
                    "Statistical conclusion validity: report denominators before percentages.",
                ],
            ),
            s(
                "Current empirical basis",
                "Use current derived results without overclaiming.",
                [
                    "Active thesis-relevant deduplicated rows: 1319.",
                    "Completed active thesis rows: 557.",
                    "Fully crossed matched subset: 11 condition groups, 220 runs.",
                    "The fully crossed subset supports strongest direct text/audio and Agent A/Agent B comparisons.",
                ],
                evidence=[
                    "See docs/THESIS_RESULT_CONFIGURATION_EFFECTS.md for current denominators.",
                    "See docs/THESIS_METRIC_VALIDITY_ASSESSMENT.md for metric-role classification.",
                ],
            ),
        ],
        terms=[
            t("Automatic evaluation", "Metric-based evaluation computed from stored evidence rather than direct human ratings.", "In this thesis, automatic evaluation is useful because route validity and constraint satisfaction can be checked objectively."),
            t("Phase-wise evidence", "Intermediate outputs from pipeline stages such as TTS, ASR, NLU, dialogue state, route validation, NLG, and final outcome.", "Phase-wise evidence makes failure explanation possible."),
            t("Outcome metric", "A metric that describes final task status.", "Examples: task success, semi-success, unsuccessful completed dialogue, execution failure."),
            t("Diagnostic metric", "A metric used to identify where and how a run degraded.", "Examples: ASR station F1, repair success, grounded proposal score, state drift."),
            t("Completed run", "A run that reaches the end of the dialogue pipeline and produces usable outcome evidence.", "Navigation failure inside a completed run is analytically useful and should not be removed."),
        ],
        references=[COMMON_REFERENCES["paradise"], COMMON_REFERENCES["communicator"], COMMON_REFERENCES["problematic"]],
    ),
    Chapter(
        number="1",
        slug="introduction",
        title="Introduction",
        function="Motivate the research problem, define the scope, state research questions, and present contributions without implementation detail.",
        argument=[
            "Spoken task-oriented systems are practically useful but difficult to evaluate.",
            "Final task success does not explain whether failure came from speech, understanding, dialogue management, route reasoning, or generation.",
            "A route task provides externally checkable correctness and therefore a useful testbed for automatic evaluation.",
            "The thesis asks how far phase-wise automatic metrics can explain successful and failed SDS interactions.",
        ],
        sections=[
            s(
                "1.1 Motivation and background",
                "Open with the relevance of spoken task-oriented systems.",
                [
                    "Mention navigation, hotline-style assistance, accessibility, and hands-free use.",
                    "Explain that speech adds timing, audibility, recognition, and repair problems.",
                    "Use route finding as a concrete example where a caller needs correct actionable instructions.",
                ],
                ["Do not start with implementation details or model names."],
            ),
            s(
                "1.2 Evaluation difficulty",
                "Explain why final success alone is insufficient.",
                [
                    "A dialogue can fail despite fluent text if a station is misrecognized.",
                    "A dialogue can semi-succeed if it finds a valid route but violates a revealed constraint.",
                    "Speech and ASR failures can be hidden if only normalized final transcripts are inspected.",
                    "Automatic metrics must therefore be phase-aware and task-grounded.",
                ],
            ),
            s(
                "1.3 Problem statement",
                "Narrow from general SDS evaluation to the thesis gap.",
                [
                    "Current automatic metrics often measure surface text, semantic similarity, or final success separately.",
                    "For spoken task-oriented dialogue, the evaluation needs to connect speech evidence, semantic state, dialogue behavior, and task outcome.",
                    "The problem is not only to score a dialogue but to make the score interpretable.",
                ],
            ),
            s(
                "1.4 Research questions",
                "State research questions so later chapters can answer them directly.",
                [
                    "RQ1: Can a controlled route-dialogue framework produce reliable evidence for automatic phase-wise SDS evaluation?",
                    "RQ2: Which metrics best distinguish successful, semi-successful, and unsuccessful completed dialogues?",
                    "RQ3: How do speech conditions affect task success compared with paired text-only controls?",
                    "RQ4: How do Agent A and Agent B backend choices affect outcome and diagnostic metrics under matched conditions?",
                    "RQ5: Which failures can be localized to speech, understanding, dialogue management, or route reasoning evidence?",
                ],
            ),
            s(
                "1.5 Contributions",
                "List concrete thesis contributions.",
                [
                    "A reproducible route-dialogue experiment framework.",
                    "A logging schema that preserves phase evidence for retrospective metrics.",
                    "A metric set separating outcome, phase diagnostics, and supplementary indicators.",
                    "A matched text/audio analysis showing the effect of speech-channel degradation.",
                    "A validity-aware interpretation guide for automatic SDS metrics.",
                ],
            ),
        ],
        terms=[
            t("Spoken dialogue system", "A dialogue system that communicates through spoken input and/or spoken output.", "This thesis models it as a sequence of observable phases."),
            t("Task-oriented dialogue", "Dialogue directed toward a concrete, externally checkable goal.", "The goal here is finding and accepting a valid route under constraints."),
            t("Route-grounded evaluation", "Evaluation where route candidates are checked against a transport network.", "This supplies independent correctness evidence."),
            t("Semi-success", "A completed dialogue with partial task achievement, such as a valid route but violated constraint.", "Semi-success is crucial because binary success hides informative failures."),
        ],
        references=[COMMON_REFERENCES["paradise"], COMMON_REFERENCES["communicator"], COMMON_REFERENCES["problematic"], COMMON_REFERENCES["spokenwoz"]],
    ),
    Chapter(
        number="2",
        slug="background_related_work",
        title="Background and Related Work",
        function="Define the technical and theoretical context: dialogue systems, spoken pipelines, task-oriented structure, LLM backends, and related evaluation traditions.",
        argument=[
            "Task-oriented dialogue provides structured goals and state.",
            "Spoken dialogue adds a speech pipeline where errors can propagate.",
            "LLM-based systems change implementation details but do not remove the need for task grounding.",
            "Related work motivates the thesis design but does not already solve phase-wise automatic evaluation for this controlled route task.",
        ],
        sections=[
            s(
                "2.1 Dialogue systems and SDS",
                "Introduce dialogue systems before narrowing to speech.",
                [
                    "Define user, system, turn, dialogue state, system action, and response.",
                    "Explain the classical SDS pipeline: user speech, ASR, NLU, dialogue state, dialogue management, NLG, TTS.",
                    "Use the pipeline as an analysis model, even when some components are implemented with LLMs.",
                ],
            ),
            s(
                "2.2 Task-oriented dialogue",
                "Show why this task can be evaluated automatically.",
                [
                    "Task-oriented systems have goals, constraints, and external task state.",
                    "For navigation, the task state is the route network and constraint set.",
                    "Slot-like variables include start station, destination, time, line names, route duration, transfer count, crowding, and delay risk.",
                ],
            ),
            s(
                "2.3 Spoken versus text dialogue",
                "Clarify why paired text/audio runs matter.",
                [
                    "Text runs test dialogue and route reasoning without speech degradation.",
                    "Audio runs test the additional TTS/ASR transmission channel.",
                    "A paired design is stronger than comparing unrelated text and audio conditions.",
                ],
            ),
            s(
                "2.4 LLM-based dialogue systems",
                "Position Agent B backends without turning the thesis into a model leaderboard.",
                [
                    "LLMs can produce flexible responses but may still hallucinate routes or ignore constraints.",
                    "Backend size is only one dimension; family, instruction tuning, provider, prompt, and decoding also matter.",
                    "The thesis compares selected feasible backends under controlled conditions.",
                ],
            ),
            s(
                "2.5 Error propagation",
                "Prepare the logic for phase-wise metrics.",
                [
                    "Speech errors can corrupt entity recognition.",
                    "Entity errors can corrupt dialogue state.",
                    "State errors can lead to invalid or constraint-violating routes.",
                    "Later metrics may detect a failure, but earlier evidence is needed to localize it.",
                ],
            ),
        ],
        terms=[
            t("Dialogue state", "The system's current structured representation of the task and conversation.", "In this thesis it includes known trip details, revealed constraints, current route, and satisfaction status."),
            t("Dialogue management", "The phase that decides the next system action.", "Examples: propose route, ask clarification, repair misunderstanding, accept final route."),
            t("Large language model backend", "A concrete LLM/model provider used to generate Agent B behavior.", "Treat it as an implementation of the evaluated system role, not as the only explanation of performance."),
            t("Error propagation", "The process by which an error in one phase affects later phases.", "For example, ASR losing a station name can produce invalid route reasoning."),
        ],
        references=[COMMON_REFERENCES["multiwoz"], COMMON_REFERENCES["sgd"], COMMON_REFERENCES["spokenwoz"], COMMON_REFERENCES["slue"], COMMON_REFERENCES["user_sim"]],
    ),
    Chapter(
        number="3",
        slug="evaluation_validity",
        title="Evaluation Concepts and Validity Threats",
        function="Define metric families, validity criteria, and why task-grounded phase metrics are needed before the methodology is described.",
        argument=[
            "Automatic metrics are useful only when the construct and evidence source are explicit.",
            "Lexical and semantic text metrics are supplementary because they cannot verify route executability.",
            "Speech metrics diagnose audio or recognition quality but do not by themselves establish task success.",
            "Validity threats must be visible before results are interpreted.",
        ],
        sections=[
            s(
                "3.1 Human versus automatic evaluation",
                "Set the scope of automatic evaluation.",
                [
                    "Human evaluation is valuable but expensive and hard to scale.",
                    "Automatic evaluation is repeatable but only valid for explicitly defined constructs.",
                    "The thesis uses automatic metrics as controlled evidence, not as a replacement for all human judgment.",
                ],
            ),
            s(
                "3.2 Construct validity",
                "Define what each metric is allowed to mean.",
                [
                    "Task success metrics measure whether the dialogue achieved the route goal.",
                    "Phase metrics measure intermediate evidence that may explain success or failure.",
                    "Outcome-confirming metrics overlap with the success definition and should not be sold as independent predictors.",
                ],
            ),
            s(
                "3.3 Lexical and semantic text metrics",
                "Use the passed XW-style depth here: explain usefulness and limits precisely.",
                [
                    "Lexical metrics compare surface overlap and can detect wording similarity or divergence.",
                    "Semantic text metrics compare meaning more flexibly than exact overlap.",
                    "Both families can miss route executability errors such as wrong station order, omitted line names, or violated constraints.",
                    "Therefore they are supplementary indicators in this thesis.",
                ],
            ),
            s(
                "3.4 Speech and semantic metrics",
                "Separate audio quality, ASR transcript quality, and task semantics.",
                [
                    "Audio quality metrics describe the signal or perceived quality.",
                    "WER describes word-level transcription errors.",
                    "Entity and semantic ASR metrics are closer to task success because they measure station, line, time, and constraint preservation.",
                    "The thesis should emphasize semantic ASR/entity metrics over WER when explaining route failures.",
                ],
            ),
            s(
                "3.5 Validity threats",
                "Make limitations part of the analysis rather than an afterthought.",
                [
                    "Internal validity: compare only matched conditions for direct model claims.",
                    "External validity: synthetic network and simulated callers limit generalization.",
                    "Reliability: metrics must be regenerable from raw logs.",
                    "Missing data: unavailable evidence must be reported as unavailable, not imputed silently.",
                ],
            ),
        ],
        terms=[
            t("Construct validity", "The degree to which a metric actually measures the concept it claims to measure.", "Route validity has strong construct validity because the network can check it directly."),
            t("Lexical metric", "A metric based on surface text overlap or token similarity.", "Useful for wording comparison, weak for route correctness."),
            t("Semantic metric", "A metric intended to compare meaning rather than only surface words.", "Still not sufficient for executable route validation."),
            t("Metric role", "The interpretation category assigned to a metric.", "Current roles include outcome-confirming, diagnostic phase, associated diagnostic, supplementary, metric quality, and execution/constant."),
        ],
        references=[COMMON_REFERENCES["bleu"], COMMON_REFERENCES["rouge"], COMMON_REFERENCES["meteor"], COMMON_REFERENCES["bertscore"], COMMON_REFERENCES["bleurt"], COMMON_REFERENCES["slue"], COMMON_REFERENCES["nisqa"], COMMON_REFERENCES["dnsmos"]],
    ),
    Chapter(
        number="4",
        slug="methodology_research_design",
        title="Methodology and Research Design",
        function="Describe the experiment precisely enough that a reader can reproduce the logic without reading the code.",
        argument=[
            "The experiment is a controlled route-dialogue task with two agents and explicit knowledge boundaries.",
            "Agent A has private goals and staged constraints; Agent B must cooperate by proposing and refining routes.",
            "Text-only and speech runs separate dialogue reasoning from speech-channel degradation.",
            "All metric-relevant evidence is captured during execution and evaluated retrospectively.",
        ],
        sections=[
            s(
                "4.1 Experimental unit and condition",
                "Define what one row of analysis means.",
                [
                    "One condition/run combines Agent A, Agent B, scenario, persona, audio persona, TTS/ASR settings, seed/repetition, and run type.",
                    "The run, not the individual turn, is the primary unit for outcome rates.",
                    "Turns provide nested evidence for diagnosis.",
                ],
            ),
            s(
                "4.2 Agent roles and knowledge boundaries",
                "Protect the experiment from hidden shared knowledge.",
                [
                    "Agent A knows start, destination, time, station/line names, and private constraints as staged goals.",
                    "Agent B knows the network and must infer Agent A's needs from what it hears.",
                    "Each agent maintains its own memory based on its own intended speech and understood transcript.",
                    "Clarification and repair must occur through dialogue, not shared hidden state.",
                ],
            ),
            s(
                "4.3 Dialogue stages",
                "Explain the staged task logic.",
                [
                    "Stage 1: establish valid route from start to destination within acceptable time.",
                    "Stage 2: reveal first additional constraint if Stage 1 is satisfied.",
                    "Stage 3: reveal second constraint if Stage 2 is satisfied.",
                    "Final: Agent A accepts the best route if goals are satisfied or classifies the result as semi-success/unsuccessful.",
                ],
            ),
            s(
                "4.4 Route task and network",
                "Describe why the network is useful for evaluation.",
                [
                    "Routes must specify station sequence, line names, and transport segments.",
                    "Optimal route is recalculated per constraint layer.",
                    "Constraints are designed to change the optimal route where possible.",
                    "This makes cooperation and route revision observable.",
                ],
            ),
            s(
                "4.5 Text and speech channel",
                "Describe paired controls.",
                [
                    "Text-only runs preserve the dialogue policy without TTS/ASR degradation.",
                    "Audio-variant runs transmit utterances through TTS and ASR.",
                    "The paired design supports estimating the performance loss caused by the speech channel.",
                ],
            ),
            s(
                "4.6 Evidence logging",
                "Show that retrospective metrics are legitimate.",
                [
                    "Log intended utterance, TTS output, ASR raw transcript, normalized understanding, corrections, memory update, route proposal, validation, timing, and outcome.",
                    "Do not aggregate away raw evidence during runtime.",
                    "Derived metrics must be traceable back to fields in the run folder.",
                ],
            ),
        ],
        terms=[
            t("Agent A", "The simulated caller/user role with private travel goals and constraints.", "UserLM is the primary thesis caller; TinyLlama can be used as a control stratum."),
            t("Agent B", "The evaluated route-information dialogue-system role.", "Agent B is instantiated by different LLM backends."),
            t("Knowledge boundary", "A rule defining what each agent can know directly.", "This preserves the authenticity of dialogue-based cooperation."),
            t("Matched condition", "A set of runs that differ only in the factor being compared.", "Matched conditions are required for direct model or channel claims."),
            t("Retrospective metric calculation", "Metrics are calculated after the run from stored evidence.", "This improves auditability and reproducibility."),
        ],
        references=[COMMON_REFERENCES["paradise"], COMMON_REFERENCES["multiwoz"], COMMON_REFERENCES["sgd"], COMMON_REFERENCES["spokenwoz"], COMMON_REFERENCES["user_sim"]],
    ),
    Chapter(
        number="5",
        slug="metric_selection",
        title="Metric Selection",
        function="Define the selected metrics, formulas, evidence requirements, interpretation, and limitations.",
        argument=[
            "Metrics are selected because they answer a research question and can be calculated from reliable evidence.",
            "Outcome metrics describe final status; diagnostic metrics explain phase behavior; supplementary metrics add context.",
            "Metrics without required evidence should be marked unavailable rather than computed from assumptions.",
            "Metric validity is assessed by role, evidence availability, and relation to outcome.",
        ],
        sections=[
            s(
                "5.1 Selection principle",
                "Explain why each metric belongs.",
                [
                    "Include only metrics with a defined construct, required logged fields, calculation rule, and interpretation boundary.",
                    "Reject or disable metrics that cannot be calculated from captured evidence.",
                    "Separate direct outcome decompositions from independent diagnostic indicators.",
                ],
            ),
            s(
                "5.2 Core formulas",
                "Give formulas compactly.",
                [
                    "Task success rate = successful completed runs / completed runs.",
                    "Semi-success rate = semi-successful completed runs / completed runs.",
                    "Route-valid rate = runs with valid accepted/proposed route / completed runs.",
                    "Constraint satisfaction rate = satisfied revealed constraints / revealed constraints.",
                    "Audio-text delta = audio success rate minus paired text success rate.",
                    "Route optimality gap = proposed route duration minus constraint-layer optimal duration.",
                ],
            ),
            s(
                "5.3 Phase metrics",
                "Map metrics to pipeline phases.",
                [
                    "TTS/audio: audio availability, speech duration, intelligibility/quality where evidence exists.",
                    "ASR: WER, station/line/time entity preservation, semantic ASR error.",
                    "NLU: constraint extraction, route entity extraction, semantic frame correctness.",
                    "Dialogue state: constraint retention, state drift, shared-state consistency where observable.",
                    "Dialogue management: clarification rate, repair success, premature answer rate, stagnation.",
                    "Agent B response: grounded proposal score, hallucinated content, actionability.",
                    "NLG: faithfulness, semantic adequacy, executable utterance rate.",
                    "Whole dialogue: task success, semi-success, turns, runtime, failure-localization candidate.",
                ],
            ),
            s(
                "5.4 Current metric validity evidence",
                "Use the generated metric validity assessment.",
                [
                    "Current completed active rows used for metric validity assessment: 557.",
                    "Strong diagnostic associations include ASR station F1, NLU route-valid rate, grounded proposal score, actionability, stagnation, and faithfulness.",
                    "Outcome-overlapping metrics should be used to decompose success, not as independent predictors.",
                    "Generic lexical metrics are supplementary unless linked to task-specific evidence.",
                ],
                evidence=[
                    "See docs/THESIS_METRIC_VALIDITY_ASSESSMENT.md.",
                    "Use docs/THESIS_METRIC_VALIDITY_TABLE.csv for exact metric roles and correlation values.",
                ],
            ),
            s(
                "5.5 Missing evidence rule",
                "Prevent invalid calculations.",
                [
                    "If audio evidence is missing, do not compute audio-quality metrics.",
                    "If a route is not parseable, route metrics should record parse failure rather than infer correctness.",
                    "If an ASR transcript is unavailable, ASR metrics are unavailable and the provider failure is part of the result.",
                ],
            ),
        ],
        terms=[
            t("Task success rate", "The proportion of completed runs ending with satisfactory task completion.", "Use completed runs as denominator when analyzing dialogue performance; separately report execution failures."),
            t("Route optimality gap", "Difference between proposed route duration and the constraint-layer optimal duration.", "Useful for route quality after basic validity is satisfied."),
            t("Constraint extraction F1", "F1 score for extracted constraints against known/revealed constraints.", "Requires captured reference constraints and extracted semantic frames."),
            t("Repair success rate", "Proportion of repair attempts that resolve a misunderstanding or missing slot.", "Useful for diagnosing whether dialogue can recover from speech or NLU errors."),
            t("Failure-localization candidate", "The earliest phase whose evidence plausibly explains the final failure.", "Diagnostic, not causal proof."),
        ],
        references=[COMMON_REFERENCES["paradise"], COMMON_REFERENCES["problematic"], COMMON_REFERENCES["bleu"], COMMON_REFERENCES["bertscore"], COMMON_REFERENCES["slue"], COMMON_REFERENCES["nisqa"], COMMON_REFERENCES["dnsmos"]],
    ),
    Chapter(
        number="6",
        slug="results",
        title="Results",
        function="Present coverage, completed-run outcomes, metric patterns, and matched comparisons in a validity-safe order.",
        argument=[
            "Report what was run and completed before interpreting rates.",
            "Use fully matched subsets for direct comparisons.",
            "Use broader completed rows for descriptive associations and metric-validity analysis.",
            "Separate execution incompleteness from unsuccessful completed dialogue.",
        ],
        sections=[
            s(
                "6.1 Run inventory and coverage",
                "Start with denominators.",
                [
                    "Report active thesis rows, completed rows, fully crossed condition groups, and matched runs.",
                    "Explain exclusions: archived, duplicate, irrelevant model, missing paired text counterpart, execution incomplete.",
                    "State that raw result folders remain unchanged; analysis views are derived.",
                ],
                evidence=[
                    "Deduplicated active thesis rows: 1319.",
                    "Completed active thesis rows: 557.",
                    "Fully crossed subset: 11 condition groups, 220 runs.",
                ],
            ),
            s(
                "6.2 Overall task outcomes",
                "Present success, semi-success, and unsuccessful completed dialogues.",
                [
                    "Do not merge semi-success into failure without explanation.",
                    "Use semi-success to show constraint-level degradation after route validity.",
                    "Report route-valid rate separately from full task success.",
                ],
            ),
            s(
                "6.3 Text versus speech",
                "Use the strongest current channel claim.",
                [
                    "In the fully crossed subset, text controls are at 100% success for each Agent A/Agent B row.",
                    "Audio variants reduce success by about 18.2 to 27.3 percentage points.",
                    "Interpret this as speech-channel degradation under the tested TTS/ASR/audio personas.",
                    "Do not generalize to all TTS or ASR systems because Piper and selected ASR settings dominate current evidence.",
                ],
            ),
            s(
                "6.4 Agent B comparison",
                "Compare backends cautiously.",
                [
                    "Use fully matched subset for direct comparison.",
                    "Use broader completed rows for descriptive runtime and feasibility.",
                    "Current broad completed rows suggest Qwen2.5 1.5B has the strongest practical profile, but this is not a pure size effect.",
                    "Large Qwen2.5 7B is valuable for comparison but much more costly in runtime.",
                ],
            ),
            s(
                "6.5 Scenario, persona, and audio-persona effects",
                "Discuss pressure conditions.",
                [
                    "Clean and nominal speech conditions are ceiling-like in current completed rows.",
                    "Severe-channel/floor conditions generate many failures and are useful for metric sensitivity.",
                    "Multi-destination errands create semi-success examples because route validity can be preserved while full goal completion fails.",
                ],
            ),
            s(
                "6.6 Metric-outcome relations",
                "Interpret indicators.",
                [
                    "Distinguish outcome-confirming metrics from diagnostic associations.",
                    "Strong indicators include route validity, constraint satisfaction, grounded proposal/actionability, ASR station/entity preservation, and stagnation.",
                    "Use metric results to explain likely failure phase, not to claim definitive causality.",
                ],
            ),
        ],
        terms=[
            t("Coverage", "The set of planned or observed conditions represented by completed evidence.", "Always state coverage before model or metric comparisons."),
            t("Fully crossed subset", "A subset where every compared factor combination is present.", "This is the safest basis for direct comparisons."),
            t("Audio-text delta", "Difference between audio and paired text success rates.", "Negative values show speech-channel degradation relative to clean text control."),
            t("Ceiling condition", "A condition where nearly all runs succeed.", "Useful for validating the pipeline but weak for distinguishing metrics."),
            t("Floor condition", "A condition where many runs fail.", "Useful for stress-testing whether metrics identify failure."),
        ],
        references=[COMMON_REFERENCES["communicator"], COMMON_REFERENCES["paradise"], COMMON_REFERENCES["problematic"], COMMON_REFERENCES["spokenwoz"], COMMON_REFERENCES["slue"]],
    ),
    Chapter(
        number="7",
        slug="discussion_conclusion",
        title="Discussion and Conclusion",
        function="Answer research questions, state supported inferences, discuss limitations, and close with realistic future work.",
        argument=[
            "The thesis can support a methodological claim: phase-wise automatic evaluation is feasible and informative in a controlled route-dialogue SDS.",
            "The evidence supports a speech-channel degradation claim in matched text/audio comparisons.",
            "The evidence supports metric usefulness claims, especially for task-grounded and semantic phase metrics.",
            "The evidence does not support universal model-size conclusions or replacement of human evaluation.",
        ],
        sections=[
            s(
                "7.1 Answer research questions",
                "Answer each RQ directly.",
                [
                    "RQ1: Yes, the framework produces analyzable phase evidence for completed runs.",
                    "RQ2: Task-grounded and semantic phase metrics are strongest; generic lexical metrics are supplementary.",
                    "RQ3: Speech variants reduce success relative to paired text controls in the fully crossed subset.",
                    "RQ4: Backend effects are observable but must be interpreted within matched coverage and runtime feasibility.",
                    "RQ5: Failure-localization candidates can be derived, but remain diagnostic rather than causal.",
                ],
            ),
            s(
                "7.2 Interpret successful, semi-successful, and unsuccessful runs",
                "Use outcome categories as analytical structure.",
                [
                    "Successful runs show that the full pipeline can preserve enough task information for correct route completion.",
                    "Semi-successful runs are especially valuable because they reveal partial route competence with constraint failure.",
                    "Unsuccessful completed runs reveal where speech, understanding, state, or route reasoning broke down.",
                ],
            ),
            s(
                "7.3 Discuss model and configuration effects",
                "Avoid overclaiming while still drawing useful conclusions.",
                [
                    "Qwen2.5 1.5B appears practically strong in current broad rows, but model family and condition coverage also matter.",
                    "Very small models remain useful baselines because they show lower-resource behavior.",
                    "Large models increase runtime/resource cost and should be justified by matched-condition improvement.",
                    "Speech pressure settings are methodologically useful because they create non-ceiling cases.",
                ],
            ),
            s(
                "7.4 Limitations",
                "Separate methodological limits from implementation limits.",
                [
                    "Simulated Agent A is controlled but not a human caller.",
                    "Synthetic network supports validation but limits ecological realism.",
                    "Current TTS/ASR provider coverage is not a universal speech-technology benchmark.",
                    "Metric correlations are descriptive and should not be interpreted as causal predictors without additional ablation.",
                ],
            ),
            s(
                "7.5 Final conclusion",
                "Close with the strongest defensible thesis claim.",
                [
                    "A controlled navigation task can provide objective task grounding for SDS evaluation.",
                    "Phase-wise logging makes automatic metrics more interpretable than final success alone.",
                    "The most useful metrics are those tied to task entities, route validity, constraint satisfaction, repair, and grounded proposal behavior.",
                    "Future work should add human validation, real microphone input, broader TTS/ASR variation, and real transit data.",
                ],
            ),
        ],
        terms=[
            t("Defensible inference", "A conclusion supported by the available evidence and stated with its denominator and limits.", "Use this phrase mentally when drafting results and discussion."),
            t("Causal claim", "A claim that one factor caused another.", "Avoid unless the experiment isolates the factor through matched design or ablation."),
            t("Descriptive association", "A relationship observed in the data without full causal control.", "Most broad configuration effects currently belong here."),
            t("Future work", "A bounded extension beyond the thesis scope.", "Keep it realistic: human validation, real microphones, broader ASR/TTS, real networks."),
        ],
        references=[COMMON_REFERENCES["paradise"], COMMON_REFERENCES["communicator"], COMMON_REFERENCES["problematic"], COMMON_REFERENCES["user_sim"], COMMON_REFERENCES["spokenwoz"]],
    ),
]


def bullet(items: list[str], indent: int = 0) -> str:
    prefix = "  " * indent + "- "
    return "\n".join(prefix + item for item in items)


def render_outline(chapter: Chapter) -> str:
    parts = [
        f"# Chapter {chapter.number}: {chapter.title} - Outline",
        "",
        "Purpose: in-depth bullet outline for drafting thesis prose. Use this file for structure and argument order.",
        "",
        "## Chapter Function",
        "",
        f"- {chapter.function}",
        "",
        "## Core Argument",
        "",
        bullet(chapter.argument),
        "",
        "## Subchapter Writing Plan",
        "",
    ]
    for section in chapter.sections:
        parts.extend(
            [
                f"### {section.title}",
                "",
                f"- Purpose: {section.purpose}",
                "- Points to write:",
                bullet(section.write, 1),
            ]
        )
        if section.evidence_hooks:
            parts.extend(["- Evidence hooks:", bullet(section.evidence_hooks, 1)])
        if section.avoid:
            parts.extend(["- Avoid:", bullet(section.avoid, 1)])
        parts.append("")
    parts.extend(
        [
            "## Minimum Quality Checklist",
            "",
            "- Every claim has a clear denominator, source, or citation.",
            "- The chapter distinguishes task outcome, phase evidence, and interpretation.",
            "- The wording stays cautious where the evidence is descriptive rather than causal.",
            "- No raw implementation detail appears unless it supports a methodological point.",
            "",
        ]
    )
    return "\n".join(parts)


def render_terms(chapter: Chapter) -> str:
    parts = [
        f"# Chapter {chapter.number}: {chapter.title} - Terminology",
        "",
        "Purpose: chapter-local definitions and explanations. Define terms before using them in thesis prose.",
        "",
    ]
    for term in chapter.terms:
        parts.extend([f"## {term.name}", "", f"- Definition: {term.definition}", "- Explanation:"])
        parts.append(bullet(term.explanation, 1))
        parts.append("")
    parts.extend(
        [
            "## Term-Use Rules",
            "",
            "- Use one term for one construct; avoid alternating synonyms for key variables.",
            "- Do not use `success`, `route validity`, and `constraint satisfaction` interchangeably.",
            "- Reserve `causal` for controlled comparisons or ablations; otherwise use `associated with` or `indicates`.",
            "- Define abbreviations at first use and prefer full terms in headings.",
            "",
        ]
    )
    return "\n".join(parts)


def render_references(chapter: Chapter) -> str:
    parts = [
        f"# Chapter {chapter.number}: {chapter.title} - References",
        "",
        "Purpose: citable papers, concrete claims they support, and boundaries for safe use.",
        "",
    ]
    for ref in chapter.references:
        parts.extend(
            [
                f"## {ref.paper}",
                "",
                "- Cite for:",
                bullet(ref.cite_for, 1),
                "- Use in this chapter:",
                bullet(ref.use_in_chapter, 1),
            ]
        )
        if ref.boundary:
            parts.extend(["- Do not use for:", bullet(ref.boundary, 1)])
        parts.append("")
    parts.extend(
        [
            "## Citation Practice",
            "",
            "- Cite the paper at the first point where its concept is needed.",
            "- Attach each citation to one specific claim, not to a whole paragraph of unrelated statements.",
            "- Prefer one strong reference per claim; add a second only when it contributes a different angle.",
            "- Pair project-specific result claims with generated result documents, not with external papers.",
            "",
        ]
    )
    return "\n".join(parts)


def readme() -> str:
    rows = [
        "| Chapter | Outline | Terms | References |",
        "| --- | --- | --- | --- |",
    ]
    for chapter in CHAPTERS:
        folder = f"chapter_{int(chapter.number):02d}_{chapter.slug}"
        rows.append(
            f"| {chapter.number}. {chapter.title} | "
            f"[outline]({folder}/01_outline.md) | "
            f"[terms]({folder}/02_terminology.md) | "
            f"[references]({folder}/03_references.md) |"
        )
    return "\n".join(
        [
            "# Thesis Chapter Workbench",
            "",
            "Purpose: split the thesis writing aid into small chapter-local files that can be opened side by side.",
            "",
            "For each chapter:",
            "",
            "- `01_outline.md` gives the argument order and bullet-point drafting plan.",
            "- `02_terminology.md` defines the terms that should be introduced in that chapter.",
            "- `03_references.md` maps citable papers to concrete claims and boundaries.",
            "",
            "The files are derived writing aids. They do not change experiment results.",
            "",
            *rows,
            "",
            "Recommended workflow:",
            "",
            "1. Open the outline, terminology, and references file for the chapter currently being written.",
            "2. Draft prose from the outline, defining terms from the terminology file before using them.",
            "3. Add citations only for the specific claims listed in the references file.",
            "4. Check the generated result documents when writing Chapters 5-7.",
            "",
        ]
    )


def write_workbench(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(readme(), encoding="utf-8")
    for chapter in CHAPTERS:
        folder = output_dir / f"chapter_{int(chapter.number):02d}_{chapter.slug}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "01_outline.md").write_text(render_outline(chapter), encoding="utf-8")
        (folder / "02_terminology.md").write_text(render_terms(chapter), encoding="utf-8")
        (folder / "03_references.md").write_text(render_references(chapter), encoding="utf-8")


def main() -> None:
    write_workbench()
    print(f"wrote thesis chapter workbench to {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
