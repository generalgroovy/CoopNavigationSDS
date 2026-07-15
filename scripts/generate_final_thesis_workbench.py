"""Generate a compact final thesis workbench from the LaTeX proposal structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "thesis_final_workbench"


@dataclass(frozen=True)
class Chapter:
    number: str
    slug: str
    title: str
    purpose: list[str]
    sections: list[tuple[str, list[str]]]
    terms: list[tuple[str, str]]
    refs: list[tuple[str, list[str]]]


CHAPTERS = [
    Chapter(
        "1",
        "introduction",
        "Introduction",
        [
            "Motivate automatic evaluation of spoken task-oriented dialogue systems.",
            "State the research gap: final task success alone does not explain phase-level failure.",
            "Define research questions and contributions without implementation detail.",
        ],
        [
            ("Motivation", ["spoken route assistance is practical", "speech introduces naturalness and accessibility", "task-oriented dialogue gives checkable goals"]),
            ("Evaluation Problem", ["spoken SDS failure can originate in TTS, ASR, NLU, state, policy, grounding, or NLG", "automatic evaluation must preserve phase evidence"]),
            ("Research Questions", ["phase-wise evidence", "metric-outcome relation", "failure localization", "backend comparison", "text versus speech"]),
            ("Contributions", ["controlled route-dialogue framework", "retrospective evidence logging", "metric validity analysis", "matched text/audio comparison"]),
        ],
        [
            ("spoken dialogue system", "dialogue system using spoken input and/or spoken output"),
            ("task-oriented dialogue", "dialogue directed toward an externally checkable goal"),
            ("phase-wise evaluation", "evaluation using intermediate pipeline evidence instead of only final outcome"),
            ("task-grounded evaluation", "evaluation based on an external task environment such as the route network"),
        ],
        [
            ("Walker et al. 1997 / PARADISE", ["task success and dialogue behavior both matter", "motivate multidimensional SDS evaluation"]),
            ("Walker, Passonneau, and Boland 2001 / DARPA Communicator", ["full SDS evaluation combines outcome, cost, and error analysis"]),
            ("SpokenWOZ / SLUE", ["speech changes task-oriented dialogue evaluation because recognition and semantic speech errors matter"]),
        ],
    ),
    Chapter(
        "2",
        "background_related_work_evaluation_foundations",
        "Background, Related Work, and Evaluation Foundations",
        [
            "Combine theory, related work, and general evaluation validity.",
            "Explain only the concepts needed to understand the experiment and metrics.",
            "Avoid a separate literature catalog chapter.",
        ],
        [
            ("Task-Oriented and Spoken Dialogue Systems", ["define dialogue system, SDS, TOD, dialogue state, system action", "explain why navigation is evaluable"]),
            ("Phase Model", ["ASR, NLU, dialogue state, dialogue management, backend grounding, NLG, TTS", "pipeline phases are also evidence boundaries"]),
            ("LLM-Based Dialogue Backends", ["LLMs blur internal module boundaries", "external evidence remains necessary for evaluation"]),
            ("Error Propagation", ["early speech errors can cause downstream task errors", "failure localization is diagnostic, not causal proof"]),
            ("Automatic Evaluation Foundations", ["human versus automatic evaluation", "lexical and semantic text metrics", "speech metrics", "task-grounded metrics", "construct validity"]),
            ("Related Work Gap", ["classical SDS evaluation", "TOD benchmarks", "spoken TOD", "user simulation", "navigation and grounded dialogue", "LLM-agent evaluation"]),
        ],
        [
            ("dialogue state", "structured representation of known goals, constraints, proposed route, and progress"),
            ("construct validity", "degree to which a metric measures the intended construct"),
            ("lexical metric", "surface text-overlap metric; useful context but not route validation"),
            ("semantic ASR metric", "speech-understanding metric focused on preserving task meaning and entities"),
            ("error propagation", "downstream effects of an earlier pipeline error"),
        ],
        [
            ("MultiWOZ and Schema-Guided Dialogue", ["structured goals, slots, domains, and state support automatic TOD evaluation"]),
            ("BLEU, ROUGE, METEOR, BERTScore, BLEURT", ["text metrics are useful but insufficient for route-grounded correctness"]),
            ("NISQA, DNSMOS, STOI", ["speech quality/intelligibility metrics are diagnostic audio evidence"]),
            ("Schatzmann et al. user simulation", ["simulated users improve repeatability but limit ecological realism"]),
        ],
    ),
    Chapter(
        "3",
        "methodology_experiment_design",
        "Methodology and Experiment Design",
        [
            "Describe CoopNavigationSDS precisely enough to reproduce the experiment logic.",
            "State condition factors, knowledge boundaries, logging, and validity controls.",
            "Do not interpret results in this chapter.",
        ],
        [
            ("Research Design", ["controlled cooperative route-dialogue task", "retrospective metric calculation", "matched text/audio comparisons"]),
            ("Experimental Unit", ["one run/condition is the main analysis row", "turns are nested diagnostic evidence", "completed and execution-incomplete runs are separated"]),
            ("Agent Roles", ["Agent A as simulated caller", "UserLM as primary caller", "TinyLlama as caller-control", "Agent B as evaluated route-information system"]),
            ("Route Task", ["start, destination, time, station and line entities", "shortest valid route under progressively revealed constraints", "constraint-layer optimal routes"]),
            ("Speech and Text Conditions", ["text-only control", "audio variant through TTS and ASR", "audio personas and speech patterns as pressure conditions"]),
            ("Evidence Logging", ["intended utterance", "TTS speech", "ASR raw transcript", "normalized understanding", "memory update", "route proposal", "validation", "timing", "outcome"]),
            ("Validity Controls", ["no hidden shared memory", "paired conditions", "deduplication", "large2 exclusion", "raw evidence preserved"]),
        ],
        [
            ("experimental unit", "one condition/run with one selected configuration"),
            ("knowledge boundary", "rule defining what each agent can know directly"),
            ("paired run", "text and audio run sharing the same non-audio configuration"),
            ("retrospective metric calculation", "metrics calculated after execution from preserved evidence"),
            ("constraint layer", "task stage where additional user constraints are revealed and optimal route changes"),
        ],
        [
            ("PARADISE", ["separate task outcome from dialogue behavior in the methodology"]),
            ("Task-oriented dialogue benchmarks", ["justify structured goals and state"]),
            ("SpokenWOZ / SLUE", ["justify speech-specific evidence capture"]),
            ("User simulation literature", ["justify controlled Agent A while noting external-validity limits"]),
        ],
    ),
    Chapter(
        "4",
        "evaluation_results_metric_analysis",
        "Evaluation, Results, and Metric Analysis",
        [
            "Define reported metrics and present empirical evidence in one chapter.",
            "Report coverage before rates and matched subsets before direct comparison.",
            "Use metric-outcome relations to explain success, semi-success, and failure.",
        ],
        [
            ("Metric Selection", ["include only metrics with evidence, formula, interpretation, and limitation", "separate outcome-confirming and diagnostic metrics"]),
            ("Coverage", ["1319 active thesis rows", "557 completed active rows", "11 fully crossed condition groups", "220 matched runs"]),
            ("Outcome Results", ["success", "semi-success", "unsuccessful completed dialogue", "execution incomplete separately"]),
            ("Text Versus Speech", ["text controls are at ceiling in the fully crossed subset", "audio variants reduce success by about 18.2 to 27.3 percentage points"]),
            ("Agent Comparisons", ["compare Agent B only in matched subsets", "Qwen2.5 1.5B strongest broad completed-row profile", "Qwen2.5 7B strong when completed but runtime-sensitive"]),
            ("Pressure Conditions", ["severe-channel/floor creates failures", "clean/nominal creates ceiling", "multi-destination errands create semi-success"]),
            ("Metric Relations and Failure Localization", ["route validity, constraint satisfaction, ASR station F1, grounded proposal, actionability, stagnation", "localization is diagnostic, not causal proof"]),
        ],
        [
            ("task success rate", "successful completed runs divided by completed runs"),
            ("semi-success", "valid or partial task progress without full constraint/task satisfaction"),
            ("audio-text delta", "audio success rate minus paired text success rate"),
            ("route optimality gap", "proposed route duration minus constraint-layer optimal route duration"),
            ("metric role", "outcome-confirming, diagnostic phase, associated diagnostic, supplementary, metric quality, or execution/constant"),
        ],
        [
            ("PARADISE and DARPA Communicator", ["justify multidimensional reporting"]),
            ("Problematic Dialogue Predictor", ["support early diagnostic evidence and failure indicators"]),
            ("ASR/SLU references", ["justify entity-level speech metrics"]),
            ("Text metric references", ["support why lexical/semantic text metrics are supplementary"]),
        ],
    ),
    Chapter(
        "5",
        "discussion_conclusion_future_work",
        "Discussion, Conclusion, and Future Work",
        [
            "Answer research questions directly.",
            "State contributions and validity boundaries.",
            "Keep future work short and tied to limitations.",
        ],
        [
            ("Research Question Answers", ["phase-wise evidence is feasible", "task-grounded and semantic metrics are strongest", "speech channel reduces success in matched comparisons", "backend ranking remains cautious"]),
            ("Interpretation", ["success alone is too coarse", "semi-success is analytically valuable", "failure localization identifies likely origin but not definitive cause"]),
            ("Validity Boundaries", ["simulated caller", "synthetic network", "selected TTS/ASR", "descriptive correlations", "matched-subset limits"]),
            ("Contribution", ["framework", "logging design", "metric validity classification", "matched text/audio analysis", "result interpretation method"]),
            ("Future Work", ["human validation", "real microphone input", "real transit data", "more languages", "broader speech-native models", "controlled ablations"]),
        ],
        [
            ("defensible inference", "claim supported by evidence, denominator, and validity boundary"),
            ("descriptive association", "observed relation without full causal control"),
            ("causal claim", "claim requiring controlled ablation or matched isolation"),
            ("future work", "bounded extension directly tied to a limitation"),
        ],
        [
            ("Classical SDS evaluation references", ["frame the final contribution"]),
            ("User simulation references", ["state caller validity boundary"]),
            ("Spoken TOD references", ["state speech-channel limits and future work"]),
            ("Metric references", ["justify which metrics were useful and which were supplementary"]),
        ],
    ),
]


def bullets(items: list[str], indent: int = 0) -> str:
    prefix = "  " * indent + "- "
    return "\n".join(prefix + item for item in items)


def render_outline(chapter: Chapter) -> str:
    lines = [
        f"# Chapter {chapter.number}: {chapter.title} - Outline",
        "",
        "Use this as the direct writing plan for the final compact thesis structure.",
        "",
        "## Function",
        "",
        bullets(chapter.purpose),
        "",
        "## Sections",
        "",
    ]
    for title, points in chapter.sections:
        lines += [f"### {chapter.number}.{len([line for line in lines if line.startswith('### ')]) + 1} {title}", "", bullets(points), ""]
    lines += [
        "## Keep Out",
        "",
        "- exhaustive metric catalogs;",
        "- raw per-run tables;",
        "- long prompt templates;",
        "- implementation details that do not support the chapter argument.",
        "",
    ]
    return "\n".join(lines)


def render_terms(chapter: Chapter) -> str:
    lines = [f"# Chapter {chapter.number}: {chapter.title} - Terminology", ""]
    for name, definition in chapter.terms:
        lines += [f"## {name}", "", f"- Definition: {definition}", ""]
    lines += [
        "## Use Rule",
        "",
        "- Define a term when it first becomes necessary for the argument.",
        "- Do not introduce terms only because they appear in the software.",
        "",
    ]
    return "\n".join(lines)


def render_refs(chapter: Chapter) -> str:
    lines = [f"# Chapter {chapter.number}: {chapter.title} - References", ""]
    for paper, uses in chapter.refs:
        lines += [f"## {paper}", "", bullets(uses), ""]
    lines += [
        "## Citation Rule",
        "",
        "- Attach each citation to one concrete claim.",
        "- Use result documents for project-specific empirical claims.",
        "",
    ]
    return "\n".join(lines)


def readme() -> str:
    rows = ["| Chapter | Outline | Terms | References |", "| --- | --- | --- | --- |"]
    for chapter in CHAPTERS:
        folder = f"chapter_{int(chapter.number):02d}_{chapter.slug}"
        rows.append(f"| {chapter.number}. {chapter.title} | [outline]({folder}/01_outline.md) | [terms]({folder}/02_terminology.md) | [references]({folder}/03_references.md) |")
    return "\n".join(
        [
            "# Final Thesis Workbench",
            "",
            "This is the compact writing target adapted from the passed `BA_Propose_2026` LaTeX project.",
            "",
            "It combines chapters where that improves readability:",
            "",
            "- background, related work, and evaluation foundations;",
            "- metric selection, results, and metric analysis;",
            "- discussion, conclusion, limitations, and future work.",
            "",
            *rows,
            "",
            "Use this workbench for actual drafting. Use `docs/thesis_chapter_workbench/` as a deeper reference.",
            "",
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(readme(), encoding="utf-8")
    for chapter in CHAPTERS:
        folder = OUT / f"chapter_{int(chapter.number):02d}_{chapter.slug}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "01_outline.md").write_text(render_outline(chapter), encoding="utf-8")
        (folder / "02_terminology.md").write_text(render_terms(chapter), encoding="utf-8")
        (folder / "03_references.md").write_text(render_refs(chapter), encoding="utf-8")
    print(f"wrote final thesis workbench to {OUT}")


if __name__ == "__main__":
    main()
