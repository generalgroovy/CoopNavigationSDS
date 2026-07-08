"""Prepare or verify project-local Hugging Face assets for Transformers Agent B jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.NaturalLanguageGeneration.model_runtime import MODEL_CACHE_DIR  # noqa: E402
from coop_navigation_sds.NaturalLanguageGeneration.models import (  # noqa: E402
    MODEL_PROFILE_SPECS,
    model_profile_metadata,
)
from scripts.progress import ProgressBar, progress_enabled  # noqa: E402


TRANSFORMERS_AGENT_B_PROFILES = {
    "small": (
        "tinyllama_1b_transformers",
        "qwen2_5_0_5b_transformers",
        "smollm2_360m_transformers",
        "smollm2_1_7b_transformers",
    ),
    "medium": (
        "qwen2_5_1_5b_transformers",
        "phi3_mini_4k_transformers",
        "gemma2_2b_it_transformers",
        "qwen3_4b_instruct_transformers",
    ),
    "large": (
        "qwen2_5_7b_transformers",
        "mistral_7b_transformers",
        "llama3_1_8b_transformers",
        "falcon3_7b_transformers",
    ),
}


def selected_profiles(tiers=(), profiles=()):
    selected = []
    for tier in tiers or ():
        normalized = str(tier).strip().lower()
        if normalized not in TRANSFORMERS_AGENT_B_PROFILES:
            raise ValueError(f"Unknown tier '{tier}'. Use one of: {', '.join(TRANSFORMERS_AGENT_B_PROFILES)}.")
        selected.extend(TRANSFORMERS_AGENT_B_PROFILES[normalized])
    selected.extend(str(profile).strip() for profile in profiles or ())
    if not selected:
        selected.extend(TRANSFORMERS_AGENT_B_PROFILES["small"])
    unknown = [profile for profile in selected if profile not in MODEL_PROFILE_SPECS]
    if unknown:
        raise ValueError(f"Unknown model profiles: {', '.join(unknown)}")
    non_transformers = [
        profile for profile in selected
        if MODEL_PROFILE_SPECS[profile].provider != "transformers"
    ]
    if non_transformers:
        raise ValueError(f"Profiles are not Transformers models: {', '.join(non_transformers)}")
    return tuple(dict.fromkeys(selected))


def local_model_dir(model_name):
    return Path(MODEL_CACHE_DIR) / str(model_name).replace("/", "--")


def model_ready(model_name):
    folder = local_model_dir(model_name)
    return folder.is_dir() and any(folder.glob("config.json")) and any(folder.glob("tokenizer*"))


def readiness_rows(profiles):
    rows = []
    for profile in profiles:
        metadata = model_profile_metadata(profile)
        model = metadata["model"]
        rows.append({
            "profile": profile,
            "model": model,
            "size_tier": metadata.get("size_tier"),
            "family": metadata.get("family"),
            "approximate_memory_gb": metadata.get("approximate_memory_gb"),
            "local_dir": str(local_model_dir(model)),
            "ready": model_ready(model),
        })
    return rows


def download_profiles(profiles, *, show_progress=True):
    from huggingface_hub import snapshot_download

    progress = ProgressBar(
        len(profiles),
        label="Transformers models",
        enabled=show_progress,
    )
    for index, profile in enumerate(profiles, start=1):
        metadata = model_profile_metadata(profile)
        model = metadata["model"]
        destination = local_model_dir(model)
        progress.update(index - 1, message=f"downloading {profile}")
        print(f"PREPARING | {profile} | {model}", flush=True)
        snapshot_download(
            repo_id=model,
            local_dir=destination,
            local_dir_use_symlinks=False,
        )
        print(f"READY     | {profile} | {destination}", flush=True)
        progress.update(index, message=f"ready {profile}")
    progress.finish(message="all selected assets ready")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", action="append", choices=tuple(TRANSFORMERS_AGENT_B_PROFILES))
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="Select all registered Transformers Agent B proposals.")
    parser.add_argument("--download", action="store_true", help="Download missing selected model assets.")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress display.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    tiers = tuple(TRANSFORMERS_AGENT_B_PROFILES) if args.all else tuple(args.tier or ())
    profiles = selected_profiles(tiers, args.profile)
    before = readiness_rows(profiles)
    missing = [row["profile"] for row in before if not row["ready"]]
    if args.download and missing:
        download_profiles(
            missing,
            show_progress=progress_enabled(args.json, args.no_progress),
        )
    rows = readiness_rows(profiles)
    ready = all(row["ready"] for row in rows)
    if args.json:
        print(json.dumps({
            "model_cache_dir": MODEL_CACHE_DIR,
            "ready": ready,
            "models": rows,
        }, indent=2, sort_keys=True))
    else:
        print(f"Model cache: {MODEL_CACHE_DIR}")
        for row in rows:
            state = "READY" if row["ready"] else "MISSING"
            print(f"{state:<7} | {row['size_tier']:<6} | {row['profile']:<32} | {row['model']}")
        if missing and not args.download:
            print("Run again with --download to fetch only missing selected assets.")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
