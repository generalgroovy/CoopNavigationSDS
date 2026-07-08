"""Prepare or verify the six controlled Agent B Ollama model conditions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from urllib import parse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coop_navigation_sds.Configuration.model_matrix import (
    AGENT_B_MODEL_SIZE_TREATMENTS,
    agent_b_model_platform_dir,
    agent_b_ollama_store_dir,
    model_catalog_folder,
    model_store_platform,
    model_size_treatment,
    models_for_size_treatments,
    resolve_agent_b_model_store,
)
from coop_navigation_sds.Configuration.travel import OLLAMA_BASE_URL
from coop_navigation_sds.NaturalLanguageGeneration.models import (
    ensure_ollama_models_ready,
    ollama_executable,
    ollama_model_inventory,
)
from scripts.progress import ProgressBar, progress_enabled  # noqa: E402


def selected_model_names(tiers=(), models=()):
    """Resolve an explicit model selection or all models in selected tiers."""
    tiers = tuple(tiers or (() if models else AGENT_B_MODEL_SIZE_TREATMENTS))
    selected = list(models_for_size_treatments(tiers))
    for model in models or ():
        model = str(model).strip()
        if model and model not in selected:
            selected.append(model)
    return tuple(selected)


def pull_models(models, *, executable=None, base_url=OLLAMA_BASE_URL, models_dir=None, show_progress=True):
    """Download only selected Ollama models that preflight reported missing."""
    executable = executable or ollama_executable()
    if not executable:
        raise RuntimeError(
            "Ollama is not installed. Install Ollama, then rerun this command."
        )
    environment = dict(os.environ)
    parsed = parse.urlparse(str(base_url or OLLAMA_BASE_URL))
    if parsed.netloc:
        environment["OLLAMA_HOST"] = parsed.netloc
    environment["OLLAMA_MODELS"] = str(resolve_agent_b_model_store(models_dir))
    progress = ProgressBar(
        len(models),
        label="Ollama models",
        enabled=show_progress,
    )
    for index, model in enumerate(models, start=1):
        progress.update(index - 1, message=f"pulling {model}")
        print(f"PULLING  | {model}", flush=True)
        completed = subprocess.run(
            [str(executable), "pull", str(model)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "unknown provider error").strip()
            raise RuntimeError(f"Ollama could not pull '{model}': {detail[-1000:]}")
        print(f"READY    | {model}", flush=True)
        progress.update(index, message=f"ready {model}")
    progress.finish(message="all selected models ready")


def initialize_platform_folders():
    """Create uncluttered Windows and Linux roots before provider downloads."""
    for system_name in ("windows", "linux"):
        platform_dir = agent_b_model_platform_dir(system_name)
        (platform_dir / "ollama").mkdir(parents=True, exist_ok=True)
        inventory_path = platform_dir / "inventory.json"
        if not inventory_path.exists():
            inventory_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "platform": system_name,
                        "models_dir": str(agent_b_ollama_store_dir(system_name)),
                        "models": [],
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )


def write_local_catalog(platform_key, selected_models, inventory):
    """Write size-first metadata without duplicating Ollama's model blobs."""
    platform_dir = agent_b_model_platform_dir(platform_key)
    records = {record["name"]: record for record in inventory.get("model_records", ())}
    rows = []
    for model in selected_models:
        record = records.get(model, {})
        row = {
            "model": model,
            "size_tier": model_size_treatment(model) or "custom",
            "ready": bool(record),
            "digest": record.get("digest"),
            "size_bytes": record.get("size_bytes"),
            "modified_at": record.get("modified_at"),
            "provider_details": record.get("details", {}),
        }
        rows.append(row)
        if model_size_treatment(model):
            catalog_dir = platform_dir / model_catalog_folder(model)
            catalog_dir.mkdir(parents=True, exist_ok=True)
            (catalog_dir / "model.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    document = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform_key,
        "base_url": inventory.get("base_url"),
        "models_dir": inventory.get("models_dir"),
        "models": rows,
    }
    (platform_dir / "inventory.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def readiness_rows(selected_models, installed_models):
    """Return stable status rows suitable for console or JSON output."""
    installed = set(installed_models)
    return [
        {
            "model": model,
            "size_tier": model_size_treatment(model) or "custom",
            "ready": model in installed,
        }
        for model in selected_models
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify or pull the controlled Agent B Ollama model matrix."
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=tuple(AGENT_B_MODEL_SIZE_TREATMENTS),
        help="Size tier to prepare; repeat for multiple tiers. Defaults to all tiers.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Additional exact Ollama model name; repeat when needed.",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull only missing selected models. Without this flag, perform a status check.",
    )
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL)
    parser.add_argument(
        "--models-dir",
        help="Ollama store override. Defaults to .model-providers/agent_b/<platform>/ollama.",
    )
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress display.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable status.")
    args = parser.parse_args(argv)

    initialize_platform_folders()
    platform_key = model_store_platform()
    models_dir = resolve_agent_b_model_store(args.models_dir)
    selected = selected_model_names(args.tier or (), args.model)
    try:
        inventory = ollama_model_inventory(
            args.base_url,
            timeout_sec=args.timeout_sec,
            models_dir=models_dir,
        )
        installed = inventory["available_models"]
    except RuntimeError as exc:
        if not args.pull:
            rows = readiness_rows(selected, ())
            if args.json:
                print(json.dumps(
                    {"models": rows, "ready": False, "error": str(exc)},
                    indent=2,
                ))
            else:
                print(f"ERROR   | Ollama | {exc}")
                print("Install/start Ollama, or rerun with --pull after installation.")
            return 2
        installed = ()

    missing = tuple(model for model in selected if model not in installed)
    if args.pull and missing:
        try:
            pull_models(
                missing,
                base_url=args.base_url,
                models_dir=models_dir,
                show_progress=progress_enabled(args.json, args.no_progress),
            )
            status = ensure_ollama_models_ready(
                args.base_url,
                selected,
                timeout_sec=args.timeout_sec,
                models_dir=models_dir,
            )
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            if args.json:
                print(json.dumps(
                    {
                        "models": readiness_rows(selected, installed),
                        "ready": False,
                        "error": str(exc),
                    },
                    indent=2,
                ))
            else:
                print(f"ERROR   | model preparation | {exc}")
            return 2
        installed = status["available_models"]

    try:
        final_inventory = ollama_model_inventory(
            args.base_url,
            timeout_sec=args.timeout_sec,
            models_dir=models_dir,
        )
    except RuntimeError:
        final_inventory = {
            "base_url": args.base_url,
            "models_dir": str(models_dir),
            "available_models": tuple(installed),
            "model_records": (),
        }
    write_local_catalog(platform_key, selected, final_inventory)

    rows = readiness_rows(selected, installed)
    if args.json:
        print(json.dumps({
            "platform": platform_key,
            "models_dir": str(models_dir),
            "catalog": str(agent_b_model_platform_dir(platform_key) / "inventory.json"),
            "models": rows,
            "ready": all(row["ready"] for row in rows),
        }, indent=2))
    else:
        for row in rows:
            state = "READY" if row["ready"] else "MISSING"
            print(f"{state:<7} | {row['size_tier']:<6} | {row['model']}")
        if missing and not args.pull:
            print("Run again with --pull to download only the missing models.")
    return 0 if all(row["ready"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
