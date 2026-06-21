"""Check a CoopNavigationSDS preset without downloading or loading model weights."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coop_navigation_sds.Configuration.pipeline import component_status  # noqa: E402


def transformers_cached(model_name):
    path = Path(model_name)
    if path.exists():
        return True
    prepared = ROOT / ".speech-providers" / "models" / "huggingface" / model_name.replace("/", "--")
    if prepared.is_dir():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_name, "config.json")
        return isinstance(cached, str) and Path(cached).is_file()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        default="linux_userlm_tinyllama_chattts_faster_whisper",
    )
    args = parser.parse_args()
    path = ROOT / "coop_navigation_sds" / "Configuration" / "presets" / f"{args.preset}.job"
    if not path.is_file():
        raise SystemExit(f"Preset not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))["config"]
    model_cached = transformers_cached(config["model_name"])
    checks = [
        (
            "Agent model",
            model_cached,
            "cached locally"
            if model_cached
            else f"prepare {config['model_name']} before running",
        ),
    ]
    for kind, key in (("tts", config["tts_engine"]), ("asr", config["asr_engine"])):
        status = component_status(kind, key, config)
        checks.append((f"{kind.upper()} {key}", status.available, status.reason))
    for label, ready, detail in checks:
        print(f"{'READY' if ready else 'MISSING'} | {label} | {detail}")
    if not all(ready for _label, ready, _detail in checks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
