"""Prepare all local providers and model assets before an experiment run."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.progress import ProgressBar, progress_enabled  # noqa: E402

ASSET_MANIFEST = ROOT / "coop_navigation_sds" / "Configuration" / "model_assets.json"
READINESS_FILE = ROOT / ".speech-providers" / "readiness.json"


def _destination(spec):
    return (ROOT / spec["destination"]).resolve()


def _required_present(spec):
    root = _destination(spec)
    return root.exists() and all(any(root.rglob(name)) for name in spec.get("required", ()))


def _run(command):
    subprocess.run([str(value) for value in command], cwd=ROOT, check=True)


def _provider_python(engine):
    manifest = ROOT / ".speech-providers" / "providers.json"
    try:
        entry = json.loads(manifest.read_text(encoding="utf-8"))["providers"][engine]
        configured = entry["python"] if isinstance(entry, dict) else entry
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        return candidate if candidate.is_file() else None
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _download_archive(spec):
    destination = _destination(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = (
        ".tar.bz2" if spec["source"].endswith(".tar.bz2")
        else ".tar.gz" if spec["source"].endswith(".tar.gz")
        else ".zip"
    )
    with tempfile.TemporaryDirectory(prefix="coop_navigation_assets_") as tmpdir:
        archive = Path(tmpdir) / f"asset{suffix}"
        urllib.request.urlretrieve(spec["source"], archive)
        extract_root = Path(tmpdir) / "extract"
        extract_root.mkdir()
        if suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extract_root)
        elif suffix == ".tar.bz2":
            with tarfile.open(archive, "r:bz2") as bundle:
                bundle.extractall(extract_root, filter="data")
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(extract_root, filter="data")
        roots = [path for path in extract_root.iterdir()]
        source = roots[0] if len(roots) == 1 and roots[0].is_dir() else extract_root
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def _prepare_asset(key, spec):
    destination = _destination(spec)
    destination.mkdir(parents=True, exist_ok=True)
    kind = spec["kind"]
    if kind == "huggingface_snapshot":
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=spec["source"],
            local_dir=destination,
            allow_patterns=spec.get("patterns"),
        )
    elif kind == "huggingface_file":
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=spec["source"],
            filename=spec["filename"],
            local_dir=destination,
        )
    elif kind == "piper_voice":
        _run((sys.executable, "-m", "piper.download_voices", "--data-dir", destination, spec["source"]))
    elif kind == "faster_whisper":
        from faster_whisper import WhisperModel
        WhisperModel(spec["source"], device="cpu", compute_type="int8", download_root=str(destination))
    elif kind == "archive":
        _download_archive(spec)
    elif kind == "github_release_archive":
        request = urllib.request.Request(
            f"https://api.github.com/repos/{spec['source']}/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "CoopNavigationSDS"},
        )
        with urllib.request.urlopen(request) as response:
            release = json.load(response)
        asset = next(item for item in release["assets"] if item["name"] == spec["asset"])
        _download_archive({**spec, "source": asset["browser_download_url"]})
    else:
        raise ValueError(f"Unsupported asset kind for {key}: {kind}")


def prepare(download=True, asset_timeout_seconds=600, show_progress=True):
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    results = {}
    assets = {**manifest["models"], **manifest.get("executables", {})}
    platform_assets = [
        (key, spec) for key, spec in assets.items()
        if platform.system() in spec.get("platforms", [platform.system()])
    ]
    progress = ProgressBar(
        len(platform_assets),
        label="Provider assets",
        enabled=show_progress,
    )
    for index, (key, spec) in enumerate(platform_assets, start=1):
        progress.update(index - 1, message=f"checking {key}")
        try:
            if download and not _required_present(spec):
                progress.update(index - 1, message=f"preparing {key}")
                print(f"PREPARING | {key} | {spec['source']}", flush=True)
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--asset", key],
                    cwd=ROOT,
                    check=True,
                    timeout=asset_timeout_seconds,
                )
            ready = _required_present(spec)
            results[key] = {"ready": ready, "destination": str(_destination(spec)), "error": ""}
            progress.update(index, message=f"{'ready' if ready else 'missing'} {key}")
        except Exception as exc:
            results[key] = {
                "ready": False,
                "destination": str(_destination(spec)),
                "error": f"{type(exc).__name__}: {exc}",
            }
            progress.update(index, message=f"failed {key}")
    progress.finish(message="asset readiness recorded")
    report = {
        "schema_version": 1,
        "platform": platform.system(),
        "python": sys.version,
        "models": results,
        "ready": all(item["ready"] for item in results.values()),
    }
    READINESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    READINESS_FILE.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _register_prepared_whisper_cpp()
    return report


def _register_prepared_whisper_cpp():
    provider_root = ROOT / ".speech-providers"
    executable_name = "whisper-cli.exe" if platform.system() == "Windows" else "whisper-cli"
    executable = next((provider_root / "whisper_cpp" / "bin").rglob(executable_name), None)
    model = provider_root / "models" / "whisper.cpp" / "ggml-base.en.bin"
    if executable is None or not model.is_file():
        return
    manifest = provider_root / "providers.json"
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        document = {"schema_version": 1, "providers": {}}
    document.setdefault("providers", {})["whisper_cpp"] = {
        "executable": str(executable.resolve()),
        "model": str(model.resolve()),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check existing assets without downloading.")
    parser.add_argument("--asset", help=argparse.SUPPRESS)
    parser.add_argument("--asset-timeout-seconds", type=int, default=600)
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress display.")
    args = parser.parse_args()
    if args.asset:
        manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
        assets = {**manifest["models"], **manifest.get("executables", {})}
        _prepare_asset(args.asset, assets[args.asset])
        return
    report = prepare(
        download=not args.check,
        asset_timeout_seconds=max(30, args.asset_timeout_seconds),
        show_progress=progress_enabled(False, args.no_progress),
    )
    for key, item in report["models"].items():
        print(f"{'READY' if item['ready'] else 'MISSING'} | {key} | {item['error'] or item['destination']}")
    print(f"Readiness manifest: {READINESS_FILE}")
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
