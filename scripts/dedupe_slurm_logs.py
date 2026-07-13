"""Create compact, deduplicated Slurm log summaries for result transport.

The raw ``slurm/logs`` directory can contain thousands of near-identical array
task logs. This script keeps analysis useful without pushing redundant bulk:
it groups logs by logical job name while ignoring Slurm job and task suffixes,
records one small representative sample per group, and writes machine-readable
CSV summaries under ``results/general``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_TAIL_BYTES = 24_000
LOG_SUFFIX_RE = re.compile(r"-(\d+)(?:_\d+)?\.(out|err)$")


def logical_log_name(path: Path) -> str:
    return LOG_SUFFIX_RE.sub(r".\2", path.name)


def read_tail(path: Path, limit: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > limit:
                handle.seek(max(0, size - limit))
            data = handle.read(limit)
    except OSError as exc:
        return f"<unreadable: {exc}>"
    return data.decode("utf-8", errors="replace")


def signature(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\b\d{6,}(?:_\d+)?\b", "<id>", line)
        line = re.sub(r"/beegfs/[^\s]+", "<path>", line)
        line = re.sub(r"C:\\[^\s]+", "<path>", line)
        line = re.sub(r"\d{4}-\d{2}-\d{2}[T ][^\s]+", "<time>", line)
        lines.append(line)
        if len(lines) >= 8:
            break
    payload = "\n".join(lines) or "<empty>"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12], payload


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="slurm/logs")
    parser.add_argument("--output-dir", default="results/general")
    parser.add_argument("--tail-bytes", type=int, default=DEFAULT_TAIL_BYTES)
    parser.add_argument("--sample-limit", type=int, default=1)
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "slurm_log_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(log_dir.glob("*")):
        if path.is_file() and path.suffix in {".out", ".err"}:
            groups[logical_log_name(path)].append(path)

    group_rows = []
    signature_rows = []
    for group_name, paths in sorted(groups.items()):
        total_size = sum(path.stat().st_size for path in paths)
        newest = max(path.stat().st_mtime for path in paths)
        nonempty = [path for path in paths if path.stat().st_size > 0]
        sample_paths = nonempty[: max(0, args.sample_limit)]
        sample_files = []
        signatures = []
        for index, path in enumerate(sample_paths, start=1):
            text = read_tail(path, args.tail_bytes)
            digest, message = signature(text)
            signatures.append(digest)
            sample_name = f"{group_name}.{index}.txt".replace("/", "_")
            sample_path = sample_dir / sample_name
            sample_path.write_text(
                f"source={path}\nlogical_group={group_name}\nsignature={digest}\n\n{text}",
                encoding="utf-8",
            )
            sample_files.append(str(sample_path))
            signature_rows.append({
                "logical_group": group_name,
                "source_file": str(path),
                "signature": digest,
                "message_sample": message,
                "sample_file": str(sample_path),
            })
        group_rows.append({
            "logical_group": group_name,
            "file_count": len(paths),
            "nonempty_file_count": len(nonempty),
            "total_size_bytes": total_size,
            "newest_mtime": int(newest),
            "representative_signatures": ";".join(sorted(set(signatures))),
            "sample_files": ";".join(sample_files),
        })

    write_csv(output_dir / "slurm_log_group_summary.csv", group_rows)
    write_csv(output_dir / "slurm_error_signature_summary.csv", signature_rows)
    text_summary = [
        "Slurm log summary",
        "=================",
        "",
        f"source_log_dir: {log_dir}",
        f"logical_groups: {len(group_rows)}",
        f"representative_error_signatures: {len(signature_rows)}",
        "",
        "Raw Slurm logs are intentionally not required for analysis transport.",
        "Use the CSV files and representative samples in this folder first.",
    ]
    (output_dir / "slurm_error_summary.txt").write_text("\n".join(text_summary) + "\n", encoding="utf-8")
    print(f"groups: {output_dir / 'slurm_log_group_summary.csv'}", flush=True)
    print(f"signatures: {output_dir / 'slurm_error_signature_summary.csv'}", flush=True)
    print(f"samples: {sample_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
