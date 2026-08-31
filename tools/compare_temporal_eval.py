#!/usr/bin/env python3
"""Compare temporal occupancy metrics with the project BaseLine.

The historical scripts under ``/data/jxy/projects/tools`` extract metrics
from individual ``IoU at ...`` lines.  Temporal evaluation logs now finish
with a single Python dictionary, for example::

    {'IoU': [...], 'mIoU': [...], 'dynamic_mIoU': [...], ...}

This script parses that final dictionary from both the current experiment
logs and the BaseLine log, then writes a common summary and a delta table.
Only the Python standard library is required.

Example (from the repository root)::

    python tools/compare_temporal_eval.py \
        --logs-dir logs \
        --output-dir work_dirs/dsqe-ddp-32-baseline56-b2/eval_results

Individual logs can be supplied instead of ``--logs-dir`` with one or more
``--log`` arguments.  The BaseLine path defaults to the memory-off BaseLine
under ``/data/jxy/projects`` and can be overridden with ``--baseline-log``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


HORIZONS = (0, 1, 2, 3)
METRICS = ("IoU", "mIoU", "dynamic_mIoU", "static_mIoU")
DEFAULT_BASELINE_LOG = (
    "/data/jxy/projects/work_dirs/sparseworld-traj-memory-only/"
    "eval_epoch56_memory_off.log"
)
DEFAULT_WORK_DIR = (
    "work_dirs/dsqe-ddp-32-baseline56-b2"
)


def _as_metric_list(value: Any, name: str, source: Path) -> List[float]:
    """Validate and normalize one metric list from a parsed log dictionary."""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{source}: metric {name!r} is not a list")
    if len(value) != len(HORIZONS):
        raise ValueError(
            f"{source}: metric {name!r} has {len(value)} values; "
            f"expected {len(HORIZONS)} (0s-3s)"
        )
    result: List[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{source}: metric {name!r}[{index}] is boolean")
        try:
            result.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: metric {name!r}[{index}] is not numeric: {item!r}"
            ) from exc
    return result


def parse_metrics(log_path: Path) -> Dict[str, Any]:
    """Read the last metric dictionary from an evaluation log.

    Searching from the end makes this robust to progress output and to logs
    that contain an intermediate metric dictionary before the final result.
    ``ast.literal_eval`` is used instead of ``eval`` so the log is never
    executed as code.
    """

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"Unable to read log {log_path}: {exc}") from exc

    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if "IoU" not in line or "mIoU" not in line:
            continue
        start = line.find("{")
        end = line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            candidate = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if not isinstance(candidate, dict) or "IoU" not in candidate:
            continue

        metrics: Dict[str, Any] = {
            "IoU": _as_metric_list(candidate["IoU"], "IoU", log_path),
            "mIoU": _as_metric_list(candidate["mIoU"], "mIoU", log_path),
        }
        for name in ("dynamic_mIoU", "static_mIoU"):
            if name in candidate and candidate[name] is not None:
                metrics[name] = _as_metric_list(candidate[name], name, log_path)
        if "classes" in candidate:
            try:
                metrics["classes"] = int(candidate["classes"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{log_path}: classes is not an integer: "
                    f"{candidate['classes']!r}"
                ) from exc
        return metrics

    raise ValueError(
        f"No final metric dictionary containing IoU/mIoU found in {log_path}"
    )


def _checkpoint_from_name(path: Path) -> str:
    """Infer an epoch label from common evaluation-log file names."""

    match = re.search(r"(?:final[_-])?epoch[_-]?(\d+)", path.name, re.I)
    if match:
        return f"epoch_{match.group(1)}"
    return path.stem


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _record(
    model: str,
    checkpoint: str,
    source: Path,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "model": model,
        "checkpoint": checkpoint,
        "source": str(source.resolve()),
    }
    for metric in METRICS:
        values = metrics.get(metric)
        for horizon, value in zip(HORIZONS, values or []):
            record[f"{metric}_{horizon}s"] = value
    record["classes"] = metrics.get("classes")
    return record


def _discover_logs(logs_dir: Path, work_dir: Path) -> List[Path]:
    """Discover current experiment logs without including the BaseLine log."""

    candidates: List[Path] = []
    if logs_dir.is_dir():
        # Prefer the DSQE validation naming convention.  If no such files are
        # present, fall back to all logs so the script remains useful for a
        # renamed experiment.
        candidates = sorted(logs_dir.glob("dsqe*.log"))
        if not candidates:
            candidates = sorted(logs_dir.glob("*.log"))
    if not candidates and work_dir.is_dir():
        candidates = sorted(work_dir.rglob("*.log"))
    return candidates


def _filter_metric_logs(paths: Iterable[Path]) -> List[Path]:
    """Drop training/incomplete logs when discovery is automatic."""

    valid: List[Path] = []
    for path in paths:
        try:
            parse_metrics(path)
        except ValueError:
            continue
        valid.append(path)
    return valid


def _deduplicate_logs(paths: Iterable[Path]) -> List[Path]:
    """Keep the newest readable log for each inferred checkpoint label."""

    newest: Dict[str, Path] = {}
    for path in paths:
        label = _checkpoint_from_name(path)
        previous = newest.get(label)
        if previous is None:
            newest[label] = path
            continue
        try:
            if path.stat().st_mtime >= previous.stat().st_mtime:
                newest[label] = path
        except OSError:
            newest[label] = path

    def sort_key(path: Path) -> tuple:
        match = re.search(r"epoch[_-]?(\d+)", _checkpoint_from_name(path))
        if match:
            return (0, int(match.group(1)), str(path))
        return (1, 0, str(path))

    return sorted(newest.values(), key=sort_key)


def _summary_fields() -> List[str]:
    fields = ["model", "checkpoint", "source"]
    for metric in METRICS:
        fields.extend(f"{metric}_{horizon}s" for horizon in HORIZONS)
    fields.append("classes")
    return fields


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {}
            for field in fields:
                value = row.get(field)
                if field == "classes":
                    output[field] = "" if value is None else str(value)
                elif field.startswith(("IoU_", "mIoU_", "dynamic_mIoU_", "static_mIoU_")):
                    output[field] = _format_number(value)
                elif field.startswith("delta_"):
                    output[field] = _format_number(value)
                else:
                    output[field] = "" if value is None else str(value)
            writer.writerow(output)


def _delta_record(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {
        "model": current["model"],
        "checkpoint": current["checkpoint"],
        "source": current["source"],
    }
    for metric in METRICS:
        for horizon in HORIZONS:
            field = f"{metric}_{horizon}s"
            baseline_value = baseline.get(field)
            current_value = current.get(field)
            delta[f"delta_{field}"] = (
                None
                if baseline_value is None or current_value is None
                else round(float(current_value) - float(baseline_value), 2)
            )
    return delta


def _delta_fields() -> List[str]:
    fields = ["model", "checkpoint", "source"]
    for metric in METRICS:
        fields.extend(f"delta_{metric}_{horizon}s" for horizon in HORIZONS)
    return fields


def _print_report(rows: Sequence[Dict[str, Any]], deltas: Sequence[Dict[str, Any]]) -> None:
    print("Temporal evaluation comparison")
    print("model/checkpoint                 IoU 0s  IoU 1s  IoU 2s  IoU 3s  "
          "mIoU 0s mIoU 1s mIoU 2s mIoU 3s")
    for row in rows:
        values = [row.get(f"IoU_{horizon}s") for horizon in HORIZONS]
        miou = [row.get(f"mIoU_{horizon}s") for horizon in HORIZONS]
        label = f"{row['model']}/{row['checkpoint']}"
        print(
            f"{label:<32} "
            + " ".join(f"{value:7.2f}" if value is not None else "    n/a" for value in values)
            + " "
            + " ".join(f"{value:7.2f}" if value is not None else "    n/a" for value in miou)
        )
    if deltas:
        print("\nDelta versus BaseLine (current - BaseLine), IoU / mIoU by horizon:")
        for delta in deltas:
            iou = [delta.get(f"delta_IoU_{horizon}s") for horizon in HORIZONS]
            miou = [delta.get(f"delta_mIoU_{horizon}s") for horizon in HORIZONS]
            label = f"{delta['model']}/{delta['checkpoint']}"
            print(
                f"{label:<32} "
                + "IoU "
                + ", ".join("n/a" if value is None else f"{value:+.2f}" for value in iou)
                + " | mIoU "
                + ", ".join("n/a" if value is None else f"{value:+.2f}" for value in miou)
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=Path(DEFAULT_BASELINE_LOG),
        help="BaseLine evaluation log (default: %(default)s)",
    )
    parser.add_argument(
        "--log",
        dest="logs",
        type=Path,
        action="append",
        help="Current experiment evaluation log; repeat for multiple checkpoints",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory used to discover current logs when --log is omitted",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(DEFAULT_WORK_DIR),
        help="Fallback work directory used for log discovery",
    )
    parser.add_argument(
        "--model",
        default="DSQE",
        help="Label for current experiment rows (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-name",
        default="BaseLine",
        help="Label for the BaseLine row (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <work-dir>/eval_results)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not collapse repeated logs with the same inferred checkpoint",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    baseline_path = args.baseline_log.expanduser()
    if not baseline_path.is_file():
        print(f"ERROR: BaseLine log not found: {baseline_path}", file=sys.stderr)
        return 2

    explicit_logs = args.logs is not None
    log_paths = args.logs or _discover_logs(args.logs_dir, args.work_dir)
    if not explicit_logs:
        log_paths = _filter_metric_logs(log_paths)
    if not log_paths:
        print(
            "ERROR: no current evaluation logs found; pass one or more --log "
            "arguments or set --logs-dir",
            file=sys.stderr,
        )
        return 2
    log_paths = [path.expanduser() for path in log_paths]
    if not args.keep_duplicates:
        log_paths = _deduplicate_logs(log_paths)

    try:
        baseline_metrics = parse_metrics(baseline_path)
        baseline_row = _record(
            args.baseline_name,
            _checkpoint_from_name(baseline_path) or "epoch_56",
            baseline_path,
            baseline_metrics,
        )
        current_rows = []
        for path in log_paths:
            metrics = parse_metrics(path)
            current_rows.append(
                _record(args.model, _checkpoint_from_name(path), path, metrics)
            )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    rows = [baseline_row] + current_rows
    deltas = [_delta_record(row, baseline_row) for row in current_rows]
    output_dir = args.output_dir or args.work_dir / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    delta_path = output_dir / "delta_vs_baseline.csv"
    json_path = output_dir / "summary.json"
    _write_csv(summary_path, rows, _summary_fields())
    _write_csv(delta_path, deltas, _delta_fields())

    payload = {
        "baseline": baseline_row,
        "current": current_rows,
        "delta_vs_baseline": deltas,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_report(rows, deltas)
    print(f"\nBaseLine source: {baseline_path.resolve()}")
    print(f"Summary CSV:     {summary_path.resolve()}")
    print(f"Delta CSV:       {delta_path.resolve()}")
    print(f"Summary JSON:    {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
