#!/usr/bin/env python3
"""Inspect detector runs; aggregate only complete, matched, independently seeded runs.

The terminal statistic is a descriptive mean of trailing evaluations, not an
unbiased estimate or independent replication: neighboring checkpoints correlate.
The maximum remains separate because checkpoint selection and comparing training
recipes answer different questions. Legacy metrics.jsonl and baseline run.json
remain readable; research aggregation requires verified launch manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from agrinav.training.run_manifest import file_sha256, read_manifest

REPORTED_METRICS = (
    "val/AP",
    "val/AP50",
    "val/AP75",
    "val/AP_small",
    "val/AR100",
    "val/AP[rice_protect]",
    "val/AP[weed_target]",
)
DEFAULT_TAIL = 3


class RunSummaryError(RuntimeError):
    """A run cannot be inspected or included in a research aggregate."""


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric(value: Any) -> bool:
    return _number(value) and 0 <= value <= 1


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunSummaryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunSummaryError(f"{path} must contain a JSON object")
    return value


def _load_rows(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        raise RunSummaryError(f"no metrics.jsonl at {metrics_path}")
    rows = []
    for line_no, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunSummaryError(f"{metrics_path}:{line_no} is not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise RunSummaryError(f"{metrics_path}:{line_no} must contain a JSON object")
        rows.append(row)
    return rows


def _eval_rows(rows: list[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    previous = 0
    evals = []
    for row in rows:
        epoch = row.get("epoch")
        if type(epoch) is not int or epoch <= previous:
            raise RunSummaryError(f"{source}: epochs must be positive, unique and increasing")
        previous = epoch
        if "val/AP" not in row:
            continue
        if not _metric(row["val/AP"]):
            raise RunSummaryError(f"{source}: epoch {epoch} has invalid val/AP {row['val/AP']!r}")
        evals.append(row)
    if not evals:
        raise RunSummaryError(f"{source} has {len(rows)} rows but none carry 'val/AP'")
    return evals


def load_eval_rows(metrics_path: Path) -> list[dict[str, Any]]:
    return _eval_rows(_load_rows(metrics_path), metrics_path)


def _baseline_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    epochs = run.get("epochs")
    if not isinstance(epochs, list) or any(not isinstance(row, dict) for row in epochs):
        raise RunSummaryError("baseline run.json must contain an epochs array of objects")
    rows = []
    for epoch in epochs:
        row = {
            "epoch": epoch.get("epoch"),
            "train/total_loss": epoch.get("train_loss"),
            "eval_protocol": epoch.get("eval_protocol"),
            "run_id": (run.get("provenance") or {}).get("run_id"),
        }
        if epoch.get("val_ap") is not None:
            row["val/AP"] = epoch["val_ap"]
        metrics = epoch.get("eval_metrics", {})
        if not isinstance(metrics, dict):
            raise RunSummaryError("baseline eval_metrics must be an object")
        row.update(metrics)
        rows.append(row)
    return rows


def summarise_run(run_dir: Path, tail: int = DEFAULT_TAIL) -> dict[str, Any]:
    """Inspect current and legacy runs; expose reasons research aggregation is blocked."""
    if type(tail) is not int or tail < 1:
        raise RunSummaryError(f"tail must be >= 1, got {tail}")
    baseline = None
    source = run_dir / "metrics.jsonl"
    if source.exists():
        rows = _load_rows(source)
    elif (run_dir / "run.json").exists():
        source = run_dir / "run.json"
        baseline = _json_object(source)
        rows = _baseline_rows(baseline)
    else:
        raise RunSummaryError(f"no metrics.jsonl or baseline run.json at {run_dir}")
    evals = _eval_rows(rows, source)
    window = evals[-tail:]
    issues: list[str] = []
    record: dict[str, Any] = {
        "run": run_dir.name,
        "path": str(run_dir),
        "source": source.name,
        "evaluations": len(evals),
        "epochs_evaluated": [r["epoch"] for r in evals],
        "terminal_window_epochs": [r["epoch"] for r in window],
        "terminal_window_short": len(window) < tail,
        "research_issues": issues,
    }
    for key in REPORTED_METRICS:
        present = [r[key] for r in evals if _metric(r.get(key))]
        in_window = [r[key] for r in window if _metric(r.get(key))]
        # Missing secondary metrics must not silently change the averaging window.
        record[f"terminal_{key}"] = (
            statistics.mean(in_window) if len(in_window) == len(window) else None
        )
        record[f"max_{key}"] = max(present) if present else None
        if any(key in r and not _metric(r[key]) for r in evals):
            issues.append(f"invalid or unavailable {key} (including COCO -1 sentinels)")
    ap = [r["val/AP"] for r in evals]
    record["peak_epoch"] = evals[ap.index(max(ap))]["epoch"]
    record["final_epoch"] = evals[-1]["epoch"]
    record["peaked_before_end"] = record["peak_epoch"] != record["final_epoch"]
    record["terminal_window_sd"] = (
        statistics.stdev(r["val/AP"] for r in window) if len(window) > 1 else None
    )
    final_loss = rows[-1].get("train/total_loss")
    record["final_train_loss"] = final_loss if _number(final_loss) else None
    if "train/total_loss" in rows[-1] and not _number(final_loss):
        issues.append("invalid final training loss")
    status = (
        _json_object(run_dir / "status.json")
        if (run_dir / "status.json").exists()
        else baseline or {}
    )
    record["completed"] = status.get("completed")
    record["provenance"] = status.get("provenance")
    if status.get("completed") is not True:
        issues.append("completion not verified")
    if len(window) < tail:
        issues.append("terminal window shorter than requested")
    manifest = None
    try:
        manifest = read_manifest(run_dir, record["provenance"] or {})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        issues.append(f"launch manifest not verified: {exc}")
    if manifest is not None:
        artifacts = status.get("artifact_sha256", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        if source.name not in artifacts or not any(name.endswith(".pth") for name in artifacts):
            issues.append("metric log or checkpoint hashes missing")
        for name, expected in artifacts.items():
            try:
                if Path(name).name != name or file_sha256(run_dir / name) != expected:
                    issues.append(f"artifact checksum mismatch: {name}")
            except OSError:
                issues.append(f"artifact missing: {name}")
        record["recipe_sha256"] = manifest["recipe_sha256"]
        record["seed"] = manifest["seed"]
        record["protocol"] = manifest["protocol"]
        planned = manifest["config"].get("num_epochs")
        if manifest["config"].get("diagnostic"):
            issues.append("diagnostic run is not a research comparison")
        if (
            type(planned) is not int
            or type(status.get("epochs_planned")) is not int
            or type(status.get("epochs_completed")) is not int
            or status.get("epochs_planned") != planned
            or status.get("epochs_completed") != planned
            or rows[-1]["epoch"] != planned
        ):
            issues.append("completed schedule does not match launch manifest")
        if record["epochs_evaluated"] != manifest["eval_epochs"]:
            issues.append("evaluation epochs do not match launch schedule")
        if type(manifest["seed"]) is not int:
            issues.append("seed missing or invalid")
        if manifest["resume_from"] or manifest["start_epoch"] != 1:
            issues.append("resumed run requires a separate continuity audit")
        if any(
            not manifest["data"][split].get("annotation_sha256")
            for split in ("train", "validation")
        ):
            issues.append("training or validation annotations lack hashes")
        initialization = manifest["recipe"]["initialization"]
        if initialization["external_model"] or initialization["untracked_callable"]:
            issues.append("model initialization not fully tracked")
        if any(r.get("run_id") != manifest["run_id"] for r in rows):
            issues.append("metric rows belong to a different or unknown run")
        if any(r.get("eval_protocol") != manifest["protocol"] for r in evals):
            issues.append("evaluation protocol differs from launch manifest")
    record["research_eligible"] = not issues
    return record


def aggregate(records: list[dict[str, Any]], metric: str = "val/AP") -> dict[str, Any]:
    """Across-seed spread within one recipe, never across experimental arms."""
    if not records:
        return {"n": 0}
    seeds = set()
    identities = set()
    values = []
    for record in records:
        if not record.get("research_eligible"):
            raise RunSummaryError(
                f"{record.get('run')}: research aggregate refused: "
                + "; ".join(record.get("research_issues", ["unverified record"]))
            )
        value = record.get(f"terminal_{metric}")
        if not _metric(value):
            raise RunSummaryError(
                f"{record['run']}: no complete valid terminal window for {metric}"
            )
        seed = record.get("seed")
        if type(seed) is not int or seed in seeds:
            raise RunSummaryError(f"duplicate or invalid seed: {seed}")
        seeds.add(seed)
        recipe = record.get("recipe_sha256")
        if not recipe:
            raise RunSummaryError("missing recipe identity")
        identities.add((recipe, tuple(record["terminal_window_epochs"])))
        values.append(value)
    if len(identities) != 1:
        raise RunSummaryError(
            "mixed recipes, data, code, environments, protocols or terminal schedules"
        )
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "cv_percent": 100 * sd / mean if sd is not None and mean else None,
        "values": values,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--aggregate", action="store_true", help="aggregate one verified recipe across seeds"
    )
    args = parser.parse_args(argv)
    records = []
    skipped = False
    for path in args.run_dirs:
        try:
            records.append(summarise_run(path, args.tail))
        except RunSummaryError as exc:
            print(f"SKIP {path}: {exc}")
            skipped = True
    if not records:
        print("no runs could be summarised")
        return 1
    print(f"{'run':<24} {'evals':>5} {'terminal AP':>12} {'max AP':>9}  research aggregate")
    for record in records:
        print(
            f"{record['run']:<24} {record['evaluations']:>5} "
            f"{record['terminal_val/AP']:>12.4f} {record['max_val/AP']:>9.4f}  "
            + ("eligible" if record["research_eligible"] else "; ".join(record["research_issues"]))
        )
    print(f"terminal = mean of last {args.tail} evaluations; epochs are correlated, not replicates")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    if args.aggregate:
        try:
            if skipped:
                raise RunSummaryError(
                    "some requested runs could not be read; refusing a partial aggregate"
                )
            result = aggregate(records)
        except RunSummaryError as exc:
            print(f"AGGREGATE REFUSED: {exc}")
            return 1
        print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
