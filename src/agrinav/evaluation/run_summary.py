#!/usr/bin/env python3
"""Summarise training runs from ``metrics.jsonl`` without selecting on the maximum.

Why this exists
---------------
The 2026-09-03 ablation reported each run as ``max(val/AP)`` over its logged
epochs. That is the **maximum of a noisy sequence**, and it has three properties
that make it the wrong number to compare arms with:

1. it is biased upward — ``E[max of k] = mu + a_k * sigma`` — so every reported
   AP was inflated;
2. the bias grows with an arm's own noise, so a noisier arm wins a maximum
   contest even when its true curve is identical. In that ablation
   ``sd(C_no_atss)`` was 2.0x ``sd(A_baseline)``, and the bound on the resulting
   differential bias (0.0089 AP) exceeded the effect being reported (0.0048);
3. runs with different ``val_ap_interval`` or epoch counts get different ``k``,
   so an 8-epoch run (k=4) and an 18-epoch run (k=9) are not even the same
   estimator.

``terminal_ap`` reports the mean of the last ``k`` evaluations instead: unbiased
for end-of-schedule quality, averages evaluation noise down by ``sqrt(k)``, and
comparable across runs whose schedules match.

Checkpoint *selection* by best validation AP is untouched and remains correct —
you do want the best weights. What changes is the number a run is **scored** by.

This module also reads the metric keys that actually exist. Two typos in the
ablation drivers (``avg_loss`` for ``train/total_loss``, ``val/AP-small`` for
``val/AP_small``) silently discarded the loss curves and every small-object AP,
in an experiment justified by small objects. The keys here are asserted against
the file rather than assumed.

CLI::

    python -m agrinav.evaluation.run_summary <run-dir> [<run-dir> ...]
    python -m agrinav.evaluation.run_summary --tail 3 --json out.json runs/*/
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

#: Metrics reported per run. Aggregate AP alone hides the two failure modes this
#: detector actually has: weed AP runs 3-5x below rice, and AP75 runs ~20x below
#: AP50, so localisation -- not detection -- dominates the error.
REPORTED_METRICS: tuple[str, ...] = (
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
    """A run directory cannot be summarised."""


def load_eval_rows(metrics_path: Path) -> list[dict[str, Any]]:
    """Rows from ``metrics.jsonl`` that carry a validation AP, in epoch order."""
    if not metrics_path.exists():
        raise RunSummaryError(f"no metrics.jsonl at {metrics_path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RunSummaryError(f"{metrics_path}:{line_no} is not valid JSON: {exc}") from exc
    evals = [r for r in rows if isinstance(r.get("val/AP"), (int, float))]
    if not evals:
        raise RunSummaryError(
            f"{metrics_path} has {len(rows)} rows but none carry 'val/AP' — "
            "the run logged no validation evaluation"
        )
    return sorted(evals, key=lambda r: r.get("epoch", 0))


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarise_run(run_dir: Path, tail: int = DEFAULT_TAIL) -> dict[str, Any]:
    """Terminal and (for contrast) maximum statistics for one run directory.

    Args:
        run_dir: a checkpoint directory containing ``metrics.jsonl``.
        tail: how many trailing evaluations the terminal statistic averages.

    Returns:
        A record carrying, per metric in :data:`REPORTED_METRICS`, the terminal
        mean and the maximum, plus the epochs those came from and any provenance
        recorded in ``status.json``.
    """
    if tail < 1:
        raise RunSummaryError(f"tail must be >= 1, got {tail}")

    evals = load_eval_rows(run_dir / "metrics.jsonl")
    window = evals[-tail:]
    record: dict[str, Any] = {
        "run": run_dir.name,
        "path": str(run_dir),
        "evaluations": len(evals),
        "epochs_evaluated": [r.get("epoch") for r in evals],
        "terminal_window_epochs": [r.get("epoch") for r in window],
        # A window shorter than requested is not an error, but it changes the
        # estimator, so say so rather than let it pass unnoticed.
        "terminal_window_short": len(window) < tail,
    }

    for key in REPORTED_METRICS:
        present = [r[key] for r in evals if isinstance(r.get(key), (int, float))]
        in_window = [r[key] for r in window if isinstance(r.get(key), (int, float))]
        record[f"terminal_{key}"] = _mean(in_window)
        record[f"max_{key}"] = max(present) if present else None

    ap_values = [r["val/AP"] for r in evals]
    record["peak_epoch"] = evals[ap_values.index(max(ap_values))].get("epoch")
    record["final_epoch"] = evals[-1].get("epoch")
    # If AP peaked before the end, the schedule was not still improving at the
    # point it stopped -- worth surfacing, since "still climbing" was asserted
    # about runs whose AP had already turned over.
    record["peaked_before_end"] = record["peak_epoch"] != record["final_epoch"]
    window_ap = [r["val/AP"] for r in window if isinstance(r.get("val/AP"), (int, float))]
    record["terminal_window_sd"] = statistics.stdev(window_ap) if len(window_ap) > 1 else None

    losses = [
        r["train/total_loss"] for r in evals if isinstance(r.get("train/total_loss"), (int, float))
    ]
    record["final_train_loss"] = losses[-1] if losses else None

    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
        record["provenance"] = status.get("provenance")
        record["completed"] = status.get("completed")
    else:
        record["provenance"] = None
        record["completed"] = None
    return record


def aggregate(records: list[dict[str, Any]], metric: str = "val/AP") -> dict[str, Any]:
    """Mean and sd of the terminal statistic across runs (i.e. across seeds)."""
    values = [
        r[f"terminal_{metric}"]
        for r in records
        if isinstance(r.get(f"terminal_{metric}"), (int, float))
    ]
    if not values:
        return {"n": 0}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "cv_percent": (100 * sd / mean) if sd is not None and mean else None,
        "values": values,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="checkpoint directories")
    parser.add_argument(
        "--tail",
        type=int,
        default=DEFAULT_TAIL,
        help=f"evaluations averaged for the terminal statistic (default {DEFAULT_TAIL})",
    )
    parser.add_argument("--json", type=Path, help="also write the records here")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="report mean +/- sd of the terminal AP across the given runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    records: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        try:
            records.append(summarise_run(run_dir, tail=args.tail))
        except RunSummaryError as exc:
            print(f"  SKIP {run_dir}: {exc}")

    if not records:
        print("no runs could be summarised")
        return 1

    header = (
        f"{'run':<20}{'evals':>6}{'peak@':>7}{'term AP':>9}{'max AP':>8}"
        f"{'AP50':>8}{'AP75':>8}{'AP_sm':>8}{'rice':>8}{'weed':>8}"
    )
    print(header)
    for r in records:

        def fmt(key: str, prefix: str = "terminal_") -> str:
            value = r.get(f"{prefix}{key}")
            return f"{value:.4f}" if isinstance(value, (int, float)) else "   -  "

        peak = f"{r['peak_epoch']}{'*' if r['peaked_before_end'] else ''}"
        print(
            f"{r['run']:<20}{r['evaluations']:>6}{peak:>7}{fmt('val/AP'):>9}"
            f"{fmt('val/AP', 'max_'):>8}{fmt('val/AP50'):>8}{fmt('val/AP75'):>8}"
            f"{fmt('val/AP_small'):>8}{fmt('val/AP[rice_protect]'):>8}"
            f"{fmt('val/AP[weed_target]'):>8}"
        )
    print(
        f"\nterminal = mean of the last {args.tail} evaluations; "
        "max = the biased statistic, shown for contrast"
    )
    if any(r["peaked_before_end"] for r in records):
        print("* AP peaked before the final evaluation — the run was not still improving")

    if args.aggregate:
        agg = aggregate(records)
        if agg["n"] > 1:
            print(
                f"\nacross {agg['n']} runs: terminal AP {agg['mean']:.4f}"
                f" +/- {agg['sd']:.4f}  (CV {agg['cv_percent']:.1f}%)"
            )
        else:
            print(f"\nonly {agg['n']} run — no spread to report")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
