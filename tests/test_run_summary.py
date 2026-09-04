"""The run summariser must not select on the maximum, and must not invent data.

Written after an ablation reported every run as `max(val/AP)` over its logged
epochs — a statistic whose upward bias grows with the arm's own noise, so the
noisier arm wins even with an identical true curve. These tests pin the
properties that make the replacement trustworthy: the terminal statistic ignores
an early spike, a short window is flagged rather than silently changing the
estimator, and a missing metric reads as absent rather than as zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agrinav.evaluation.run_summary import (
    REPORTED_METRICS,
    RunSummaryError,
    aggregate,
    load_eval_rows,
    main,
    summarise_run,
)


def _write_run(
    tmp_path: Path,
    ap_by_epoch: dict[int, float],
    *,
    name: str = "run",
    extra: dict[int, dict[str, float]] | None = None,
    status: dict | None = None,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for epoch, ap in ap_by_epoch.items():
        row = {"epoch": epoch, "val/AP": ap, "train/total_loss": 5.0 - epoch * 0.1}
        row.update((extra or {}).get(epoch, {}))
        lines.append(json.dumps(row))
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status is not None:
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return run_dir


# --- the core property: no selection on the maximum -------------------------


def test_terminal_ignores_an_early_spike(tmp_path: Path) -> None:
    """A lucky mid-run evaluation must not become the run's score."""
    run = _write_run(tmp_path, {2: 0.01, 4: 0.90, 6: 0.05, 8: 0.05, 10: 0.05})
    rec = summarise_run(run, tail=3)
    assert rec["max_val/AP"] == pytest.approx(0.90)
    assert rec["terminal_val/AP"] == pytest.approx(0.05)


def test_terminal_is_the_mean_of_the_last_k(tmp_path: Path) -> None:
    run = _write_run(tmp_path, {2: 0.10, 4: 0.20, 6: 0.30, 8: 0.40})
    assert summarise_run(run, tail=3)["terminal_val/AP"] == pytest.approx(0.30)
    assert summarise_run(run, tail=2)["terminal_val/AP"] == pytest.approx(0.35)
    assert summarise_run(run, tail=1)["terminal_val/AP"] == pytest.approx(0.40)


def test_a_noisier_run_does_not_win_on_the_terminal_statistic(tmp_path: Path) -> None:
    """The exact failure mode that motivated this module."""
    # Both average 0.10 over the trailing window; the noisy one additionally
    # spikes to 0.30 early, which is exactly what a max-statistic rewards.
    steady = _write_run(tmp_path, {2: 0.10, 4: 0.10, 6: 0.10, 8: 0.10}, name="steady")
    noisy = _write_run(tmp_path, {2: 0.30, 4: 0.02, 6: 0.18, 8: 0.10}, name="noisy")
    steady_rec, noisy_rec = summarise_run(steady), summarise_run(noisy)
    # On the max, the noisy run looks better; on the terminal mean they tie.
    assert noisy_rec["max_val/AP"] > steady_rec["max_val/AP"]
    assert noisy_rec["terminal_val/AP"] == pytest.approx(steady_rec["terminal_val/AP"])


# --- honesty about the estimator --------------------------------------------


def test_short_window_is_flagged(tmp_path: Path) -> None:
    """Fewer evaluations than requested is a different estimator; say so."""
    run = _write_run(tmp_path, {2: 0.10, 4: 0.20})
    rec = summarise_run(run, tail=3)
    assert rec["terminal_window_short"] is True
    assert rec["terminal_window_epochs"] == [2, 4]
    full = summarise_run(_write_run(tmp_path, {2: 0.1, 4: 0.2, 6: 0.3}, name="full"), tail=3)
    assert full["terminal_window_short"] is False


def test_peak_before_end_is_reported(tmp_path: Path) -> None:
    """'Still climbing' was asserted about runs whose AP had already turned over."""
    turned = _write_run(tmp_path, {2: 0.1, 4: 0.3, 6: 0.2}, name="turned")
    rec = summarise_run(turned)
    assert rec["peak_epoch"] == 4
    assert rec["final_epoch"] == 6
    assert rec["peaked_before_end"] is True

    rising = _write_run(tmp_path, {2: 0.1, 4: 0.2, 6: 0.3}, name="rising")
    assert summarise_run(rising)["peaked_before_end"] is False


def test_window_sd_needs_two_points(tmp_path: Path) -> None:
    run = _write_run(tmp_path, {2: 0.1, 4: 0.2, 6: 0.3})
    assert summarise_run(run, tail=1)["terminal_window_sd"] is None
    assert summarise_run(run, tail=3)["terminal_window_sd"] is not None


# --- missing data reads as missing, never as zero ---------------------------


def test_absent_metric_is_none_not_zero(tmp_path: Path) -> None:
    """A key the run never logged must not be reported as 0.0."""
    run = _write_run(tmp_path, {2: 0.1, 4: 0.2, 6: 0.3})
    rec = summarise_run(run)
    assert rec["terminal_val/AP_small"] is None
    assert rec["max_val/AP_small"] is None
    assert rec["terminal_val/AP"] is not None


def test_all_reported_metrics_are_extracted_when_present(tmp_path: Path) -> None:
    """The driver typos dropped AP_small silently; every key is asserted here."""
    extra = {e: {k: 0.5 for k in REPORTED_METRICS if k != "val/AP"} for e in (2, 4, 6)}
    run = _write_run(tmp_path, {2: 0.1, 4: 0.2, 6: 0.3}, extra=extra)
    rec = summarise_run(run)
    for key in REPORTED_METRICS:
        assert rec[f"terminal_{key}"] is not None, key
        assert rec[f"max_{key}"] is not None, key


def test_underscore_not_hyphen_is_the_small_object_key() -> None:
    """`val/AP-small` is the typo that discarded every small-object number."""
    assert "val/AP_small" in REPORTED_METRICS
    assert "val/AP-small" not in REPORTED_METRICS


# --- failure behaviour -------------------------------------------------------


def test_missing_metrics_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RunSummaryError, match="no metrics.jsonl"):
        summarise_run(tmp_path / "absent")


def test_run_with_no_evaluation_is_an_error(tmp_path: Path) -> None:
    run = tmp_path / "noeval"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        json.dumps({"epoch": 1, "train/total_loss": 4.0}) + "\n", encoding="utf-8"
    )
    with pytest.raises(RunSummaryError, match="none carry 'val/AP'"):
        summarise_run(run)


def test_malformed_line_names_the_line(tmp_path: Path) -> None:
    run = tmp_path / "bad"
    run.mkdir()
    (run / "metrics.jsonl").write_text('{"epoch": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(RunSummaryError, match=r":2 is not valid JSON"):
        load_eval_rows(run / "metrics.jsonl")


def test_tail_must_be_positive(tmp_path: Path) -> None:
    run = _write_run(tmp_path, {2: 0.1})
    with pytest.raises(RunSummaryError, match="tail must be"):
        summarise_run(run, tail=0)


# --- provenance and aggregation ---------------------------------------------


def test_provenance_is_carried_through_when_status_has_it(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        {2: 0.1, 4: 0.2},
        status={"completed": True, "provenance": {"git_commit": "abc123", "seed": 42}},
    )
    rec = summarise_run(run)
    assert rec["provenance"]["git_commit"] == "abc123"
    assert rec["completed"] is True


def test_absent_status_is_not_an_error(tmp_path: Path) -> None:
    rec = summarise_run(_write_run(tmp_path, {2: 0.1, 4: 0.2}))
    assert rec["provenance"] is None


def test_aggregate_reports_spread_across_runs(tmp_path: Path) -> None:
    runs = [
        _write_run(tmp_path, {2: v, 4: v, 6: v}, name=f"r{i}")
        for i, v in enumerate((0.10, 0.20, 0.30))
    ]
    agg = aggregate([summarise_run(r) for r in runs])
    assert agg["n"] == 3
    assert agg["mean"] == pytest.approx(0.20)
    assert agg["sd"] == pytest.approx(0.10)
    assert agg["cv_percent"] == pytest.approx(50.0)


def test_aggregate_of_one_run_has_no_sd(tmp_path: Path) -> None:
    agg = aggregate([summarise_run(_write_run(tmp_path, {2: 0.1, 4: 0.1}))])
    assert agg["n"] == 1 and agg["sd"] is None


# --- CLI ---------------------------------------------------------------------


def test_cli_writes_json_and_succeeds(tmp_path: Path) -> None:
    run = _write_run(tmp_path, {2: 0.1, 4: 0.2, 6: 0.3})
    out = tmp_path / "nested" / "summary.json"
    assert main([str(run), "--json", str(out), "--aggregate"]) == 0
    records = json.loads(out.read_text(encoding="utf-8"))
    assert records[0]["terminal_val/AP"] == pytest.approx(0.20)


def test_cli_skips_a_bad_run_but_reports_the_good_ones(tmp_path: Path) -> None:
    good = _write_run(tmp_path, {2: 0.1, 4: 0.2}, name="good")
    assert main([str(good), str(tmp_path / "absent")]) == 0


def test_cli_fails_when_nothing_can_be_summarised(tmp_path: Path) -> None:
    assert main([str(tmp_path / "absent")]) == 1
