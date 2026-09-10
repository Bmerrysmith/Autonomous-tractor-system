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
from agrinav.training.run_manifest import finish_run, start_run


def _write_run(
    tmp_path: Path,
    ap_by_epoch: dict[int, float],
    *,
    name: str = "run",
    extra: dict[int, dict[str, float]] | None = None,
    status: dict | None = None,
    seed: int | None = None,
    config_extra: dict | None = None,
) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"max_detections": 100, "use_soft_nms": False}
    provenance = None
    if seed is not None:

        class Dataset(list):
            pass

        dataset = Dataset([1])
        dataset.ann_file = tmp_path / "annotations.json"
        dataset.ann_file.write_text('{"images": []}', encoding="utf-8")
        provenance = start_run(
            {
                "num_epochs": max(ap_by_epoch),
                "val_ap_interval": 2,
                "seed": seed,
                **(config_extra or {}),
            },
            run_dir,
            trainer="fixture",
            train_dataset=dataset,
            val_dataset=dataset,
            protocol=protocol,
            device="cpu",
        )
        status = {
            "completed": True,
            "epochs_planned": max(ap_by_epoch),
            "epochs_completed": max(ap_by_epoch),
            "provenance": provenance,
            **(status or {}),
        }
    lines = []
    for epoch, ap in ap_by_epoch.items():
        row = {"epoch": epoch, "val/AP": ap, "train/total_loss": 5.0 - epoch * 0.1}
        if provenance:
            row.update(run_id=provenance["run_id"], eval_protocol=protocol)
        row.update((extra or {}).get(epoch, {}))
        lines.append(json.dumps(row))
    (run_dir / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status is not None:
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    if provenance:
        (run_dir / "fixture.pth").write_bytes(b"fixture checkpoint")
        finish_run(run_dir, status, ["fixture.pth"])
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
        _write_run(tmp_path, {2: v, 4: v, 6: v}, name=f"r{i}", seed=i)
        for i, v in enumerate((0.10, 0.20, 0.30))
    ]
    agg = aggregate([summarise_run(r) for r in runs])
    assert agg["n"] == 3
    assert agg["mean"] == pytest.approx(0.20)
    assert agg["sd"] == pytest.approx(0.10)
    assert agg["cv_percent"] == pytest.approx(50.0)


def test_aggregate_of_one_run_has_no_sd(tmp_path: Path) -> None:
    agg = aggregate([summarise_run(_write_run(tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42))])
    assert agg["n"] == 1 and agg["sd"] is None


# --- CLI ---------------------------------------------------------------------


def test_cli_writes_json_and_succeeds(tmp_path: Path) -> None:
    run = _write_run(tmp_path, {2: 0.1, 4: 0.2, 6: 0.3}, seed=42)
    out = tmp_path / "nested" / "summary.json"
    assert main([str(run), "--json", str(out), "--aggregate"]) == 0
    records = json.loads(out.read_text(encoding="utf-8"))
    assert records[0]["terminal_val/AP"] == pytest.approx(0.20)


def test_cli_skips_a_bad_run_but_reports_the_good_ones(tmp_path: Path) -> None:
    good = _write_run(tmp_path, {2: 0.1, 4: 0.2}, name="good")
    assert main([str(good), str(tmp_path / "absent")]) == 0


def test_cli_fails_when_nothing_can_be_summarised(tmp_path: Path) -> None:
    assert main([str(tmp_path / "absent")]) == 1


@pytest.mark.parametrize("ap", [float("nan"), float("inf"), -1, 1.01, True, "0.1", None])
def test_invalid_primary_metric_is_rejected(tmp_path, ap):
    run = _write_run(tmp_path, {2: ap})
    with pytest.raises(RunSummaryError, match="invalid val/AP"):
        summarise_run(run)


@pytest.mark.parametrize("row", [[], 42, None])
def test_non_object_row_is_rejected(tmp_path, row):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(RunSummaryError, match="JSON object"):
        load_eval_rows(path)


@pytest.mark.parametrize("epochs", [[2, 2], [4, 2], [0, 2], [True, 2], [1.5, 2]])
def test_invalid_epoch_sequence_is_rejected(tmp_path, epochs):
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps({"epoch": e, "val/AP": 0.1}) for e in epochs))
    with pytest.raises(RunSummaryError, match="epochs must be"):
        load_eval_rows(path)


def test_partial_secondary_window_does_not_change_estimator(tmp_path):
    run = _write_run(tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, extra={2: {"val/AP_small": 0.4}})
    assert summarise_run(run)["terminal_val/AP_small"] is None


def test_coco_sentinel_is_unavailable_and_blocks_research(tmp_path):
    run = _write_run(tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42, extra={6: {"val/AP_small": -1}})
    record = summarise_run(run)
    assert record["terminal_val/AP_small"] is None
    with pytest.raises(RunSummaryError, match="sentinels"):
        aggregate([record])


@pytest.mark.parametrize("status", [{"completed": False}, {"epochs_completed": 4}])
def test_incomplete_run_cannot_enter_aggregate(tmp_path, status):
    run = _write_run(tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42, status=status)
    with pytest.raises(RunSummaryError, match="aggregate refused"):
        aggregate([summarise_run(run)])


def test_duplicate_seed_is_not_replication(tmp_path):
    runs = [
        _write_run(tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42, name=name) for name in ("a", "b")
    ]
    with pytest.raises(RunSummaryError, match="duplicate"):
        aggregate([summarise_run(run) for run in runs])


def test_mixed_recipes_are_not_pooled(tmp_path):
    a = _write_run(
        tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42, name="a", config_extra={"img_size": 512}
    )
    b = _write_run(
        tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=43, name="b", config_extra={"img_size": 640}
    )
    with pytest.raises(RunSummaryError, match="mixed recipes"):
        aggregate([summarise_run(a), summarise_run(b)])


def test_missing_scheduled_evaluation_blocks_research(tmp_path):
    run = _write_run(tmp_path, {2: 0.1, 6: 0.1, 8: 0.1}, seed=42)
    assert "evaluation epochs do not match launch schedule" in summarise_run(run)["research_issues"]


def test_manifest_tampering_blocks_research(tmp_path):
    run = _write_run(tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42)
    path = next(run.glob("manifest-*.json"))
    manifest = json.loads(path.read_text())
    manifest["seed"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(RunSummaryError, match="checksum"):
        aggregate([summarise_run(run)])


def test_legacy_baseline_run_json_is_readable_but_not_verified(tmp_path):
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "config": {"num_epochs": 6},
                "epochs": [{"epoch": e, "val_ap": e / 10, "train_loss": 1.0} for e in (2, 4, 6)],
            }
        )
    )
    record = summarise_run(tmp_path)
    assert record["terminal_val/AP"] == pytest.approx(0.4)
    assert record["terminal_val/AP_small"] is None
    assert not record["research_eligible"]


def test_cli_refuses_partial_aggregate(tmp_path):
    run = _write_run(tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42)
    assert main([str(run), str(tmp_path / "absent"), "--aggregate"]) == 1


def test_final_loss_uses_last_training_row(tmp_path):
    run = _write_run(tmp_path, {2: 0.1})
    with (run / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps({"epoch": 3, "train/total_loss": 0.25}) + "\n")
    assert summarise_run(run)["final_train_loss"] == 0.25


def test_modified_checkpoint_cannot_enter_aggregate(tmp_path):
    run = _write_run(tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42)
    (run / "fixture.pth").write_bytes(b"different weights")
    with pytest.raises(RunSummaryError, match="artifact checksum"):
        aggregate([summarise_run(run)])


def test_overfit_diagnostic_cannot_be_pooled_as_research(tmp_path):
    run = _write_run(
        tmp_path, {2: 0.1, 4: 0.1, 6: 0.1}, seed=42, config_extra={"diagnostic": "overfit"}
    )
    with pytest.raises(RunSummaryError, match="diagnostic run"):
        aggregate([summarise_run(run)])
