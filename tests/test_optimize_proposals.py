"""Tests for ``optimize_proposals``: candidate selection, fragments, and safety.

The load-bearing assertions here are the ones that check what the module
REFUSES to do -- substitute a candidate kind, mint a truth marker, default a
label, or delete an object -- and the round-trip through
``triage_proposals.read_feature_rows``, which is the actual downstream contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from agrinav.data import triage_proposals
from agrinav.data.optimize_proposals import (
    DEFAULT_MIN_COMPONENT_FRACTION,
    FRAGMENT_POLICY_DROP_SMALL,
    FRAGMENT_POLICY_KEEP_ALL,
    FRAGMENT_POLICY_LARGEST,
    SKIP_DEGENERATE_BOX,
    SKIP_RULE_KIND_NOT_IN_SHARD,
    OptimizeProposalsError,
    Options,
    apply_fragment_policy,
    candidates_per_box_of,
    main,
    optimize,
    read_shard_rows,
    select_for_row,
)
from agrinav.data.sam_box_to_mask import RAW_SCHEMA_VERSION

pytestmark = pytest.mark.unit

WIDTH, HEIGHT = 40, 40
BOX_XYWH = [4.0, 4.0, 20.0, 20.0]


def _rle(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as maskutils

    encoded = maskutils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    return {
        "size": [int(v) for v in encoded["size"]],
        "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts,
    }


def _blob(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _candidate(index: int, kind: str, mask: np.ndarray, score: float) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "kind": kind,
        "multimask_index": index if kind == "multimask" else None,
        "jitter_index": index - 3 if kind == "jitter" else None,
        "prompt_box": list(BOX_XYWH),
        "sam_pred_iou": score,
        "area_px": int(np.count_nonzero(mask)),
        "rle": _rle(mask),
    }


def make_row(
    *,
    sha: str = "a" * 64,
    ann_id: int = 11,
    image_id: int = 1,
    label: str = "rice_protect",
    single_mask: np.ndarray | None = None,
    with_single_mask: bool = True,
    candidates_per_box: int | None = None,
    candidate_kinds: Sequence[str] | None = None,
    box: Sequence[float] = tuple(BOX_XYWH),
    degenerate: bool = False,
) -> dict[str, Any]:
    """One raw ``sam_box_to_mask`` row, with real RLE candidates."""
    chosen = single_mask if single_mask is not None else _blob(6, 6, 22, 22)
    candidates: list[dict[str, Any]] = []
    if not degenerate:
        for index in range(3):
            candidates.append(
                _candidate(index, "multimask", _blob(5, 5, 20 + index, 20), 0.9 - 0.1 * index)
            )
        for index in range(3, 8):
            candidates.append(_candidate(index, "jitter", _blob(6, 6, 22, 21 + index % 2), 0.8))
        if with_single_mask:
            candidates.append(_candidate(8, "single_mask", chosen, 0.95))
    kinds = (
        list(candidate_kinds)
        if candidate_kinds is not None
        else (
            ["multimask", "jitter", "single_mask"] if with_single_mask else ["multimask", "jitter"]
        )
    )
    per_box = (
        candidates_per_box if candidates_per_box is not None else (9 if with_single_mask else 8)
    )
    thresholds: dict[str, Any] = {
        "candidate_kinds": kinds,
        "candidates_per_box": per_box,
        "candidate_selection": "deferred_to_optimize_proposals",
    }
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "key": f"{sha}|coco-ann-{ann_id}",
        "source_image_sha256": sha,
        "coco_image_id": image_id,
        "file_name": f"img_{image_id}.jpg",
        "width": WIDTH,
        "height": HEIGHT,
        "group_id": "rice_detection:family_a",
        "capture_family": "family_a",
        "source_object_id": f"coco-ann-{ann_id}",
        "coco_annotation_id": ann_id,
        "label": label,
        "prompt_box_clipped": [] if degenerate else list(box),
        "degenerate_reason": "prompt_box_degenerate_after_clip" if degenerate else None,
        "candidate_count": len(candidates),
        "provenance": {
            "thresholds": thresholds,
            "review_status": "unreviewed",
            "annotator_id": None,
            "reviewer_id": None,
            "original_proposal": {
                "source_object_id": f"coco-ann-{ann_id}",
                "coco_annotation_id": ann_id,
                "coco_category_id": 1 if label == "rice_protect" else 2,
                "coco_bbox_xywh": list(box),
                "candidates": candidates,
            },
        },
        "_source_shard": "test-shard",
    }


def write_shard(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    payload = [{k: v for k, v in row.items() if k != "_source_shard"} for row in rows]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- #
# Candidate count is read, never assumed
# --------------------------------------------------------------------------- #
def test_candidates_per_box_is_read_from_the_shard() -> None:
    assert candidates_per_box_of(make_row()) == 9
    assert candidates_per_box_of(make_row(with_single_mask=False)) == 8
    assert candidates_per_box_of(make_row(candidates_per_box=12)) == 12


def test_a_shard_that_does_not_state_its_candidate_count_is_a_hard_error() -> None:
    row = make_row()
    del row["provenance"]["thresholds"]["candidates_per_box"]
    with pytest.raises(OptimizeProposalsError) as excinfo:
        candidates_per_box_of(row)
    message = str(excinfo.value)
    assert "candidates_per_box" in message
    assert "8-candidate" in message and "9-candidate" in message


def test_eight_candidate_shard_yields_none_for_the_single_mask_rule() -> None:
    """The whole point: no substitution, and the caller is told which case it is."""
    candidate, reason = select_for_row(make_row(with_single_mask=False), "single_mask_exact_box")
    assert candidate is None
    assert reason == SKIP_RULE_KIND_NOT_IN_SHARD


def test_a_nine_candidate_shard_selects_the_single_mask_candidate() -> None:
    candidate, reason = select_for_row(make_row(), "single_mask_exact_box")
    assert reason is None
    assert candidate is not None
    assert candidate["kind"] == "single_mask"
    assert candidate["candidate_index"] == 8


def test_a_degenerate_box_is_reported_as_such_not_as_a_missing_kind() -> None:
    candidate, reason = select_for_row(make_row(degenerate=True), "single_mask_exact_box")
    assert candidate is None
    assert reason == SKIP_DEGENERATE_BOX


def test_an_unservable_shard_aborts_rather_than_substituting() -> None:
    rows = [make_row(with_single_mask=False)]
    with pytest.raises(OptimizeProposalsError) as excinfo:
        optimize(rows, Options())
    message = str(excinfo.value)
    assert "single_mask" in message
    assert "--allow-rule-unavailable" in message
    assert "multimask_argmax_pred_iou" in message


def test_acknowledged_unservable_shard_emits_bbox_only_and_counts_it() -> None:
    rows = [make_row(with_single_mask=False)]
    result = optimize(rows, Options(allow_rule_unavailable=True))
    annotation = result["coco"]["annotations"][0]
    assert annotation["segmentation"] == []
    assert annotation["bbox"] == BOX_XYWH
    assert annotation["agrinav_selection"]["candidate_kind"] is None
    assert annotation["agrinav_selection"]["skip_reason"] == SKIP_RULE_KIND_NOT_IN_SHARD
    assert result["manifest"]["skips"][SKIP_RULE_KIND_NOT_IN_SHARD] == 1
    assert result["manifest"]["counts"]["objects_demoted_to_bbox"] == 1
    # The object survives as a human box: nothing was deleted.
    assert result["manifest"]["counts"]["objects_emitted"] == 1


def test_the_multimask_rule_still_works_on_an_eight_candidate_shard() -> None:
    rows = [make_row(with_single_mask=False)]
    result = optimize(rows, Options(selection_rule="multimask_argmax_pred_iou"))
    selection = result["coco"]["annotations"][0]["agrinav_selection"]
    assert selection["candidate_kind"] == "multimask"
    assert selection["candidate_index"] == 0  # highest sam_pred_iou
    assert result["coco"]["annotations"][0]["segmentation"]


# --------------------------------------------------------------------------- #
# Selection is deterministic and recorded
# --------------------------------------------------------------------------- #
def test_selection_rule_and_candidate_index_are_recorded_on_every_object() -> None:
    rows = [make_row(), make_row(ann_id=12, with_single_mask=False)]
    result = optimize(rows, Options(allow_rule_unavailable=True))
    for annotation in result["coco"]["annotations"]:
        assert annotation["agrinav_selection"]["selection_rule"] == "single_mask_exact_box"
    for feature in result["features"]:
        assert feature["selection"]["selection_rule"] == "single_mask_exact_box"
    chosen = [f for f in result["features"] if f["selection"]["candidate_kind"]]
    assert [f["selection"]["candidate_index"] for f in chosen] == [8]


def test_two_runs_over_one_shard_are_byte_identical(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "shard.jsonl", [make_row(), make_row(ann_id=12)])
    outputs = []
    for run in ("a", "b"):
        code = main(
            [
                "--raw-shard",
                str(shard),
                "--out-coco",
                str(tmp_path / f"{run}.coco.json"),
                "--out-features",
                str(tmp_path / f"{run}.jsonl"),
                "--out-manifest",
                str(tmp_path / f"{run}.manifest.json"),
            ]
        )
        assert code == 0
        outputs.append(
            (
                (tmp_path / f"{run}.coco.json").read_bytes(),
                (tmp_path / f"{run}.jsonl").read_bytes(),
            )
        )
    assert outputs[0] == outputs[1]


def test_object_order_does_not_depend_on_input_order() -> None:
    rows = [make_row(ann_id=12), make_row(ann_id=11), make_row(ann_id=13)]
    forward = optimize(rows, Options())
    backward = optimize(list(reversed(rows)), Options())
    assert [a["id"] for a in forward["coco"]["annotations"]] == [11, 12, 13]
    assert forward["features"] == backward["features"]


# --------------------------------------------------------------------------- #
# Nothing may read as truth
# --------------------------------------------------------------------------- #
def test_no_truth_marker_reaches_either_output() -> None:
    result = optimize([make_row(), make_row(ann_id=12, label="weed_target")], Options())
    blob = json.dumps(result["coco"]) + json.dumps(result["features"])
    for forbidden in ("verified_empty", "treatment_eligible", "annotator_id", "reviewer_id"):
        assert forbidden not in blob
    for annotation in result["coco"]["annotations"]:
        assert annotation["review_status"] == "unreviewed_proposal"
    for feature in result["features"]:
        assert feature["review_status"] == "unreviewed"


def test_review_status_vocabularies_are_not_crossed() -> None:
    """``unreviewed_proposal`` is not an ``annotation_record.v1`` enum member."""
    result = optimize([make_row()], Options())
    assert result["coco"]["annotations"][0]["review_status"] not in (
        triage_proposals.REVIEW_STATUSES_SNAPSHOT
    )
    assert result["features"][0]["review_status"] in triage_proposals.REVIEW_STATUSES_SNAPSHOT


def test_an_unmapped_label_is_a_counted_drop_that_aborts() -> None:
    rows = [make_row(), make_row(ann_id=12, label="rice_plant")]
    with pytest.raises(OptimizeProposalsError) as excinfo:
        optimize(rows, Options())
    assert "rice_plant" in str(excinfo.value)
    assert "--allow-unmapped-labels" in str(excinfo.value)


def test_an_unmapped_label_is_never_defaulted_into_a_known_class() -> None:
    rows = [make_row(), make_row(ann_id=12, label="rice_plant")]
    result = optimize(rows, Options(allow_unmapped_labels=True))
    assert [a["category_id"] for a in result["coco"]["annotations"]] == [1]
    assert result["manifest"]["label_drops"] == {"label='rice_plant'": 1}
    assert result["manifest"]["counts"]["objects_dropped_unmapped_label"] == 1


def test_weed_box_count_in_is_recorded_from_the_input_side() -> None:
    """The count is taken before any drop, so a later filter stays detectable.

    ``triage_proposals`` escalates to T1 when this disagrees with the number of
    ``weed_target`` rows it sees. This stage cannot itself lose a
    ``weed_target`` (that label is always mapped and every object is emitted),
    so the field is a tripwire for stages downstream of here, not a check on
    this one -- but it has to carry the input-side number for that to work.
    """
    rows = [
        make_row(ann_id=11, label="weed_target"),
        make_row(ann_id=12, label="weed_target"),
        make_row(ann_id=13, label="rice_plant"),
    ]
    result = optimize(rows, Options(allow_unmapped_labels=True))
    assert len(result["features"]) == 2
    assert {f["weed_box_count_in"] for f in result["features"]} == {2}


# --------------------------------------------------------------------------- #
# Fragment policy
# --------------------------------------------------------------------------- #
def _fragmented(main_size: int = 12, speck: bool = True) -> np.ndarray:
    mask = _blob(6, 6, 6 + main_size, 6 + main_size)
    if speck:
        mask[30:31, 30:31] = True  # 1 px against 144: 0.7%
    else:
        mask[26:34, 26:34] = True  # 64 px against 144: 30.8%
    return mask


def test_drop_small_components_removes_the_speck() -> None:
    result = apply_fragment_policy(
        _fragmented(), FRAGMENT_POLICY_DROP_SMALL, DEFAULT_MIN_COMPONENT_FRACTION
    )
    assert result.components_total == 2
    assert result.components_dropped == 1
    assert result.dropped_area_fraction < DEFAULT_MIN_COMPONENT_FRACTION
    assert not result.mask[30, 30]


def test_drop_small_components_preserves_the_substantial_tail() -> None:
    """The measured ~2% tail is real structure; the default must not delete it."""
    result = apply_fragment_policy(
        _fragmented(speck=False), FRAGMENT_POLICY_DROP_SMALL, DEFAULT_MIN_COMPONENT_FRACTION
    )
    assert result.components_total == 2
    assert result.components_dropped == 0
    assert result.mask[30, 30]


def test_largest_component_deletes_the_tail_and_says_so() -> None:
    result = apply_fragment_policy(_fragmented(speck=False), FRAGMENT_POLICY_LARGEST, 0.01)
    assert result.components_dropped == 1
    assert result.dropped_area_fraction > 0.3
    assert not result.mask[30, 30]


def test_keep_all_drops_nothing() -> None:
    result = apply_fragment_policy(_fragmented(), FRAGMENT_POLICY_KEEP_ALL, 0.01)
    assert result.components_dropped == 0
    assert result.dropped_area_fraction == 0.0
    assert result.mask[30, 30]


def test_an_unknown_fragment_policy_raises_instead_of_defaulting() -> None:
    with pytest.raises(OptimizeProposalsError, match="unknown fragment policy"):
        apply_fragment_policy(_fragmented(), "largest", 0.01)


def test_a_surviving_second_component_becomes_a_second_ring_and_is_counted() -> None:
    rows = [make_row(single_mask=_fragmented(speck=False))]
    result = optimize(rows, Options())
    segmentation = result["coco"]["annotations"][0]["segmentation"]
    assert len(segmentation) == 2
    assert result["manifest"]["fragmentation"]["objects_multi_ring"] == 1


def test_fragmentation_does_not_reach_reason_codes() -> None:
    """``reason_codes`` means auto_reject in triage; a real mask must not carry one."""
    result = optimize([make_row(single_mask=_fragmented(speck=False))], Options())
    feature = result["features"][0]
    assert feature["reason_codes"] == []
    assert feature["fragment"]["components_total"] == 2
    assert feature["geometry_type"] == "polygon"


def test_a_demoted_object_carries_a_reason_code_so_triage_rejects_the_mask() -> None:
    result = optimize([make_row(degenerate=True)], Options())
    feature = result["features"][0]
    assert feature["reason_codes"] == [SKIP_DEGENERATE_BOX]
    assert feature["geometry_type"] == "bbox"


# --------------------------------------------------------------------------- #
# The downstream contract: triage must be able to read what we emit
# --------------------------------------------------------------------------- #
def test_feature_rows_parse_in_triage_proposals(tmp_path: Path) -> None:
    rows = [make_row(), make_row(ann_id=12, label="weed_target")]
    result = optimize(rows, Options())
    path = tmp_path / "proposal_features_v1.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["features"]),
        encoding="utf-8",
    )
    parsed = triage_proposals.read_feature_rows(path)
    assert len(parsed) == 2
    assert {row.label for row in parsed} == {"rice_protect", "weed_target"}
    assert all(row.group_id == "rice_detection:family_a" for row in parsed)
    assert all(row.geometry_type == "polygon" for row in parsed)


def test_every_required_triage_feature_is_present_for_a_normal_object(tmp_path: Path) -> None:
    result = optimize([make_row()], Options())
    path = tmp_path / "features.jsonl"
    path.write_text(json.dumps(result["features"][0], sort_keys=True) + "\n", encoding="utf-8")
    row = triage_proposals.read_feature_rows(path)[0]
    assert row.has_required_features()
    missing = [name for name in triage_proposals.REQUIRED_FEATURES if row.feature(name) is None]
    assert missing == []


def test_the_pixel_features_triage_needs_for_promotion_are_absent_by_construction(
    tmp_path: Path,
) -> None:
    """No image pixels are read here, so family B/C cannot be produced.

    That is the safe direction: ``triage_proposals`` needs at least one
    independent corroborating signal to promote anything, so a feature sidecar
    from this stage alone can never reach ``bulk_confirm``.
    """
    result = optimize([make_row()], Options())
    path = tmp_path / "features.jsonl"
    path.write_text(json.dumps(result["features"][0], sort_keys=True) + "\n", encoding="utf-8")
    row = triage_proposals.read_feature_rows(path)[0]
    for name in (triage_proposals.F_VEG_IN, triage_proposals.F_VEG_GAIN):
        assert row.feature(name) is None
    assert row.feature(triage_proposals.F_GRABCUT_IOU) is None


def test_neighbour_features_are_computed_within_an_image() -> None:
    rows = [
        make_row(ann_id=11, label="rice_protect", single_mask=_blob(6, 6, 22, 22)),
        make_row(ann_id=12, label="weed_target", single_mask=_blob(10, 10, 26, 26)),
    ]
    result = optimize(rows, Options())
    rice = next(f for f in result["features"] if f["label"] == "rice_protect")
    assert rice["features"]["cross_class_ioa_max"] > 0
    assert rice["features"]["rice_over_weed_box_ioa"] is not None
    weed = next(f for f in result["features"] if f["label"] == "weed_target")
    assert weed["features"]["rice_over_weed_box_ioa"] is None


# --------------------------------------------------------------------------- #
# IO and CLI
# --------------------------------------------------------------------------- #
def test_duplicate_object_keys_across_shards_are_refused(tmp_path: Path) -> None:
    first = write_shard(tmp_path / "a.jsonl", [make_row()])
    second = write_shard(tmp_path / "b.jsonl", [make_row()])
    with pytest.raises(OptimizeProposalsError, match="duplicate object key"):
        read_shard_rows([first, second])


def test_a_foreign_schema_version_is_refused(tmp_path: Path) -> None:
    row = make_row()
    row["schema_version"] = "agrinav.something_else.v1"
    path = write_shard(tmp_path / "shard.jsonl", [row])
    with pytest.raises(OptimizeProposalsError, match="schema_version"):
        read_shard_rows([path])


def test_cli_writes_coco_features_and_manifest(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "shard.jsonl", [make_row(), make_row(ann_id=12)])
    coco_path = tmp_path / "out.coco.json"
    features_path = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "out.manifest.json"
    assert (
        main(
            [
                "--raw-shard",
                str(shard),
                "--out-coco",
                str(coco_path),
                "--out-features",
                str(features_path),
                "--out-manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    assert [c["name"] for c in coco["categories"]] == ["rice_protect", "weed_target"]
    assert len(coco["annotations"]) == 2
    assert len(coco["images"]) == 1
    assert coco["info"]["selection_rule"] == "single_mask_exact_box"
    assert coco["info"]["fragment_policy"] == FRAGMENT_POLICY_DROP_SMALL
    assert "NOT training truth" in coco["info"]["WARNING"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["objects_with_polygon"] == 2
    assert manifest["candidates_per_box_declared"] == [9]
    assert len(features_path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_cli_refuses_to_clobber_without_overwrite(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "shard.jsonl", [make_row()])
    coco_path = tmp_path / "out.coco.json"
    coco_path.write_text("{}", encoding="utf-8")
    argv = [
        "--raw-shard",
        str(shard),
        "--out-coco",
        str(coco_path),
        "--out-features",
        str(tmp_path / "out.jsonl"),
    ]
    assert main(argv) == 2
    assert coco_path.read_text(encoding="utf-8") == "{}"
    assert main([*argv, "--overwrite"]) == 0


def test_same_class_crowding_names_the_partner_for_collision_grouping() -> None:
    """``triage_proposals`` groups a collision from this id; both survive."""
    rows = [
        make_row(ann_id=11, single_mask=_blob(6, 6, 22, 22)),
        make_row(ann_id=12, single_mask=_blob(7, 7, 23, 23)),
    ]
    result = optimize(rows, Options())
    first = next(f for f in result["features"] if f["coco_annotation_id"] == 11)
    assert first["features"]["same_class_neighbor_annotation_id"] == "coco-ann-12"
    assert first["features"]["same_class_mask_iou_max"] > 0.5
    assert result["manifest"]["counts"]["objects_emitted"] == 2


def test_a_shard_whose_provenance_undercounts_its_payload_is_refused() -> None:
    """The declared count is load-bearing, so it has to describe the payload."""
    row = make_row(candidates_per_box=4)
    with pytest.raises(OptimizeProposalsError) as excinfo:
        optimize([row], Options())
    assert "declares candidates_per_box=4" in str(excinfo.value)
    assert "9 candidates" in str(excinfo.value)


def test_fewer_candidates_than_declared_is_legal() -> None:
    """A skipped jitter pass is normal; only an overcount is a contradiction."""
    row = make_row(candidates_per_box=20)
    result = optimize([row], Options())
    assert result["features"][0]["candidates_per_box_declared"] == 20
    assert result["features"][0]["candidates_available"] == 9


def test_no_output_is_written_when_one_destination_would_be_clobbered(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "shard.jsonl", [make_row()])
    features_path = tmp_path / "out.jsonl"
    features_path.write_text("existing\n", encoding="utf-8")
    coco_path = tmp_path / "out.coco.json"
    assert (
        main(
            [
                "--raw-shard",
                str(shard),
                "--out-coco",
                str(coco_path),
                "--out-features",
                str(features_path),
            ]
        )
        == 2
    )
    assert not coco_path.exists()
    assert features_path.read_text(encoding="utf-8") == "existing\n"
