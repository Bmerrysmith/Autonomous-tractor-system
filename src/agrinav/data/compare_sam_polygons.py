#!/usr/bin/env python3
"""Compare a fresh SAM raw-candidate shard against the existing polygon proposals.

THE QUESTION THIS ANSWERS
-------------------------
``agrinav_intake_2026-07-21/deliverable/detection/RICE/annotations/`` already
holds a full set of box-prompted SAM polygons. Re-seeding them costs GPU hours,
so the pilot has to establish whether new polygons are actually *better*. If
they are not, the honest outcome is to keep the old ones and spend the hours
elsewhere.

THERE IS NO MASK GROUND TRUTH, AND THIS MODULE DOES NOT PRETEND OTHERWISE
------------------------------------------------------------------------
Nobody has hand-drawn a rice or weed polygon in this project yet. Both sets are
unreviewed machine output, so "better" cannot be measured directly. What can be
measured are proxies, and they are reported as proxies:

* ``box_iou``   - IoU between the tight bbox of the mask and the *human* box.
                  The human box is the only reviewed geometry in the pipeline.
                  A mask that fills its prompt box scores ~1.0; a mask that
                  collapses to a fragment, or bleeds past the box, scores low.
                  This is the primary proxy, and it is *directional*: a higher
                  value is only better up to the point where the mask is simply
                  the rectangle again (see ``fill``).
* ``fill``      - mask area / box area. Distinguishes "tracks the object" from
                  "returned the whole box". A value near 1.0 alongside a
                  ``box_iou`` near 1.0 means SAM gave back the rectangle and
                  added no information over the box that was already there.
* ``containment`` - fraction of mask pixels inside the prompt box. Mask pixels
                  outside the human box are geometry no human asserted.
* ``jitter_agreement`` - mean IoU of the 5 deterministic box-jitter masks
                  against the selected mask. Only the new set has this; it is a
                  stability signal, not a quality signal.
* ``iou_new_old`` - agreement between the two sets. High agreement means the
                  re-seed would change little; that is itself a finding.

MASK SELECTION IS A COMPARISON-ONLY HEURISTIC
---------------------------------------------
``sam_box_to_mask`` deliberately emits all 8 candidates and defers selection to
``optimize_proposals``. To compare anything, one has to be chosen. This module
picks ``argmax(sam_pred_iou)`` over the 3 ``multimask`` candidates - SAM's own
default - and says so in the output. It does NOT write that choice anywhere a
downstream stage could read it as a decision, and it emits no annotation
records.

EXIF-REORIENTED IMAGES ARE EXCLUDED, NOT SILENTLY COMPARED
----------------------------------------------------------
The phase-2 rebuild applied ``ImageOps.exif_transpose`` to 214 images. For
orientation 3 (180 degrees) the width and height are unchanged while every
pixel moves, so a dimension check would not catch it and the comparison would
silently score a rotated mask against an unrotated one. Those images are
therefore dropped by the rebuild's own ``exif_reoriented`` flag and counted.

CLI::

    python -m agrinav.data.compare_sam_polygons \\
        --raw-shard artifacts/sam_reseed/sam_raw/pilot_shard_000.jsonl \\
        --proposals-json artifacts/sam_reseed/pilot_proposals_unreviewed.coco.json \\
        --legacy-annotations-dir <intake>/detection/RICE/annotations \\
        --legacy-splits train,valid,test \\
        --out-json reports/metrics/sam_reseed_pilot_iou.json

``--legacy-splits`` names splits of the *legacy* export, whose partition was
rebuilt; a legacy 'test' file may hold images that are train/valid in the
current build. The sealed set is defined by the current build's membership, and
this module only ever looks up images that are already in ``--proposals-json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

#: Boxes are considered the same human assertion only at near-identity. The
#: rebuild copied boxes verbatim from the legacy export, so a genuine match is
#: ~1.0; anything looser risks pairing two different plants.
BOX_MATCH_MIN_IOU = 0.99

#: Candidate-selection rules this module can compare under. CLOSED dict: an
#: unknown rule is an error, never a fallback to the default, because silently
#: comparing two different rules is how a re-seed gets judged on the wrong
#: geometry. Selection itself properly belongs to ``optimize_proposals``; these
#: exist only so a comparison has something to score.
SELECTION_RULES: dict[str, str] = {
    "multimask_argmax_pred_iou": (
        "argmax(sam_pred_iou) over the multimask candidates -- SAM's default, "
        "and the only rule expressible before the single_mask candidate existed"
    ),
    "single_mask_exact_box": (
        "the multimask_output=False candidate on the unjittered human box "
        "(kind='single_mask'); the configuration SAM is trained for on "
        "unambiguous box prompts"
    ),
}
DEFAULT_SELECTION_RULE = "multimask_argmax_pred_iou"


def _decode_rle(rle: dict[str, Any]) -> np.ndarray:
    from pycocotools import mask as maskutils

    counts = rle["counts"]
    payload = {
        "size": [int(v) for v in rle["size"]],
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }
    return maskutils.decode(payload).astype(bool)


def _decode_polygon(segmentation: Any, height: int, width: int) -> np.ndarray | None:
    """Decode a COCO polygon / RLE segmentation to a boolean mask."""
    from pycocotools import mask as maskutils

    if not segmentation:
        return None
    if isinstance(segmentation, dict):
        return _decode_rle(segmentation)
    polygons = [np.asarray(p, dtype=np.float64) for p in segmentation if len(p) >= 6]
    if not polygons:
        return None
    rles = maskutils.frPyObjects([p.tolist() for p in polygons], height, width)
    merged = maskutils.merge(rles)
    return maskutils.decode(merged).astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape:
        return None
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else None


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def xywh_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in bbox)
    return (x, y, x + w, y + h)


def box_pixel_mask(bbox_xyxy: Sequence[float], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x0 = max(0, int(np.floor(bbox_xyxy[0])))
    y0 = max(0, int(np.floor(bbox_xyxy[1])))
    x1 = min(width, int(np.ceil(bbox_xyxy[2])))
    y1 = min(height, int(np.ceil(bbox_xyxy[3])))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def select_multimask(candidates: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest self-reported IoU among the multimask heads."""
    best = None
    for candidate in candidates:
        if candidate.get("kind") != "multimask":
            continue
        score = candidate.get("sam_pred_iou")
        if score is None:
            continue
        if best is None or score > best["sam_pred_iou"]:
            best = candidate
    return best


def select_single_mask(candidates: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """The one ``kind='single_mask'`` candidate, if the shard carries one.

    Returns ``None`` for a shard produced before that candidate existed, which
    the caller counts rather than silently substituting another candidate.
    """
    for candidate in candidates:
        if candidate.get("kind") == "single_mask":
            return candidate
    return None


def select_candidate(candidates: Iterable[dict[str, Any]], rule: str) -> dict[str, Any] | None:
    """Closed dispatch. No ``else``, no default rule."""
    materialised = list(candidates)
    if rule == "multimask_argmax_pred_iou":
        return select_multimask(materialised)
    if rule == "single_mask_exact_box":
        return select_single_mask(materialised)
    raise ValueError(f"unknown selection rule {rule!r}; choose one of {sorted(SELECTION_RULES)}")


def component_profile(mask: np.ndarray) -> tuple[int, float]:
    """Return (component count, area fraction NOT in the largest component).

    The second number is what decides whether fragmentation is a reviewer cost
    or a post-processing triviality: a mask whose extra components are 0.5% of
    its area is one largest-component call away from clean, whereas one where
    they are 30% has real structure outside the main blob and a human has to
    look.
    """
    import cv2

    arr = np.asarray(mask).astype(np.uint8)
    total, labels = cv2.connectedComponents(arr, connectivity=8)
    count = int(total) - 1
    if count <= 1:
        return count, 0.0
    areas = sorted((int(np.count_nonzero(labels == i)) for i in range(1, total)), reverse=True)
    summed = float(sum(areas))
    return count, (summed - areas[0]) / summed if summed else 0.0


def count_components(mask: np.ndarray) -> int:
    """8-connected component count. A fragmented mask is a quality signal.

    Every legacy polygon is a single part by construction (one polygon, no
    holes), so a multi-component raster is geometry the legacy format could not
    have expressed at all.
    """
    import cv2

    total, _ = cv2.connectedComponents(np.asarray(mask).astype(np.uint8), connectivity=8)
    return int(total) - 1  # label 0 is background


def summarise(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"n": 0}
    ordered = sorted(clean)

    def percentile(fraction: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 4)

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "median": round(statistics.median(ordered), 4),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def load_legacy_index(
    annotations_dir: Path, splits: Sequence[str], want_sha: set[str], want_name: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Index legacy images by sha256 (preferred) and basename (fallback)."""
    by_sha: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    found_in: dict[str, int] = {}
    for split in splits:
        path = annotations_dir / f"instances_{split}.coco.json"
        if not path.is_file():
            raise FileNotFoundError(f"legacy annotation file not found: {path}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        keep_ids: dict[int, dict[str, Any]] = {}
        for image in doc["images"]:
            sha = str(image.get("sha256") or "").lower()
            name = Path(image["file_name"]).name
            if sha in want_sha or name in want_name:
                record = {
                    "legacy_split": split,
                    "file_name": name,
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "sha256": sha,
                    "annotations": [],
                }
                keep_ids[image["id"]] = record
                if sha:
                    by_sha.setdefault(sha, record)
                by_name.setdefault(name, record)
                found_in[split] = found_in.get(split, 0) + 1
        for ann in doc["annotations"]:
            owner = keep_ids.get(ann["image_id"])
            if owner is not None:
                owner["annotations"].append(ann)
        del doc
    return {"by_sha": by_sha, "by_name": by_name}, found_in


def compare(
    *,
    raw_shard: Path,
    proposals_json: Path,
    legacy_dir: Path,
    legacy_splits: Sequence[str],
    selection_rule: str = DEFAULT_SELECTION_RULE,
) -> dict[str, Any]:
    """Pair every new candidate with its legacy polygon and score both."""
    if selection_rule not in SELECTION_RULES:
        raise ValueError(
            f"unknown selection rule {selection_rule!r}; "
            f"choose one of {sorted(SELECTION_RULES)}"
        )
    proposals = json.loads(proposals_json.read_text(encoding="utf-8"))
    images_by_sha = {str(i["sha256"]).lower(): i for i in proposals["images"]}
    want_sha = set(images_by_sha)
    want_name = {Path(i["file_name"]).name for i in proposals["images"]}
    legacy, found_in = load_legacy_index(legacy_dir, legacy_splits, want_sha, want_name)

    rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with raw_shard.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows_by_image[str(row["source_image_sha256"]).lower()].append(row)

    counters = {
        "images_in_shard": len(rows_by_image),
        "images_compared": 0,
        "images_excluded_exif_reoriented": 0,
        "images_excluded_no_legacy_match": 0,
        "images_excluded_dimension_mismatch": 0,
        "images_matched_by_sha256": 0,
        "images_matched_by_filename_only": 0,
        "objects_in_shard": 0,
        "objects_matched": 0,
        "objects_unmatched_no_legacy_box": 0,
        "objects_skipped_no_candidate_for_rule": 0,
        "objects_skipped_legacy_polygon_empty": 0,
        "objects_skipped_new_mask_empty": 0,
    }
    per_object: list[dict[str, Any]] = []

    for sha, rows in rows_by_image.items():
        counters["objects_in_shard"] += len(rows)
        image_meta = images_by_sha.get(sha)
        if image_meta is None:
            counters["images_excluded_no_legacy_match"] += 1
            continue
        if image_meta.get("exif_reoriented"):
            counters["images_excluded_exif_reoriented"] += 1
            continue
        record = legacy["by_sha"].get(sha)
        matched_by = "sha256"
        if record is None:
            record = legacy["by_name"].get(Path(image_meta["file_name"]).name)
            matched_by = "filename"
        if record is None:
            counters["images_excluded_no_legacy_match"] += 1
            continue
        width, height = int(image_meta["width"]), int(image_meta["height"])
        if (record["width"], record["height"]) != (width, height):
            counters["images_excluded_dimension_mismatch"] += 1
            continue
        counters["images_compared"] += 1
        counters[
            (
                "images_matched_by_sha256"
                if matched_by == "sha256"
                else "images_matched_by_filename_only"
            )
        ] += 1

        legacy_by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in record["annotations"]:
            legacy_by_category[ann["category_id"]].append(ann)
        consumed: set[int] = set()

        for row in rows:
            human_box = xywh_to_xyxy(row["provenance"]["original_proposal"]["coco_bbox_xywh"])
            category_id = row["provenance"]["original_proposal"]["coco_category_id"]
            best_ann, best_iou = None, 0.0
            for ann in legacy_by_category.get(category_id, []):
                if ann["id"] in consumed:
                    continue
                overlap = box_iou(human_box, xywh_to_xyxy(ann["bbox"]))
                if overlap > best_iou:
                    best_ann, best_iou = ann, overlap
            if best_ann is None or best_iou < BOX_MATCH_MIN_IOU:
                counters["objects_unmatched_no_legacy_box"] += 1
                continue
            consumed.add(best_ann["id"])
            counters["objects_matched"] += 1

            candidates = row["provenance"]["original_proposal"]["candidates"]
            chosen = select_candidate(candidates, selection_rule)
            if chosen is None:
                counters["objects_skipped_no_candidate_for_rule"] += 1
                continue
            new_mask = _decode_rle(chosen["rle"])
            old_mask = _decode_polygon(best_ann.get("segmentation"), height, width)
            if old_mask is None or not old_mask.any():
                counters["objects_skipped_legacy_polygon_empty"] += 1
                continue
            if not new_mask.any():
                counters["objects_skipped_new_mask_empty"] += 1
                continue

            new_components, new_nonlargest = component_profile(new_mask)
            old_components, old_nonlargest = component_profile(old_mask)
            box_mask = box_pixel_mask(human_box, height, width)
            box_area = float(np.count_nonzero(box_mask))
            new_bbox, old_bbox = mask_bbox(new_mask), mask_bbox(old_mask)
            jitters: list[float] = [
                value
                for value in (
                    mask_iou(_decode_rle(c["rle"]), new_mask)
                    for c in candidates
                    if c.get("kind") == "jitter"
                )
                if value is not None
            ]

            per_object.append(
                {
                    "label": row["label"],
                    "category_id": category_id,
                    "box_area_px": box_area,
                    "new_box_iou": box_iou(new_bbox, human_box) if new_bbox else 0.0,
                    "old_box_iou": box_iou(old_bbox, human_box) if old_bbox else 0.0,
                    "new_fill": float(np.count_nonzero(new_mask)) / box_area if box_area else None,
                    "old_fill": float(np.count_nonzero(old_mask)) / box_area if box_area else None,
                    "new_containment": (
                        float(np.count_nonzero(new_mask & box_mask))
                        / float(np.count_nonzero(new_mask))
                    ),
                    "old_containment": (
                        float(np.count_nonzero(old_mask & box_mask))
                        / float(np.count_nonzero(old_mask))
                    ),
                    "iou_new_old": mask_iou(new_mask, old_mask),
                    "new_components": new_components,
                    "old_components": old_components,
                    "new_nonlargest_area_fraction": round(new_nonlargest, 6),
                    "old_nonlargest_area_fraction": round(old_nonlargest, 6),
                    # identity, so a downstream renderer can find these pixels
                    # again without re-deriving the pairing
                    "image_sha256": sha,
                    "file_name": image_meta["file_name"],
                    "width": width,
                    "height": height,
                    "source_object_id": row["source_object_id"],
                    "coco_annotation_id": row["coco_annotation_id"],
                    "legacy_annotation_id": best_ann["id"],
                    # REQUIRED with the id: legacy COCO annotation ids restart
                    # at 1 in every split file, so a bare id is ambiguous and
                    # resolves to a different object in a different image.
                    "legacy_split": record["legacy_split"],
                    "human_box_xyxy": [round(v, 3) for v in human_box],
                    "new_candidate_index": chosen["candidate_index"],
                    "new_candidate_kind": chosen["kind"],
                    "new_outside_box_fraction": (
                        float(np.count_nonzero(new_mask & ~box_mask))
                        / float(np.count_nonzero(new_mask))
                    ),
                    "old_outside_box_fraction": (
                        float(np.count_nonzero(old_mask & ~box_mask))
                        / float(np.count_nonzero(old_mask))
                    ),
                    "new_jitter_agreement": (
                        round(statistics.fmean(jitters), 4) if jitters else None
                    ),
                    "new_sam_pred_iou": chosen.get("sam_pred_iou"),
                    "legacy_recorded_mask_box_iou": (best_ann.get("attributes") or {}).get(
                        "mask_box_iou"
                    ),
                }
            )

    return {
        "counters": counters,
        "legacy_images_found_in_split": found_in,
        "selection_rule": selection_rule,
        "objects": per_object,
    }


def aggregate(result: dict[str, Any]) -> dict[str, Any]:
    """Distributions overall and per class, plus the paired new-vs-old delta."""
    objects = result["objects"]
    metrics = (
        "new_box_iou",
        "old_box_iou",
        "new_fill",
        "old_fill",
        "new_containment",
        "old_containment",
        "new_outside_box_fraction",
        "old_outside_box_fraction",
        "new_components",
        "old_components",
        "new_nonlargest_area_fraction",
        "old_nonlargest_area_fraction",
        "iou_new_old",
        "new_jitter_agreement",
        "new_sam_pred_iou",
    )

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n_objects": len(subset)}
        for metric in metrics:
            out[metric] = summarise([o[metric] for o in subset if o.get(metric) is not None])
        paired = [
            o["new_box_iou"] - o["old_box_iou"]
            for o in subset
            if o.get("new_box_iou") is not None and o.get("old_box_iou") is not None
        ]
        out["box_iou_delta_new_minus_old"] = summarise(paired)
        if paired:
            out["fraction_new_better_box_iou"] = round(
                sum(1 for d in paired if d > 0) / len(paired), 4
            )
            out["fraction_new_worse_box_iou"] = round(
                sum(1 for d in paired if d < 0) / len(paired), 4
            )
            out["fraction_tied_within_0.01"] = round(
                sum(1 for d in paired if abs(d) <= 0.01) / len(paired), 4
            )
        for side in ("new", "old"):
            counts = [o[f"{side}_components"] for o in subset if o.get(f"{side}_components")]
            if counts:
                out[f"{side}_multi_component_rate"] = round(
                    sum(1 for c in counts if c >= 2) / len(counts), 4
                )
                out[f"{side}_5plus_component_rate"] = round(
                    sum(1 for c in counts if c >= 5) / len(counts), 4
                )
        return out

    per_class = {
        label: block([o for o in objects if o["label"] == label])
        for label in sorted({o["label"] for o in objects})
    }
    small = [o for o in objects if o["box_area_px"] < 32 * 32]
    rule = result.get("selection_rule", DEFAULT_SELECTION_RULE)
    return {
        "selection_rule": rule,
        "selection_rule_description": SELECTION_RULES[rule],
        "box_match_min_iou": BOX_MATCH_MIN_IOU,
        "counters": result["counters"],
        "legacy_images_found_in_split": result["legacy_images_found_in_split"],
        "overall": block(objects),
        "per_class": per_class,
        "small_objects_lt_32x32_px": block(small),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare new SAM candidates to legacy polygons")
    ap.add_argument("--raw-shard", required=True, type=Path)
    ap.add_argument("--proposals-json", required=True, type=Path)
    ap.add_argument("--legacy-annotations-dir", required=True, type=Path)
    ap.add_argument("--legacy-splits", default="train,valid,test")
    ap.add_argument(
        "--selection-rule",
        default=DEFAULT_SELECTION_RULE,
        choices=sorted(SELECTION_RULES),
        help="which candidate to score; comparison-only, asserts no decision",
    )
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-objects-jsonl", type=Path, default=None)
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.legacy_splits.split(",") if s.strip()]
    try:
        result = compare(
            raw_shard=args.raw_shard,
            proposals_json=args.proposals_json,
            legacy_dir=args.legacy_annotations_dir,
            legacy_splits=splits,
            selection_rule=args.selection_rule,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = aggregate(result)
    report["inputs"] = {
        "raw_shard": str(args.raw_shard),
        "proposals_json": str(args.proposals_json),
        "legacy_annotations_dir": str(args.legacy_annotations_dir),
        "legacy_splits": splits,
        "selection_rule": args.selection_rule,
    }
    report["warning"] = (
        "Both mask sets are UNREVIEWED machine output. There is no mask ground "
        "truth here; every figure is a proxy against the human box."
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_objects_jsonl:
        args.out_objects_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_objects_jsonl.open("w", encoding="utf-8") as handle:
            for obj in result["objects"]:
                handle.write(json.dumps(obj, sort_keys=True) + "\n")

    print(f"Wrote {args.out_json}")
    print(f"  selection_rule: {report['selection_rule']}")
    for key, value in report["counters"].items():
        print(f"  {key}: {value}")
    print(f"  overall n_objects: {report['overall']['n_objects']}")
    for metric in ("new_box_iou", "old_box_iou", "iou_new_old"):
        stats = report["overall"][metric]
        if stats.get("n"):
            print(f"  {metric}: median={stats['median']} p10={stats['p10']} mean={stats['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
