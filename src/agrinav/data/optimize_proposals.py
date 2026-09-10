#!/usr/bin/env python3
"""Select ONE SAM candidate per human box and emit CVAT-importable COCO polygons.

WHY THIS FILE EXISTS
====================
``sam_box_to_mask`` deliberately emits every candidate it computed and selects
none of them, recording ``candidate_selection:
"deferred_to_optimize_proposals"``. Three modules then assume this stage exists:
``triage_proposals`` (which wants ``proposal_features_v1.jsonl``),
``compare_sam_polygons`` (which owns the *comparison-only* selection vocabulary)
and the raw sidecar itself. Nothing wrote it, so a raw shard could not reach
CVAT at all: CVAT imports COCO 1.0 polygon JSON, and the shard holds RLE
candidate sets. This stage is the missing hop, and it needs no GPU.

Two outputs, both unreviewed:

``--out-coco``      COCO polygon JSON for CVAT import. One annotation per input
                    box, always -- an object whose mask could not be selected is
                    emitted with ``segmentation: []`` and keeps the human box,
                    because deleting the object would delete a human assertion.
``--out-features``  ``proposal_features_v1.jsonl``: one pixel-free feature row
                    per object, in the shape ``triage_proposals.parse_feature_row``
                    accepts.

NOTHING HERE IS TRUTH
=====================
Selecting a machine mask is not reviewing it. Every emitted annotation carries
``review_status`` ``"unreviewed_proposal"`` (the COCO proposal layer's
vocabulary) and every feature row carries ``"unreviewed"`` (the
``annotation_record.v1`` enum member). No code path here writes
``verified_empty``, ``treatment_eligible``, ``annotator_id`` or ``reviewer_id``,
and ``test_no_truth_marker_reaches_either_output`` asserts the emitted bytes
contain none of them.

Labels are copied from the source box through ``LABEL_TO_CATEGORY_ID``, a CLOSED
dict with no ``else`` and no ``.get(..., default)``. An unmapped label is a
COUNTED DROP that aborts the run unless ``--allow-unmapped-labels`` is passed --
the same discipline (and the same failure it guards against) as
``sam_box_to_mask.COCO_CATEGORY_TO_LABEL``.

THE SELECTION RULE, AND THE MEASUREMENT BEHIND IT
=================================================
Default ``--selection-rule single_mask_exact_box``: the ``multimask_output=False``
pass on the *unjittered* human box (``kind="single_mask"``).

Measured on 1,238 paired objects of the re-pilot shard
(``reports/summaries/sam_reseed_repilot_2026-09-01.md``): it recovers the human
box better than ``argmax(sam_pred_iou)`` over the multimask heads on **68.4% of
objects**, at median box IoU **0.774** against 0.759 for the legacy polygons a
re-seed would replace. A box is an unambiguous prompt, which is the case the
single-mask head is trained for; the multimask head exists to resolve
whole/part/subpart ambiguity that a box has already resolved.

CANDIDATE COUNT IS READ, NEVER ASSUMED
======================================
``thresholds.candidates_per_box`` is read from each shard. Shards written before
the single-mask candidate existed carry 8 candidates and NO ``single_mask``
kind. For those rows ``select_for_row`` returns ``None`` and the object is
counted under ``rule_kind_not_in_shard`` -- it is never quietly served a
different candidate kind, because a silent substitution would report
single-mask geometry that is actually multimask-argmax geometry, and the two
were measured 6.4 points of box IoU apart. A shard that cannot serve the
requested rule at all aborts the run unless ``--allow-rule-unavailable`` is
passed. A shard carrying no ``candidates_per_box`` is a hard error: there is no
safe default to guess.

FRAGMENT POLICY IS EXPLICIT, NOT SILENTLY ALWAYS-ON
===================================================
16.2% of ``single_mask`` masks have >= 2 connected components. Of 945 extra
components, **850 (89.9%) are < 1% of their mask area** and the median mask
loses **0.57%** of its area to a largest-component-only rule -- free cleanup.
But p90 is 22.1% and the max 72.0%: for the **27 of 1,238** objects with >= 5%
of area outside the largest component, a blanket largest-component rule deletes
real structure (plausibly a second leaf blade of the same weed). So the policy
is a named, defaulted, counted option:

``drop_small_components`` (DEFAULT)  drop each connected component smaller than
                          ``--fragment-min-component-fraction`` (default 0.01)
                          of the mask area. Cleans the median object and
                          PRESERVES the tail: a 22% component survives.
``keep_all``              drop nothing. Every component becomes a ring.
``largest_component``     keep only the largest, always. Deletes the tail's
                          structure; available, but not the default.

Every dropped component is counted per object (``components_dropped``,
``dropped_area_fraction``) and in the manifest.

KNOWN ROUND-TRIP LIMIT (counted, not hidden)
============================================
``cvat_export_to_records._polygon_ring`` raises on a multi-ring polygon: "the
wire format holds one ring per object". Any object this stage emits with more
than one ring therefore has to be split into separate shapes in CVAT before it
can come back. The alternative -- emitting one annotation per component -- would
invent objects a human never drew, and the other alternative -- keeping only the
largest -- deletes geometry. Emitting the rings and COUNTING them
(``objects_multi_ring``) is the only option that neither invents nor deletes.
Holes cannot be expressed by a single COCO ring either; interior contours are
dropped and counted as ``objects_with_holes_flattened``.

CLI::

    python -m agrinav.data.optimize_proposals \\
        --raw-shard artifacts/detector_v1/sam_raw/shard_000.jsonl \\
        --out-coco artifacts/detector_v1/cvat_import_unreviewed.coco.json \\
        --out-features artifacts/detector_v1/proposal_features_v1.jsonl \\
        --out-manifest artifacts/detector_v1/optimize_proposals_manifest.json

Deterministic: two runs over one shard set produce byte-identical outputs.
Needs no GPU, no checkpoint and no image files -- only the shard.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from agrinav.data.compare_sam_polygons import SELECTION_RULES, mask_bbox, select_candidate
from agrinav.data.sam_box_to_mask import RAW_SCHEMA_VERSION

FEATURES_SCHEMA_VERSION = "agrinav.proposal_features.v1"
MANIFEST_SCHEMA_VERSION = "agrinav.optimize_proposals_manifest.v1"

#: Ontology label -> COCO category id. CLOSED dict, mirroring (and inverting)
#: ``sam_box_to_mask.COCO_CATEGORY_TO_LABEL``. No ``else``, no default: an
#: unmapped label is counted and dropped, never renamed into a known class.
LABEL_TO_CATEGORY_ID: dict[str, int] = {"rice_protect": 1, "weed_target": 2}

RICE_LABEL = "rice_protect"
WEED_LABEL = "weed_target"

#: Emitted geometry vocabulary, matching ``cvat_export_to_records``.
GEOMETRY_POLYGON = "polygon"
GEOMETRY_BBOX = "bbox"

#: The review-status vocabularies are DIFFERENT and must not be crossed. The
#: COCO proposal layer uses ``unreviewed_proposal`` (see
#: ``coco_boxes_to_proposals``); ``annotation_record.v1`` uses ``unreviewed``.
COCO_REVIEW_STATUS = "unreviewed_proposal"
RECORD_REVIEW_STATUS = "unreviewed"

DEFAULT_SELECTION_RULE = "single_mask_exact_box"

FRAGMENT_POLICY_KEEP_ALL = "keep_all"
FRAGMENT_POLICY_DROP_SMALL = "drop_small_components"
FRAGMENT_POLICY_LARGEST = "largest_component"
FRAGMENT_POLICIES: dict[str, str] = {
    FRAGMENT_POLICY_DROP_SMALL: (
        "drop every connected component below --fragment-min-component-fraction of "
        "the mask area; keeps substantial extra structure"
    ),
    FRAGMENT_POLICY_KEEP_ALL: "drop nothing; every component becomes a polygon ring",
    FRAGMENT_POLICY_LARGEST: "keep only the largest component; deletes the measured ~2% tail",
}
DEFAULT_FRAGMENT_POLICY = FRAGMENT_POLICY_DROP_SMALL
#: 89.9% of measured extra components sit below this fraction of their mask area.
DEFAULT_MIN_COMPONENT_FRACTION = 0.01

#: Skip reasons. A skipped object still reaches both outputs, demoted to bbox.
SKIP_DEGENERATE_BOX = "degenerate_box_no_candidates"
SKIP_RULE_KIND_NOT_IN_SHARD = "rule_kind_not_in_shard"
SKIP_RULE_KIND_MISSING_ON_OBJECT = "rule_kind_missing_on_object"
SKIP_EMPTY_MASK = "selected_mask_empty"
SKIP_NO_POLYGON = "mask_yielded_no_polygon_ring"
SKIP_REASONS: tuple[str, ...] = (
    SKIP_DEGENERATE_BOX,
    SKIP_RULE_KIND_NOT_IN_SHARD,
    SKIP_RULE_KIND_MISSING_ON_OBJECT,
    SKIP_EMPTY_MASK,
    SKIP_NO_POLYGON,
)

#: A COCO polygon ring needs at least three points.
MIN_RING_POINTS = 3


class OptimizeProposalsError(Exception):
    """Raised when the inputs cannot be turned into proposals safely. Fatal."""


# --------------------------------------------------------------------------- #
# Shard reading
# --------------------------------------------------------------------------- #
def read_shard_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read raw candidate rows from one or more shards.

    Raises:
        OptimizeProposalsError: on unreadable JSON, an unexpected
            ``schema_version``, or a duplicate ``key`` across the inputs. A
            duplicate is never resolved by keeping one row: two rows for one
            object may disagree, and picking one silently picks a geometry.
    """
    rows: list[dict[str, Any]] = []
    origin: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise OptimizeProposalsError(f"raw shard not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                where = f"{path}:{line_number}"
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise OptimizeProposalsError(f"{where}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise OptimizeProposalsError(f"{where}: expected a JSON object")
                version = row.get("schema_version")
                if version != RAW_SCHEMA_VERSION:
                    raise OptimizeProposalsError(
                        f"{where}: schema_version {version!r} is not {RAW_SCHEMA_VERSION!r}; "
                        "this stage reads sam_box_to_mask raw candidate shards only"
                    )
                key = str(row.get("key") or "")
                if not key:
                    raise OptimizeProposalsError(f"{where}: row has no 'key'")
                if key in origin:
                    raise OptimizeProposalsError(
                        f"{where}: duplicate object key {key!r}, first seen at {origin[key]}. "
                        "Overlapping shards must be de-duplicated before selection: two rows "
                        "for one object may carry different geometry."
                    )
                origin[key] = where
                row["_source_shard"] = str(path)
                rows.append(row)
    if not rows:
        raise OptimizeProposalsError(
            f"no candidate rows found in {[str(p) for p in paths]}; nothing to select from"
        )
    return rows


def candidates_per_box_of(row: Mapping[str, Any]) -> int:
    """Read ``provenance.thresholds.candidates_per_box``. Never assume 8 or 9.

    Raises:
        OptimizeProposalsError: when the shard does not state its own candidate
            count. Guessing it is exactly the failure this stage exists to
            avoid: 8-candidate shards have no ``single_mask`` candidate and
            9-candidate shards do.
    """
    provenance = row.get("provenance")
    thresholds = provenance.get("thresholds") if isinstance(provenance, Mapping) else None
    declared = thresholds.get("candidates_per_box") if isinstance(thresholds, Mapping) else None
    if not isinstance(declared, int) or isinstance(declared, bool) or declared < 0:
        raise OptimizeProposalsError(
            f"object {row.get('key')!r} from {row.get('_source_shard')!r} carries no integer "
            "provenance.thresholds.candidates_per_box. Refusing to assume a candidate count: "
            "an 8-candidate shard has no 'single_mask' candidate and a 9-candidate one does, "
            "and substituting the wrong kind changes the geometry that reaches a reviewer."
        )
    return declared


def declared_candidate_kinds(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """``thresholds.candidate_kinds`` if the shard states it, else ``None``."""
    provenance = row.get("provenance")
    thresholds = provenance.get("thresholds") if isinstance(provenance, Mapping) else None
    kinds = thresholds.get("candidate_kinds") if isinstance(thresholds, Mapping) else None
    if isinstance(kinds, list) and all(isinstance(kind, str) for kind in kinds):
        return tuple(kinds)
    return None


def row_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    provenance = row.get("provenance")
    original = provenance.get("original_proposal") if isinstance(provenance, Mapping) else None
    candidates = original.get("candidates") if isinstance(original, Mapping) else None
    if not isinstance(candidates, list):
        return []
    return [c for c in candidates if isinstance(c, dict)]


#: Selection rule -> the candidate ``kind`` it draws from. CLOSED, and it is the
#: only place the two vocabularies meet.
RULE_TO_KIND: dict[str, str] = {
    "single_mask_exact_box": "single_mask",
    "multimask_argmax_pred_iou": "multimask",
}


def select_for_row(row: Mapping[str, Any], rule: str) -> tuple[dict[str, Any] | None, str | None]:
    """Select one candidate for one object, or return ``(None, skip_reason)``.

    Dispatch is delegated to :func:`compare_sam_polygons.select_candidate`, whose
    closed dispatch raises on an unknown rule rather than falling back.

    Returns:
        ``(candidate, None)`` on success, else ``(None, reason)`` where reason is
        one of :data:`SKIP_REASONS`. There is NO branch that returns a candidate
        of a kind the rule did not ask for.
    """
    candidates = row_candidates(row)
    if not candidates:
        return None, SKIP_DEGENERATE_BOX
    chosen = select_candidate(candidates, rule)
    if chosen is not None:
        return chosen, None
    wanted = RULE_TO_KIND[rule]
    declared = declared_candidate_kinds(row)
    kind_is_expected = wanted in declared if declared is not None else False
    if kind_is_expected:
        return None, SKIP_RULE_KIND_MISSING_ON_OBJECT
    return None, SKIP_RULE_KIND_NOT_IN_SHARD


# --------------------------------------------------------------------------- #
# Mask handling
# --------------------------------------------------------------------------- #
def decode_rle(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode a COCO RLE dict to a boolean HxW mask."""
    from pycocotools import mask as maskutils

    counts = rle["counts"]
    payload = {
        "size": [int(v) for v in rle["size"]],
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }
    return np.asarray(maskutils.decode(payload)).astype(bool)


@dataclass(frozen=True)
class FragmentResult:
    """What the fragment policy did to one mask. Every field is reported."""

    mask: np.ndarray
    components_total: int
    components_kept: int
    components_dropped: int
    dropped_area_fraction: float

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "components_total": self.components_total,
            "components_kept": self.components_kept,
            "components_dropped": self.components_dropped,
            "dropped_area_fraction": round(self.dropped_area_fraction, 6),
        }


def apply_fragment_policy(
    mask: np.ndarray, policy: str, min_component_fraction: float
) -> FragmentResult:
    """Apply the named fragment policy to one mask.

    Args:
        mask: boolean HxW mask.
        policy: a key of :data:`FRAGMENT_POLICIES`. Unknown policies raise --
            there is no default branch, because a mistyped policy silently
            falling back to "keep everything" would misreport what was emitted.
        min_component_fraction: used by ``drop_small_components`` only. A
            component is dropped when its area is strictly below this fraction
            of the mask's total area.

    Returns:
        A :class:`FragmentResult` carrying the surviving mask and the counts.
    """
    import cv2

    if policy not in FRAGMENT_POLICIES:
        raise OptimizeProposalsError(
            f"unknown fragment policy {policy!r}; choose one of {sorted(FRAGMENT_POLICIES)}"
        )
    arr = np.asarray(mask).astype(np.uint8)
    total_area = int(np.count_nonzero(arr))
    if total_area == 0:
        return FragmentResult(arr.astype(bool), 0, 0, 0, 0.0)

    label_count, labels = cv2.connectedComponents(arr, connectivity=8)
    areas = {index: int(np.count_nonzero(labels == index)) for index in range(1, label_count)}
    components_total = len(areas)
    if policy == FRAGMENT_POLICY_KEEP_ALL or components_total <= 1:
        return FragmentResult(arr.astype(bool), components_total, components_total, 0, 0.0)

    if policy == FRAGMENT_POLICY_LARGEST:
        largest = max(areas, key=lambda index: (areas[index], -index))
        keep = {largest}
    else:
        threshold = min_component_fraction * total_area
        keep = {index for index, area in areas.items() if area >= threshold}
        if not keep:
            # Every component is below threshold (possible for a mask split into
            # many equal specks). Keeping the largest is the least-destructive
            # outcome that still returns geometry; it is counted like any drop.
            keep = {max(areas, key=lambda index: (areas[index], -index))}

    kept_mask = np.isin(labels, sorted(keep))
    dropped_area = total_area - int(np.count_nonzero(kept_mask))
    return FragmentResult(
        mask=kept_mask,
        components_total=components_total,
        components_kept=len(keep),
        components_dropped=components_total - len(keep),
        dropped_area_fraction=dropped_area / total_area,
    )


def mask_to_polygons(mask: np.ndarray) -> tuple[list[list[int]], int]:
    """Convert a boolean mask to COCO polygon rings.

    Returns:
        ``(rings, holes_dropped)``. Only exterior contours become rings: a COCO
        polygon ring cannot express a hole, so interior contours are dropped and
        counted rather than silently filled without a record. Rings with fewer
        than :data:`MIN_RING_POINTS` points are dropped (they enclose no area).
    """
    import cv2

    arr = np.asarray(mask).astype(np.uint8)
    contours, hierarchy = cv2.findContours(arr, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings: list[list[int]] = []
    holes = 0
    for index, contour in enumerate(contours):
        is_hole = hierarchy is not None and int(hierarchy[0][index][3]) >= 0
        if is_hole:
            holes += 1
            continue
        points = contour.reshape(-1, 2)
        if points.shape[0] < MIN_RING_POINTS:
            continue
        rings.append([int(value) for point in points for value in point])
    return rings, holes


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _iou(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape:
        return None
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return None
    return int(np.count_nonzero(a & b)) / union


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _clipped_box_xyxy(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """The human box after ``sam_box_to_mask``'s clip, as xyxy."""
    clipped = row.get("prompt_box_clipped")
    if not isinstance(clipped, list) or len(clipped) != 4:
        return None
    x, y, w, h = (float(v) for v in clipped)
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


def multimask_margin(candidates: Sequence[Mapping[str, Any]]) -> float | None:
    """Top-1 minus top-2 ``sam_pred_iou`` across the multimask heads.

    ``None`` when fewer than two multimask candidates carry a score -- a missing
    required feature fails open to ``human_review`` in ``triage_proposals``,
    which is the safe direction; a zero would read as "the heads agreed".
    """
    scores = sorted(
        (
            float(c["sam_pred_iou"])
            for c in candidates
            if c.get("kind") == "multimask" and c.get("sam_pred_iou") is not None
        ),
        reverse=True,
    )
    if len(scores) < 2:
        return None
    return scores[0] - scores[1]


def jitter_agreement(candidates: Sequence[Mapping[str, Any]], selected: np.ndarray) -> float | None:
    """Mean IoU of the jitter masks against the selected mask, or ``None``."""
    values: list[float] = []
    for candidate in candidates:
        if candidate.get("kind") != "jitter":
            continue
        rle = candidate.get("rle")
        if not isinstance(rle, dict):
            continue
        value = _iou(decode_rle(rle), selected)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


@dataclass
class ObjectOutcome:
    """One object's fully-resolved selection outcome. Ordering key included."""

    row: dict[str, Any]
    label: str
    category_id: int
    selection_rule: str
    selected: dict[str, Any] | None
    skip_reason: str | None
    rings: list[list[int]] = field(default_factory=list)
    mask: np.ndarray | None = None
    fragment: FragmentResult | None = None
    holes_dropped: int = 0
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def geometry_type(self) -> str:
        return GEOMETRY_POLYGON if self.rings else GEOMETRY_BBOX

    @property
    def order_key(self) -> tuple[str, int, str]:
        return (
            str(self.row.get("source_image_sha256") or ""),
            int(self.row.get("coco_annotation_id") or 0),
            str(self.row.get("source_object_id") or ""),
        )

    def selection_record(self, fragment_policy: str) -> dict[str, Any]:
        """What was chosen and by which rule. Recorded on every object.

        ``selection_rule`` stays populated even when nothing was selected: the
        run still applied that rule, and a reader has to be able to tell "this
        rule found nothing" from "no rule was applied".
        """
        selected = self.selected or {}
        return {
            "selection_rule": self.selection_rule,
            "candidate_index": selected.get("candidate_index"),
            "candidate_kind": selected.get("kind"),
            "skip_reason": self.skip_reason,
            "fragment_policy": fragment_policy if self.fragment is not None else None,
        }


def _mask_fraction_in_box(mask: np.ndarray, box: Sequence[float] | None) -> float | None:
    """Fraction of a mask's pixels that fall inside an axis-aligned xyxy box."""
    if box is None:
        return None
    area = int(np.count_nonzero(mask))
    if area == 0:
        return None
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = int(np.ceil(box[2])), int(np.ceil(box[3]))
    return int(np.count_nonzero(mask[y0:y1, x0:x1])) / area


def _geometry_features(
    outcome: ObjectOutcome, candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Family-A geometry features, all derived from the selected mask alone."""
    features: dict[str, Any] = {}
    box = _clipped_box_xyxy(outcome.row)
    mask = outcome.mask
    if mask is None or box is None:
        return features
    area_px = int(np.count_nonzero(mask))
    features["area_px"] = area_px
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    features["fill"] = area_px / box_area if box_area > 0 else None
    containment = _mask_fraction_in_box(mask, box)
    if containment is not None:
        features["containment"] = containment
        features["leak"] = 1.0 - containment
        bounds = mask_bbox(mask)
        features["box_iou"] = _box_iou(bounds, box) if bounds is not None else None
    features["multimask_margin"] = multimask_margin(candidates)
    features["jitter_iou_mean"] = jitter_agreement(candidates, mask)
    selected = outcome.selected or {}
    features["sam_pred_iou"] = selected.get("sam_pred_iou")
    return features


def _same_class_neighbour(
    outcome: ObjectOutcome, others: Sequence[ObjectOutcome]
) -> tuple[float | None, float | None, str | None]:
    """Worst same-class crowding: ``(mask IoU, box IoU, partner id)``.

    ``triage_proposals`` flags both members of a collision and deletes neither,
    so this only has to identify the pair, not rank it.
    """
    mask, box = outcome.mask, _clipped_box_xyxy(outcome.row)
    mask_best: float | None = None
    box_best: float | None = None
    partner: str | None = None
    for other in others:
        if other.label != outcome.label:
            continue
        if mask is not None and other.mask is not None:
            value = _iou(mask, other.mask)
            if value is not None and (mask_best is None or value > mask_best):
                mask_best = value
                partner = str(other.row.get("source_object_id") or "")
        other_box = _clipped_box_xyxy(other.row)
        if box is not None and other_box is not None:
            value = _box_iou(box, other_box)
            if box_best is None or value > box_best:
                box_best = value
    return mask_best, box_best, partner


def _cross_class_overlap(outcome: ObjectOutcome, others: Sequence[ObjectOutcome]) -> float | None:
    """Largest fraction of this mask covered by an other-class mask."""
    mask = outcome.mask
    if mask is None:
        return None
    area = int(np.count_nonzero(mask))
    if area == 0:
        return None
    best: float | None = None
    for other in others:
        if other.label == outcome.label or other.mask is None:
            continue
        if other.mask.shape != mask.shape:
            continue
        overlap = int(np.count_nonzero(mask & other.mask)) / area
        if best is None or overlap > best:
            best = overlap
    return best


def _rice_over_weed(outcome: ObjectOutcome, others: Sequence[ObjectOutcome]) -> float | None:
    """Largest fraction of a rice mask sitting inside a weed box; ``None`` if not rice.

    ``triage_proposals`` defines ``rice_over_weed_box_ioa`` only by its name and
    its use (above a threshold, a rice object escalates to T0 cross-class
    intrusion). This is the reading implemented here, and it is stated rather
    than assumed: rice mask pixels inside a human weed box are the case where a
    protected object has been drawn over a spray target.
    """
    if outcome.label != RICE_LABEL or outcome.mask is None:
        return None
    best: float | None = None
    for other in others:
        if other.label != WEED_LABEL:
            continue
        overlap = _mask_fraction_in_box(outcome.mask, _clipped_box_xyxy(other.row))
        if overlap is not None and (best is None or overlap > best):
            best = overlap
    return best


def _neighbour_features(outcomes: Sequence[ObjectOutcome]) -> None:
    """Fill the per-image neighbour features in place. O(n^2) over one image."""
    for outcome in outcomes:
        if outcome.mask is None:
            continue
        others = [other for other in outcomes if other is not outcome]
        mask_iou, box_iou_value, partner = _same_class_neighbour(outcome, others)
        outcome.features["same_class_mask_iou_max"] = mask_iou
        outcome.features["same_class_box_iou_max"] = box_iou_value
        outcome.features["cross_class_ioa_max"] = _cross_class_overlap(outcome, others)
        outcome.features["rice_over_weed_box_ioa"] = _rice_over_weed(outcome, others)
        if partner:
            outcome.features["same_class_neighbor_annotation_id"] = partner


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Options:
    """Resolved run options. Every one of these is recorded in the manifest."""

    selection_rule: str = DEFAULT_SELECTION_RULE
    fragment_policy: str = DEFAULT_FRAGMENT_POLICY
    fragment_min_component_fraction: float = DEFAULT_MIN_COMPONENT_FRACTION
    allow_unmapped_labels: bool = False
    allow_rule_unavailable: bool = False


def _round(value: Any, places: int = 6) -> Any:
    """Round floats for byte-stable output; pass everything else through."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, places)
    return value


def _resolve_object(
    row: dict[str, Any], options: Options
) -> tuple[ObjectOutcome | None, str | None]:
    """Select, clean and polygonise one object. ``(None, label)`` on label drop."""
    label = str(row.get("label") or "")
    category_id = LABEL_TO_CATEGORY_ID.get(label)
    if category_id is None:
        return None, label
    declared = candidates_per_box_of(row)
    available = len(row_candidates(row))
    if available > declared:
        # Fewer than declared is legal (a jitter pass can be skipped). More means
        # the shard's own provenance disagrees with its payload, and every count
        # this stage reports downstream is derived from that provenance.
        raise OptimizeProposalsError(
            f"object {row.get('key')!r} carries {available} candidates but its shard declares "
            f"candidates_per_box={declared}. The sidecar's provenance does not describe its "
            "own contents; refusing to select from it."
        )
    selected, skip_reason = select_for_row(row, options.selection_rule)
    outcome = ObjectOutcome(
        row=row,
        label=label,
        category_id=category_id,
        selection_rule=options.selection_rule,
        selected=selected,
        skip_reason=skip_reason,
    )
    if selected is None:
        return outcome, None
    rle = selected.get("rle")
    if not isinstance(rle, dict):
        outcome.skip_reason = SKIP_EMPTY_MASK
        outcome.selected = None
        return outcome, None
    raw_mask = decode_rle(rle)
    if not raw_mask.any():
        outcome.skip_reason = SKIP_EMPTY_MASK
        return outcome, None
    fragment = apply_fragment_policy(
        raw_mask, options.fragment_policy, options.fragment_min_component_fraction
    )
    rings, holes = mask_to_polygons(fragment.mask)
    if not rings:
        outcome.skip_reason = SKIP_NO_POLYGON
        return outcome, None
    outcome.mask = fragment.mask
    outcome.fragment = fragment
    outcome.rings = rings
    outcome.holes_dropped = holes
    return outcome, None


def _coco_annotation(outcome: ObjectOutcome, options: Options) -> dict[str, Any]:
    """One COCO annotation. Bbox-only when no mask survived selection."""
    row = outcome.row
    human_box = [float(v) for v in (row.get("prompt_box_clipped") or [])]
    annotation: dict[str, Any] = {
        "id": int(row["coco_annotation_id"]),
        "image_id": int(row["coco_image_id"]),
        "category_id": outcome.category_id,
        "iscrowd": 0,
        "annotation_origin": "sam_box_to_mask_candidate_selected",
        "review_status": COCO_REVIEW_STATUS,
        "human_box_xywh": human_box,
        "agrinav_selection": outcome.selection_record(options.fragment_policy),
    }
    if outcome.rings and outcome.mask is not None:
        bounds = mask_bbox(outcome.mask)
        assert bounds is not None  # a non-empty ring implies a non-empty mask
        annotation["segmentation"] = [list(ring) for ring in outcome.rings]
        annotation["bbox"] = [
            int(bounds[0]),
            int(bounds[1]),
            int(bounds[2] - bounds[0]),
            int(bounds[3] - bounds[1]),
        ]
        annotation["area"] = int(np.count_nonzero(outcome.mask))
    else:
        annotation["segmentation"] = []
        annotation["bbox"] = human_box
        annotation["area"] = float(human_box[2] * human_box[3]) if len(human_box) == 4 else 0.0
    return annotation


def _feature_row(
    outcome: ObjectOutcome, options: Options, weed_box_count_in: int
) -> dict[str, Any]:
    """One ``proposal_features_v1`` row, in the shape ``triage_proposals`` parses.

    ``reason_codes`` is NOT a "please look at this" channel:
    ``triage_proposals.rule_mask_demoted`` sends any object with a non-empty
    ``reason_codes`` straight to ``auto_reject``, which DISCARDS the mask. So it
    carries only genuine mask failures (no candidate, empty mask, no ring) --
    exactly the objects that have no mask to keep. Fragmentation facts travel as
    ordinary feature values instead, so a fragmented-but-real mask is not thrown
    away by a note about it.
    """
    row = outcome.row
    reason_codes = [outcome.skip_reason] if outcome.skip_reason else []
    payload: dict[str, Any] = {
        "schema_version": FEATURES_SCHEMA_VERSION,
        "record_id": f"img:{row.get('source_image_sha256')}",
        "annotation_id": str(row.get("source_object_id") or ""),
        "source_object_id": row.get("source_object_id"),
        "label": outcome.label,
        "group_id": row.get("group_id") or "",
        "capture_family": row.get("capture_family") or "",
        "source_image_sha256": row.get("source_image_sha256") or "",
        "geometry_type": outcome.geometry_type,
        "reason_codes": reason_codes,
        "review_status": RECORD_REVIEW_STATUS,
        "weed_box_count_in": weed_box_count_in,
        "coco_image_id": row.get("coco_image_id"),
        "coco_annotation_id": row.get("coco_annotation_id"),
        "file_name": row.get("file_name"),
        "candidates_available": len(row_candidates(row)),
        "candidates_per_box_declared": candidates_per_box_of(row),
        "selection": outcome.selection_record(options.fragment_policy),
        "features": {name: _round(value) for name, value in outcome.features.items()},
    }
    if outcome.fragment is not None:
        payload["fragment"] = outcome.fragment.summary
        payload["polygon_rings"] = len(outcome.rings)
        payload["holes_dropped"] = outcome.holes_dropped
    return payload


def optimize(rows: Sequence[dict[str, Any]], options: Options) -> dict[str, Any]:
    """Select one candidate per object and build both outputs plus a manifest.

    Args:
        rows: raw candidate rows, as returned by :func:`read_shard_rows`.
        options: resolved run options.

    Returns:
        ``{"coco": ..., "features": [...], "manifest": ...}``. Ordering is fully
        determined by ``(source_image_sha256, coco_annotation_id,
        source_object_id)``, so two runs over one shard set are byte-identical.

    Raises:
        OptimizeProposalsError: on an unmapped label (unless acknowledged), or
            when the shard cannot serve the requested selection rule at all.
    """
    if options.selection_rule not in SELECTION_RULES:
        raise OptimizeProposalsError(
            f"unknown selection rule {options.selection_rule!r}; "
            f"choose one of {sorted(SELECTION_RULES)}"
        )
    if options.fragment_policy not in FRAGMENT_POLICIES:
        raise OptimizeProposalsError(
            f"unknown fragment policy {options.fragment_policy!r}; "
            f"choose one of {sorted(FRAGMENT_POLICIES)}"
        )

    label_drops: dict[str, int] = {}
    outcomes: list[ObjectOutcome] = []
    weed_in: dict[str, int] = {}
    for row in rows:
        sha = str(row.get("source_image_sha256") or "")
        if str(row.get("label") or "") == WEED_LABEL:
            weed_in[sha] = weed_in.get(sha, 0) + 1
        outcome, dropped_label = _resolve_object(row, options)
        if outcome is None:
            key = f"label={dropped_label!r}"
            label_drops[key] = label_drops.get(key, 0) + 1
            continue
        outcomes.append(outcome)

    if label_drops and not options.allow_unmapped_labels:
        raise OptimizeProposalsError(
            f"unmapped source labels {label_drops}; these boxes would be dropped from the "
            "CVAT import entirely. Extend LABEL_TO_CATEGORY_ID, or pass "
            "--allow-unmapped-labels to acknowledge the loss (it stays counted either way)."
        )

    outcomes.sort(key=lambda outcome: outcome.order_key)
    by_image: dict[str, list[ObjectOutcome]] = {}
    for outcome in outcomes:
        by_image.setdefault(str(outcome.row.get("source_image_sha256") or ""), []).append(outcome)
    for image_outcomes in by_image.values():
        for outcome in image_outcomes:
            outcome.features.update(_geometry_features(outcome, row_candidates(outcome.row)))
        _neighbour_features(image_outcomes)

    skip_counts: dict[str, int] = {reason: 0 for reason in SKIP_REASONS}
    for outcome in outcomes:
        if outcome.skip_reason:
            skip_counts[outcome.skip_reason] += 1
    if skip_counts[SKIP_RULE_KIND_NOT_IN_SHARD] and not options.allow_rule_unavailable:
        raise OptimizeProposalsError(
            f"{skip_counts[SKIP_RULE_KIND_NOT_IN_SHARD]} object(s) come from a shard that "
            f"carries no {RULE_TO_KIND[options.selection_rule]!r} candidate, so "
            f"--selection-rule {options.selection_rule!r} cannot be served for them. They were "
            "NOT served a different candidate kind. Re-run with --selection-rule "
            "multimask_argmax_pred_iou for pre-single_mask shards, or pass "
            "--allow-rule-unavailable to emit them as bbox-only proposals."
        )

    features = [
        _feature_row(
            outcome, options, weed_in.get(str(outcome.row.get("source_image_sha256") or ""), 0)
        )
        for outcome in outcomes
    ]
    coco = _build_coco(outcomes, options)
    manifest = _build_manifest(rows, outcomes, options, skip_counts, label_drops)
    return {"coco": coco, "features": features, "manifest": manifest}


def _build_coco(outcomes: Sequence[ObjectOutcome], options: Options) -> dict[str, Any]:
    images: dict[int, dict[str, Any]] = {}
    annotations: list[dict[str, Any]] = []
    for outcome in outcomes:
        row = outcome.row
        image_id = int(row["coco_image_id"])
        if image_id not in images:
            images[image_id] = {
                "id": image_id,
                "file_name": row.get("file_name"),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "sha256": row.get("source_image_sha256"),
                "group_id": row.get("group_id"),
                "capture_family": row.get("capture_family"),
                "review_status": COCO_REVIEW_STATUS,
            }
        annotations.append(_coco_annotation(outcome, options))
    return {
        "info": {
            "description": ("SAM candidate selection over UNREVIEWED human boxes, for CVAT import"),
            "WARNING": (
                "NOT training truth and NOT reviewed. Every polygon is machine geometry "
                "selected by a fixed rule over a human box; every annotation is "
                f"review_status={COCO_REVIEW_STATUS!r}. A human must approve each mask in "
                "CVAT before any of it becomes truth."
            ),
            "selection_rule": options.selection_rule,
            "selection_rule_description": SELECTION_RULES[options.selection_rule],
            "fragment_policy": options.fragment_policy,
            "fragment_policy_description": FRAGMENT_POLICIES[options.fragment_policy],
            "fragment_min_component_fraction": options.fragment_min_component_fraction,
        },
        "categories": [
            {"id": category_id, "name": name}
            for name, category_id in sorted(LABEL_TO_CATEGORY_ID.items(), key=lambda kv: kv[1])
        ],
        "images": [images[key] for key in sorted(images)],
        "annotations": annotations,
    }


def _build_manifest(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[ObjectOutcome],
    options: Options,
    skip_counts: Mapping[str, int],
    label_drops: Mapping[str, int],
) -> dict[str, Any]:
    with_polygon = [o for o in outcomes if o.rings]
    multi_ring = sum(1 for o in with_polygon if len(o.rings) > 1)
    four_point = sum(1 for o in with_polygon if len(o.rings) == 1 and len(o.rings[0]) == 8)
    holes = sum(o.holes_dropped for o in with_polygon)
    dropped_components = sum(
        o.fragment.components_dropped for o in with_polygon if o.fragment is not None
    )
    fragmented = sum(
        1 for o in with_polygon if o.fragment is not None and o.fragment.components_total > 1
    )
    declared = sorted({candidates_per_box_of(row) for row in rows})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "options": {
            "selection_rule": options.selection_rule,
            "fragment_policy": options.fragment_policy,
            "fragment_min_component_fraction": options.fragment_min_component_fraction,
            "allow_unmapped_labels": options.allow_unmapped_labels,
            "allow_rule_unavailable": options.allow_rule_unavailable,
        },
        "shards": sorted({str(row.get("_source_shard")) for row in rows}),
        "candidates_per_box_declared": declared,
        "counts": {
            "objects_in": len(rows),
            "objects_emitted": len(outcomes),
            "objects_with_polygon": len(with_polygon),
            "objects_demoted_to_bbox": len(outcomes) - len(with_polygon),
            "objects_dropped_unmapped_label": sum(label_drops.values()),
            "images": len({str(o.row.get("source_image_sha256")) for o in outcomes}),
            "by_label": {
                label: sum(1 for o in outcomes if o.label == label)
                for label in sorted(LABEL_TO_CATEGORY_ID)
            },
        },
        "skips": dict(sorted(skip_counts.items())),
        "label_drops": dict(sorted(label_drops.items())),
        "fragmentation": {
            "objects_fragmented_before_policy": fragmented,
            "components_dropped": dropped_components,
            "objects_multi_ring": multi_ring,
            "objects_with_holes_flattened": holes,
            "objects_single_four_point_ring": four_point,
        },
        "notes": [
            "Nothing in either output is reviewed; no truth marker is written.",
            "A multi-ring object must be split into separate shapes in CVAT: "
            "cvat_export_to_records rejects a multi-ring polygon on the way back.",
        ],
    }


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    _guard_existing(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool) -> None:
    _guard_existing(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _guard_existing(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise OptimizeProposalsError(f"{path} exists; pass --overwrite to replace it")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--raw-shard",
        required=True,
        action="append",
        type=Path,
        help="sam_box_to_mask raw candidate JSONL; repeat for several shards",
    )
    parser.add_argument("--out-coco", required=True, type=Path, help="COCO polygon JSON for CVAT")
    parser.add_argument(
        "--out-features", required=True, type=Path, help="proposal_features_v1.jsonl for triage"
    )
    parser.add_argument("--out-manifest", type=Path, default=None, help="run summary JSON")
    parser.add_argument(
        "--selection-rule",
        default=DEFAULT_SELECTION_RULE,
        choices=sorted(SELECTION_RULES),
        help=f"candidate selection rule (default: {DEFAULT_SELECTION_RULE})",
    )
    parser.add_argument(
        "--fragment-policy",
        default=DEFAULT_FRAGMENT_POLICY,
        choices=sorted(FRAGMENT_POLICIES),
        help=f"connected-component policy (default: {DEFAULT_FRAGMENT_POLICY})",
    )
    parser.add_argument(
        "--fragment-min-component-fraction",
        type=float,
        default=DEFAULT_MIN_COMPONENT_FRACTION,
        help=(
            "drop_small_components threshold as a fraction of mask area "
            f"(default: {DEFAULT_MIN_COMPONENT_FRACTION})"
        ),
    )
    parser.add_argument(
        "--allow-unmapped-labels",
        action="store_true",
        help="acknowledge and continue past source labels with no ontology category",
    )
    parser.add_argument(
        "--allow-rule-unavailable",
        action="store_true",
        help="emit bbox-only proposals for shards that cannot serve the selection rule",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    options = Options(
        selection_rule=args.selection_rule,
        fragment_policy=args.fragment_policy,
        fragment_min_component_fraction=args.fragment_min_component_fraction,
        allow_unmapped_labels=args.allow_unmapped_labels,
        allow_rule_unavailable=args.allow_rule_unavailable,
    )
    outputs = [args.out_coco, args.out_features]
    if args.out_manifest is not None:
        outputs.append(args.out_manifest)
    try:
        # Guard every destination BEFORE the work: refusing to clobber after a
        # full pass costs the whole pass, and leaves some outputs written.
        for destination in outputs:
            _guard_existing(destination, overwrite=args.overwrite)
        rows = read_shard_rows(args.raw_shard)
        result = optimize(rows, options)
        _write_json(args.out_coco, result["coco"], overwrite=args.overwrite)
        _write_jsonl(args.out_features, result["features"], overwrite=args.overwrite)
        if args.out_manifest is not None:
            _write_json(args.out_manifest, result["manifest"], overwrite=args.overwrite)
    except OptimizeProposalsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    manifest = result["manifest"]
    print("Wrote UNREVIEWED proposals (machine geometry over human boxes, NOT truth)")
    print(f"  coco     : {args.out_coco}")
    print(f"  features : {args.out_features}")
    for key, value in manifest["counts"].items():
        print(f"  {key}: {value}")
    for key, value in manifest["fragmentation"].items():
        print(f"  {key}: {value}")
    skipped = {key: value for key, value in manifest["skips"].items() if value}
    if skipped:
        print(f"  demoted to bbox: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
