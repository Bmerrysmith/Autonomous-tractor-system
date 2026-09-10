#!/usr/bin/env python3
"""Measure which SAM2.1 candidate configuration best recovers the human box.

WHY
---
``sam_box_to_mask`` emits 8 candidates per box: 3 from ``multimask_output=True``
on the exact human box, and 5 from ``multimask_output=False`` on deterministic
+/-3% box jitters. The set therefore never contains the one configuration SAM's
own documentation recommends for an *unambiguous* prompt such as a box:
``multimask_output=False`` on the **unjittered** box. ``multimask_output=True``
exists to resolve ambiguity (whole object / part / subpart) and is aimed at
single-point prompts; picking among its three heads by ``argmax(pred_iou)`` is a
different, weaker rule than asking the model for its single best answer.

That gap is worth a number rather than an argument, because it decides whether a
full re-seed produces better geometry than the polygons that already exist or
worse. This probe answers exactly that and nothing else.

WHAT IT IS NOT
--------------
It emits no proposals, no candidates, no annotation records and no masks to
disk - only summary statistics. It cannot mint truth because it writes no
geometry at all. It is a measurement, and its output is only ever an input to a
human decision about how to configure the real stage.

CLI::

    python -m agrinav.data.sam_selection_probe \\
        --coco-zip artifacts/sam_reseed/pilot_images.zip \\
        --proposals-json artifacts/sam_reseed/pilot_proposals_unreviewed.coco.json \\
        --model-revision 665f8e2ad61cf5f53d65644ff27c8ee525124610 \\
        --out-json reports/metrics/sam_selection_probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agrinav.data.compare_sam_polygons import box_iou, mask_bbox, summarise, xywh_to_xyxy
from agrinav.data.sam2_predictor import (
    DEFAULT_DTYPE,
    PredictorCounters,
    build_predictor,
    resolve_pinned_revision,
)
from agrinav.data.sam_box_to_mask import (
    SamPreflightError,
    clip_box,
    index_proposals,
    preflight_image,
)
from agrinav.data.sam_reseed_pilot import assert_no_sealed_split


def _stats_for_mask(mask: np.ndarray, human_box: Sequence[float]) -> dict[str, float]:
    area = float(np.count_nonzero(mask))
    if area == 0.0:
        return {"box_iou": 0.0, "fill": 0.0, "containment": 0.0, "empty": 1.0}
    height, width = mask.shape
    x0 = max(0, int(np.floor(human_box[0])))
    y0 = max(0, int(np.floor(human_box[1])))
    x1 = min(width, int(np.ceil(human_box[2])))
    y1 = min(height, int(np.ceil(human_box[3])))
    box_mask = np.zeros_like(mask)
    if x1 > x0 and y1 > y0:
        box_mask[y0:y1, x0:x1] = True
    box_area = float(np.count_nonzero(box_mask))
    bbox = mask_bbox(mask)
    return {
        "box_iou": box_iou(bbox, human_box) if bbox else 0.0,
        "fill": area / box_area if box_area else 0.0,
        "containment": float(np.count_nonzero(mask & box_mask)) / area,
        "empty": 0.0,
    }


def probe(
    *,
    coco_zip: Path,
    proposals: dict[str, Any],
    model_id: str,
    revision: str,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    """Score ``multimask_output=False`` and ``=True`` on the exact human box."""
    assert_no_sealed_split(proposals)
    pin = resolve_pinned_revision(model_id, revision)
    units, drops = index_proposals(proposals)
    if drops:
        raise SamPreflightError(f"unmapped/orphan annotations {drops}; refusing to probe")

    counters = PredictorCounters()
    predictor = build_predictor(
        model_id=model_id,
        revision=pin["requested_revision"],
        device=device,
        dtype=dtype,
        counters=counters,
        verify_revision=False,
    )

    collected: dict[str, dict[str, list[float]]] = {
        rule: {"box_iou": [], "fill": [], "containment": [], "empty": []}
        for rule in ("single_mask_exact_box", "multimask_argmax_pred_iou")
    }
    per_label: dict[str, dict[str, list[float]]] = {}
    boxes_scored = 0
    started = time.perf_counter()

    with zipfile.ZipFile(coco_zip, "r") as archive:
        for unit in units:
            image = preflight_image(unit, archive.read(unit["file_name"]))
            width, height = int(unit["width"]), int(unit["height"])
            predictor.set_image(image)
            for box in unit["boxes"]:
                clipped = clip_box(box["bbox"], width, height)
                if clipped is None:
                    continue
                human_box = xywh_to_xyxy(clipped)
                prompt = np.asarray(human_box, dtype=np.float32)

                single, _, _ = predictor.predict(box=prompt, multimask_output=False)
                multi, scores, _ = predictor.predict(box=prompt, multimask_output=True)
                chosen = multi[int(np.argmax(np.asarray(scores).reshape(-1)))]

                for rule, mask in (
                    ("single_mask_exact_box", single[0]),
                    ("multimask_argmax_pred_iou", chosen),
                ):
                    stats = _stats_for_mask(np.asarray(mask).astype(bool), human_box)
                    for key, value in stats.items():
                        collected[rule][key].append(value)
                    bucket = per_label.setdefault(f"{box['label']}|{rule}", {"box_iou": []})
                    bucket["box_iou"].append(stats["box_iou"])
                boxes_scored += 1

    wall = time.perf_counter() - started
    paired = [
        a - b
        for a, b in zip(
            collected["single_mask_exact_box"]["box_iou"],
            collected["multimask_argmax_pred_iou"]["box_iou"],
        )
    ]
    return {
        "model_pin": pin,
        "dtype": dtype,
        "boxes_scored": boxes_scored,
        "images": len(units),
        "wall_seconds": round(wall, 3),
        "predictor_counters": counters.as_dict(),
        "by_rule": {
            rule: {
                metric: summarise(values) for metric, values in metrics.items() if metric != "empty"
            }
            | {"empty_mask_rate": round(statistics.fmean(metrics["empty"]), 5)}
            for rule, metrics in collected.items()
        },
        "by_label_box_iou": {
            key: summarise(value["box_iou"]) for key, value in sorted(per_label.items())
        },
        "single_minus_multimask_box_iou": summarise(paired),
        "fraction_single_better": (
            round(sum(1 for d in paired if d > 0) / len(paired), 4) if paired else None
        ),
        "note": (
            "Measurement only. No mask, candidate or annotation record is written. "
            "Both rules operate on the same human box; neither asserts truth."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe SAM2.1 candidate configurations")
    ap.add_argument("--coco-zip", required=True, type=Path)
    ap.add_argument("--proposals-json", required=True, type=Path)
    ap.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=DEFAULT_DTYPE)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args(argv)

    try:
        report = probe(
            coco_zip=args.coco_zip,
            proposals=json.loads(args.proposals_json.read_text(encoding="utf-8")),
            model_id=args.model_id,
            revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
        )
    except SamPreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    print(f"  boxes_scored: {report['boxes_scored']}  wall_seconds: {report['wall_seconds']}")
    for rule, metrics in report["by_rule"].items():
        print(
            f"  {rule}: box_iou median={metrics['box_iou']['median']} "
            f"containment median={metrics['containment']['median']} "
            f"fill median={metrics['fill']['median']}"
        )
    print(f"  single-minus-multimask box_iou: {report['single_minus_multimask_box_iou']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
