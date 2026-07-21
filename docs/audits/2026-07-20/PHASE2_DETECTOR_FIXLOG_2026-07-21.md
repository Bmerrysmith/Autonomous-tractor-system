# Phase 2 Fix Log — Detector Correctness (`models/weeddet_v6b.py`)

**Date:** 2026-07-21
**Scope:** the detector half of roadmap **Gate 4** ("Fix code correctness before training").
**Basis:** July 20 deep audit §5–6 and the Gate 4 required-code-changes table.
**Method:** test-first per `active/ACTIVE_NOTES.md`. New suite: `tests/test_weeddet_v6b_fixes.py` (9 tests). Full suite **77 tests green**.

---

## Fixed

| Audit ID | Defect | Fix | Proof |
|---|---|---|---|
| **P0-1** | Translation augmentation moved image content and boxes in **opposite** directions. Pillow's affine tuple maps output→input, so `(1,0,dx,0,1,dy)` shifts content **left** for positive `dx` while boxes were shifted **right**. Systematically corrupted ~half of augmented samples (and invalidated T7d). | New module-level `translate_image_and_boxes()` negates the offsets in the affine tuple so content and boxes both move `(+dx, +dy)`. | `test_positive/negative_translation_moves_content_and_box_together` — rendered content bbox and target box agree within **1 px** on every edge, in both directions |
| **P1-10** | Box keep-masks were not carried through to `labels`; `_augment` never even received them, and a prefix fallback (`labels[:keep.sum()]`) could attach the **wrong class** to a surviving box. | `_augment` and `translate_image_and_boxes` now take and return `labels`, filtered by the same mask. Prefix fallback deleted. | `test_labels_follow_boxes_when_one_is_clipped_away` — the surviving object keeps its **original** label, not a prefix value |
| **P1-8** | `letterbox_pil` returned one nominal scale while the integer resize produced slightly different actual x/y scales; forward and inverse disagreed by up to ~1 px. | Returns exact `scale_x, scale_y`; `unpad_boxes` and both dataset call sites use them per-axis. | `test_roundtrip_within_one_pixel_on_non_square_image`, `test_scales_match_actual_resized_dimensions` |
| **P1-3** | Forced-positive assignment used advanced indexing (`pos[gt_best_anchor] = True`), so two GTs choosing the same best anchor silently **orphaned** one of them. | New `_force_one_positive_per_gt()`: strongest GT claims a contested anchor, the loser takes its next-best *unclaimed* anchor. Applied to **both** the ATSS path and the fixed-threshold fallback (which previously had no guarantee at all), plus a post-condition assert. | `test_colliding_best_anchor_does_not_orphan_a_gt` |
| **P1-2** | ATSS top-k ran over anchors, but all 12 shapes at a cell share one centre — so top-9 could return 9 shapes at **one** cell instead of 9 nearby locations, destroying the spatial neighbourhood ATSS needs for its mean+std IoU threshold. | Select top-k distinct **cells** per level (shapes-per-cell inferred from shared centres), then take every shape at those cells. | `test_atss_candidates_span_distinct_cells`, `test_anchor_shape_permutation_preserves_cell_selection` |
| **P1-1** | `VariFocalLoss` does not implement published VFL/IACS (it uses `target*|target-p|^gamma` with hard 1.0 positive targets), so confidence is not trained to rank by localisation quality. | Renamed to **`HardTargetFocalLikeLoss`** with an honest docstring; `VariFocalLoss` kept as a backwards-compatible alias. | truthful naming; real VFL remains a separate future ablation |
| **P1-9** | `load_imagenet_backbone` returned `(0, 0)` on a fetch failure, so an "ImageNet" run could silently be a scratch run. | Raises `RuntimeError` — fails closed. | code path now unrepresentable |

Verified end-to-end: training-mode forward produces finite `cls_loss`/`reg_loss`/`total_loss` (including an **empty-GT** image), eval-mode returns detections.

---

## Still open in Gate 4 (not addressed here)

- **P1-5** Python-loop NMS → replace with `torchvision.ops.batched_nms` (latency).
- **P1-4** 258,048 anchors/image (memory + pre-NMS candidate pressure).
- **P1-6** three conflicting postprocessing protocols → one canonical decode/postprocess shared by val/test/inference/export.
- **P1-7** Smooth L1 + CIoU combined without measured balance.
- **Evaluator**: standard COCO AP@[.50:.95] maxDets=100 as the primary metric (custom maxDets 300 needs a separate name).
- **P0-2/P0-3** inference runtime — remains quarantined in `_archive/unsafe_inference/`.

These do not block the *correctness* of training geometry and assignment, but they do block a defensible published result.

> **Note:** T7d and all earlier detector runs remain **invalid as baselines** — they trained with the P0-1 corrupted transform on a capture-series-contaminated split. Re-run on the new grouped split (`data/manifests/detector_split_v1.json`) after the remaining Gate 4 items.
