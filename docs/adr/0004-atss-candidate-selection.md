# ADR 0004: Select ATSS candidates as top-k cells, then the best-IoU shape

## Status

Accepted — 2026-08-28

## Context

WeedDet reached AP 0.0833 on the phase-2 validation split against 0.1693 for a
stock torchvision Faster R-CNN with matched ImageNet initialisation — a 2.03x
gap. The shape of that gap was more informative than its size:

| metric | WeedDet | Faster R-CNN (ImageNet) |
|---|---|---|
| AR@100 | 0.2296 | 0.2944 |
| AP50 | 0.3277 | 0.5460 |
| AP75 | 0.0112 | 0.0557 |
| AP75 / AP50 | **0.034** | **0.102** |
| detections / image | 99.7 | 73.4 |

Recall was roughly intact while localisation quality was three times worse at the
same recall, and the detector was saturating COCO's `maxDets=100` cap. The
AP75/AP50 ratio is also initialisation-invariant — 0.1036 for the COCO-init
baseline, 0.1020 for the ImageNet-init one — so the cause was structural rather
than a matter of pretraining or training budget.

`WeedDetLoss._assign_atss` built its ATSS candidate pool by selecting the top-k
nearest feature-map **cells** and then expanding each selected cell to **every
co-located anchor shape**. With 12 shapes per location and k=9 over three levels,
the `mean + std` threshold that ATSS uses to decide positives was computed over
roughly 324 candidates spanning every aspect ratio and scale, rather than over a
neighbourhood of comparable anchors. The statistic collapsed and the positive set
filled with mediocre anchors.

A second, compounding defect sat underneath it. The number of co-located shapes
was derived at runtime by comparing anchor centres for float equality. Centres
are computed as `(x1 + x2) * 0.5`, which is not exact in float32, so 2 of the 12
shapes lost their centre to 1-ULP round-off and the count came back **10** at
every level. The selector then strode a period-12 array in steps of 10, silently
selecting windows that straddled two grid cells at a rotating shape phase.

Measured on 2034 real ground-truth boxes from the phase-2 rebuild train split,
letterboxed to 512 exactly as training does:

| candidate rule | pos/GT | mean assigned IoU | ≥0.75 | ≥0.5 |
|---|---|---|---|---|
| shipped (`legacy`) | 45.71 | 0.5006 | 0.0127 | 0.4211 |
| literal mmdet (`priors`) | 4.92 | 0.5990 | 0.0799 | 0.8469 |
| **`cells_best_shape`** | **4.58** | **0.6376** | **0.1096** | **0.9754** |
| `priors_iou_tiebreak` | 4.92 | 0.6244 | 0.1006 | 0.9288 |
| best anchor available | — | 0.7346 | 0.4258 | 0.9946 |

The decisive column is the last one: the shipped assigner trained **58% of its
positives at IoU < 0.5** — below the threshold AP50 itself uses — while anchors
averaging 0.73 went unselected. Fixing only the float bug makes this worse
(45.7 → 53.9 positives per object), because it restores the full 12-shape
expansion the design intended.

## Decision

Replace the candidate build with **`cells_best_shape`**: for each ground-truth
box and each FPN level, take the top-k nearest grid cells, then admit the single
highest-IoU anchor shape at each of those cells. The number of shapes per
location is taken structurally from `AnchorGenerator.num_shapes`
(`len(aspect_ratios) * len(scales)`) and never re-derived from anchor geometry.

Add `atss_candidate_mode` to `WeedDetLoss` and `WeedDet` with four values:
`cells_best_shape` (default), `priors`, `priors_iou_tiebreak`, and `legacy`.

## Alternatives considered

**Literal MMDetection ATSS (`priors`).** The reference implementation selects
top-k priors per level by centre distance over every prior in that level. This is
the obvious port and it is a large improvement over the shipped behaviour, but
MMDetection's ATSS assumes **one prior per location**. With 12 co-located shapes
sharing an identical centre, "top-k nearest priors" is decided by tie-breaking on
index order rather than by geometry — the selection becomes shape-order-sensitive
and the spatial neighbourhood the `mean + std` statistic is supposed to summarise
is lost, because all k candidates can come from a single cell. It scores worse
than `cells_best_shape` on every column measured.

**Distance ties broken by IoU (`priors_iou_tiebreak`).** Repairs the tie-breaking
without restoring the neighbourhood. Lands between the two, as expected.

**Fixing only the `per_cell` float derivation.** Rejected: measured as a
regression (45.7 → 53.9 positives per object). The float bug was masking part of
the design defect, not causing it.

**Reducing the anchor set to one shape per location.** Would make the literal
MMDetection semantics correct by construction, but discards the tall-object
aspect ratios (0.2, 0.33, 0.5) that this dataset needs — ground truth is
predominantly tall, with a median box of 20.8 x 41.9 px.

## Consequences

- Positives per object fall roughly 9x. Both classification and regression losses
  are normalised by `num_pos`, so the normalised loss and its gradients rise by
  about the same factor. The default `grad_clip` was raised from 0.5 to 100.0 in
  the same change set; at 0.5 the overfit gate's 2-epoch loss *rises*
  (5.4257 → 5.6939) and at 100.0 it falls (5.4257 → 5.4100).
- `base_lr` is deliberately **unchanged** at 0.003. Total gradient norm into the
  head rises only ~1.33x at the initialisation operating point, against roughly
  3.3x of measured headroom. Read `grad_norm/p99` and `train/cls_loss_neg` on
  epoch 1 before changing it.
- The effective weight of the classification loss's negative term returns to its
  designed value. With `atss_all_neg=True` the negative term runs over all
  anchor x class entries divided by `num_pos`, so inflating `num_pos` had been
  running `alpha = 0.75` at an effective ~0.087. **Do not also change `alpha`** —
  that would flip two factors at once.
- No checkpoint compatibility break: assignment is a training-time decision and
  no parameter shapes change.
- `legacy` is retained so the table above can be regenerated, and the
  pre-fix runs remain explicable.
- Not yet validated on GPU. Every number here is a static measurement of
  assignment quality on real ground truth. Whether it moves AP is an open
  question that one 60-epoch run answers.

## References

- `src/agrinav/models/weeddet_v6b.py` — `AnchorGenerator.num_shapes`,
  `WeedDetLoss._assign_atss`
- `tests/test_atss_assignment.py` — behavioural guards for the properties above
- MMDetection `ATSSAssigner`, `mmdet/models/task_modules/assigners/atss_assigner.py`
- Zhang et al., *Bridging the Gap Between Anchor-based and Anchor-free Detection
  via Adaptive Training Sample Selection*, CVPR 2020 (arXiv:1912.02424)
- ADR 0003 — one canonical detection postprocessor
