# Benchmark Comparison Plan

**Status:** deferred by intent. **Opened:** 2026-08-25.

## The intent

Once AgriNav has favorable results, compare them against published rice/weed
detection papers to measure how the system actually stands. This file holds the
rules that comparison must follow and the candidate papers as they accumulate,
so the comparison is assembled from a standing protocol rather than from
whichever numbers happen to look good at the time.

**Gate: no published comparison goes in the manuscript until**

1. a same-protocol baseline has been run (`agrinav baseline-detector`) — an AP
   with nothing to be measured against is not a result. See `docs/baselines.md`;
2. the operating threshold is frozen on validation and the sealed test split has
   been evaluated exactly once;
3. every number in the table carries its IoU threshold and its split.

## Why this file exists — the GE-YOLO case

Comparing AgriNav's phase-2 val AP of 0.0924 against GE-YOLO's "93.1% mAP"
suggests a 10x gap. Almost none of that gap is model quality:

- **93.1% is mAP@0.5.** The paper states no IoU threshold. The comparable
  AgriNav number is AP50 = 0.3665, not AP = 0.0924.
- **~2.8 objects per image** in their data against **33.2** in ours.
- **3 weed species, rice as background.** We also detect rice, which is 87.4%
  of our annotations.
- **Stock YOLOv8 already scores 91.2%** on their data; their architecture
  contributes +1.9 points.

A comparison that reports "0.0924 vs 0.931" is not wrong about the arithmetic
and is wrong about everything else.

## Rules for any row added to the comparison table

1. **Metric with threshold, always.** `AP50`, `AP75`, `AP@[.50:.95]` — never a
   bare "mAP". If a paper does not state its threshold, record the inferred one
   and mark it inferred.
2. **State the split the number came from** (test / val / unclear) and whether
   the test split was used for tuning.
3. **Record task density and object scale** — objects per image and median box
   area — because these dominate cross-paper AP differences.
4. **Record the class set.** Weed-species discrimination and rice-vs-weed
   detection are different tasks.
5. **Record initialization** (COCO-detection / ImageNet / scratch / other) and
   the training budget.
6. **Note split hygiene**: was augmentation applied before or after the split;
   are video-frame families or augmented siblings shared across splits. Mark
   unverifiable when the paper does not say.
7. **Report our own numbers under the same disclosure**, including the ones
   that make us look worse. Our grouped, leakage-checked splits cost us AP
   relative to papers that split randomly, and that trade is part of the result.

## Candidates

| paper | reported | metric | task | notes |
|---|---:|---|---|---|
| [chen2025geyolo] GE-YOLO, Appl. Sci. 15(5):2823 | 93.1% | mAP@0.5 (inferred) | 3 weed species, ~2.8 obj/img | YOLOv8 baseline 91.2%; augmentation/split order unstated |
| _(add as found)_ | | | | |

Also cited inside that paper, worth retrieving as further candidates:
GTCBS-YOLOv5s (six rice-field weed species, 91.1% mAP), YOLO-EP
(*Pomacea canaliculata* on rice), Cotton-YOLO.

## AgriNav numbers as they stand (2026-08-25) — provisional, validation split

| | AP@[.50:.95] | AP50 | AP75 | AR100 | AP rice | AP weed |
|---|---:|---:|---:|---:|---:|---:|
| WeedDet, RiceSEG init, 60 ep, lr 0.003 | 0.0924 | 0.3665 | 0.0131 | 0.2273 | 0.1528 | 0.0321 |
| Faster R-CNN, COCO init | _run in progress_ | | | | | |

**Validation split, not test.** Not publishable as a headline. The test split
has never been scored and stays that way until the gate above is met.
