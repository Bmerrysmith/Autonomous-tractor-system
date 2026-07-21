# Archive: v2 — WeedDet Initial Colab Runs

**Status:** Superseded (contained critical Bug 1 in early runs).<br>
**Platform:** Google Colab<br>
**Dates:** April 7–21, 2026

## What This Was
First implementation of the custom WeedDet architecture (weeddet_for_VSCode.py) in Colab. Multiple runs during this phase. Early runs were completely broken due to Bug 1.

## Critical Bug 1 Discovered Here
**The most costly bug in the project.**

- Symptom: loss → 0.0000 from epoch 2. Training ran to completion with no errors.
- Cause: VOC XML bounding box coordinates in original pixel space, NOT rescaled after image resize to 1000×600.
- Effect: IoU(ground_truth, anchors) = 0 → zero positive anchors → zero gradient → silent failure.
- Fix: Added `boxes *= tW/orig_w, tH/orig_h` in `WeedDataset.__getitem__`.
- How to verify: `assert max(boxes[:,[0,2]]) <= 1000 and max(boxes[:,[1,3]]) <= 600`

## Also: Bug 2 (Double-Scaling)
- `FixedWeedDataset` wrapper applied the coordinate scaling a second time on top of already-fixed `WeedDataset`
- Effect: coordinates scaled by (scale_factor)² instead of scale_factor
- Fix: `FixedWeedDataset` entirely removed

## Drive Files
- `weeddet_training_fixed.ipynb` (private Drive artifact; ID intentionally omitted)
- `Weed_Det_Training_v2.ipynb` (private Drive artifact; ID intentionally omitted)
- `weeddet_checkpoints/` (private Drive artifact; ID intentionally omitted) — early checkpoints
- `weeddet_v2_checkpoints/` (private Drive artifact; ID intentionally omitted) — later checkpoints
