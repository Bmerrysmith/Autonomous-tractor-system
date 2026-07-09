# Archive: v3 — WeedDet Extended Training (94 Epochs)

**Status:** Superseded. Best checkpoint still valid for reference.  
**Platform:** Google Colab  
**Dates:** April 23–28, 2026

## What This Was
Extended training run after Bug 1 and Bug 2 fixes. Ran for 94 epochs on the merged dataset (~1,596 images with real per-plant boxes). Produced the best quantitative result prior to v5.

## Architecture
- Script: `weeddet_for_VSCode.py`
- `anchor_base_scale=4` (NOTE: v4/v5 use 6 — checkpoint NOT compatible)
- Input: 1000×600
- All other params per Peng et al. 2022

## Training Config
- Epochs: 94
- Dataset: merged COCO datasets (~5,544 total, ~1,596 with real per-plant boxes, rest filtered)
- Annotation filtering: placeholder boxes (≥95% image area) excluded
- 2× dataset repeat per epoch

## Results
- **Best val_loss: 0.7437** (epoch 60 and epoch 40+ on extended run with 4k dataset added)
- Qualitative: confidence scores 0.55–0.84 on paddy, aerial, and post-flood imagery

## Drive Checkpoints
- `weeddet_v3_best.pth` (Drive ID: 1L5pdF3_Crd4Nw5byjIXWMAqqNZZtIP3-)
- `weeddet_v3_epoch94.pth` (Drive ID: 1c4rAkuo9KIC3PLdA3lztHztyoHqdmP08)

## Other Files Created
- `rice_detection_demo.mp4` — demo video (Drive ID: 1hFilZGUM7MrvN13kmSNpKnpsE7DGlC_q)
- `AgriNav_Final_Report.pptx` — final report presentation
- `AgriNav_Progress_Report.pptx` — progress presentation

## Why Superseded
- `anchor_base_scale=4` vs v4/v5 `anchor_base_scale=6` — checkpoints incompatible
- mAP evaluation showed near-0% due to cross-dataset domain gap (NOT a code bug)
- `weeddet_Latest.py` is the updated script for v4/v5
