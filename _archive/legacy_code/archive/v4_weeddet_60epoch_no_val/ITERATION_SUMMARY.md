# Archive: v4 — WeedDet 60-Epoch Run (No Val Split)

**Status:** Superseded. Training complete but evaluation broken — replaced by v5.<br>
**Platform:** Google Colab (Tesla T4)<br>
**Dates:** May 29 – June 1, 2026

## What This Was
Retrain with `weeddet_Latest.py` and `anchor_base_scale=6`. 60 epochs on `rice_detection_for_export.v1i.voc`. First use of the updated script. Training completed but had three post-training bugs.

## Training Config
```
Script:      weeddet_Latest.py
Dataset:     rice_detection_for_export.v1i.voc (1,347 images)
Val split:   NONE — val.txt was empty (0 pairs)
Epochs:      60
anchor_base_scale: 6
IMG_SIZE:    512
LR:          0.001 cosine → 0.00005
grad_clip:   0.5
freeze_bn:   True
```

## Loss Progression
| Epoch | Avg Loss |
|---|---|
| 1 | 1.0448 |
| 10 | 0.8476 |
| 20 | 0.8123 |
| 30 | 0.7903 |
| 60 | TBD (best checkpoint saved, full run completed) |

## Checkpoint
- `weeddet_best.pth` in `MyDrive/weeddet_v4_checkpoints/`
- Private Drive folder; ID intentionally omitted

## Bugs Found Post-Training (all 3 fixed in v5)

### Bug 1: Checkpoint naming mismatch
- Training saved: `weeddet_best.pth`
- Visual check cell looked for: `weeddet_v4_best.pth`
- Fix: `V4_CKPT = f'{CKPT_DIR}/weeddet_best.pth'`

### Bug 2: Wrong module import in visual check
- Training used: `import weeddet_Latest as wd`
- Cell 4 used: `import weeddet_for_VSCode as wd`
- Fix: Change to `import weeddet_Latest as wd`

### Bug 3: Empty val split
- Dataset had all images in `train/`, no `valid/` folder with matching XMLs
- Result: val.txt empty, no val loss curve generated
- Fix: Build explicit 80/20 split before training (done in v5 Cell 1)

## Why Superseded
All three bugs above. Replaced by v5 which fixes them all.

## Notebook
`weeddet_trainingV4.ipynb` (in this folder)
