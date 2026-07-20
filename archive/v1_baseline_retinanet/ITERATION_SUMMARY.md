# Archive: v1 — Kaggle Baseline RetinaNet

**Status:** Superseded. Do not use for paper results.<br>
**Platform:** Kaggle<br>
**Date:** Early April 2026

## What This Was
Proof-of-concept using torchvision's pretrained RetinaNet ResNet-50 FPN — NOT the custom WeedDet architecture. Used to validate the rice detection concept before building WeedDet from scratch.

## Architecture
- `torchvision.models.detection.retinanet_resnet50_fpn`
- Backbone: pretrained ImageNet weights
- Detection heads: random init, trained from scratch
- num_classes=2 (0=background, 1=rice)

## Training Config
- Epochs: 12, SGD lr=0.001, momentum=0.9, weight_decay=1e-4
- StepLR decay ×0.1 after epoch 9, grad clip max_norm=1.0, batch 2

## Datasets
- rice-hainan.coco (~1,347 images)
- rice_detection_for_export.coco
- 80/20 split, seed 42

## Why Superseded
- Not the WeedDet architecture from Peng et al. 2022
- No formal mAP numbers kept
- Replaced by custom WeedDet implementation

## Key Lessons
- Kaggle `/kaggle/input/` dirs are read-only — write to `/kaggle/working/`
- `category_id=0` is background in RetinaNet — use `category_id=1` for rice
- Use an approved device-flow or narrowly scoped credential for GitHub access; never paste tokens into notebooks.

## GitHub File
`notebooks/rice_detection_fixed.ipynb`
