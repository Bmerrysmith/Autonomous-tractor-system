# AgriNav — Weed Detection & Plant Identification Subsystem

**Author:** Benjamin Merryman-Smith | Florida Gulf Coast University  
**Project:** AgriNav Autonomous Tractor  
**Papers:** Peng et al. 2022 (WeedDet) · Jiang et al. 2020 (GCN Weed Recognition)

---

## What This Is

Weed detection subsystem for an autonomous paddy-field tractor.  
**Detection logic is inverted:** the model finds rice plants (protected class).  
Anything not inside a high-confidence rice box = spray target.  
A hardcoded rice veto in the confidence gate ensures rice is never sprayed.

---

## Repository Structure

```
agrinav-weed-detection/
├── models/
│   └── weeddet_for_VSCode.py     ← Complete WeedDet model (train + infer)
├── data/
│   ├── step1_extract.py          ← Inspect and extract dataset zip
│   ├── auto_annotate.py          ← Grounding DINO auto-annotation
│   ├── coco_to_voc.py            ← Convert COCO JSON → PASCAL VOC XML
│   └── step2_split.py            ← Validate annotations, generate train/val split
├── training/
│   └── colab_weeddet_train.py    ← Full Colab training cell (includes FixedWeedDataset)
├── inference/
│   └── inference_rice.py         ← Run inference + green/red visualization
└── notebooks/
    └── rice_detection_fixed.ipynb ← Kaggle baseline RetinaNet notebook
```

---

## Quick Start

### Option A — Run on Kaggle (baseline RetinaNet)
1. Upload `rice_detection_fixed.ipynb` to Kaggle
2. Add datasets: `bennymerryman/rice-detection` (both COCO datasets)
3. Run all cells — trains 12 epochs, saves `rice_retinanet.pth`

### Option B — Train WeedDet on Colab
1. Upload `merged_dataset.zip` to `MyDrive/`
2. Upload `weeddet_for_VSCode.py` to `/content/`
3. Paste `training/colab_weeddet_train.py` into a Colab cell and run

### Option C — Run locally (RTX 4060 or better)
```bash
pip install torch torchvision pillow tqdm transformers

# Auto-annotate images with Grounding DINO
python data/auto_annotate.py --images ./images --output ./annotations

# Convert COCO JSON to VOC XML
python data/coco_to_voc.py --json merged.json --images ./images --output ./voc_dataset

# Generate train/val split
python data/step2_split.py --root ./voc_dataset

# Train
python models/weeddet_for_VSCode.py  # runs build check
# Then call train() or train_with_progress() from weeddet_for_VSCode

# Inference
python inference/inference_rice.py \
    --model ./checkpoints/weeddet_best.pth \
    --image ./test_field.jpg
```

---

## Critical Bugs (both FIXED)

### Bug 1 — Box coordinate space mismatch (loss → 0.0000)

**Symptom:** Loss = 0.0000 from epoch 2. Training runs normally but learns nothing.

**Cause:** VOC XML bounding box coordinates are in original image pixel space.  
WeedDet resizes images to (600, 1000). Boxes were NOT being rescaled to match.  
Result: IoU(ground_truth, anchors) = 0 → no positive anchors → no gradient.

**Fix:** `WeedDataset.__getitem__` in `weeddet_for_VSCode.py` now applies:
```python
boxes[:, [0,2]] *= tW / orig_w   # scale x-coords to target width
boxes[:, [1,3]] *= tH / orig_h   # scale y-coords to target height
```

### Bug 2 — Double-scaling in FixedWeedDataset (boxes shrink incorrectly)

**Symptom:** Detections appear but are systematically mis-sized.

**Cause:** `FixedWeedDataset` in `colab_weeddet_train.py` applied bounding box
coordinate scaling a SECOND time on top of the fix already in `WeedDataset`.
Boxes ended up scaled by (scale_factor)² — shrinking dramatically for large images.

**Fix:** `FixedWeedDataset` and its monkey-patch have been REMOVED.
`colab_weeddet_train.py` now includes a runtime sanity check that verifies
box coordinates are within [0, 1000] × [0, 600] after loading.

---

## Datasets

| Dataset | Images | Source | Used For |
|---|---|---|---|
| rice-hainan | ~1,347 | Roboflow / Kaggle | WeedDet training |
| rice_detection_for_export | ~250 | Roboflow / Kaggle | WeedDet training |
| Severity labeled (4 classes) | ~3,947 | Various | GCN only (filtered from WeedDet) |

Merged COCO: ~5,544 images total. ~1,596 have real per-plant boxes.  
Full-image placeholder boxes from severity datasets are filtered by `coco_to_voc.py`.

---

## Architecture (WeedDet)

```
Input: 1000×600 RGB
  ↓
Det-ResNet-50 (modified stem: two 3×3 convs + DetResidualBlock)
  ↓  C3 (512ch), C4 (1024ch), C5 (2048ch)
eFPN — produces only P3/P4/P5 (no P6/P7, saves 7.71M params)
  ↓  256ch per level
ERetina-Head — 1 conv (64ch) + Large Separable Conv (k=7)
  ↓
WeedDetLoss: SmoothL1 + GIoU (regression) + VariFocal (classification)
Anchors: base_scale=6, 3 aspect ratios × 3 scales = 9 per location
```

Training: SGD lr=0.01, batch=2, 12 epochs, warmup 500 iters,  
          MultiStepLR decay at [8, 11], freeze BatchNorm, grad clip 1.0

---

## Papers

1. Peng et al. (2022) — "Weed Detection in Paddy Field Using an Improved RetinaNet Network"  
   *Computers and Electronics in Agriculture*, vol. 199, p. 107179.

2. Jiang et al. (2020) — "CNN Feature Based Graph Convolutional Network for Weed and Crop Recognition in Smart Farming"  
   *Computers and Electronics in Agriculture*, vol. 174, p. 105450.
