# WeedDet Latest + V5 Training Optimization Report

## Executive Summary

This report converts the WeedDet deep-dive into an export-ready Markdown plan for improving rice-plant detection mAP, especially AP@0.75 and AP@0.50:0.95.

The current model has a coherent RetinaNet-style design, but its performance is likely being limited by a combination of training-loop bugs, postprocessing choices, small-object anchor mismatch, weak validation protocol, and insufficient modern detection features.

The most important fixes are:

1. Use `loss_dict["total_loss"]` instead of `sum(loss_dict.values())`.
2. Stop mixing `test` data into the train/validation split.
3. Replace wraparound translation augmentation.
4. Restore the intended 2× dataset repeat per epoch.
5. Evaluate mAP with low score threshold, higher max detections, and NMS sweeps.
6. Tune anchors using actual rice-box size statistics.
7. Consider P2 features or sliced/tiled training if rice plants are small in image space.
8. Benchmark against a pretrained detector such as YOLOv8, VFNet, GFL, RT-DETR, or MMDetection RetinaNet.

---

## 1. Current Model Summary

`weeddet_Latest.py` implements a custom single-class rice detector based on WeedDet. The file describes a RetinaNet-style architecture with:

| Area | Current Design |
|---|---|
| Input | 512×512 letterboxed RGB image |
| Backbone | Det-ResNet-50 |
| Neck | eFPN with P3, P4, P5 |
| Head | ERetinaHead with large separable convolution |
| Anchors | 9 anchors/location, base scale 6, strides 8/16/32 |
| Classification loss | VariFocal Loss |
| Regression loss | SmoothL1 + GIoU |
| Assignment | positive IoU ≥ 0.5, negative IoU < 0.4 |
| NMS | hard NMS at 0.25 |
| Max detections | 300 |
| Dataset | Pascal VOC XML |
| Class count | 1 class: Rice |

The model file correctly includes letterbox utilities and inverse box unpadding, which is critical for avoiding coordinate artifacts.

---

## 2. Biggest Implementation Problems

### 2.1 Notebook Loss Is Likely Double-Counted

If the V5 notebook uses:

```python
loss_dict = model(imgs, targets)
loss = sum(loss_dict.values())
```

that is incorrect because the model already returns:

```python
{
    "cls_loss": ...,
    "reg_loss": ...,
    "total_loss": cls_loss + reg_loss
}
```

So summing all values optimizes:

```python
cls_loss + reg_loss + total_loss
```

which equals:

```python
2 * (cls_loss + reg_loss)
```

Use this instead:

```python
loss = loss_dict["total_loss"]
```

Do this in both training and validation loss calculations.

---

### 2.2 Dataset Split Hygiene

The training notebook should not pool `train`, `valid`, `val`, and `test` together and reshuffle them.

That makes mAP unreliable because the test set stops being a true holdout.

Recommended policy:

| Split | Use |
|---|---|
| train | training |
| valid | training or validation, depending dataset source |
| val | validation |
| test | final holdout only |

For paper-grade evaluation, keep `test` untouched until the final model.

---

### 2.3 Translation Augmentation Can Corrupt Labels

The current augmentation uses `ImageChops.offset`, which wraps pixels around image borders. This is bad for detection because rice pixels can reappear on the opposite edge without corresponding boxes.

Replace wraparound translation with affine translation using constant fill:

```python
from PIL import Image

def safe_translate(img, boxes, dx, dy, fill=(114, 114, 114), min_size=2):
    w, h = img.size

    img = img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, dx, 0, 1, dy),
        fillcolor=fill,
    )

    if len(boxes) == 0:
        return img, boxes

    boxes = boxes.clone().float()
    boxes[:, [0, 2]] += dx
    boxes[:, [1, 3]] += dy

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

    keep = ((boxes[:, 2] - boxes[:, 0]) >= min_size) & (
        (boxes[:, 3] - boxes[:, 1]) >= min_size
    )

    return img, boxes[keep]
```

---

## 3. NMS and Postprocessing Problems

The current decoder uses:

```python
nms_thr = 0.25
max_dets = 300
```

That is very aggressive for dense rice scenes.

Dense agricultural images may contain many overlapping or adjacent plants. A hard NMS threshold of 0.25 can suppress valid neighboring rice plants.

For mAP evaluation, use:

```python
score_thr = 0.001
nms_thr = 0.45
max_dets = 1000
```

Run an NMS sweep:

```python
for nms_thr in [0.25, 0.35, 0.45, 0.55, 0.65]:
    evaluate_map(model, val_loader, score_thr=0.001, nms_thr=nms_thr, max_dets=1000)
```

Use different thresholds for visualization and evaluation:

| Use Case | Score Threshold | NMS | Max Detections |
|---|---:|---:|---:|
| mAP evaluation | 0.001 | sweep 0.25–0.65 | 1000 |
| debugging | 0.05 | 0.45 | 1000 |
| visual demo | 0.25–0.50 | 0.45 | 300 |

---

## 4. Small-Object and Anchor Analysis

Current anchors:

| Level | Stride | Base Scale | Nominal Square Anchor |
|---|---:|---:|---:|
| P3 | 8 | 6 | 48×48 |
| P4 | 16 | 6 | 96×96 |
| P5 | 32 | 6 | 192×192 |

If many rice boxes are smaller than 48 pixels in width or height after letterboxing, then P3/P4/P5 with base scale 6 is probably too coarse.

### Required Box Statistics Script

```python
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

def collect_box_stats(voc_root):
    ann_dir = Path(voc_root) / "annotations"
    widths, heights, areas, ratios = [], [], [], []

    for xml_path in ann_dir.glob("*.xml"):
        root = ET.parse(xml_path).getroot()

        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue

            x1 = float(bb.find("xmin").text)
            y1 = float(bb.find("ymin").text)
            x2 = float(bb.find("xmax").text)
            y2 = float(bb.find("ymax").text)

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            if w > 1 and h > 1:
                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                ratios.append(w / h)

    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    ratios = np.array(ratios)

    print(f"Boxes: {len(widths)}")
    print(f"Median width: {np.median(widths):.2f}")
    print(f"Median height: {np.median(heights):.2f}")
    print(f"Median area: {np.median(areas):.2f}")
    print(f"Median aspect ratio: {np.median(ratios):.2f}")

    for s in [16, 24, 32, 48, 64, 96]:
        pct = np.mean((widths < s) | (heights < s)) * 100
        print(f"Pct with width or height < {s}px: {pct:.1f}%")
```

### Anchor Experiments

| Experiment | Change |
|---|---|
| A | Current base scale 6 |
| B | Base scale 4 |
| C | Base scale 3 |
| D | Add P2 stride-4 feature map |
| E | Use k-means anchors from actual box sizes |

A likely improved anchor setup:

```python
AnchorGenerator(
    base_scale=4,
    aspect_ratios=(0.5, 1.0, 2.0),
    scales=(1.0, 2**(1/3), 2**(2/3)),
    strides=(8, 16, 32),
)
```

If boxes are very small, add P2:

| Level | Stride | Base Scale 4 Anchor |
|---|---:|---:|
| P2 | 4 | 16×16 |
| P3 | 8 | 32×32 |
| P4 | 16 | 64×64 |
| P5 | 32 | 128×128 |

---

## 5. Loss and Assignment Improvements

The current loss stack is:

```python
classification = VariFocalLoss
regression = SmoothL1 + GIoU
assignment = fixed IoU thresholds
```

This is not terrible, but modern detectors usually gain mAP from better assignment and localization modeling.

Recommended progression:

| Priority | Upgrade | Why |
|---:|---|---|
| 1 | Fix current loss bug | Required before any fair test |
| 2 | Try CIoU or DIoU instead of GIoU | Better center/aspect localization |
| 3 | Add ATSS assignment | More robust positives for small objects |
| 4 | Test Task-Aligned Assignment | Aligns classification and localization |
| 5 | Move to GFL/DFL-style head | Better box localization distribution |
| 6 | Compare VFNet/GFL baseline | Stronger implementation of quality-aware detection |

Generalized Focal Loss and Distribution Focal Loss were designed to address classification-quality inconsistency and bounding-box uncertainty in dense one-stage detection. GFL-style methods are especially relevant because the current code already uses a quality-aware scoring idea. citeturn396314academia2

---

## 6. Feature Pyramid / Architecture Improvements

Current eFPN uses only P3/P4/P5. That is efficient, but it may be too weak for dense small rice plants.

Recommended architecture experiments:

| Priority | Change | Expected Benefit |
|---:|---|---|
| 1 | Add P2 | Better small-object recall |
| 2 | Use pretrained backbone | Faster convergence and better features |
| 3 | Replace eFPN with PAN-FPN | Better bottom-up localization |
| 4 | Try BiFPN | Weighted multi-scale fusion |
| 5 | Try deformable conv in neck/head | Better irregular plant geometry |
| 6 | Use VFNet/GFL/RT-DETR baseline | Stronger modern detector comparison |

The biggest architectural concern is that the custom Det-ResNet-50 appears to be trained from scratch. For a relatively small rice dataset, a pretrained backbone is likely much more important than custom architectural purity.

---

## 7. Tiling / Sliced Training and Inference

Small rice plants may occupy too few pixels at 512×512. Slicing/tiled inference can make each plant effectively larger in the model input.

SAHI is specifically designed for small-object detection by slicing high-resolution images into overlapping patches. In reported experiments, SAHI improved AP by 6.8, 5.1, and 5.3 points for FCOS, VFNet, and TOOD respectively, and slicing-aided fine-tuning increased gains further. citeturn396314academia0

Recommended settings:

| Tile Size | Overlap | Use |
|---:|---:|---|
| 512 | 20% | memory-safe |
| 640 | 20% | better context |
| 768 | 20–25% | if GPU allows |

Best experiment:

1. train on sliced tiles
2. validate on sliced inference
3. merge predictions back to full-image coordinates
4. use Soft-NMS or weighted box fusion

---

## 8. Training Protocol Recommendations

### Corrected Training Loop Core

```python
REPEAT_FACTOR = 2

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    for repeat in range(REPEAT_FACTOR):
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = [
                {k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)

            loss_dict = model(imgs, targets)
            loss = loss_dict["total_loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    scheduler.step()
```

### Use AMP if Available

```python
scaler = torch.amp.GradScaler("cuda")

with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss_dict = model(imgs, targets)
    loss = loss_dict["total_loss"]

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
scaler.step(optimizer)
scaler.update()
```

### Add EMA

Exponential moving average often stabilizes detector evaluation.

```python
ema_decay = 0.9998
```

Evaluate the EMA weights, not just the raw training weights.

---

## 9. Evaluation Protocol for Paper-Grade mAP

Report:

- AP@0.50
- AP@0.75
- AP@0.50:0.95
- AR@100
- AR@300
- AR@1000
- detections per image
- score histogram
- NMS sensitivity
- PR curve
- qualitative false positive / false negative grid

Evaluation should use low score threshold:

```python
score_thr = 0.001
```

Do not evaluate mAP using only high-confidence detections. That can destroy recall and lower AP.

---

## 10. Highest-Impact Experiment Order

### Phase 0: No-Retrain Diagnostics

1. Evaluate current checkpoint with `score_thr=0.001`.
2. Raise `max_dets` to 1000.
3. Sweep NMS from 0.25 to 0.65.
4. Plot score histogram.
5. Count detections/image.
6. Plot GT box-size distribution.

### Phase 1: Clean WeedDet Retrain

Patch:

- loss double-counting
- split hygiene
- translation augmentation
- 2× repeat
- mAP checkpointing
- low-threshold evaluation
- max detections 1000

### Phase 2: Anchor/Resolution Experiments

| Run | Change |
|---|---|
| A | base scale 6, 512 |
| B | base scale 4, 512 |
| C | base scale 4, 640 |
| D | base scale 4, 768 |
| E | P2 + base scale 4 |
| F | tiled 512 inference |
| G | tiled training + inference |

### Phase 3: Modern Baselines

Train at least one:

- YOLOv8s or YOLOv8m
- MMDetection RetinaNet
- MMDetection GFL
- MMDetection VFNet
- RT-DETR

If a pretrained baseline strongly outperforms WeedDet, the custom architecture is the bottleneck.

---

## 11. Final Ranked Recommendations

| Rank | Recommendation | Expected mAP Impact |
|---:|---|---|
| 1 | Fix notebook loss calculation | Very high |
| 2 | Clean train/val/test split | Very high for valid mAP |
| 3 | Replace wraparound augmentation | High |
| 4 | Relax NMS and increase max detections | High |
| 5 | Evaluate with low score threshold | High |
| 6 | Tune anchors from box statistics | High |
| 7 | Add P2 if boxes are small | High |
| 8 | Use tiled inference/training | High for small objects |
| 9 | Use pretrained backbone | High |
| 10 | Benchmark VFNet/GFL/YOLOv8 | High |
| 11 | Add EMA | Medium |
| 12 | Try CIoU/DIoU/EIoU | Medium |
| 13 | Add ATSS/task-aligned assignment | Medium-high |
| 14 | Replace eFPN with PAN/BiFPN | Medium-high |
| 15 | Add DFL/GFL-style localization | Medium-high |

---

## 12. Bottom Line

The model should not be judged yet based on the current v5 result. The current pipeline likely has enough training, augmentation, postprocessing, and evaluation issues to suppress mAP even if the architecture is usable.

The best next move is:

**Run one clean WeedDet retrain after fixing the notebook loss, split hygiene, augmentation, 2× repeat, and evaluation settings. Then compare it against a pretrained modern detector baseline.**

If the cleaned WeedDet still fails, the most likely next high-gain path is:

**pretrained backbone + smaller anchors/P2 + sliced training/inference + quality-aware detector such as VFNet/GFL.**

---

## References

- `weeddet_Latest.py` project implementation: fileciteturn0file0
- SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection. citeturn396314academia0
- Generalized Focal Loss. citeturn396314academia2
- Generalized Focal Loss V2. citeturn396314academia1
