# Implementation Roadmap: Next Steps for AgriNav Rice Detection

**Prepared:** June 29, 2026  
**Current Performance:** AP@0.50=0.1662, AP@0.75=0.0043, val_loss=0.6157  
**Goal:** AP@0.75 > 0.10 (3x improvement)

---

## Phase 1: Add P2 Feature Level (CRITICAL)

### Why P2?
- 43% of your boxes are smaller than 24px
- Current P3 (stride 8) creates 32px anchors (too large)
- P2 (stride 4) creates 12px anchors (perfect for small objects)
- Modern detectors (YOLOv8, EfficientDet, Faster R-CNN) all use P2

### Timeline: 1-2 weeks
- Day 1-2: Modify model architecture
- Day 3-7: Training (20-30 epochs)
- Day 8: Evaluation + results

### Implementation Steps

#### Step 1: Add P2 Output to Backbone
**File:** `weeddet_LatestGPT.py`, around line 689 (WeedDet class)

Find this in `__init__`:
```python
class WeedDet(nn.Module):
    def __init__(self, num_classes=1, anchor_base_scale=3, lsc_k=7, use_atss=True):
        ...
        self.neck = eFPN(256)
        self.head = ERetinaHead(256, num_classes, 12, lsc_k)  # 12 = 4 ratios × 3 scales
```

Change to:
```python
class WeedDet(nn.Module):
    def __init__(self, num_classes=1, anchor_base_scale=3, lsc_k=7, use_atss=True, use_p2=True):
        ...
        self.use_p2 = use_p2
        self.neck = eFPN(256, use_p2=True)
        
        # 20 anchors = 5 aspect_ratios × 4 scales (for P2/P3/P4/P5)
        num_anchors = 5 * 4 if use_p2 else 4 * 3
        self.head = ERetinaHead(256, num_classes, num_anchors, lsc_k)
```

#### Step 2: Modify eFPN to Output P2
**File:** `weeddet_LatestGPT.py`, around line 352 (eFPN class)

Current:
```python
class eFPN(nn.Module):
    def forward(self, c3, c4, c5):
        # ... only P3, P4, P5
        return p3, p4, p5
```

Change to:
```python
class eFPN(nn.Module):
    def __init__(self, in_ch, use_p2=True):
        super().__init__()
        self.use_p2 = use_p2
        if use_p2:
            self.p2_conv = nn.Conv2d(512, 256, 1)  # C3 is 512 channels
        # ... rest of FPN
        
    def forward(self, c3, c4, c5, c2=None):
        # ... existing P3, P4, P5 code ...
        if self.use_p2 and c2 is not None:
            p2 = self.p2_conv(c2)  # or appropriate fusion
            return p2, p3, p4, p5
        return p3, p4, p5
```

#### Step 3: Update Anchor Generator
**File:** `weeddet_LatestGPT.py`, around line 400 (AnchorGenerator)

Current:
```python
class AnchorGenerator:
    def __init__(self, base_scale=3, aspect_ratios=(0.2, 0.33, 0.5, 1.0)):
        self.strides = (8, 16, 32)
        self.scales = (1.0, 2**(1/3), 2**(2/3))
```

Change to:
```python
class AnchorGenerator:
    def __init__(self, base_scale=3, aspect_ratios=(0.2, 0.33, 0.5, 1.0), use_p2=True):
        if use_p2:
            self.strides = (4, 8, 16, 32)
            self.scales = (1.0, 2**(1/3), 2**(2/3), 2**(2/3))  # 4 scales for 4 levels
        else:
            self.strides = (8, 16, 32)
            self.scales = (1.0, 2**(1/3), 2**(2/3))
        self.aspect_ratios = aspect_ratios
        self.base_scale = base_scale
```

#### Step 4: Update Forward Pass
**File:** `weeddet_LatestGPT.py`, around line 707 (WeedDet.forward)

Current:
```python
def forward(self, images, targets=None):
    ...
    c3, c4, c5 = self.backbone(images)
    p3, p4, p5 = self.neck(c3, c4, c5)
    features = (p3, p4, p5)
    ...
```

Change to:
```python
def forward(self, images, targets=None):
    ...
    feats = self.backbone(images)  # Returns (c2, c3, c4, c5) if c2 available
    if self.use_p2 and len(feats) == 4:
        c2, c3, c4, c5 = feats
        p2, p3, p4, p5 = self.neck(c3, c4, c5, c2)
        features = (p2, p3, p4, p5)
    else:
        c3, c4, c5 = feats[-3:]
        p3, p4, p5 = self.neck(c3, c4, c5)
        features = (p3, p4, p5)
    ...
```

#### Step 5: Update Training Code
**File:** `weeddet_trainingV5GPT_fixed.ipynb`, Cell 3

```python
model = wd.WeedDet(
    num_classes=NUM_CLASSES,
    anchor_base_scale=ANCHOR_BASE_SCALE,
    lsc_k=LSC_K,
    use_atss=USE_ATSS,
    use_p2=True,  # ← NEW
).to(device)
```

### Testing P2

After modifying code, verify with:
```python
# Quick sanity check
model = wd.WeedDet(num_classes=1, use_p2=True)
dummy_img = torch.randn(1, 3, 512, 512)
dummy_target = {'boxes': torch.randn(5, 4), 'labels': torch.ones(5)}

loss_dict = model([dummy_img], [dummy_target])
print(f"Loss: {loss_dict}")  # Should work without errors
```

### Expected Results After P2

```
Before P2:
  AP@0.50:      0.1662
  AP@0.75:      0.0043
  AP@0.50:0.95: 0.0389
  Det/img:      48.8

After P2 (estimated):
  AP@0.50:      0.17-0.19   (slight dip due to more anchors, but fewer FP)
  AP@0.75:      0.08-0.15   (MAJOR improvement)
  AP@0.50:0.95: 0.08-0.12   (2-3x improvement)
  Det/img:      28-32       (closer to true object count)
```

---

## Phase 2: Verify Soft-NMS Impact (1 day, OPTIONAL)

### Why Check?
- Your eval uses `use_soft_nms=True`
- Soft-NMS decays scores instead of removing boxes
- COCO mAP evaluation might not like decayed scores

### Quick Test

**In your Cell 6 evaluation:**

```python
# Current
results = eval_model._decode(
    cls_l, regs, anchors, ishape,
    score_thr=EVAL_SCORE_THR,
    nms_thr=NMS_THR,
    max_dets=MAX_DETS,
    output_thr=EVAL_SCORE_THR,
    use_soft_nms=True,  # ← Try False
)
```

Run twice: once with `use_soft_nms=True`, once with `False`.

**Expected outcome:**
- Hard NMS might give slightly higher AP@0.75
- If so, update training code to use hard NMS

---

## Phase 3: Data Quality Audit (2 days, OPTIONAL)

### Why Audit?
- 10.6% of boxes are <10% margin to edge
- Want to confirm these are genuine (not over-tight annotations)

### Process

```python
import numpy as np
from PIL import Image, ImageDraw
import random

# Sample 20 tight-margin boxes
tight_boxes = [...]  # Load from your dataset analysis

# Visualize
for stem in random.sample(tight_boxes, min(20, len(tight_boxes))):
    img = Image.open(f'{FLAT_ROOT}/images/{stem}.jpg')
    draw = ImageDraw.Draw(img)
    
    # Draw GT boxes in GREEN
    # Draw model predictions in RED
    
    # Check: Does the green box actually cover the rice plant?
    img.show()
```

**Decision Matrix:**
- If boxes look accurate → Accept, continue
- If boxes look loose → Flag as annotation error, lower regression weight
- If boxes look too tight → Accept, model should still try to tighten

---

## Phase 4: Benchmark YOLOv8 (Optional, 2-3 weeks)

### Purpose
Confirm WeedDet is the right architecture for your data.

### Quick Comparison

```python
# Load YOLOv8s (pre-trained on COCO)
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

# Fine-tune on your rice data
results = model.train(
    data='rice_data.yaml',  # Your data in COCO format
    epochs=30,
    imgsz=512,
    batch=2,
)

# Compare AP
print(f"YOLOv8s AP@0.75: {results.box.ap75}")
print(f"WeedDet AP@0.75: {your_ap75}")
```

---

## Detailed Timeline

```
Week 1: Architecture Modifications
  Mon-Tue   : Modify model code (P2 backbone, FPN, head)
  Wed-Thu   : Test on dummy data, fix bugs
  Fri       : Verify code runs on actual dataset

Week 2: Training & Evaluation
  Mon-Fri   : Train 20-30 epochs on full dataset
            : Monitor loss curves
            : Save best checkpoint

Week 3: Results & Decisions
  Mon-Tue   : Full evaluation on test set
  Wed       : Phase 2 (soft-NMS check)
  Thu-Fri   : Decisions on Phase 3/4
```

---

## Files to Modify

| File | Changes | Difficulty |
|------|---------|------------|
| `weeddet_LatestGPT.py` | Add P2 to backbone, FPN, anchor gen | Medium |
| `weeddet_trainingV5GPT_fixed.ipynb` | Add `use_p2=True` to model init | Easy |
| `Cell 6 (evaluation)` | Verify soft-NMS setting | Easy |

---

## Backup Plan (If P2 Doesn't Help)

If AP@0.75 is still low after P2:

1. **Check loss balance:** Maybe classification loss is drowning regression
   ```python
   cls_weight = 1.0
   reg_weight = 5.0  # Increase regression emphasis
   ```

2. **Try different assignment:** Switch from ATSS to FCOS or OTA
   ```python
   use_atss = False
   use_fcos = True
   ```

3. **Add more training data:** 1.3k images might be small
   - Augment with mosaic, copy-paste
   - Scrape more rice field images

4. **Benchmark YOLOv8:** If WeedDet plateaus, try modern baseline

---

## Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| AP@0.50:0.95 | 0.0389 | 0.08+ | ⚠️ Need 2x |
| AP@0.50 | 0.1662 | 0.18+ | ⚠️ Need slight gain |
| AP@0.75 | 0.0043 | 0.10+ | 🔴 **Critical** |
| Det/img | 48.8 | ~30 | ⚠️ Need reduction |
| Train time | N/A | <2hrs/epoch | ✓ OK with AMP |

---

## Resources & References

- **WeedDet paper:** Peng et al. (2022) — your baseline
- **P2 examples:** YOLOv5, EfficientDet, Faster R-CNN
- **Feature pyramid:** FPN paper (Lin et al. 2017)
- **ATSS:** Zhu et al. (2021)
- **CIoU:** Zheng et al. (2021)

---

## Decision Checklist

Before starting Phase 1:
- [ ] Understand why P2 helps (stride 4 anchors for small objects)
- [ ] Backup current `weeddet_LatestGPT.py` before modifying
- [ ] Verify training notebook runs on current code
- [ ] Decide: modify code yourself or request implementation?
- [ ] Confirm 2-3 week timeline is acceptable
- [ ] Plan for checkpoint storage (GPU memory)

---

**Next Action:** Reply with answers to decision checklist, then Phase 1 starts.

