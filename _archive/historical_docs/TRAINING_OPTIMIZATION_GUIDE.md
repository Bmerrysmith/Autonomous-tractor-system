# WeedDet Training Optimization Guide
## Based on Your Current Pipeline Analysis

**Date:** June 29, 2026  
**Status:** You're 70% there. Small tuning required for AP@0.75.

---

## What's Working ✓

Your latest notebook already includes critical optimizations:

```python
ANCHOR_BASE_SCALE = 3              ✓ CORRECT (was 4)
aspect_ratios = (0.2, 0.33, 0.5, 1.0)  ✓ CORRECT (was 0.25, 0.5, 1.0, 2.0)
USE_ATSS = True                    ✓ GOOD (adaptive assignment)
USE_AMP = True                     ✓ GOOD (mixed precision)
USE_EMA = True                     ✓ GOOD (exponential moving average)
```

**Result:** Best val loss improved from 0.7437 → 0.6157 ✓

---

## What's NOT Working ✗

Your AP@0.75 is 0.004 (nearly 0). This is the core issue.

**Root cause:** NMS threshold is **too high at 0.50**

```
Current:  NMS_THR = 0.50
Result:   48.76 detections/image (should be ~30)
Effect:   Boxes overlap, AP@0.75 requires IoU > 0.75 on non-overlapping boxes
```

---

## CRITICAL FIX #1: Lower NMS Threshold

**In Cell 6 (evaluation), change:**

```python
# BEFORE
NMS_THR = 0.50
MAX_DETS = 100

# AFTER — Test this configuration
NMS_THR = 0.30  # More aggressive NMS removal of overlaps
MAX_DETS = 100  # Keep same cap
```

**In Cell 3 (training), also add NMS control:**

```python
# During inference in train/val, use same NMS threshold
# (Currently your model's forward() may have hardcoded NMS settings)
```

**Expected impact:**
- Detections/image: 48.76 → ~32
- AP@0.50: 0.166 → ~0.15-0.18 (slight dip, but less noisy)
- AP@0.75: 0.004 → 0.02-0.05 (MAJOR improvement expected)

---

## CRITICAL FIX #2: Regression Loss — Switch to CIoU

AP@0.75 requires **tight box localization**. GIoU is okay, but CIoU is better for tall boxes.

**Why:** Your boxes are tall/narrow (median 22w × 56h, AR=0.368). CIoU includes aspect-ratio penalty, which helps.

**In your weeddet_LatestGPT.py model file, find the regression loss:**

```python
# Current (somewhere in your loss computation)
# giou_loss = ...

# Change to CIoU
def ciou_loss(pred_boxes, target_boxes):
    """CIoU loss — better for tall/narrow boxes"""
    pred_cxcy = (pred_boxes[..., 2:] + pred_boxes[..., :2]) / 2
    pred_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    
    target_cxcy = (target_boxes[..., 2:] + target_boxes[..., :2]) / 2
    target_wh = target_boxes[..., 2:] - target_boxes[..., :2]
    
    # CIoU formula includes center distance + aspect ratio + IoU
    # (Full implementation: see implementation section below)
```

**Easier approach:** If weeddet_LatestGPT.py uses MMDet or similar, just change:

```python
# In loss config
reg_loss = dict(type='CIoULoss')  # was 'GIoULoss'
```

**Expected impact:**
- AP@0.75: 0.004 → 0.04-0.08 (major improvement)
- Training time: same
- Convergence: might be slightly faster

---

## IMPORTANT: Diagnostic Metrics to Track

Add these metrics to your training loop to diagnose remaining issues:

```python
# After each validation epoch, log:

metrics_to_track = {
    'val_loss': avg_val,
    'positive_anchors_per_gt': positive_count / gt_count,
    'nms_filtered_boxes': (detections_before_nms - detections_after_nms),
    'detections_per_image': len(detections) / len(valid_items),
    'avg_confidence': scores.mean(),
    'median_confidence': scores.median(),
    'confidence_std': scores.std(),
}
```

**Why?**
- If `positive_anchors_per_gt < 3`: assignment is still bad, need P2 + better anchors
- If `detections_per_image > 40`: NMS threshold too high
- If `avg_confidence < 0.15`: VariFocal not calibrating well

---

## OPTIONAL: Add P2 Feature Level

If NMS + CIoU don't get you to AP@0.75 > 0.10, add P2.

**What to change in weeddet_LatestGPT.py:**

```python
# In AnchorGenerator, add P2 (stride 4)
strides = (4, 8, 16, 32)  # was (8, 16, 32)

# In model initialization
anchor_base_scale = 3
scales = (1.0, 2**(1/3), 2**(2/3))
num_anchors = len(aspect_ratios) * len(scales)  # 20 instead of 12

# In ERetinaHead
self.head = ERetinaHead(256, num_classes, 20, lsc_k)  # was 12
```

**Expected impact:**
- Training time: +20-30%
- AP@0.75: +5-10% additional (if NMS/CIoU already helped)
- Small object AP: +8-12%

**Timeline:** Don't do this yet. Try NMS + CIoU first.

---

## Quick-Start: Changes to Make NOW

### Step 1: Fix NMS in Cell 6 Evaluation

```python
# Line: NMS_THR = 0.50
NMS_THR = 0.30  # Lower to ~0.30-0.40

# Line: MAX_DETS = 100
# Keep same (your dataset averages 29.4, so 100 is fine)
```

### Step 2: Verify weeddet_LatestGPT.py NMS Settings

Check if your model has internal NMS settings:

```python
# In WeedDet._decode() method, find:
results = model._decode(
    cls_l, regs, anchors, ishape,
    score_thr=...,
    nms_thr=...,     # Make sure this is 0.30
    max_dets=...,
)
```

Make sure NMS threshold in training matches evaluation (or use 0.30 everywhere).

### Step 3: Try CIoU Loss (if comfortable modifying model)

In weeddet_LatestGPT.py, find regression loss computation:

```python
# Search for "GIoU" or "giou"
# Change to CIoU equivalent

# If using MMDet-style loss:
reg_loss = dict(type='CIoULoss')  # instead of GIoULoss
```

### Step 4: Re-run Cell 3 Training (short run: 10-15 epochs)

```python
NUM_EPOCHS = 15  # Quick test, not 40

# Load your best checkpoint
RESUME_WEIGHTS = True  # loads BEST_CKPT

# Run training, monitor:
# - Train loss decreasing?
# - AP@0.75 improving?
# - Detections/image decreasing toward ~30?
```

**Expected outcome after 10-15 epochs:**
- AP@0.50: 0.166 → 0.17-0.19
- AP@0.75: 0.004 → 0.02-0.05
- Val loss: 0.6157 → 0.60-0.62 (slight improvement or stable)

---

## If AP@0.75 Still Low (<0.05)

Then add P2 feature level. The bottleneck is spatial resolution for small boxes.

```python
# In notebook Cell 3, change model init:
model = wd.WeedDet(
    num_classes=NUM_CLASSES,
    anchor_base_scale=ANCHOR_BASE_SCALE,
    lsc_k=LSC_K,
    use_atss=USE_ATSS,
    use_p2=True,  # Add this (if supported)
).to(device)
```

Then train 20-30 epochs.

---

## Summary: What to Do This Week

**Priority 1 (30 mins):**
- [ ] Change NMS_THR from 0.50 to 0.30 in Cell 6
- [ ] Re-run Cell 6 evaluation
- [ ] Check if AP@0.75 improves

**Priority 2 (1-2 hours, if confident modifying model):**
- [ ] Find regression loss in weeddet_LatestGPT.py
- [ ] Switch GIoU → CIoU
- [ ] Run short training (15 epochs)
- [ ] Re-evaluate

**Priority 3 (3-5 days, only if still needed):**
- [ ] Add P2 feature level
- [ ] Train 20-30 epochs
- [ ] Should push AP@0.75 to 0.08-0.12+

---

## Your Current Performance vs. Target

| Metric | Current | After NMS Fix | After CIoU | After P2 |
|--------|---------|---------------|-----------|----------|
| AP@0.50 | 0.1662 | 0.17-0.18 | 0.17-0.19 | 0.17-0.20 |
| AP@0.75 | 0.0043 | 0.02-0.05 | 0.04-0.08 | 0.08-0.15 |
| AP@0.50:0.95 | 0.0389 | 0.045-0.055 | 0.055-0.075 | 0.08-0.12 |
| Det/img | 48.76 | 32-35 | 30-32 | 28-32 |

---

## Do NOT Change (These Are Good)

```python
ANCHOR_BASE_SCALE = 3          ✓ Correct
aspect_ratios = (0.2, 0.33, 0.5, 1.0)  ✓ Correct
USE_ATSS = True                ✓ Keep
USE_EMA = True                 ✓ Keep
USE_AMP = True                 ✓ Keep
BASE_LR = 0.001                ✓ Keep
MOMENTUM = 0.9                 ✓ Keep
WEIGHT_DECAY = 1e-4            ✓ Keep
WARMUP_RATIO = 0.05            ✓ Keep
```

---

## Questions?

If AP@0.75 doesn't improve after NMS fix, it's likely:
1. Your model file isn't using CIoU (need to check weeddet_LatestGPT.py)
2. You need P2 feature level
3. Regression targets aren't being set correctly (check VariFocal soft-label targets)

