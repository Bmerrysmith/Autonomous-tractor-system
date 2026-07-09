# AgriNav — Master Project History
## Extreme-Depth Timeline & Technical Record

**Author:** Benjamin Merryman-Smith  
**Project:** AgriNav — Autonomous Paddy Field Tractor, Weed Detection Subsystem  
**Course:** CEN 4930, FGCU Whitaker College of Engineering, Spring 2026  
**Document generated:** 2026-06-01  
**Sources:** Conversation history, GitHub repo, Google Drive, project memory

---

## Table of Contents

1. [Project Origins & Team Structure](#1-project-origins--team-structure)
2. [Core System Architecture](#2-core-system-architecture)
3. [Phase 1 — Literature Review & Design (March 2026)](#3-phase-1--literature-review--design-march-2026)
4. [Phase 2 — Baseline RetinaNet on Kaggle (Early April 2026)](#4-phase-2--baseline-retinanet-on-kaggle-early-april-2026)
5. [Phase 3 — WeedDet v1/v2 Initial Colab Runs (April 2026)](#5-phase-3--weeddet-v1v2-initial-colab-runs-april-2026)
6. [Phase 4 — Critical Bug Discovery & Resolution](#6-phase-4--critical-bug-discovery--resolution)
7. [Phase 5 — WeedDet v3, Extended Training (Late April 2026)](#7-phase-5--weeddet-v3-extended-training-late-april-2026)
8. [Phase 6 — Dataset Expansion & mAP Problem](#8-phase-6--dataset-expansion--map-problem)
9. [Phase 7 — Paper Submission, Professor Feedback, Revisions](#9-phase-7--paper-submission-professor-feedback-revisions)
10. [Phase 8 — WeedDet v4, Colab Retraining (May–June 2026)](#10-phase-8--weeddet-v4-colab-retraining-mayjune-2026)
11. [Phase 9 — WeedDet v5 (Current)](#11-phase-9--weeddet-v5-current)
12. [Complete Bug Log](#12-complete-bug-log)
13. [Training Run Log](#13-training-run-log)
14. [Model Architecture Evolution](#14-model-architecture-evolution)
15. [Dataset History](#15-dataset-history)
16. [Google Drive File Inventory](#16-google-drive-file-inventory)
17. [GitHub Repository Structure](#17-github-repository-structure)
18. [Open Items & Next Steps](#18-open-items--next-steps)

---

## 1. Project Origins & Team Structure

### Course Context
CEN 4930 — Introduction to Autonomous Driving Systems, Spring 2026, FGCU. The capstone deliverable is an IEEE conference-format research paper. The professor is an active IEEE reviewer.

### Team: AgriNav
| Member | Role | Status |
|---|---|---|
| **Benjamin Merryman-Smith** | Perception stack — data pipeline, WeedDet object detection, discrimination/spray module | Active |
| Bilal Dogutas | Lane detection (LiDAR/EKF) | Active |
| Tony Nguyen | Lightweight CNN-FPN detection pipeline integration | Active |
| Krish Shah | Localization, row following, GNSS/EKF | Active |
| ~~Anthony Raphael~~ | ~~Mission controller/path planner~~ | **Removed** — contribution not completed; removed from paper |

### Why Benny Owns the Perception Stack
Benny's coursework research focused on the WeedDet paper (Peng et al. 2022) and the GCN weed recognition paper (Jiang et al. 2020). He was best-positioned to implement the detection model and had the hardware (RTX 4060) for local development alongside Colab/Kaggle for training.

### Strategic Context
The professor is an active IEEE reviewer. A strong paper + recommendation letter is strategically valuable for Benny's target roles at Waymo, Aurora, NVIDIA, Bosch, Continental, Aptiv, and defense contractors.

---

## 2. Core System Architecture

### Full Pipeline (as designed)
```
RGB Camera + LiDAR + GNSS
         │
    ┌────┴────────────────────┐
    │                         │
Lane Detection            Object Detection
(Bilal/Krish — LiDAR/EKF) (Benny — WeedDet)
    │                         │
    │              Bounding Boxes (rice)
    │                         │
    └──────────────┬──────────┘
                   │
         Object Discrimination
         (rice veto logic)
                   │
         Path Planning / Mission Controller
         (Anthony — NOT IMPLEMENTED)
                   │
              Spray Actuator
```

### Benny's Subsystem: The Inverted Spray Approach
**Core insight:** Instead of detecting weeds (unknown, diverse, hard to label), detect rice (known, consistent, well-labeled). Anything outside a high-confidence rice detection box = spray candidate.

**Safety asymmetry:** False negatives on rice → crop damage. Therefore the system is biased toward protecting rice at all costs. A hard rice-veto rule prevents herbicide actuation on any detected rice region regardless of downstream logic.

**This is the paper's central novel contribution framing.**

### ROS Node Architecture (designed)
- `weeddet_inference_node` — camera frames → bounding boxes + class labels
- `lidar_roi_subscriber` — row boundaries from lane detection → masked ROI
- `world_coordinate_projector` — pixel positions + LiDAR depth → weed (x,y) world coords
- `confidence_gate` — filters detections before mission controller
- `lidar_row_detection_node` (Bilal/Krish) — row boundaries + centerline
- `ekf_localization_node` (Krish) — global pose

---

## 3. Phase 1 — Literature Review & Design (March 2026)

### Key Dates
- ~March 25, 2026: Benny produces outline technical analysis comparing WeedDet and GCN papers
- ~March 31, 2026: Architecture planning doc created in Drive ("Architecture for Autonomous Tractor")
- ~March 25, 2026: Presentation transcript doc created

### Work Done
Benny analyzed two research papers in depth:

**Paper 1: Peng et al. (2022) — WeedDet**
- Title: "Weed Detection in Paddy Field Using an Improved RetinaNet Network"
- Journal: Computers and Electronics in Agriculture, vol. 199, p. 107179
- Key contributions: Det-ResNet backbone (replaces 7×7 stem + MaxPool with 2×3×3 convs + DetResidualBlock), eFPN (P3/P4/P5 only, saves 7.71M params vs standard FPN), ERetinaHead (1 conv 64ch + LSC k=7), VariFocal Loss + GIoU + SmoothL1
- Result: 94.1 mAP @ 24.3 fps on 6408-image 9-class paddy field dataset
- Decision: Use as primary architecture

**Paper 2: Jiang et al. (2020) — GCN**
- Title: "CNN Feature Based Graph Convolutional Network for Weed and Crop Recognition in Smart Farming"
- Journal: Computers and Electronics in Agriculture, vol. 174, p. 105450
- Key contributions: Semi-supervised learning with CNN feature extraction + graph propagation
- Results: 97.80 / 99.37 / 98.93 / 96.51% accuracy across 4 weed datasets
- Decision: Originally planned for severity classification layer; ultimately removed from paper (not implemented)

**Paper 3: Lu et al. (2021) — Improved CenterNet**
- Used as additional reference for the architecture background

### Initial Architecture Decision
Benny proposed the "inverted detection" logic — detect rice as protected class rather than directly detecting weeds. This became the core novel framing.

### Proposal Doc
Team roles defined. Benny proposed: Tony Nguyen → object detection; Bilal + Krish → lane detection + path planning; Benny → object recognition/discrimination; Anthony → mission controller.

---

## 4. Phase 2 — Baseline RetinaNet on Kaggle (Early April 2026)

### What Was Built
A baseline notebook (`rice_detection_fixed.ipynb`, now in `archive/v1_baseline_retinanet/`) using torchvision's pretrained RetinaNet ResNet-50 FPN — NOT the custom WeedDet architecture. This served as a baseline proof-of-concept.

### Platform
Kaggle, GPU notebook. Datasets loaded from `bennymerryman/rice-detection` (COCO format).

### Architecture
```
torchvision.models.detection.retinanet_resnet50_fpn
- weights=None (detection heads random)
- weights_backbone='DEFAULT' (pretrained ImageNet backbone)
- num_classes=2 (0=background, 1=rice)
```

### Training Config
- 12 epochs, SGD lr=0.001, momentum=0.9, weight_decay=1e-4
- StepLR: decay ×0.1 after epoch 9
- Grad clip max_norm=1.0
- Batch size 2

### Datasets Used
- DATASET_1: `rice-hainan.coco` (primary)
- DATASET_2: `rice_detection_for_export.coco`
- Merged: 80/20 train/val split, seed 42
- category_id=1 for rice (0 reserved for background)

### Key Issues Discovered
- Kaggle input directories are read-only → must copy files to `/kaggle/working/` before writing
- GitHub fine-grained tokens don't work in Colab → use classic personal access tokens stored in Colab Secrets

### Outcome
Baseline working. Demonstrated the rice detection concept. Transitioned to custom WeedDet architecture.

---

## 5. Phase 3 — WeedDet v1/v2 Initial Colab Runs (April 2026)

### Timeline
- April 7, 2026: `weeddet_for_VSCode.py` first committed to Drive (`weeddet_checkpoints/`)
- April 7–8, 2026: `weeddet_training_fixed.ipynb` and early Colab runs
- April 8, 2026: `weeddet_training_fixed.ipynb` modified
- April 20–21, 2026: Second round, `Weed_Det_Training_v2.ipynb`, `rice-dataset-two.zip` added to Drive

### What Was Built
Full custom WeedDet implementation (`weeddet_for_VSCode.py`) — 1,077 lines, 39KB. Complete from scratch PyTorch implementation of the Peng et al. (2022) architecture.

### Architecture Implemented
```
Input: 1000×600 RGB
  ↓
Det-ResNet-50
  - Modified stem: 3×3 conv (3→16) + 3×3 conv (16→16) + DetResidualBlock (16→32, stride=2)
  - Replaces standard 7×7 conv + MaxPool
  - 4 bottleneck stages: layer1 (256ch), layer2/C3 (512ch), layer3/C4 (1024ch), layer4/C5 (2048ch)
  ↓
eFPN (Efficient FPN)
  - Lateral 1×1 projections: C3→256, C4→256, C5→256
  - Top-down pathway with nearest-neighbor upsample
  - Output: P3, P4, P5 (256ch each) — NO P6/P7 (saves 7.71M params)
  ↓
ERetinaHead
  - Shared: 1×Conv(256→64) + LargeSeparableConv(k=7)
  - LargeSeparableConv: depthwise k×1 → depthwise 1×k → pointwise 1×1
  - Per-level: cls_head (64→num_classes×9), reg_head (64→4×9)
  ↓
AnchorGenerator
  - base_scale=4 (later changed to 6 in v4/v5)
  - 3 aspect ratios × 3 scales = 9 anchors/location
  - Strides: 8, 16, 32
  ↓
WeedDetLoss
  - SmoothL1 + GIoU (regression)
  - VariFocal Loss (classification, alpha=0.75, gamma=2.0)
  - IACS-style soft labels
  - iou_threshold=0.4, neg_iou_threshold=0.3
```

### Initial Training Config
```python
base_lr=0.01, momentum=0.9, weight_decay=0.0001
batch_size=2, num_epochs=12
warmup_iters=500, lr_decay_epochs=[8, 11]
img_size=(600, 1000)
freeze_bn=True, grad_clip=1.0
dataset 2× repeat per epoch
```

### Critical Problem Discovered — Bug 1
Training appeared to run normally but loss collapsed to 0.0000 from epoch 2. No error thrown. Entire training run silent failure. See Bug Log Section 12.

---

## 6. Phase 4 — Critical Bug Discovery & Resolution

### Bug 1: Bounding Box Coordinate Space Mismatch (MOST COSTLY BUG)

**Discovery:** Loss = 0.0000 from epoch 2. Training appeared normal. Model learned nothing.

**Root cause:** VOC XML bounding box coordinates are stored in original image pixel space (e.g., 4000×3000). WeedDet resizes images to (600, 1000). Boxes were NOT being rescaled after resize. Result: box coords like (2000, 1500, 3000, 2000) on a 1000×600 image → IoU between ground truth and anchors = 0 → zero positive anchor assignments → zero gradient → loss collapses silently.

**The insidious part:** PyTorch's loss functions don't assert that boxes are within image bounds. The training loop ran to completion, logged plausible-looking loss values on other components, and saved checkpoints. Everything looked normal.

**Fix applied in `WeedDataset.__getitem__`:**
```python
# CRITICAL: rescale boxes to match resized image
if len(boxes):
    boxes = boxes.clone()
    boxes[:, [0,2]] *= tW / orig_w   # scale x-coords to target width (1000)
    boxes[:, [1,3]] *= tH / orig_h   # scale y-coords to target height (600)
```

**Verification rule:** After this fix, always sanity-check that `max(boxes[:,[0,2]]) <= 1000` and `max(boxes[:,[1,3]]) <= 600` before trusting any loss values. The training script includes a runtime assertion for this.

---

### Bug 2: Double-Scaling (FixedWeedDataset Wrapper)

**Discovery:** After Bug 1 fix, a `FixedWeedDataset` wrapper class was created that applied the coordinate scaling again on top of the now-fixed `WeedDataset`. This squared the scale factor — boxes shrank dramatically for large images.

**Root cause:** `FixedWeedDataset` wrapped `WeedDataset` which already applied the transform, then applied the same transform a second time. Effect: coordinates scaled by `(tW/orig_w)²` instead of `tW/orig_w`.

**Fix:** `FixedWeedDataset` entirely removed. `WeedDataset.__getitem__` is the single source of truth for coordinate scaling.

**Lesson:** Never wrap a dataset class that already applies coordinate transforms with another class that applies the same transforms.

---

### Bug 3: Anchor Scale Checkpoint Incompatibility

**Discovery:** When `anchor_base_scale` was changed from 4 to 6 (v4), existing checkpoints became incompatible — not because of weight shape changes but because the anchor offsets no longer matched the learned regression targets.

**Fix:** Fresh training required from scratch whenever anchor scale changes.

---

### Other Platform Bugs Discovered
| Bug | Platform | Fix |
|---|---|---|
| Kaggle input dirs read-only | Kaggle | Copy everything to `/kaggle/working/` |
| GitHub fine-grained tokens rejected | Colab | Use classic PATs in Colab Secrets |
| `category_id=0` treated as background | RetinaNet/WeedDet | Use `category_id=1` for rice |
| Dataset dir structure varies by export format | Roboflow | Normalize with symlink flat directory |
| `labels/` vs `annotations/` dir naming | Roboflow VOC export | Rename with `os.rename()` at setup |
| `weeddet_v4_best.pth` not found (v4 visual check) | Colab | Training saved `weeddet_best.pth`, cell looked for `weeddet_v4_best.pth` |
| Wrong module import in visual check | Colab | v4 Cell 4 imported `weeddet_for_VSCode`, training used `weeddet_Latest` |

---

## 7. Phase 5 — WeedDet v3, Extended Training (Late April 2026)

### Timeline
- April 23, 2026: Rice datasets uploaded to Drive (`rice_detection_for_export.v1i.voc.zip`, `Rice_Classification.v1i.voc.zip`)
- April 24, 2026: `Weed_Det_Training_v3.ipynb` — extended training to 94 epochs
- April 24, 2026: `weeddet_v3_best.pth` and `weeddet_v3_epoch94.pth` saved to Drive
- April 28, 2026: `weeddet_for_VSCode.py` updated; `rice_detection_demo.mp4` created
- April 29, 2026: `AgriNav_Final_Report.pptx` and `AgriNav_Progress_Report.pptx` created

### Training Run: v3
- **Dataset:** Merged COCO datasets (~1,596 images with real per-plant boxes, filtered from ~5,544 total)
- **Architecture:** WeedDet custom (weeddet_for_VSCode.py), anchor_base_scale=4
- **Epochs:** 94 (extended beyond 12)
- **Best val_loss:** 0.7437
- **Checkpoints:** `weeddet_v3_best.pth`, `weeddet_v3_epoch94.pth`
- **Qualitative result:** Confidence scores 0.55–0.84 on paddy, aerial, and post-flood imagery

### Demo Video
`rice_detection_demo.mp4` created showing inference results. Uploaded to Drive.

### mAP Problem First Encountered
Attempts to compute formal mAP using a separate evaluation dataset showed near-0% results. This was eventually diagnosed as a **domain gap** (annotation style differences between training and evaluation datasets), NOT a code bug. Same-distribution evaluation is the correct approach.

---

## 8. Phase 6 — Dataset Expansion & mAP Problem

### Datasets Used Throughout Project

| Dataset | Images | Annotations | Source | Format | Notes |
|---|---|---|---|---|---|
| rice-hainan (COCO) | ~1,347 | Per-plant boxes | Roboflow/Kaggle | COCO JSON | Primary Kaggle baseline |
| rice_detection_for_export.v1i.voc | ~1,347 | Per-plant VOC XML | Roboflow | VOC XML | Used in all Colab training |
| Rice_Classification.v1i.voc | ~4,041 | Per-plant VOC XML | Roboflow | VOC XML | Larger dataset, added in v3+ |
| Severity labeled (4 classes) | ~3,947 | Full-image boxes | Various | Mixed | GCN only — filtered out (placeholder boxes) |

**Critical annotation detail:** Full-image "placeholder" boxes (area ≥ 95% of image) from severity datasets are filtered by `coco_to_voc.py` before training. Only real per-plant boxes are used.

### The mAP Domain Gap Problem
- Cross-dataset evaluation (train on dataset A, evaluate on dataset B) produced ~0% mAP
- This is a well-known issue: annotation style varies across Roboflow exports (box tightness, labeling conventions)
- The model works qualitatively (correct visual detections) but formal metrics fail cross-dataset
- **Solution:** Evaluate on a held-out split of the same training distribution
- **Status as of 2026-06-01:** Same-distribution AP@0.5 and AP@0.75 still need to be run — this is the single most important blocking item for paper submission

### mAP Eval Notebook
`mAP eval.ipynb` created ~May 3, 2026 in Drive. Earlier attempt. Results not reliable due to domain gap.

---

## 9. Phase 7 — Paper Submission, Professor Feedback, Revisions

### Timeline
- ~May 2, 2026: Assignment 3 completed (`Assignment_3_CEN4930_completed`)
- May 2, 2026: `colab_train.ipynb` created (new training notebook iteration)
- May 3, 2026: `mAP eval.ipynb` created
- ~May 19, 2026: "Changes to research paper" doc created — major professor feedback round

### Paper Feedback Rounds
Two major rounds of feedback from the professor. Key themes:

**Round 1: Structural Issues**
1. Paper reads as multiple mini-papers stitched together, not unified
2. System Overview still inside Introduction — must be its own section after Related Work
3. Architecture figure in Introduction — must move to System Architecture section
4. Technical Contributions at wrong position — must be end of Introduction
5. No standalone Motivation section
6. Module sections are major `\section{}` — should be `\subsection{}` under System Architecture
7. Author names in section titles — must be removed
8. Individual Author Contributions in paper body — must move to after References
9. Bug-fix narratives in paper — must be removed
10. Multiple Motivation sections — must be consolidated

**Round 2: Content Issues**
1. mAP <1%, weed AP 0.0 — paper needs real same-distribution numbers
2. No detection visualization figures (bounding boxes on images)
3. "Estimated 30-50% ROI compute reduction" — needs actual measured number
4. GNSS outage position error not quantified
5. LiDAR-camera bridge implementation status unclear (implemented vs. proposed)
6. Abstract still AI-polished, overclaims
7. "GPU training resolves this" too confident — needs actual measurement
8. Discrimination logic text has contradictions
9. "Weeds physically cannot grow on row surfaces" — biologically too absolute
10. ROS vs ROS 2 inconsistency

**Structural Changes Made (Completed)**
- [x] Removed mission controller / Anthony Raphael references
- [x] Removed GCN section and related work subsection
- [x] Fixed broken LaTeX in author contributions block
- [x] Removed AI-tone markers from bookend sections
- [x] Replaced AI-generated architecture figure with draw.io version (confirm status)
- [x] Consolidated structure per feedback

**Still Open (as of 2026-06-01)**
- [ ] Same-distribution mAP (AP@0.5, AP@0.75) — HIGHEST PRIORITY
- [ ] System Overview still partially inside Introduction
- [ ] Architecture figure needs to move to System Architecture section
- [ ] Standalone Motivation section not yet added
- [ ] Module sections not yet converted to subsections
- [ ] Unified Experiments/Results section not yet created
- [ ] Author Contributions not yet added after References
- [ ] Abstract still overclaims (says "four modules", intro says "three")
- [ ] Discrimination logic contradiction not fixed
- [ ] LiDAR-camera bridge status table not added
- [ ] ROS version inconsistency not resolved

---

## 10. Phase 8 — WeedDet v4, Colab Retraining (May–June 2026)

### Timeline
- May 29, 2026: `weeddet_Latest.py` pushed to Drive (`weeddet_v2_checkpoints/`)
- May 29, 2026: `weeddet_trainingV4.ipynb` created, training launched
- June 1, 2026: Training completed (60 epochs)

### v4 Training Summary
```
Script:      weeddet_Latest.py (loaded from Drive)
Dataset:     rice_detection_for_export.v1i.voc only (1,347 images)
Val split:   NONE — val.txt was empty (0 pairs). Critical omission.
Platform:    Google Colab, Tesla T4 GPU
Epochs:      60
anchor_base_scale: 6 (changed from 4 — breaks prior checkpoint compatibility)
Batch:       2
LR:          0.001 cosine → 0.00005
grad_clip:   0.5
freeze_bn:   True
```

**Loss progression:**
| Epoch | Avg Loss |
|---|---|
| 1 | 1.0448 |
| 5 | 0.8666 |
| 10 | 0.8476 |
| 15 | 0.8252 |
| 20 | 0.8123 |
| 25 | 0.8008 |
| 30 | 0.7903 |
| 33 (last logged) | 0.7923 |
| 60 | TBD (run completed, checkpoint saved) |

**Best checkpoint saved as:** `weeddet_best.pth` (in `weeddet_v4_checkpoints/`)  
Note: NOT `weeddet_v4_best.pth` — this caused the visual check cell to fail.

### v4 Bugs Found (Post-Training)

**Bug: Checkpoint naming mismatch**
- Training saves: `weeddet_best.pth`
- Cell 4 (visual check) looks for: `weeddet_v4_best.pth`
- Result: "checkpoint not found" on every run
- Fix: Change `V4_CKPT = f'{CKPT_DIR_V4}/weeddet_best.pth'`

**Bug: Wrong module import in visual check**
- Training uses: `import weeddet_Latest as wd`
- Cell 4 imports: `import weeddet_for_VSCode as wd`
- Result: Wrong model loaded (different file, potentially different architecture)
- Fix: Change import to `import weeddet_Latest as wd`

**Issue: No val split**
- `val.txt` was empty (0 pairs) because the dataset's folder structure had all images in `train/` with no separate `valid/` folder containing matching XMLs
- No validation loss curve was generated
- No formal evaluation possible from this run

---

## 11. Phase 9 — WeedDet v5 (Current)

### What Changed
- Proper 80/20 train/val split implemented (seed 42, shuffled before split)
- Manual training loop replacing `train_with_progress()` — allows per-epoch val loss computation
- Val loss tracked every epoch alongside train loss
- Loss curve plot saved to Drive (`loss_curve.png`)
- mAP eval cell added (pycocotools, AP@0.5 and AP@0.75, same-distribution val set)
- Visual check fixed: correct checkpoint path (`weeddet_v5_best.pth`), correct import (`weeddet_Latest`)
- GT boxes loaded from VOC XML for side-by-side comparison

### v5 Config
```python
NUM_EPOCHS        = 12
ANCHOR_BASE_SCALE = 6
LSC_K             = 7
IMG_SIZE          = 512
BATCH_SIZE        = 2
BASE_LR           = 0.001
MIN_LR            = 0.00005
GRAD_CLIP         = 0.5
WARMUP_ITERS      = 200
FREEZE_BN         = True
SAVE_EVERY        = 4
DATASET:          rice_detection_for_export.v1i.voc (~1,347 images)
SPLIT:            80/20 (~1,077 train / ~270 val)
```

### Status (2026-06-01)
- Notebook: `active/weeddet_trainingV5.ipynb` — ready to run in Colab
- Training: NOT YET RUN
- Expected output: `weeddet_v5_best.pth` in `MyDrive/weeddet_v5_checkpoints/`

---

## 12. Complete Bug Log

| # | Bug | Phase | Symptom | Root Cause | Fix | Impact |
|---|---|---|---|---|---|---|
| 1 | Box coordinate space mismatch | Phase 3 | loss→0.0000, model learns nothing | VOC XML coords in orig pixel space, not rescaled after resize to 1000×600 | `boxes *= tW/orig_w, tH/orig_h` in `WeedDataset.__getitem__` | **CRITICAL** — multiple training runs wasted |
| 2 | Double-scaling (FixedWeedDataset) | Phase 3 | Boxes shrank dramatically | Wrapper applied scale transform on already-scaled dataset | Remove FixedWeedDataset entirely | HIGH — incorrect detections |
| 3 | Anchor scale checkpoint incompatibility | Phase 4/8 | Checkpoint loads but predicts garbage | `anchor_base_scale` change breaks learned regression targets | Fresh training from scratch | MEDIUM |
| 4 | category_id=0 as background | Phase 2 | Rice class ignored by model | RetinaNet/WeedDet reserves 0 for background | Use `category_id=1` for rice | HIGH |
| 5 | Kaggle read-only input | Phase 2 | File write error | `/kaggle/input/` is read-only | Write all outputs to `/kaggle/working/` | LOW |
| 6 | GitHub fine-grained token rejected | All | Authentication failure in Colab | Fine-grained tokens not compatible | Use classic PATs in Colab Secrets | LOW |
| 7 | Cross-dataset mAP collapse | Phase 6 | AP@0.5 ≈ 0% | Annotation style domain gap | Evaluate on same-distribution held-out split | HIGH — paper blocker |
| 8 | v4 checkpoint naming mismatch | Phase 8 | "checkpoint not found" | Training saves `weeddet_best.pth`, cell looks for `weeddet_v4_best.pth` | Fix `V4_CKPT` path | MEDIUM |
| 9 | v4 wrong module import | Phase 8 | Wrong model loaded | Cell 4 imports `weeddet_for_VSCode`, training used `weeddet_Latest` | Change import | MEDIUM |
| 10 | v4 empty val split | Phase 8 | No val loss curve | Dataset had all images in `train/`, no `valid/` folder | Build proper 80/20 split before training | HIGH |

---

## 13. Training Run Log

| Run | Script | Dataset | Epochs | Platform | Best Val Loss | anchor_base_scale | Notes |
|---|---|---|---|---|---|---|---|
| v1 (Kaggle baseline) | torchvision RetinaNet | rice-hainan + rice_detection COCO | 12 | Kaggle | — | N/A (torchvision) | Proof of concept |
| v2 (WeedDet initial) | weeddet_for_VSCode.py | rice_detection_for_export VOC | ~12 | Colab | ~0.0 (Bug 1) | 4 | SILENT FAILURE — Bug 1 |
| v2b (after Bug 1 fix) | weeddet_for_VSCode.py | rice_detection_for_export VOC | ~12 | Colab | Unknown | 4 | First real training |
| v3 (extended) | weeddet_for_VSCode.py | Merged (~1,596 real boxes) | 94 | Colab | **0.7437** | 4 | Best prior result; qualitative 0.55-0.84 confidence |
| v4 | weeddet_Latest.py | rice_detection_for_export only (1,347) | 60 | Colab (T4) | ~0.79 at ep30 (train only) | **6** | No val split; visual check broken; checkpoint naming bug |
| **v5 (current)** | weeddet_Latest.py | rice_detection_for_export (1,077 train / 270 val) | 12 | Colab | **TBD** | 6 | Proper val split; val loss per epoch; mAP eval cell |

---

## 14. Model Architecture Evolution

### v1 → v3: weeddet_for_VSCode.py (anchor_base_scale=4)
- Input: 1000×600 (paper spec)
- Full WeedDet implementation per Peng et al. 2022
- VariFocal + GIoU + SmoothL1 loss
- 2× dataset repeat per epoch
- MultiStepLR decay at [8, 11]

### v4 → v5: weeddet_Latest.py (anchor_base_scale=6)
- Input: 512 (letterbox)
- Added `letterbox_pil()` and `unpad_boxes()` utilities
- Cosine annealing instead of MultiStepLR
- Changed anchor_base_scale: 4 → 6 (breaks v3 checkpoint compat)
- Added `_get_logits()` and modified `_decode()` for external eval
- `IMAGENET_MEAN`, `IMAGENET_STD` exported for use in inference cells

### Key Architectural Constants (v5)
```
Parameters: 26.37M total, 26.32M trainable (BN frozen)
Backbone:   Det-ResNet-50 (modified stem)
Neck:       eFPN (P3/P4/P5 only)
Head:       ERetinaHead (64ch + LSC k=7)
Loss:       VariFocal (cls) + GIoU + SmoothL1 (reg)
Anchors:    9 per location (3 ratios × 3 scales), base_scale=6, strides=[8,16,32]
```

---

## 15. Dataset History

### rice_detection_for_export.v1i.voc
- ~1,347 images with per-plant VOC XML annotations
- Source: Roboflow (`rice_detection_for_export.v1i.voc.zip`)
- Drive ID: `1KHdX2tduaeLDC9QEHTHMGUOc374Ys8Ft`
- Used in: v3, v4, v5
- Class: `Rice` (normalized to this in all pipelines)

### Rice_Classification.v1i.voc
- ~4,041 images with per-plant VOC XML annotations
- Source: Roboflow (`Rice_Classification.v1i.voc.zip`)
- Drive ID: `1HztyoHqdmP08ZaFsRwszY-DfMwKEgASJ`
- Used in: v3 (merged), planned for future larger runs

### rice-hainan (COCO)
- ~1,347 images
- Used only in Kaggle baseline (v1)
- Kaggle path: `bennymerryman/rice-detection`

### Annotation Filtering
The `coco_to_voc.py` converter filters out placeholder boxes:
```python
def _is_placeholder(bbox, img_w, img_h, thresh=0.95):
    x, y, w, h = bbox
    return (w * h) / (img_w * img_h + 1e-6) >= thresh
```
Boxes covering ≥95% of the image are severity labels, not per-plant boxes, and are excluded.

---

## 16. Google Drive File Inventory

| File | Type | Modified | Drive ID | Notes |
|---|---|---|---|---|
| weeddet_trainingV4.ipynb | Colab | 2026-06-01 | 1ao9PE4UI7peRnTAHWbo2LX-EP6Ww6FJq | v4 training (has bugs) |
| weeddet_trainingV5.ipynb | Colab | 2026-06-01 | 1r5gjFAPcICGxCddr174-x9tAvYs3HyOj | **CURRENT** |
| Weed Det Training | Colab | 2026-05-29 | 1jS1pUJxpuAdhUmbAL9Qrv5yGa_bE7PJs | Early training notebook |
| weeddet_Latest.py | Python | 2026-05-29 | 1WPJ0ocfCtjrBxfv4udwt2caLtZ7ck-IV | **ACTIVE model** (in weeddet_v2_checkpoints/) |
| weeddet_for_VSCode.py | Python | 2026-04-28 | 1vxf29yeHuZiP75U-r0uTw9EDee-sJnTa | Older model |
| weeddet_v5_checkpoints/ | Folder | 2026-06-01 | 1sYMaWVemAACZjGc2KGX8ouU-m-TaTB1q | v5 checkpoint target |
| weeddet_v4_checkpoints/ | Folder | 2026-05-29 | 1Uk2mVQtuP2TxUn93miumAPchLL0m7j39 | v4 checkpoints |
| weeddet_v2_checkpoints/ | Folder | 2026-04-20 | 1yRIIemlDkhSAsAM0KKAz_2_SoiVjy3RH | weeddet_Latest.py here |
| weeddet_checkpoints/ | Folder | 2026-04-07 | 1MmcPXlO1wt7Y6HTvGfs5o3ag4A-pIf6F | Early v1 checkpoints |
| Weed_Det_Training_v3.ipynb | Colab | 2026-04-24 | 1FXrpnFOtDspk0Wy0BlpxvsS8YwVVYh1e | v3 (94 epoch) |
| Weed_Det_Training_v2.ipynb | Colab | 2026-04-21 | 1dswZo1d6rhF8wd_kRN-00DVwZXY93Lhm | v2 |
| weeddet_training_fixed.ipynb | Colab | 2026-04-08 | 1fp_154i5ifOGxKps-CRkcpCi4HQjLgPB | First fixed run |
| weeddet_training.ipynb | Colab | 2026-05-29 | 1ZnB64RuYbnqFw7_DammIrvnIbf2qU-RI | General training nb |
| AgriNav_Final_Report.pptx | Slides | 2026-04-29 | 1KoYsrhCWTcASQf9DW2iOENuOi0SA4bQM | Final report presentation |
| AgriNav_Progress_Report.pptx | Slides | 2026-04-29 | 1NTHncDo4bmZvMHzqgv-J5RQzPlnkbVdi | Progress presentation |
| Tractor_Team_Proposal.docx | Doc | 2026-04-12 | 1QIl38F7H7_aCjaWbzasZ12NLLF8UpmWe | Team proposal |
| Architecture for Autonomous Tractor | GDoc | 2026-03-31 | 1Yg8q1w8hS2qNdyguzjAz5AyPUeOkurTHT35IgcscuGc | Early planning doc |
| Changes to research paper | GDoc | 2026-05-19 | 1-xBkJVoiux7mu0zZbvLK5dY4mXtVzON5BhPG3buk48w | Professor feedback |
| Outline technical analysis | GDoc | 2026-03-25 | 1iSt948yNEh1AxczXNAhOuzUb0RRoe7xkFTR8kyv_CzQ | Paper analysis |
| rice_detection_for_export.v1i.voc.zip | Zip | 2026-04-23 | 1KHdX2tduaeLDC9QEHTHMGUOc374Ys8Ft | Dataset (~1,347) |
| Rice_Classification.v1i.voc.zip | Zip | 2026-04-23 | 1HztyoHqdmP08ZaFsRwszY-DfMwKEgASJ | Dataset (~4,041) |
| rice_detection_demo.mp4 | Video | 2026-04-28 | 1hFilZGUM7MrvN13kmSNpKnpsE7DGlC_q | Demo video |
| weeddet_v3_best.pth | Checkpoint | 2026-04-24 | 1L5pdF3_Crd4Nw5byjIXWMAqqNZZtIP3- | v3 best (val_loss 0.7437) |
| weeddet_v3_epoch94.pth | Checkpoint | 2026-04-24 | 1c4rAkuo9KIC3PLdA3lztHztyoHqdmP08 | v3 epoch 94 |
| mAP eval.ipynb | Colab | 2026-05-03 | 18EOcaWlvw_wHN0vVQRfteFGKlFbUmXyJ | Old mAP eval (unreliable) |
| colab_train.ipynb | Colab | 2026-05-02 | 1A16W0gi2vLvGSWMRySaCcK8_aVq4gYmj | Training notebook |
| Assignment_3_CEN4930_completed | GDoc | 2026-05-02 | 1PS7bcbUgz1VTUHJLoZgl3CB4FASLmUR2cb5pTgAV4HY | Course assignment |
| Benjamin_Merryman_Resume_Updated | GDoc | 2026-04-18 | 1Puoz2Ac0KtomNxsb_OGQroAJPqQ-NVMQ | Resume |
| presentation transcript | GDoc | 2026-03-25 | 1rQIpLzCNERHf4WZNlUI79mZOKVhGHYEmpnrvMNhZBfc | Presentation notes |

---

## 17. GitHub Repository Structure

**URL:** https://github.com/Bmerrysmith/Autonomous-tractor-system  
**Branch:** master  
**Language:** Python 76%, Jupyter Notebook 24%  
**Commits:** 7 total  
**Open PRs:** 1

```
Autonomous-tractor-system/
├── .github/workflows/
├── data/
│   ├── step1_extract.py          — Inspect and extract dataset zip
│   ├── auto_annotate.py          — Grounding DINO auto-annotation (NOT in final pipeline)
│   ├── coco_to_voc.py            — Convert COCO JSON → PASCAL VOC XML
│   └── step2_split.py            — Validate annotations, generate train/val split
├── inference/
│   └── inference_rice.py         — Run inference + green/red visualization
├── models/
│   └── weeddet_for_VSCode.py     — Complete WeedDet model (train + infer), 1077 lines, 39KB
├── notebooks/
│   └── rice_detection_fixed.ipynb — Kaggle baseline RetinaNet notebook
├── training/
│   └── colab_weeddet_train.py    — Full Colab training cell
├── .gitignore
├── README.md
├── qodana.yaml
└── requirements.txt
```

**Note on auto_annotate.py:** This file exists in the repo but Grounding DINO was NOT used in the final pipeline. Benny explicitly confirmed this when reviewing project history.

---

## 18. Open Items & Next Steps

### Blocking (must complete before paper submission)
1. **Run WeedDet v5** — open `active/weeddet_trainingV5.ipynb` in Colab, verify `SCRIPT_DIR` path, run all cells
2. **Get AP@0.5 and AP@0.75** from Cell 6 (mAP eval). Even 40% is acceptable — demonstrates the model learned.
3. **Confirm draw.io architecture figure** is finalized and inserted into Overleaf LaTeX

### Paper Structural Fixes (high priority)
4. Move System Overview out of Introduction → new `\section{System Architecture}` after Related Work
5. Move architecture figure to top of System Architecture section
6. Add standalone `\section{Motivation}` after Introduction
7. Convert module `\section{}` → `\subsection{}` under System Architecture
8. Create unified `\section{Experiments and Results}`
9. Add `\section*{Author Contributions}` after References
10. Fix abstract: "four modules" vs "three modules" inconsistency
11. Fix discrimination logic text (rice veto contradiction)
12. Add LiDAR-camera bridge implementation status table
13. Resolve ROS vs ROS 2 vs ROS Noetic inconsistency

### Technical Cleanup
14. Run same-distribution eval if v5 mAP still low — consider using Rice_Classification.v1i.voc (larger dataset)
15. Get actual measured ROI compute reduction (run with/without mask, time it)
16. Get GNSS drift number from Krish/Bilal (position error after 20-second outage)

### Career Prep
17. Update resume with v5 training results once available
18. Secure strong recommendation letter from professor (tie to paper quality)

---

*Document compiled 2026-06-01 from: project conversation history, GitHub Autonomous-tractor-system repo (master branch), Google Drive (41 files indexed), and project memory.*

---

## 19. Phase 10 — COCO Migration, RiceSEG Pretraining & V7 (July 2026)
*(Appended 2026-07-09. Detail lives in `COCO_MIGRATION_2026-07-07.md` and `RESEARCH_PLAN_DETECTION_ACCURACY.md`.)*

### Key dates
- Jul 2: v6 era begins (`weeddet_v6_checkpoints/` created)
- Jul 5: `weeddet_v6b.py` (F3 revert + `CocoWeedDataset`)
- Jul 7: **VOC → COCO migration.** `split_coco.py` produces `rice_detection_coco_split.zip` —
  dHash near-dup clustering, 80/10/10, seed 42 → 1079/134/134. The 0.166 v5GPT baseline is retired
  (leaky VOC-era split, not comparable).
- Jul 7–8: RiceSEG pretraining built (`riceseg_pretrain.py`, `run_pretrain.ipynb`) →
  `riceseg_backbone.pth` (94 MB, keys = `WeedDet.backbone.*`). Rationale: ImageNet fills only ~92%
  of Det-ResNet-50; in-domain seg pretraining fills all of it incl. the custom stem.
- Jul 8: V7 riceseg run #1 (100 ep, COCO split) → AP@50 0.0088. FAIL.
- Jul 8 night: **diagnosis.** Probe shows boxes localize well (median best-IoU 0.69) but scores cap
  at ~0.06–0.21 → classification-target bug: VFL positives were targeted at anchor-GT assignment IoU
  (tiny for these small boxes). Hard-target patch (positives → 1.0) validated on overfit-16: AP@50 0.60.
- Jul 9: fix baked into `weeddet_v6b.py` as `WeedDetLoss.cls_hard_target = True`; re-uploaded to Drive.
  V7 riceseg run #2 started 10:15 UTC — killed ~ep 61: AP@50 ~0.000, det/img frozen at 239.6.
  Fix works in overfit config but not full run; suspects: EMA-evaluated model, AMP, GRAD_CLIP 0.5
  (hard targets ≈4× loss magnitude). Both EMA and raw weights are saved in `_best.pth` for comparison.
  Run #2 overwrote run #1's checkpoints (filename collision — add run tags).

### V7 protocol (fixed)
Fresh runs only. `BACKBONE_INIT ∈ {scratch, imagenet, riceseg}`. IMG_SIZE 512, 1-class rice
(weed cat: 285/45/28 boxes — too sparse), anchor_base_scale 3, LSC k=7, ATSS, SGD 0.001 cosine,
100 ep, EMA 0.999, AMP, clip 0.5, seed 42. Selection: val AP@50 every 2 ep.
Eval: pycocotools, score_thr 0.01, hard NMS 0.5, maxDets 300. Test split exactly once at the end.

### Open items (2026-07-09)
1. EMA vs raw eval of run #2 best + prediction visualization
2. De-EMA/de-AMP/unclipped 20-ep bisection run if raw is also dead
3. **Scratch control on COCO split** (the real baseline) → imagenet ablation → test eval once → YOLOv8s
