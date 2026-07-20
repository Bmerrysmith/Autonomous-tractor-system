# AgriNav — Autonomous Paddy Field Tractor · Project Tracker

> **Historical project tracker — superseded 2026-07-20.** This file preserves earlier project
> decisions; it is not current safety or deployment guidance. The complement-of-rice actuation
> concept recorded below is rejected and must not be implemented. See `README.md`,
> `active/ACTIVE_NOTES.md`, and the July 20 audit for the current fail-closed status.

**Course:** CEN 4930 — Introduction to Autonomous Driving Systems, Spring 2026<br>
**Institution:** U.A. Whitaker College of Engineering, Florida Gulf Coast University<br>
**GitHub:** `github.com/Bmerrysmith/Autonomous-tractor-system` (public, `master` branch)
**Last updated:** 2026-07-09 *(V7/COCO era — see `COCO_MIGRATION_2026-07-07.md` for the live plan; sections below marked v5-era are historical)*

---

## Team

| Member | Module |
|---|---|
| **Benny Merryman-Smith** | Perception stack — data pipeline, WeedDet object detection, discrimination/spray module |
| Bilal Dogutas | Lane detection (LiDAR/EKF) |
| Tony Nguyen | Lightweight CNN-FPN detection pipeline integration |
| Krish Shah | Localization, row following, GNSS/EKF |
| ~~Tony Raphael~~ | ~~Mission controller/path planner~~ — *removed; contribution not completed* |

---

## Historical Core Design Concept (Rejected)

The project previously proposed an **inverted spray approach**: detect rice as a protected class
and treat everything outside high-confidence detections as a spray target. This is unsafe because
a missed rice detection would become positive spray evidence. No actuation path may use this logic;
the inference prototype is disabled until an independently validated, fail-closed design exists.

> Historical framing only: false negatives on rice create crop-damage risk, not permission to spray.

---

## WeedDet Model

| Parameter | Value |
|---|---|
| Backbone | Det-ResNet-50 |
| Neck | eFPN |
| Head | ERetinaHead w/ Large Separable Convolution (LSC k=7) |
| Loss | VariFocal + GIoU + SmoothL1 |
| Anchor assignment | IACS |
| `iou_threshold` | 0.4 |
| `neg_iou_threshold` | 0.3 |
| `anchor_base_scale` | 4 |
| `nms_thr` | 0.3 |
| Gradient clip | `max_norm=5.0` |
| Best checkpoint val_loss | **0.7437** @ epoch 60 (Kaggle) / epoch 40+ (Colab v3 w/ 4k dataset); v4 best TBD (epoch 60 complete, loss ~0.79 at epoch 30, trending down) |
| `anchor_base_scale` *(v4)* | **6** (changed from 4 — v4 checkpoints incompatible with prior runs) |

**Qualitative inference results:** confidence scores 0.55–0.84 on paddy, aerial, and post-flood imagery.

### Training Run Log

| Run | Platform | Epochs | Val loss | mAP AP@0.5 | mAP AP@0.75 | Notes |
|---|---|---|---|---|---|---|
| Kaggle v1 | Kaggle | 60 | 0.7437 | — | — | Best prior checkpoint |
| Colab v3 | Colab | 40+ | 0.7437 | — | — | Extended run w/ 4k dataset added |
| **Colab v4** | **Colab** | **60** | **TBD** | **TBD** | **TBD** | ✅ Training complete. Loss 1.0448→~0.79 (epoch 30), full 60-epoch best in `weeddet_best.pth`. 0 val pairs (no val split in dataset). **Visual check broken — see bugs below.** |
| v5GPT | Colab | — | — | **0.166** | — | Scratch baseline, VOC-era split (leaky — retired as comparison) |
| v6b imagenet | Colab | — | — | 0.017 | — | ImageNet + apply_bn_policy = known-bad F1+F2 combo |
| V7 riceseg #1 | Colab | 100 | — | 0.0088 | 0.000 | Jul 8, COCO split, soft VFL targets — INVALID (VFL target bug) |
| V7 riceseg #2 | Colab | ~61 (killed) | — | ~0.000 | 0.000 | Jul 9, hard-target fix in v6b.py; det/img frozen — EMA/grad-clip suspect |
| V7 scratch | Colab | — | — | **PENDING** | — | Real baseline on COCO split — run next |

---

## Datasets

| Dataset | Images | Format |
|---|---|---|
| Rice_Classification.v1i.voc (primary) | ~4,041 | VOC XML → COCO |
| rice_detection_for_export.v1i.voc (supplementary) | ~1,347 | VOC XML → COCO |

- Annotations sourced via Roboflow.
- `category_id=1` for rice; `category_id=0` reserved for background (RetinaNet convention).
- Train/val split: 80/20, random seed 42.

---

## Training Platforms

- **Google Colab** — Drive-backed checkpoints at `/content/drive/MyDrive/`
- **Kaggle** — GPU notebook; outputs to `/kaggle/working/` (input dirs are read-only)
- **Local** — VSCode on Windows, RTX 4060; PowerShell for Git

---

## Paper Status

**Format:** IEEE conference, LaTeX on Overleaf.<br>
**Stage:** Two rounds of professor feedback completed; iterative LaTeX revisions applied.

### Completed revisions
- [x] Removed all mission controller / control unit references and Anthony Raphael's contributions
- [x] Consolidated structure per feedback (Technical Contributions → end of Introduction; System Overview as its own section; Individual Author Contributions after bibliography; single Motivation section)
- [x] Fixed broken LaTeX syntax in author contributions block
- [x] Removed GCN section and related-work subsection
- [x] Replaced AI-generated architecture figure with draw.io version *(required — confirm final status)*
- [x] Removed AI-tone markers from bookend sections

### ⚠️ Open items (blocking submission)
- [ ] **Real same-distribution mAP** — AP@0.5 and AP@0.75 required by professor. Current cross-dataset mAP is ~0% due to annotation-style domain gap (not a code defect); same-distribution evaluation is the correct metric to report. **Next step: fix Cell 4 bugs (see below), then run eval + qualitative visual check on Colab v4 checkpoint.**
- [ ] **Confirm draw.io architecture figure** is finalized and inserted into the LaTeX file.
- [ ] **Final structural check** — verify all revisions are fully applied in the submitted `.tex` file.

---

## Known Bugs & Hard-Won Lessons

| Issue | Root cause | Fix |
|---|---|---|
| Loss reads 0.0 but training looks normal | Bounding box coordinates never rescaled after image resize → zero IoU → zero positive anchors | Always verify box coords are within image bounds and IoU is non-zero before trusting loss |
| Double-scaling bug | Dataset class applied coordinate transforms; outer wrapper applied same transforms again, squaring the error silently | Don't wrap an already-transforming dataset with the same transform |
| Checkpoint incompatibility after anchor scale change | Anchor scale changes alter head weight shapes | Requires fresh training from scratch |
| Cross-dataset mAP collapse (~0%) | Annotation-style domain gap between training and evaluation sets | Evaluate on same-distribution data |
| Kaggle write errors | Input directories are read-only | Copy all files to `/kaggle/working/` before writing |
| GitHub access in Colab | Historical authentication failures | Use current GitHub-supported, least-privilege, short-lived authentication; never place credentials in notebooks or repository files |
| `category_id=0` treated as background | RetinaNet convention reserves 0 for background | Use `category_id=1` for rice class |
| **v4 visual check always fails** | Cell 4 looks for `weeddet_v4_best.pth` but training saved `weeddet_best.pth` | Change `V4_CKPT` to `f'{CKPT_DIR_V4}/weeddet_best.pth'` |
| **v4 visual check import error** | Cell 4 imports `weeddet_for_VSCode` but training used `weeddet_Latest` | Change import to `import weeddet_Latest as wd` |
| **v4 has no validation curve** | Dataset had no val split — `val.txt` was empty (0 pairs) | Add a val split or use a separate held-out set for proper evaluation |

---

## Future Work (Not Implemented)

- GCN severity classifier
- RRT* path planner with inverted weed-aware cost function
- LiDAR-camera fusion bridge
- Jetson AGX Orin deployment

---

## Key References

- Peng et al. (2022) — WeedDet
- Jiang et al. (2020) — GCN weed recognition
- Lu et al. (2021) — Improved CenterNet

---

## Career Notes

Target roles: computer vision, perception, or ML engineer.<br>
Target companies: Waymo, Aurora, NVIDIA, Bosch, Continental, Aptiv, defense contractors.<br>
The AgriNav paper and a strong recommendation letter from the professor (active IEEE reviewer) are strategically important for job applications and potential MS admission.
