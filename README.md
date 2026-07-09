# AgriNav — Project Root

**Author:** Benjamin Merryman-Smith | FGCU Whitaker College of Engineering
**Course:** CEN 4930 — Introduction to Autonomous Driving Systems, Spring 2026
**GitHub:** https://github.com/Bmerrysmith/Autonomous-tractor-system

---

## Start here

0. `TEST_LOG.md` — **one-test-at-a-time tracker with interaction analysis (CHECK BEFORE EVERY RUN)**
1. `COCO_MIGRATION_2026-07-07.md` — V7 action log + next plan (see also `AUDIT_2026-07-09_V7.md`)
2. `RESEARCH_PLAN_DETECTION_ACCURACY.md` — the detection-accuracy research plan + results table
3. `AgriNav_Project_Tracker.md` — team/course tracker
4. `MASTER_PROJECT_HISTORY.md` — full timeline (Phases 1–10)

## Folder structure

```
agrinav_full/
├── README.md                          ← you are here
├── COCO_MIGRATION_2026-07-07.md       ← action log + next plan (READ FIRST)
├── RESEARCH_PLAN_DETECTION_ACCURACY.md
├── AgriNav_Project_Tracker.md
├── MASTER_PROJECT_HISTORY.md
│
├── data/                              ← VOC-era scripts (split_coco.py NOT yet in repo — see below)
├── models/
│   ├── weeddet_v6b.py                 ← CURRENT model file (synced from Drive 2026-07-09, has cls_hard_target fix)
│   └── weeddet_for_VSCode.py          ← legacy
├── training/
│   ├── riceseg_pretrain.py            ← RiceSEG → WeedDet backbone pretraining (synced 2026-07-09)
│   └── colab_weeddet_train.py         ← legacy
├── active/
│   ├── ACTIVE_NOTES.md                ← what to do next
│   └── (V5-era notebooks — the live V7 notebook is on Colab, see Drive links below)
├── archive/                           ← v1–v4 iterations
├── paper/                             ← IEEE paper revision docs
└── drive_links/GOOGLE_DRIVE_INVENTORY.md
```

⚠️ **Not yet in this repo** (add when convenient): `data/split_coco.py`, `run_pretrain.ipynb`,
`weeddet_trainingV7_coco.ipynb` (live on Colab, changes with each run), `USING_RICESEG_BACKBONE.md`.

## Pipeline (all fresh-run)

1. `split_coco.py` → leakage-safe COCO train/valid/test (done → `rice_detection_coco_split.zip` on Drive)
2. `run_pretrain.ipynb` (Drive) → `riceseg_backbone.pth` (done, 94 MB)
3. `weeddet_trainingV7_coco.ipynb` (Colab), `BACKBONE_INIT ∈ {scratch, riceseg, imagenet}` → train + COCO eval

## Quick status (2026-07-09)

| Item | Status |
|---|---|
| COCO leakage-safe split (1079/134/134) | ✅ delivered |
| RiceSEG backbone pretrained + exported | ✅ `riceseg_backbone.pth` |
| VFL classification-target bug | ✅ diagnosed + fixed (`cls_hard_target=True`), validated on overfit-16 (AP@50 0.60) |
| V7 riceseg full runs | ❌ both failed (0.0088 pre-fix; ~0.000 post-fix with frozen det/img) — EMA/grad-clip under suspicion |
| Scratch control on COCO split | ⚠️ NOT RUN — blocking; this is the real baseline |
| imagenet ablation, test-split eval, YOLOv8s baseline | ⚠️ pending |
| Paper: same-distribution AP@0.5/0.75 | ⚠️ still the blocking item |

## Google Drive (active folders)

| Folder | Path | Contents |
|---|---|---|
| v2 | `MyDrive/weeddet_v2_checkpoints/` | `weeddet_v6b.py` + `rice_detection_coco_split.zip` |
| v6 | `MyDrive/weeddet_v6_checkpoints/` | `riceseg_backbone.pth` |
| v7 | `MyDrive/weeddet_v7_checkpoints/` | V7 training checkpoints + curves |
| pretrain | `MyDrive/riceseg_pretraining/` | `riceseg_pretrain.py`, `RiceSEG.zip`, `run_pretrain.ipynb` |

See `drive_links/GOOGLE_DRIVE_INVENTORY.md` for IDs.
