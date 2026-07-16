# AgriNav — Project Root

**Author:** Benjamin Merryman-Smith | FGCU Whitaker College of Engineering
**Course:** CEN 4930 — Introduction to Autonomous Driving Systems, Spring 2026
**GitHub:** https://github.com/Bmerrysmith/Autonomous-tractor-system

---

## Start here

1. `HANDOFF_2026-07-16.md` — full session handoff: everything done, known issues, next steps
2. `TEST_LOG.md` — one-test-at-a-time tracker with verdicts T0–T7c (CHECK BEFORE EVERY RUN)
3. `AUDIT_2026-07-09_V7.md` — pipeline audit + cleared-suspects list (don't re-investigate those)
4. `RESEARCH_FORGETTING_DEFENSES.md` — literature recipe behind the frozen-backbone approach
5. `RESEARCH_PLAN_DETECTION_ACCURACY.md` — protocol + Part C results table

## Folder structure (cleaned 2026-07-16)

```
agrinav_full/
├── README.md                          ← you are here
├── HANDOFF_2026-07-16.md              ← session handoff (READ FIRST)
├── TEST_LOG.md                        ← test protocol + verdicts
├── AUDIT_2026-07-09_V7.md             ← pipeline audit
├── RESEARCH_FORGETTING_DEFENSES.md    ← forgetting-defense literature note
├── RESEARCH_PLAN_DETECTION_ACCURACY.md
├── COCO_MIGRATION_2026-07-07.md       ← V7-era action log
├── AgriNav_Project_Tracker.md         ← team/course tracker
├── MASTER_PROJECT_HISTORY.md          ← full timeline (Phases 1–10)
│
├── active/
│   ├── weeddet_trainingV7_1_T1.ipynb  ← THE training notebook (currently configured T7d)
│   └── ACTIVE_NOTES.md                ← what to do next
├── models/
│   └── weeddet_v6b.py                 ← CURRENT model (T1 fix: atss_all_neg; cls_hard_target)
├── training/
│   └── riceseg_pretrain.py            ← RiceSEG → WeedDet backbone pretraining
├── data/
│   ├── auto_annotate.py
│   ├── rice_training_curated/         ← 245 curated RiceSEG images (aerial 95 / front 150)
│   │   ├── manifest.csv               ← per-image QA scores
│   │   └── AUDIT_REPORT.md            ← curation method + site table
│   └── rice_training_curated*.zip     ← source archives (git-ignored; possible duplicates — verify & dedupe)
├── inference/                         ← inference_rice.py
├── paper/                             ← IEEE paper revision docs
├── drive_links/GOOGLE_DRIVE_INVENTORY.md
└── archive/
    ├── v1_baseline_retinanet/         ← Kaggle baseline (+ its notebook)
    ├── v2–v4 snapshots/
    ├── v5_era/                        ← ALL v5-era code + notebooks (moved 2026-07-16)
    └── voc_era/                       ← VOC-era data scripts (coco_to_voc, step1/2)
```

## Quick status (2026-07-16)

| Item | Status |
|---|---|
| **Best stable model** | **val AP@50 0.1040 / AP@75 0.0116** — T7: frozen riceseg backbone, 20-ep anneal (`weeddet_v7_riceseg_T7_best.pth` on Drive) |
| Scratch baseline (paper) | 0.0450 @ 20 ep (T6) → riceseg pretraining = 2.3× scratch, forgetting confirmed + defended |
| Settled config | atss_all_neg ✓ · cls_hard_target ✓ · clip 10.0 ✓ · anchors base 6 ✓ · FREEZE_BACKBONE ✓ · fast cosine anneal ✓ |
| Schedule question | CLOSED — long/flat schedules fail at any lr (T7b/T7c); fast anneal is the recipe |
| **Current ceiling** | FP flood (~300 det/img) from ATSS assignment geometry → **next: T10** (per-cell candidates) then T11 (soft quality targets) |
| Curated data | 245 RiceSEG images in `