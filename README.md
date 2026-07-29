# AgriNav / WeedDet Research Repository

**Author:** Benjamin Merryman-Smith | FGCU Whitaker College of Engineering
**Course:** CEN 4930 — Introduction to Autonomous Driving Systems, Spring 2026
**GitHub:** https://github.com/Bmerrysmith/Autonomous-tractor-system

---

> [!CAUTION]
> This repository is research in progress, not a complete autonomous-tractor or treatment system.
> The current detector and historical inference prototype are not validated for field actuation.
> No missing or uncertain detection is permission to spray; the legacy inference entry point is
> intentionally disabled. See the July 20 audit before running new experiments.

## Start here

0. `docs/GATE_STATUS.md` — **the authoritative go/no-go verdict.** Read it before spending GPU time
1. `docs/audits/2026-07-20/AGRINAV_FULL_DEEP_AUDIT_2026-07-20.md` — authoritative recovery and correctness audit
2. `docs/audits/2026-07-20/AGRINAV_DEPLOYMENT_ROADMAP_2026-07-20.md` — gated roadmap from research to deployment
3. `active/ACTIVE_NOTES.md` — current priorities and stop conditions
4. `TEST_LOG.md` — historical T0–T7c experiment record; reconcile it with the July 20 audit before reuse
5. `HANDOFF_2026-07-16.md` — historical session handoff, retained for provenance
6. `docs/research/PERCEPTION_RESEARCH_PACKAGE_2026-07-20.md` — perception-only success targets, comparable datasets, annotation stack, architecture decision, and controlled RiceSEG study
7. `docs/annotation_guide.md` — canonical crop-protection/weed-target mask rules and human-review gates
8. `docs/research/PERCEPTION_ACCEPTANCE_MATRIX_2026-07-20.md` — extracted requirements, evidence status, unresolved decisions, and exact go/no-go gates

## Setup

AgriNav is an installable package (`agrinav`, `src/` layout, Python 3.11+).

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows; use  . .venv/bin/activate  on macOS/Linux
pip install -e ".[dev]"           # runtime + training stack + lint/test tools
pre-commit install                # optional: run hooks on commit
```

Common commands (also wrapped by the `Makefile`):

```bash
agrinav --help                                          # unified CLI (data + training tools)
python -m agrinav.training.riceseg_pretrain --self-test  # contract check, no data needed
pytest                                                  # full test suite (CPU)
ruff check . && black --check .                          # lint + format
```

Install variants: `pip install -e ".[train]"` (runtime + torch/transformers, e.g. Colab —
this is what `requirements.txt` now installs); `pip install -e "."` (data-tooling base only,
no deep-learning stack). Dependencies live in `pyproject.toml`; see `docs/adr/` for the
`src/` layout and packaging/CI decisions.

## Folder structure

> The tree below is a 2026-07-20 recovery snapshot. The authoritative package
> layout is now `src/agrinav/` (see `docs/adr/0001-src-layout.md`); some historical
> root documents have since moved under `_archive/`.

```
agrinav_full/
├── README.md                          ← you are here
├── HANDOFF_2026-07-16.md              ← historical session handoff
├── TEST_LOG.md                        ← historical test protocol + verdicts
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
├── docs/
│   ├── ARTIFACT_INVENTORY.md           ← hashes and artifact dispositions
│   └── audits/2026-07-20/              ← deep audit + deployment roadmap
├── src/agrinav/                       ← installable package (import root: `agrinav`)
│   ├── cli.py                         ← unified `agrinav` CLI dispatcher
│   ├── models/weeddet_v6b.py          ← CURRENT model (held byte-stable; open Gate-4 bugs)
│   ├── training/                      ← riceseg_pretrain.py, baseline_seg_control.py
│   ├── data/                          ← annotation + dataset tooling (former scripts/*.py)
│   └── inference/                     ← disabled safety stub; no deployable runtime yet
├── data/                             ← data ARTIFACTS (not a package): manifests/, schemas/
│   └── rice_training_curated/         ← local-only RiceSEG-derived curation set
│       ├── README.md                   ← provenance, citation, and publication policy
│       ├── manifest.csv               ← per-image QA scores (versioned)
│       └── AUDIT_REPORT.md            ← curation method + site table (versioned)
│   (auto_annotate.py now lives in _archive/unsafe_inference/)
├── paper/                             ← IEEE paper revision docs
├── drive_links/GOOGLE_DRIVE_INVENTORY.md
└── archive/
    ├── v1_baseline_retinanet/         ← Kaggle baseline (+ its notebook)
    ├── v2_weeddet_initial/
    ├── v3_weeddet_94epoch/
    ├── v4_weeddet_60epoch_no_val/
    ├── v5_era/                        ← ALL v5-era code + notebooks (moved 2026-07-16)
    └── voc_era/                       ← VOC-era data scripts (coco_to_voc, step1/2)
```

## Quick status (2026-07-29)

**Go/no-go decisions live in one file: [`docs/GATE_STATUS.md`](docs/GATE_STATUS.md).**
This table summarizes; that file governs. Session narrative is in
[`docs/HANDOFF.md`](docs/HANDOFF.md).

| Item | Status |
|---|---|
| Phase 1 (RiceSEG segmentation pretraining) | Closed. Best mIoU 0.5827 @ ep30, reproduced within 0.001; backbone exported |
| Phase 2 (detector) pipeline | Runs end to end: 18/18 epochs on an A100 in ~24 min, with `status.json` / `metrics.jsonl` / atomic checkpoints |
| Phase 2 evaluation path | **Missing.** Selection is validation *loss*; multiclass decode is class-agnostic; no model-to-COCO adapter. No AP number is defensible yet |
| 2026-07-28 detector runs | **Voided** — trained on a contaminated archive with training-loss selection. See the `VOID.md` files in their Drive run directories |
| Dataset of record | Rebuilt 2026-07-29 from the source deliverable: 1,800 / 518 / 261 images, 81,201 boxes, EXIF-normalized, hashed, preflight-verified (`agrinav data-build-rice-phase2`) |
| Test split | The manifest's 261-image test split is **burned** (231 of it was trained on). A replacement must come from the never-trained-on pool |
| Historical detector result | T7 recorded val AP@50 0.1040 / AP@75 0.0116; not a deployable result or a defensible causal estimate |
| Safety | Historical spray-by-default inference is disabled; no actuation interface is present or approved |
| License | No project-wide code license has been selected; do not assume permission beyond explicitly attributed third-party material |

## Historical experiment pipeline

Do not spend additional GPU time on this pipeline until the July 20 audit's Phase 0–2 gates are closed.

1. `rice_detection_coco_split.zip` (private artifact) — historical 1079/134/134 split; the July 20 audit confirmed capture-series leakage, so rebuild it before further evaluation
2. `riceseg_backbone.pth` (Drive) — pretrained backbone, loads 342/342 tensors
3. `active/weeddet_trainingV7_1_T1.ipynb` — Cell 3 = single config; RUN_TAG per TEST_LOG; upload finished notebook for audit

## Google Drive (active)

| What | Path |
|---|---|
| Model mirror (repository file `models/weeddet_v6b.py` is canonical) | `MyDrive/weeddet_v2_checkpoints/weeddet_v6b.py` |
| Dataset zip | `MyDrive/weeddet_v2_checkpoints/rice_detection_coco_split.zip` |
| Pretrained backbone | `MyDrive/weeddet_v6_checkpoints/riceseg_backbone.pth` |
| Checkpoints + curves | `MyDrive/weeddet_v7_checkpoints/` |
| Colab notebook | `weeddet_trainingV7_1_T7.ipynb` (private Drive artifact; ID intentionally omitted) |

See `docs/ARTIFACT_INVENTORY.md` for public-safe hashes and artifact dispositions. The legacy
Drive inventory has been sanitized; private Drive IDs are intentionally not published.

## Local development

Keep the active Git checkout outside a OneDrive-synchronized folder. OneDrive previously damaged
the Git index and truncated or NUL-padded tracked documents. Use GitHub for source synchronization
and a content-addressed artifact store for datasets, notebooks with outputs, and checkpoints.
