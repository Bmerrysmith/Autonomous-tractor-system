# START HERE — AgriNav WeedDet Perception Research

**What this is:** a research project to build a rice/weed **plant classifier + localizer** (perception only). It is **not** a complete autonomous tractor and **not** wired to any sprayer. Read this file first, then follow the order below.

> ⚠️ **Safety stop conditions (never violate):**
> - No model output here may drive an actuator/sprayer.
> - "Not detected as rice" is **not** a weed and **not** permission to treat.
> - The historical `inference/inference_rice.py` is disabled/unsafe — do not use it.

---

## Where things are (the map)

| Area | Path | Status |
|---|---|---|
| **Pretraining code** (first training) | `src/agrinav/training/riceseg_pretrain.py` | ✅ fixed + tested |
| **Colab pretraining notebook** | `notebooks/riceseg_pretrain_colab.ipynb` | ✅ run this on GPU |
| **Go/no-go verdict (authoritative)** | `docs/GATE_STATUS.md` | ✅ the only place a gate is stated |
| **Detector model** | `src/agrinav/models/weeddet_v6b.py` | ⚠️ has open correctness bugs |
| **Data pipeline (masks→polygons, split)** | `src/agrinav/data/*.py` | ✅ built + tested |
| **Phase-2 dataset rebuild** | `src/agrinav/data/build_rice_phase2.py` | ✅ build + preflight + package |
| **Detector dataset card** | `docs/detector_dataset_card.md` | ⚠️ documents the RiceSEG-derived set, **not** curated RICE |
| **The split membership** | `data/manifests/detector_split_v1.json` | ✅ immutable record |
| **Results log** (paste run outputs here) | `docs/research/RICESEG_PRETRAIN_RESULTS.md` | fill after each run |
| **Authoritative audit + roadmap** | `docs/audits/2026-07-20/` | the source of truth |
| **Fix log (what's done)** | `docs/audits/2026-07-20/PHASE1_TRANSFER_LEARNING_FIXLOG_2026-07-20.md` | ✅ |
| **Data locations** | `docs/DATA_ORGANIZATION.md` | how data is arranged |
| **Historical / do-not-use code** | `_archive/` (see its README) | quarantine (recoverable) |
| **Consolidated data** (outside repo) | `~/agrinav_data/` (`archives/ RiceSEG/ derived/ out/`) | see `docs/DATA_ORGANIZATION.md` |

---

## Do this, in order

### 1. Pretrain the RiceSEG backbone (the first training batch)

**On GPU (recommended):** open `notebooks/riceseg_pretrain_colab.ipynb` in Colab, set Runtime→GPU, put `RiceSEG.zip` in `MyDrive/agrinav_data/`, run all cells. ~30–60 min. It writes `riceseg_backbone.pth` + `manifest.json` to Drive.

**Locally (CPU is slow):** first install the package — `pip install -e ".[train]"` (see README "Setup").
```powershell
python -m agrinav.training.riceseg_pretrain --self-test          # contract check
python -m agrinav.training.riceseg_pretrain --data-root <RiceSEG_folder> --overfit 8 --batch-size 4   # sanity gate
python -m agrinav.training.riceseg_pretrain --data-root <RiceSEG_folder> --epochs 30 --img-size 512 --out <out>/riceseg_backbone.pth
```
`<RiceSEG_folder>` is the folder that **contains** `global rice segmentation/`.

Then paste the printed metrics into `docs/research/RICESEG_PRETRAIN_RESULTS.md`.

### 2. Build the detector dataset

Two different datasets exist; do not confuse them.

- **RiceSEG-derived** (`docs/detector_dataset_card.md`): 245 images, 8,200 weed
  polygons from human masks, grouped split, sealed test. This is what that card
  documents.
- **Curated RICE, phase 2** (the one the detector actually trains on): 2,579
  images, 81,201 boxes, 2-class `rice_protect`/`weed_target`. Rebuild it with

  ```bash
  agrinav data-build-rice-phase2 build --source-root <deliverable>/detection/RICE --out-root <dir> --legacy-archive <old zip>
  ```

  Counts, integrity guarantees, and the burned-test warning are in
  `docs/GATE_STATUS.md`. Do **not** use the 2026-07-27 `RICE_curated_phase2.zip`:
  it mis-assigned 940 files and contained 231 intended sealed-test images.

### 3. Detector training

**Whether you may run it, and for what purpose, is answered in one place:
[`docs/GATE_STATUS.md`](docs/GATE_STATUS.md).** Short version as of 2026-07-29: a
pipeline shakedown on the rebuilt dataset is fine; a run whose numbers you intend
to quote is not, because checkpoint selection uses validation loss rather than AP
and the multiclass decode is still class-agnostic.

Notebook: `notebooks/weeddet_rice_phase2_colab.ipynb` (pinned commit, fail-closed
gates, archive integrity preflight).

---

## The one-line status

**Phase 1 is closed. The phase-2 detector *pipeline* runs end to end; its
*evaluation path* does not yet exist, so it cannot yet produce a defensible
number.** Current verdict, always: [`docs/GATE_STATUS.md`](docs/GATE_STATUS.md).
