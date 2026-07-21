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
| **Pretraining code** (first training) | `training/riceseg_pretrain.py` | ✅ fixed + tested |
| **Colab pretraining notebook** | `notebooks/riceseg_pretrain_colab.ipynb` | ✅ run this on GPU |
| **Detector model** | `models/weeddet_v6b.py` | ⚠️ has open correctness bugs |
| **Data pipeline (masks→polygons, split)** | `scripts/*.py` | ✅ built + tested |
| **Detector dataset card** | `docs/detector_dataset_card.md` | ✅ read before detector training |
| **The split membership** | `data/manifests/detector_split_v1.json` | ✅ immutable record |
| **Results log** (paste run outputs here) | `docs/research/RICESEG_PRETRAIN_RESULTS.md` | fill after each run |
| **Authoritative audit + roadmap** | `docs/audits/2026-07-20/` | the source of truth |
| **Fix log (what's done)** | `docs/audits/2026-07-20/PHASE1_TRANSFER_LEARNING_FIXLOG_2026-07-20.md` | ✅ |
| **Data locations** | `docs/DATA_ORGANIZATION.md` | how data is arranged |
| **Historical / do-not-use code** | `archive/`, old notebooks | quarantine |

---

## Do this, in order

### 1. Pretrain the RiceSEG backbone (the first training batch)

**On GPU (recommended):** open `notebooks/riceseg_pretrain_colab.ipynb` in Colab, set Runtime→GPU, put `RiceSEG.zip` in `MyDrive/agrinav_data/`, run all cells. ~30–60 min. It writes `riceseg_backbone.pth` + `manifest.json` to Drive.

**Locally (CPU is slow):**
```powershell
python training/riceseg_pretrain.py --self-test          # contract check
python training/riceseg_pretrain.py --data-root <RiceSEG_folder> --overfit 8 --batch-size 4   # sanity gate
python training/riceseg_pretrain.py --data-root <RiceSEG_folder> --epochs 30 --img-size 512 --out <out>/riceseg_backbone.pth
```
`<RiceSEG_folder>` is the folder that **contains** `global rice segmentation/`.

Then paste the printed metrics into `docs/research/RICESEG_PRETRAIN_RESULTS.md`.

### 2. Build the detector dataset (already done; regenerate if needed)

See `docs/detector_dataset_card.md`. It produced a **weed-rich, leakage-free** instance-polygon dataset from the RiceSEG human masks (8,200 real weed polygons) with a grouped train/val/test split (sealed test).

### 3. (Later) Detector training — blocked until code fixes

Do **not** train the detector yet. `models/weeddet_v6b.py` still has the audit's open bugs (translation-augmentation label corruption, ATSS, "Varifocal Loss" misnaming, evaluator). Fix those (audit §5–6, roadmap Gate 4) before any detector run.

---

## The one-line status

**Pretraining is fixed and runnable. The detector *data* is fixed. The detector *code* is not — that's the next real gate.**
