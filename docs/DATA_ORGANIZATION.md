# Data Organization

**Rule:** data and large artifacts do **not** live in Git. Git holds code, small
manifests, and this guide. Bulk data lives in one folder outside OneDrive, and a
mirror of what Colab needs lives in Google Drive.

> The audit found OneDrive corrupted the Git index and truncated files. Keep both
> the working repo **and** the data folder out of any OneDrive-synced path.

---

## Local layout (the single data root)

Proposed canonical location: `C:\Users\<you>\agrinav_data\` (outside OneDrive).

```
agrinav_data/
├── archives/                     ← raw source archives, immutable (never edit)
│   ├── RiceSEG.zip                       (sha256 0071d9f9…)
│   ├── rice_detection_for_export.v1i.coco.zip   (sha256 9da9ce72…)
│   └── paddy-rice-imagery-DatasetNinja.tar
├── RiceSEG/                      ← extracted RiceSEG (pretraining reads this)
│   └── global rice segmentation/...
├── derived/                      ← regenerable, built by scripts/
│   └── detector_v1/
│       ├── riceseg_instances.coco.json
│       ├── paddy_panicle.coco.json
│       ├── coco_proposals_unreviewed.coco.json
│       └── split_v1/{train,val,test}.coco.json + split-v1.json
└── out/                          ← training outputs
    ├── riceseg_backbone.pth
    ├── riceseg_backbone.pth.fullckpt.pth
    └── riceseg_backbone.pth.manifest.json
```

Everything under `derived/` and `out/` is reproducible from `archives/` via the
scripts — safe to delete and rebuild. `archives/` and `RiceSEG/` are the inputs.

## Google Drive layout (for Colab)

```
MyDrive/agrinav_data/
├── RiceSEG.zip                   ← the Colab notebook extracts this
└── out/                          ← Colab writes the backbone + manifest here
```

The Colab notebook (`notebooks/riceseg_pretrain_colab.ipynb`) expects
`MyDrive/agrinav_data/RiceSEG.zip` (it will also search Drive if not there).

## Verify an archive before use

```powershell
Get-FileHash <archive> -Algorithm SHA256   # must match the registry
```
Registry of every dataset + hash + license: `data/dataset_registry.v1.json`.

## What is where right now (pre-cleanup)

- Source archives currently in `Downloads/`.
- `RiceSEG/` extracted at `agrinav_data/RiceSEG/` (a training run reads it now).
- Derived COCO/splits at `<repo>/artifacts/detector_v1/` (gitignored).
- 245-image curated subset tracked in-repo at `data/rice_training_curated/`
  (local-only; do not mix into training until licensing + dedup are recorded).

Moving `Downloads/` archives into `agrinav_data/archives/` is safe. Do **not**
move `agrinav_data/RiceSEG/` while a training run is reading it.
