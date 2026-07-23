# Google Drive artifact inventory (sanitized)

This repository previously listed exact Google Drive file/folder IDs, including unrelated personal
documents. Those locators were removed from the public recovery baseline. A Drive ID is not an
authentication secret, but it may expose a link-shared artifact and is unnecessary in source
control.

## Active logical artifacts

| Artifact | Expected private location | Publication state |
|---|---|---|
| Active WeedDet model source | `MyDrive/weeddet_v2_checkpoints/weeddet_v6b.py` | Canonical source is versioned in `models/weeddet_v6b.py`; verify hashes before Colab use |
| Detector data archive | `MyDrive/weeddet_v2_checkpoints/rice_detection_coco_split.zip` | Historical only; July 20 audit found capture-series leakage |
| RiceSEG backbone | `MyDrive/weeddet_v6_checkpoints/riceseg_backbone.pth` | Private artifact; external hash/release manifest still required |
| T7 checkpoints and curves | `MyDrive/weeddet_v7_checkpoints/` | Private historical artifacts; not a release |
| T7 notebook | `weeddet_trainingV7_1_T7.ipynb` | Private historical artifact; mutable notebook output is not the run identity |
| RiceSEG pretraining bundle | `MyDrive/riceseg_pretraining/` | Private historical artifacts; audit found parser/gate defects |

## Rules

- Do not publish or infer private Drive IDs from this file.
- Do not use mutable Drive paths as model, dataset, or run identifiers.
- Before using a Drive artifact, record its content hash, source, license/access terms, producing
  code/config, and expected consumer.
- Keep unrelated personal documents out of project inventories.
- Prefer an immutable, versioned artifact manifest and release bundle as described by the July 20
  deployment roadmap.

Public-safe container hashes and cleanup dispositions are recorded in
`../docs/ARTIFACT_INVENTORY.md`.
