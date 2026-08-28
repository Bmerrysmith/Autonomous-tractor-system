# reports/metrics

Model-free measurements committed to version control. Everything here is small,
reproducible from a COCO split alone, and cheap enough to regenerate on demand.

## Which dataset each file describes

Two RICE exports exist and they are **not** interchangeable. Every number here is
labelled with the one it came from.

| export | train | valid | test | notes |
|---|---|---|---|---|
| `agrinav_intake_2026-07-21/deliverable/detection/RICE/` | 1798 / 56502 | 518 / 16587 | 263 / 8115 | source tree. Carries SAM segmentation. Its internal split is **not grouped** — effectively per-frame random, so val/test leak against train. |
| `RICE_phase2_rebuild_2026-07-29/` | 1800 / 59691 | 518 / 15226 | 261 / 6284 | **what the model trains on.** Grouped by capture-series family, EXIF-normalised, box-only. |

The rebuild is the one that matters. It is produced by applying
`grouped_split.json` to the intake tree; see `src/agrinav/data/build_rice_phase2.py`.

## Files

| file | dataset | base_scale | notes |
|---|---|---|---|
| `anchor_audit_rice_train.json` | intake (ungrouped) | 3 | **superseded.** Kept for lineage; do not cite. 56499 filtered boxes. |
| `anchor_audit_rice_valid.json` | intake (ungrouped) | 3 | **superseded.** Kept for lineage; do not cite. |
| `anchor_audit_rebuild_train_base3.json` | rebuild | 3 | the shipped anchor ladder, on the data actually trained on |
| `anchor_audit_rebuild_valid_base3.json` | rebuild | 3 | as above, validation split |
| `anchor_audit_rebuild_train_base4.json` | rebuild | 4.0 | the current config (`detector_rice_phase2.yaml`) |
| `anchor_audit_rebuild_valid_base4.json` | rebuild | 4.0 | as above, validation split |

Headline numbers, mean anchor-to-GT grid IoU and the fraction of boxes with no
anchor above IoU 0.5:

| split | base 3 | base 4.0 |
|---|---|---|
| train | 0.7357 / 0.0051 | **0.7633 / 0.0056** |
| valid | 0.7349 / 0.0051 | **0.7600 / 0.0083** |

Raising the base scale improves mean coverage by ~0.028 at essentially no cost in
uncovered boxes. See ADR 0004 and the `anchor_base_scale` comment in
`configs/training/detector_rice_phase2.yaml` for why 4.0 rather than 4.5.

## Regenerating

```bash
agrinav data-anchor-audit \
  --ann-file <rebuild>/annotations/instances_train.coco.json \
  --base-scale 4.0 \
  --out reports/metrics/anchor_audit_rebuild_train_base4.json
```

`--base-scale` accepts floats. It was typed `int` until 2026-08-28, which meant
the audit tool could not audit the config it exists to validate.

## What is deliberately not here

No per-run training metrics. Every reported AP in this project still lacks a
committed, self-describing run record tying it to a config hash, a git SHA, a
split name and a checkpoint checksum. That gap is real and is tracked separately;
do not treat the absence of run JSON here as an oversight in this directory.
