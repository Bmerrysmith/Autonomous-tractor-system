# RiceSEG Pretraining — Results Log

This file records the outcome of each RiceSEG backbone pretraining run after the
Phase 1 transfer-learning fixes (see
[`PHASE1_TRANSFER_LEARNING_FIXLOG_2026-07-20.md`](../audits/2026-07-20/PHASE1_TRANSFER_LEARNING_FIXLOG_2026-07-20.md)).

**How to fill this in:** after a run, the script writes `.../riceseg_backbone.pth.manifest.json`.
Paste the console tail and attach (or summarize) the manifest below. Most fields
below map 1:1 to keys in that manifest, so you can copy them directly.

> Reminder (audit / ACTIVE_NOTES stop conditions): a good mIoU here is **not** a
> deployable or causal result. It is a sanity signal that the backbone learned
> in-domain features. Downstream detection benefit is a separate, later gate.

---

## Run template (copy per run)

### Run: `<run-id / date>`

**Command**

```
python -B training/riceseg_pretrain.py \
  --data-root <path-to-folder-containing 'global rice segmentation'> \
  --out <out-dir>/riceseg_backbone.pth \
  --epochs 30 --batch-size 12 --img-size 512 --seed 42
```

**Condition:** `ImageNet→RiceSEG` | `random→RiceSEG (--no-imagenet)`  *(delete one)*

**Provenance (from manifest.json)**

| Field | Value |
|---|---|
| git_commit | |
| seed | |
| environment (python/torch/torchvision/numpy) | |
| device | |
| data_root | |
| tiles / train / val | |
| countries | |
| holdout_country | |
| imagenet_coverage (loaded / backbone_total) | |
| backbone_export sha256 | |

**Result**

| Field | Value |
|---|---|
| best epoch | |
| best mIoU (present classes) | |
| absent classes at best epoch | |
| per-class IoU: background | |
| per-class IoU: green_veg | |
| per-class IoU: senescent | |
| per-class IoU: panicle | |
| per-class IoU: weeds | |
| per-class IoU: duckweed | |

**Console tail**

```
<paste the last ~15 lines here>
```

**Notes / observations**

- <e.g. weeds/duckweed IoU vs. audit §9.3 baseline (weeds ~0.23, duckweed ~0.35 at epoch 27)>
- <anything unexpected: absent classes, loss not decreasing, gate failures>

---

## Reference: last recorded pretraining (audit §9.3, pre-fix, for comparison only)

The supplied historical run (before these fixes; broken country parsing, no
gate) reported best val **mIoU 0.5749 at epoch 27**, with approximate class IoUs:
background 0.90, green_veg 0.86, senescent 0.36, panicle 0.75, weeds 0.23,
duckweed 0.35. Country holdout was **not** actually operating (all tiles were
mislabelled `"global rice segmentation"`), so any country-generalization reading
from that run is invalid. Use it only as a rough sanity anchor, not a baseline.

---

## Runs

<!-- Add each completed run above the reference section using the template. -->

### Run: `baseline_deeplabv3_resnet50` (2026-07-21) — CONTROL, not a backbone

> ⚠️ This is the **`baseline_seg_control.py` control experiment**, not a WeedDet
> backbone pretraining run. It trains stock **DeepLabV3-ResNet50** on the identical
> RiceSEG split/loss/schedule to test whether the custom Det-ResNet-50 architecture
> is the weed-IoU bottleneck. Do **not** load this as a backbone
> (`load_riceseg_backbone()`). Source: Drive `MyDrive/agrinav_data/out/baseline_deeplabv3_resnet50.json`.

**Condition:** `ImageNet backbone + COCO(VOC) seg head → RiceSEG` (segmentation head reinitialised)

**Provenance (from the run JSON)**

| Field | Value |
|---|---|
| git_commit | `917991b2b94291b8100e54af0d3c0ef860a3bfbd` — **⚠️ not present in this repo** (HEAD `0e65f17`; `baseline_seg_control.py` still uncommitted). Commit the run's code state so this resolves. |
| seed | 42 |
| environment (python/torch/torchvision/numpy) | 3.12.13 / 2.11.0+cu128 / 0.26.0+cu128 / 2.0.2 |
| device | cuda |
| data_root | `/content/RiceSEG` |
| tiles / train / val | 3078 / 2769 / 309 |
| countries | China, India, Japan, Philippines, Tanzania |
| holdout_country | `null` — random **group-aware (by source photo, 0 source overlap)** 10% val ⇒ **in-distribution**, not an unseen-site test |
| config | arch deeplabv3_resnet50, epochs 30, batch 8, img 512, lr 3e-4, val_ratio 0.1, stability_window 5, params 41,996,364 |

**Result**

| Field | Value |
|---|---|
| best epoch | 29 |
| best mIoU (present classes) | 0.6170 |
| stable mIoU (epochs 26–30) | 0.6152 ± 0.0016 |
| absent classes at best epoch | none |
| per-class IoU: background | 0.8584 (stable 0.8586 ± 0.0008) |
| per-class IoU: green_veg | 0.8722 (stable 0.8716 ± 0.0005) |
| per-class IoU: senescent | 0.3591 (stable 0.3605 ± 0.0037) |
| per-class IoU: panicle | 0.7539 (stable 0.7510 ± 0.0021) |
| per-class IoU: weeds | 0.4928 (stable 0.4831 ± 0.0063) |
| per-class IoU: duckweed | 0.3657 (stable 0.3667 ± 0.0063) |

**Notes / observations**

- **Weed IoU roughly doubles vs the pre-fix custom anchor** (0.48–0.49 vs 0.23,
  §9.3 reference below) on the same data, while easy classes (green_veg, panicle)
  are near-identical → strong signal the custom Det-ResNet-50 / its recipe was a
  **major contributor to the weak weed IoU**, specifically for the hard minority
  class.
- **Not conclusive**, for three reasons: (1) the 0.23 comparison is the pre-fix
  historical run (invalid country parsing, no gate) — an *anchor, not a baseline*;
  no matched **post-fix custom** run has been logged yet, so architecture-vs-recipe
  is not cleanly isolated. (2) `holdout_country=null` ⇒ this is **in-distribution**;
  the acceptance target is weed IoU **≥0.75 on an unseen site**, which this does not
  measure and which would likely be lower. (3) Even the stock model's 0.49 is far
  below 0.75, and duckweed (0.37) / senescent (0.36) stay poor for **both** models
  ⇒ a real **data / class-imbalance ceiling** on the hard classes, not just
  architecture.
- Hygiene is otherwise good: seeded, env-pinned, group-aware split, 5-epoch
  stability window with tight std (weeds ±0.006 — a stable number, not a lucky
  epoch).
- **To make the architecture claim conclusive:** log a matched **post-fix custom
  RiceSEG** run on this same split, and ideally re-run both under a **country
  holdout** to measure against the real generalization target.

---
