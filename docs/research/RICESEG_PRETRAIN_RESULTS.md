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

_(none yet — awaiting the first post-fix training run)_
