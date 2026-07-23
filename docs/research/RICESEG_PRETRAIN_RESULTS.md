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

### Run: `2026-07-21 ImageNet→RiceSEG (production, Colab T4)`

**Condition:** `ImageNet→RiceSEG`

| Field | Value |
|---|---|
| git_commit | 77c438e |
| seed | 42 |
| environment | py 3.12.13 / torch 2.11.0+cu128 / tv 0.26.0+cu128 / numpy 2.0.2 |
| device | cuda (Tesla T4) |
| tiles / train / val | 3078 / 2769 / 309 (group-aware, 0 source overlap) |
| countries | China, India, Japan, Philippines, Tanzania |
| imagenet_coverage | 288 / 342 (0 missed) |
| backbone sha256 | 16ff484c3e772ef366cfaecede5563d6e789a128b90c357b5d99b1907551cc62 |

**Result**

| Field | Value |
|---|---|
| best epoch | 19 |
| best mIoU (present) | 0.5816 |
| background / green_veg | 0.866 / 0.869 |
| senescent / panicle | 0.337 / 0.738 |
| **weeds / duckweed** | **0.337 / 0.343** |

**Notes**

- mIoU 0.58 is driven by background + green_veg (~0.87). The safety-critical
  classes are weak: weeds/duckweed/senescent all ~0.34.
- **weeds is highly unstable** across epochs: 0.34 (ep19) then 0.12, 0.19, 0.11
  on nearby epochs while loss falls smoothly. Best-epoch weeds is partly a noisy
  peak; the last-5-epoch stable mean is **0.202 ± 0.053**.

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
| git_commit | `917991b2b94291b8100e54af0d3c0ef860a3bfbd` |
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
- Consistent with the cross-model comparison below (DeepLabV3 stable mIoU 0.615,
  stable weeds 0.483); this entry adds the full per-class breakdown for that run.

---

## Baseline architecture control (2026-07-21)

**Question.** Weeds IoU 0.34 (unstable) admits two explanations with opposite
fixes: (a) the task is *data-limited* — RiceSEG lacks weed pixels, so any model
plateaus → go annotate; or (b) the *custom Det-ResNet-50 is the limit* → change
the model before building the detector on it. `training/baseline_seg_control.py`
runs the discriminating experiment: stock pretrained segmenters on the **same
split, weights, loss, seed, and metric** (imported from `riceseg_pretrain`, not
reimplemented). Only the model varies. Colab T4, 30 epochs each.

**Stable weeds IoU (mean ± std over last 5 epochs — the honest comparison):**

| Model | stable mIoU | **stable weeds IoU** | best-epoch weeds |
|---|---|---|---|
| custom Det-ResNet-50 (ImageNet) | 0.566 ± 0.009 | **0.202 ± 0.053** | 0.337 |
| DeepLabV3-ResNet50 (COCO/VOC) | 0.615 ± 0.002 | **0.483 ± 0.006** | 0.493 |
| SegFormer-B2 (ADE20K) | 0.645 ± 0.001 | **0.522 ± 0.002** | 0.523 |

**Verdict: (b) architecture-limited — strongly, and by two independent models.**

- DeepLabV3 beats custom on stable weeds by **+0.281**; SegFormer by **+0.320**.
  Both are **5–6× the run-to-run noise band** (custom weeds std 0.053). This is
  not a marginal or seed-sensitive result.
- Both baselines are also **far more stable** (weeds std 0.006 / 0.002 vs
  custom's 0.053). The weed instability flagged above is a property of the custom
  architecture, **not** inherent task noise — a model that fits weeds well does
  not oscillate.
- The `lr=3e-4` fairness worry (possibly too high for fully-pretrained models)
  is resolved: the baselines trained cleanly at that LR and still won decisively.

**This overturns the working "data-limited" assumption** (including my own prior
guidance). RiceSEG has enough weed signal for a stock segmenter to reach ~0.52
weeds IoU. The custom backbone is leaving ~0.3 IoU on the table on the single
most safety-critical class.

**Honest limits of this experiment.**

1. **Backbone + decoder, not pure backbone.** The custom run pairs Det-ResNet-50
   with a light seg decoder; DeepLabV3 adds ASPP and SegFormer an MLP decoder.
   Part of the gap is the decoder head, not solely the backbone. It does not
   cleanly isolate the two.
2. **One seed each.** The delta is 5–6× noise so direction is not in doubt, but
   rerun `--seed 1 / --seed 2` before any irreversible architecture change.
3. **Segmentation proxy.** The detector uses an FPN + ATSS/VFL head, not this
   seg decoder. A weak seg result is a strong warning, not a detection metric.

**Actionable implication.** DeepLabV3's backbone *is* a standard torchvision
ResNet-50 — which produces the same C3/C4/C5 the WeedDet FPN consumes and loads
342/342 ImageNet tensors (vs the custom stem's 288/342). The lowest-risk next
step is to pretrain/train the detector on a **standard ResNet-50 backbone**,
keeping the existing FPN/ATSS/VFL detector design intact, rather than the custom
Det-ResNet-50 stem. That is a backbone swap, not a detector rewrite. (A ViT like
SegFormer/DINOv2 scores highest here but has no C3/C4/C5 pyramid and would need a
new neck — a larger change to weigh separately.)

Data work from [`../EXTERNAL_DATASETS_AUDIT.md`](../EXTERNAL_DATASETS_AUDIT.md)
(WeedBD V3 species set, LFS-cloned `WeedDataset` Barnyard_Grass) is still worth
doing — but it is now the *second* lever, not the first.

**Manifests:** `MyDrive/agrinav_data/out/baseline_deeplabv3_resnet50.json`,
`baseline_segformer_b2.json` (git commit, env, model id/revision, full per-epoch
history, stability block each).

---

## Training knobs added 2026-07-23 (seg pipeline diagnosis, items 1–3)

`riceseg_pretrain.py` gained optimisation levers. **Defaults reproduce the old
recipe except checkpoint selection.** Recommended optimised command:

```
python -m agrinav.training.riceseg_pretrain \
  --data-root <RiceSEG> --out out/riceseg_backbone.pth \
  --epochs 30 --batch-size 8 --img-size 512 --seed 42 \
  --loss focal_tversky --warmup-epochs 2 --backbone-lr-mult 0.1 \
  --select-metric minority
```

- **Checkpoint selection (default changed).** Exports by an EMA-smoothed mean IoU
  over `--select-classes weeds,duckweed,senescent` instead of single-epoch mIoU
  (which was dominated by background/green_veg ~0.87 and picked noisy weed
  peaks). `--select-metric miou` reproduces prior runs; the overfit gate always
  uses `miou`. New `selection`/`stability` blocks appear in the run manifest.
- **Loss (opt-in).** `--loss focal_tversky` (α0.3/β0.7/γ1.33, background excluded)
  for minority recall; or `--dice-weighted --dice-ignore-bg` on the CE+Dice loss.
  `baseline_seg_control` keeps plain `SegLoss` so the A/B stays comparable.
- **LR (opt-in).** `--warmup-epochs` linear warmup + `--backbone-lr-mult` give the
  backbone a smaller step than the fresh head; damps the early weed oscillation.
- **Recipe parity.** New-API AMP + `--num-workers`/persistent workers now match
  `baseline_seg_control`. **Batch size remains a confound** in the recorded A/B
  (custom bs 12 vs DeepLabV3 bs 8) — match it (both 8) before citing the delta.
- **Ceiling caveat.** These are model-side levers for **weeds** + stability.
  duckweed/senescent ~0.36 for *both* architectures is a **data** ceiling — needs
  more real minority instances, not loss/LR tricks.

Smoke-tested on synthetic CPU tiles (both loss paths + both selection metrics run
end-to-end; manifest + resumable checkpoint written). No full RiceSEG re-run yet.

---
