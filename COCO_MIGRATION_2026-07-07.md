# COCO Migration & V7 Era — Action Log + Next Plan

**Covers:** 2026-07-02 → 2026-07-09
**Status:** ACTIVE — read this first.
*(Reconstructed 2026-07-09 from the Colab notebook history and Drive audit; the local folder had been stale since 2026-06-01.)*

---

## Why the migration

The v5GPT baseline (AP@50 0.166) was measured on a VOC-era random 80/20 split that likely leaked
near-duplicate frames between train and val. All V7 numbers use a new leakage-safe COCO split and are
**not directly comparable to 0.166**. The scratch run on the new split is the real baseline going forward.

## What was done

1. **VOC → COCO switch** (2026-07-07). `data/split_coco.py` (run locally) produced
   `rice_detection_coco_split.zip`: dHash near-duplicate clustering, 80/10/10, seed 42 →
   **train 1079 / valid 134 / test 134**. Boxes per category: train {1: 31013, 2: 285},
   valid {1: 3669, 2: 45}, test {1: 3706, 2: 28}. Category 2 (weed) is too sparse — training 1-class rice.
   ⚠️ `split_coco.py` and the zip are NOT in this repo folder — zip is on Drive in `weeddet_v2_checkpoints/`.
2. **`models/weeddet_v6b.py`** updated as the single model file (F3 revert + `CocoWeedDataset` baked in);
   uploaded to `MyDrive/weeddet_v2_checkpoints/`. No runtime regeneration from v6.
3. **RiceSEG backbone pretraining** (`training/riceseg_pretrain.py` + `run_pretrain.ipynb` on Drive).
   Exports `riceseg_backbone.pth` (94 MB, keys = `WeedDet.backbone.*`) → `MyDrive/weeddet_v6_checkpoints/`.
   Motivation: ImageNet only fills ~92% of Det-ResNet-50 (custom stem stays random); in-domain seg
   pretraining fills everything. NEVER `apply_bn_policy` on a riceseg backbone.
4. **V7 training notebook** `weeddet_trainingV7_coco.ipynb` (Colab): fresh runs only,
   `BACKBONE_INIT ∈ {scratch, imagenet, riceseg}`, AP@50-on-valid checkpoint selection,
   fixed eval protocol (score_thr 0.01, hard NMS 0.5, maxDets 300), EMA + AMP, SGD cosine.

## Run log (V7, riceseg init)

| Run | Date | Config | Result |
|---|---|---|---|
| riceseg #1 | Jul 8, 100 ep | soft VFL targets (anchor-GT IoU) | **AP@50 0.0088** (peak ep 20, then decays). FAIL |
| overfit-16 gate | Jul 8 | + VFL hard-target patch, lr 0.005, no EMA/AMP | **AP@50 0.60** by ep 70. Pipeline can learn ✓ |
| riceseg #2 | Jul 9, killed ~ep 61 | hard fix baked into v6b.py (`cls_hard_target=True`) | AP@50 ≤0.0010 → 0.0000; **det/img frozen at 239.6** for ~25 evals. FAIL |

### Diagnosis so far

- **Probe (Jul 8):** with score threshold ignored, predicted boxes cover GT well (median best-IoU 0.69,
  85% of GT >0.5) but scores are capped at 0.05–0.21 → **classification/confidence problem, not localization**.
- **Root cause #1 (fixed):** VFL classification target for positives was the anchor-GT assignment IoU
  (~0.06 for these tiny boxes), capping confidence. Fix: hard target 1.0 for positives
  (`WeedDetLoss.cls_hard_target = True` in `models/weeddet_v6b.py`; old behaviour kept behind the flag).
  Validated by overfit-16 (0.60).
- **Open problem:** the fix works in the overfit config but NOT in the full run. Differences:
  EMA-evaluated model, AMP, GRAD_CLIP 0.5 (hard targets ≈4× loss magnitude → clipping bites harder),
  lr 0.001 vs 0.005, augmentation. The frozen det/img strongly suggests the EMA eval model stopped evolving.
  Both EMA (`state_dict`) and raw (`raw_state_dict`) weights are saved in `_best.pth` — compare them first.

## Next plan (in order)

1. Eval `raw_state_dict` vs `state_dict` (EMA) from `weeddet_v7_riceseg_best.pth` on valid — one cell.
2. If raw is also dead: 20-epoch run with `USE_EMA=False, USE_AMP=False`, `GRAD_CLIP` off or 10.0
   (= overfit config scaled up), then bisect back toward the full config.
3. **Scratch control on the COCO split** — the actual baseline. Then `imagenet` ablation.
4. Overfit-10 A/B gate (V6b) if anything still disagrees.
5. Test-split AP@0.5/0.75 exactly ONCE at the end + YOLOv8s baseline for the paper.
6. Log every result row in `RESEARCH_PLAN_DETECTION_ACCURACY.md` Part C.

## Housekeeping

- ⚠️ Checkpoint filenames collide across runs (`weeddet_v7_riceseg_best.pth` etc.) — run #2 overwrote
  run #1's best. Add a run tag (date) to `CKPT_DIR` or filenames before the next run.
- Delete `weeddet_v7_riceseg_epoch4..60.pth` (~1.6 GB, failed run) after triage.
- Duplicate `RiceSEG.zip` (231 MB) sits at Drive root — the canonical copy is in `riceseg_pretraining/`.
