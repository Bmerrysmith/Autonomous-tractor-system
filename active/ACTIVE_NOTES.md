# Active Notes — What To Do Next

**Last updated:** 2026-07-09
*(V5-era instructions removed — superseded by the COCO/V7 pipeline. See `../COCO_MIGRATION_2026-07-07.md`.)*

---

## Where things stand

V7 riceseg run #2 (with the VFL hard-target fix) was killed around epoch 61: AP@50 ~0.000 with
det/img frozen at 239.6 — the fix works in overfit-16 (AP@50 0.60) but not in the full-run config.
Prime suspects: EMA-evaluated model, AMP, GRAD_CLIP 0.5 (hard targets ≈4× loss magnitude).

## Do next, in order

1. **EMA vs raw check** — load `MyDrive/weeddet_v7_checkpoints/weeddet_v7_riceseg_best.pth`,
   run Cell 6 eval twice: once with `ckpt['state_dict']` (EMA), once with `ckpt['raw_state_dict']`.
   Also visualize predictions on 2–3 valid images (GT green / pred red).
2. **If raw is also dead:** 20-epoch riceseg run with `USE_EMA=False`, `USE_AMP=False`,
   `GRAD_CLIP=None` (overfit config scaled up). If that works, bisect back one flag at a time.
3. **Scratch control** — `BACKBONE_INIT='scratch'`, full protocol. This is the paper baseline on
   the new split. Run it regardless of 1–2.
4. `imagenet` ablation → test-split eval ONCE → YOLOv8s baseline.
5. Log every row in `../RESEARCH_PLAN_DETECTION_ACCURACY.md` Part C.

## Before the next run

- Add a run tag to checkpoint names — run #2 overwrote run #1's checkpoints.
- Delete `weeddet_v7_riceseg_epoch4..60.pth` (~1.6 GB) after the EMA/raw check.

## Live files (Colab/Drive, not in this folder)

| What | Where |
|---|---|
| V7 training notebook | Colab: `weeddet_trainingV7_coco.ipynb` (Drive ID `1sKq1EGt0eo9Ey3yyka3wabOoe25NNaBb`) |
| Model file | `MyDrive/weeddet_v2_checkpoints/weeddet_v6b.py` (repo copy synced 2026-07-09) |
| Dataset zip | `MyDrive/weeddet_v2_checkpoints/rice_detection_coco_split.zip` |
| RiceSEG backbone | `MyDrive/weeddet_v6_checkpoints/riceseg_backbone.pth` |
