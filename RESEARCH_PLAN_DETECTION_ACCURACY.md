# Research Plan — Detection Accuracy (V7, COCO split)

*(Reconstructed 2026-07-09 from notebook/Drive state — replace with the original if you have it.)*

## Question

Does in-domain RiceSEG segmentation pretraining of the Det-ResNet-50 backbone improve WeedDet
detection accuracy over (a) training from scratch and (b) ImageNet initialization?

## Part A — Fixed protocol

- Data: `rice_detection_coco_split.zip` — dHash leakage-safe 80/10/10, seed 42 (1079/134/134). Never re-split.
- Classes: 1-class `rice` (weed category too sparse: 285/45/28 boxes).
- Model: `models/weeddet_v6b.py` as-is. IMG_SIZE 512, anchor_base_scale 3, LSC k=7, ATSS, `cls_hard_target=True`.
- Train: fresh run, 100 ep, SGD lr 0.001 cosine → 1e-5, warmup 5%, batch 2, EMA 0.999, AMP, grad clip 0.5, seed 42.
- Selection: val AP@50, eval every 2 epochs.
- Eval: pycocotools, score_thr 0.01, hard NMS 0.5, maxDets 300. Valid split during development;
  **test split exactly once** for the paper.

## Part B — Conditions

| Condition | Init | Notes |
|---|---|---|
| scratch | random, all BN trainable | control (v5GPT config); the real baseline on this split |
| imagenet | ImageNet + `apply_bn_policy` | known-bad F1+F2 combo — ablation only |
| riceseg | `riceseg_backbone.pth`, all BN trainable | never `apply_bn_policy` |
| YOLOv8s | pretrained, fine-tuned | external baseline for the paper |

## Part C — Results log

| Date | Condition | Split | AP@0.50 | AP@0.75 | AP@[.5:.95] | Notes |
|---|---|---|---|---|---|---|
| 2026-07-08 | riceseg (soft VFL targets) | valid | 0.0088 | 0.0000 | 0.0012 | run #1, pre-fix. INVALID — VFL target bug |
| 2026-07-09 | riceseg (hard targets) | valid | ≤0.0010 | 0.0000 | — | run #2, killed ~ep 61. det/img frozen — EMA/clip suspect |
| | scratch | valid | | | | PENDING — run before any more riceseg attempts |
| | imagenet | valid | | | | PENDING |
| | best condition | test | | | | run ONCE at the end |
| | YOLOv8s | test | | | | PENDING |

Reference (not comparable): v5GPT scratch AP@50 0.166 on the retired leaky VOC-era split.
