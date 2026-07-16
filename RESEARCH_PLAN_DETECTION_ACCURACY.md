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
| 2026-07-09 | riceseg + T1 (all-neg) | valid | 0.0017 | 0.0000 | 0.0002 | T1 20-ep gate. FAIL — collapse-to-prior (det/img 239.6). Next: T2 grad-clip |
| 2026-07-09 | riceseg + T1 + clip 10 (T2) | valid | 0.0412 | 0.0102 | 0.0114 | ep-4 peak then decay. AP large = 0.000 → anchors too small (T4). Best clean-split number so far |
| 2026-07-09 | riceseg + T1 + clip 10 + anchors 6 (T4) | valid | **0.1327** | 0.0061 | 0.0323 | ep-4 EMA peak; ≈ old leaky 0.166 baseline on CLEAN split. AP large 0.000, AP@75 ~0. Next: T5 lr 0.0005 |
| 2026-07-09 | riceseg + lr 5e-4 (T5) | valid | **0.1406** | 0.0123 | 0.0352 | new best @ ep 4; monotonic decay after → lr exonerated; forgetting hypothesis → T6 scratch control |
| 2026-07-09 | **scratch control (T6)** | valid | 0.0450 | 0.0009 | 0.0080 | slow climb, peak ep 8, no spike-collapse → forgetting CONFIRMED. riceseg transient = 3.1× scratch. Paper baseline @ 20 ep |
| 2026-07-14 | **riceseg FROZEN backbone (T7)** | valid | 0.1040 | 0.0051 | 0.0230 | STABLE — no decay, raw≈EMA, still rising @ ep 20. 2.3× scratch with 2.88M trainable params. → T7b 100 ep |
| | scratch | valid | | | | PENDING — run before any more riceseg attempts |
| | imagenet | valid | | | | PENDING |
| | best condition | test | | | | run ONCE at the end |
| | YOLOv8s | test | | | | PENDING |

Reference (not comparable): v5GPT scratch AP@50 0.166 on the retired leaky VOC-era split.
