# Gate status — the one authoritative answer to "can I run this?"

**Last updated: 2026-07-29.**

This file exists because three documents used to answer the same question three
different ways: `START_HERE.md` said "do not train the detector yet",
`docs/HANDOFF.md` reported four completed phase-2 training runs, and `README.md`
carried a status table dated 2026-07-20. A reader could follow any of them and
get a different answer.

**Rule: this file is the only place a go/no-go decision is stated.** Every other
document links here instead of restating it. `docs/HANDOFF.md` remains the
narrative session log; this is the standing verdict.

---

## Right now

| Activity | Verdict | Why |
|---|---|---|
| Phase-1 RiceSEG segmentation pretraining | **DONE, closed** | best mIoU 0.5827 @ ep30, reproduced within 0.001 mIoU. `docs/research/RICESEG_PRETRAIN_RESULTS.md` |
| Phase-2 detector: pipeline shakedown on the rebuilt dataset | **GO** | trainer completes 18/18 on an A100 in ~24 min; artifacts (`status.json`, `metrics.jsonl`, `weeddet_last.pth`) are unambiguous |
| Phase-2 detector: any run whose numbers will be quoted | **NO-GO** | selection is validation *loss*, not AP; multiclass decode is still class-agnostic; no canonical model-to-COCO evaluator |
| Any evaluation on the 261-image test split from `grouped_split.json` | **NO-GO, permanently** | 231 of its 261 images were trained on by the voided 2026-07-28 runs. It is burned. A replacement must be built from the never-trained-on pool |
| Training on `RICE_curated_phase2.zip` (the 2026-07-27 archive) | **NO-GO** | 940 mis-assigned files, 231 intended-test images inside, 233 intended train/valid images missing. Superseded by the rebuild below |
| Citing any metric from the 2026-07-28 runs | **NO-GO** | both voided; see the `VOID.md` files in their Drive run directories |
| Public release of code or data | **NO-GO** | no project license selected; the Roboflow RICE source's licence is unrecorded |
| Any actuation / spray path | **NO-GO, permanently** | perception research only. See the safety stop conditions in `START_HERE.md` |

## Dataset of record

`agrinav data-build-rice-phase2` rebuilds it from the source deliverable
(`agrinav_intake_2026-07-21/deliverable/detection/RICE/`). Verified on 2026-07-29:

| Split | Images | Boxes | rice_protect | weed_target | EXIF-normalized |
|---|---:|---:|---:|---:|---:|
| train | 1,800 | 59,691 | 52,194 | 7,497 | 35 |
| valid | 518 | 15,226 | 13,201 | 2,025 | 80 |
| test (**burned**) | 261 | 6,284 | 5,355 | 929 | 99 |

Full card: [`docs/rice_phase2_dataset_card.md`](rice_phase2_dataset_card.md).
Packaged training archive (train + valid only): `RICE_phase2_rebuild.zip`,
644,891,718 bytes, sha256
`57484b9d30a062be4011e24eec9f898d9c571dcc4bef3cee487d9865f190db27`. Superseded and
banned: `RICE_curated_phase2.zip`, sha256 `2161e069…a19fa3ab`.

3 annotations rejected by the sanitation rule (all out-of-bounds by more than
1 px) and 115 clipped; both itemized in `reports/rejected_annotations.json`.
Zero duplicate image hashes. `preflight` re-verifies per-image hashes, decoded
dimensions against the COCO records, in-bounds boxes, cross-split duplicates, and
stray files — and fails closed.

Known residual, not a blocker but not "leakage-free" either: 3 re-derived
capture-family/frame-block groups straddle a split boundary, because the source
grouping cut video sequences into 40-frame blocks. Quantify or guard-band this
before making a generalization claim.

## What clears the "quotable numbers" gate

In order. Each is independently verifiable.

1. **Class-aware decode.** Expand `(anchor, class)` candidates; suppress per class
   (`torchvision.ops.batched_nms` or a validated per-class Soft-NMS). Regression
   test with overlapping rice and weed boxes.
2. **One canonical model-to-COCO adapter** — inverse letterbox *with clipping*,
   class-id mapping, score threshold, NMS, max-detections — wired into training.
3. **Selection on validation AP** (AP50, AP@[.50:.95], per-class AP, AP-small,
   AR), replacing validation loss. Keep `maxDets=100` as the comparable primary;
   label anything else separately.
4. **A decoded-AP overfit gate** on the production construction path, replacing
   the current `final_loss < initial_loss` check.
5. **A replacement test split**: whole groups drawn only from images with
   `trained_on_legacy: false` in `manifests/split_membership.json`, with the
   grouping rule recorded in an ADR.
6. **Matched BN policy** across the ImageNet/RiceSEG arms (`--bn-policy trainable`
   on the ImageNet control; `auto` reproduces the old two-factor confound).

## Standing engineering debt that does not block a shakedown

EXIF normalization is applied at dataset build time, so the loader's missing
`exif_transpose` no longer corrupts *this* dataset — but the loader is still
wrong for any other source. Also open: silent CUDA-to-CPU downgrade, `assert`-based
split guards, `unpad_boxes` without clipping, unaugmented negatives, stale
`CLASS_NAMES = ['Rice']`, `PyYAML` undeclared, Black version split between CI and
`[dev]`, no coverage threshold, `weeddet_v6b.py` excluded from lint. Full list and
evidence: `docs/audits/2026-07-29/AUDIT_VERIFICATION_2026-07-29.md`.
