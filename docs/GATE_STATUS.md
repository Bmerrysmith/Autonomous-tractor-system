# Gate status — the one authoritative answer to "can I run this?"

**Last updated: 2026-07-31.**

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
| Phase-2 detector: pipeline shakedown on the rebuilt dataset | **GO FOR A FRESH RERUN** | the validation-loader batch-size crash is fixed and regression-tested; the two 2026-07-30 attempts stopped before recording epoch 1 and are not usable runs |
| Phase-2 detector: a run reporting validation AP on the rebuilt split | **GO FOR A FRESH RERUN** | decode is class-aware, the model-to-COCO adapter is wired, validation AP selects the checkpoint (`val_ap_interval`), and validation now uses an explicit positive batch size |
| Phase-2 detector: a **2-epoch pilot** with the corrected BN freeze + instrumentation | **GO** | freeze scope is fixed and fail-closed (57 of 58 under an injected backbone), and the run now records the gradient-norm distribution, the positive/negative loss split and per-epoch train-vs-eval BN parity. This is the run that produces the evidence for the two open questions below |
| Phase-2 detector: another **full 18-epoch run** | **NO-GO until the pilot reports** | two unknowns would ride along: `grad_clip=0.5` fired on 3150 of 3150 steps in the 2026-07-30 run, so the clip and not the LR schedule set the step size, and one trainable head BN can still open a train/eval gap. Both are answered by two epochs of `metrics.jsonl`; neither is answered by spending 90 more minutes of A100 time |
| Phase-2 detector: a headline accuracy claim | **NO-GO** | the baseline harness exists but **no baseline has been run**, so there is still nothing to claim *against*; and no external farm/season set, so nothing supports a generalization claim |
| Evaluating the **2026-07-28 checkpoints** on the 261-image test split | **NO-GO, permanently** | 231 of its 261 images were inside the archive those runs consumed — 179 as training data, 52 more in the archive's valid folder. Burned *for those weights*. Both runs are void anyway |
| Evaluating a **freshly trained** checkpoint on that same test split | **GO** | contamination is a property of the weights, not the images. A model trained from scratch on the correctly-rebuilt split has never seen its own test set, and no metric was ever computed on those images — no evaluator existed until 2026-07-29. Corrected 2026-07-29; an earlier note here said "permanently burned", which was too strong |
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
| test (see verdict table above) | 261 | 6,284 | 5,355 | 929 | 99 |

Full card: [`docs/rice_phase2_dataset_card.md`](rice_phase2_dataset_card.md).
Packaged training archive (train + valid only): `RICE_phase2_rebuild.zip`,
644,892,580 bytes, sha256
`40eb6370f41eeb53333918cfbeb55d3696a848067e2c96a389a8e1508be3fd03` (repackaged
2026-07-30 to correct a stale metadata string; no image, annotation or per-file
hash changed). Superseded and banned: `RICE_curated_phase2.zip`, sha256
`2161e069…a19fa3ab`.

3 annotations rejected by the sanitation rule (all out-of-bounds by more than
1 px) and 115 clipped; both itemized in `reports/rejected_annotations.json`.
Zero duplicate image hashes. `preflight` re-verifies per-image hashes, decoded
dimensions against the COCO records, in-bounds boxes, cross-split duplicates, and
stray files — and fails closed.

## 2026-07-30 validation-loader incident

The runs `weeddet_rice_20260730_204639` and
`weeddet_rice_20260730_205448` are failed launch attempts, not training results.
Both initialized correctly but wrote no `metrics.jsonl`, checkpoint, or
`status.json`.

Root cause: the CLI defaults carried `val_batch_size: null`, and the model passed
that value directly to PyTorch. `batch_size=None` disables DataLoader automatic
batching, so the first validation sample reached the list-based collator as a raw
`(image, target)` tuple and raised `KeyError(0)`. Training completed its first
training pass but crashed before validation could record epoch 1.

The driver now resolves a null validation batch size to the training batch size;
the model boundary validates the value defensively; the Phase-2 YAML sets
`val_batch_size: 8` explicitly; and a production-path regression test runs a
CLI-built train-plus-validation epoch. The Colab notebook also persists stderr
to the Drive run directory and refuses to treat a missing status file as
completion. Verified locally: 308 tests plus 16 subtests pass.

Known residual, not a blocker but not "leakage-free" either: 3 re-derived
capture-family/frame-block groups straddle a split boundary, because the source
grouping cut video sequences into 40-frame blocks. Quantify or guard-band this
before making a generalization claim.

## What clears the "quotable numbers" gate

**Done (2026-07-29):**

1. ~~**Class-aware decode.**~~ `agrinav.inference.postprocess` expands every
   `(anchor, class)` pair above threshold and suppresses **within** a class only,
   via `torchvision.ops.batched_nms` or per-class Soft-NMS. Top-k is per class,
   because a shared cap is one the 6.8:1 majority class wins. Regression-tested
   with overlapping rice and weed boxes; `WeedDet._decode` now delegates to it.
2. ~~**One canonical model-to-COCO adapter.**~~ `agrinav.evaluation.runner` —
   inverse letterbox *with clipping*, class-id mapping, threshold, NMS,
   max-detections — plus `agrinav evaluate-detector` for offline scoring. Tested
   end to end: a stub predicting exactly the ground truth scores AP 1.0, and
   shifting the boxes or swapping the class map moves it.
3. ~~**Selection on validation AP.**~~ `val_ap_interval` runs the canonical decode
   over the val split every N epochs and selects `best` on COCO AP (higher is
   better), replacing validation loss. `maxDets` stays at 100; a nonstandard value
   is flagged and its primary AP is the `-1.0` sentinel rather than a number.

**Still required:**

4. ~~**A decoded-AP overfit gate.**~~ Done 2026-07-31. `--overfit` now runs the
   canonical decode through `evaluate_coco_detections` on the **eval-mode** path
   over the memorised images and gates on AP50, AR@100 and train-vs-eval-mode
   confidence parity. `final_loss < initial_loss` is kept only as a secondary
   condition: it passed for the 2026-07-28 checkpoints, whose loss fell for 14
   epochs and which then decoded AP 0.0000. Ground truth is subset to the images
   the run actually kept, or recall would report `N/total` rather than the
   model's recall. The three thresholds
   (`--overfit-min-ap50` 0.50, `--overfit-min-recall` 0.50,
   `--overfit-max-conf-ratio` 3.0) are **provisional smoke floors, not quality
   standards** — set them from pilot evidence (CLAUDE.md §38). AP50 0.50 sits
   below the 0.6+ this configuration previously reached on overfit-16; the
   parity bound comes from the observed ~47x failure (0.9367 vs ~0.02).
5. ~~**A replacement test split.**~~ Not needed, and not buildable as originally
   described. Measured 2026-07-29: of the 781 never-trained-on images, only **6**
   sit in a re-derived group containing no trained-on image, and those 6 carry
   **zero weed boxes**. "Whole clean groups" yields an empty test set. It is also
   unnecessary — the manifest's own 261-image test split is valid for any model
   trained from scratch on the rebuilt data, because no metric was ever computed
   on it and contamination lives in the 2026-07-28 weights, which are void.
   Retrain from scratch and use it; do not evaluate the old checkpoints on it.
6. **Matched BN policy** across the ImageNet/RiceSEG arms. `auto` reproduces the
   old two-factor confound; set the policy explicitly on both arms.

   **BatchNorm is currently the top defect, not just a confound.** The
   2026-07-30 run (correct dataset, correct pin, AP-selected) scored
   `val/AP 0.0054`. Traced to eval-mode BN statistics, measured on identical
   weights over a fixed 40-image val subset:

   | condition | AP | AP50 | AR100 |
   |---|---:|---:|---:|
   | as-shipped running stats | 0.0000 | 0.0001 | 0.0009 |
   | batch stats (control, not deployable) | 0.0134 | 0.0713 | 0.0799 |
   | recalibrated running stats (precise-BN) | 0.0000 | 0.0000 | 0.0004 |

   Recalibration failing is the informative part: the buffers do not merely hold
   stale values, no fixed statistic reproduces the network's behaviour. That is
   the small-batch BN failure mode — 58 BN layers at batch 8. Note the control is
   only AP50 0.071, so BN is necessary but not sufficient; the model is weak even
   at its best. References: `docs/BIBLIOGRAPHY.md` §2.

   `--bn-policy freeze_pretrained` **was a silent no-op on the RiceSEG arm** until
   2026-07-31: `build_config` forces `pretrained_backbone=False` whenever
   `--riceseg-backbone` is used, and that flag was what decided freezing, so it
   froze **0 of 58** layers while logging that the policy applied. Fixed; a
   `freeze_pretrained` that freezes nothing now raises, and the frozen/trainable
   counts are recorded in the run config.

   That first fix took it from 0 to **48 of 58**, which was still wrong. The
   freeze predicate was the name list `_PRETRAINED_PREFIXES`, which answers "what
   can torchvision resnet50 weights fill?" — not "what holds loaded statistics?".
   `load_riceseg_backbone` fills the entire backbone (all 342 `backbone.*`
   tensors, buffers included, or it raises), so 9 backbone BN layers
   (`backbone.stem.*`, `backbone.layer1.0.*`) held in-domain statistics and were
   still being updated from batches of 8. Corrected 2026-07-31: the predicate now
   reads each module's running buffers (`bn_carries_pretrained_stats`), giving
   **57 of 58** under an injected backbone and **48 of 58** under ImageNet — both
   correct, and the two arms legitimately differ because the arms loaded
   different things. The one layer never frozen is `head.shared.2.seq.3`, the
   randomly initialised head BN; freezing that would pin it to an identity op.
   `--bn-freeze-scope` (`imagenet` | `backbone`) makes the expected set a
   fail-closed assertion, so a partial load aborts instead of quietly freezing
   fewer layers. The set is resolved **once, before the first step**, and reused:
   after one epoch a randomly initialised BN no longer looks randomly
   initialised, so re-deriving it mid-run would sweep the head BN in.

   Open, and the reason a pilot is still needed before a full run: 1 trainable BN
   remains in the shared head trunk, so a train/eval mode gap can still originate
   there. Swapping it for GroupNorm is the alternative, deliberately not taken yet
   — it changes the `state_dict` and breaks compatibility with every existing
   checkpoint, so it should be an evidence-driven ablation, not a precaution. The
   per-epoch `parity/*` metrics are what will localise it.

7. **Same-protocol baselines** — the *harness* now exists
   (`agrinav baseline-detector`, `configs/training/baseline_det_control.yaml`):
   stock torchvision `fasterrcnn_resnet50_fpn_v2`, `retinanet_resnet50_fpn_v2`
   and `fcos_resnet50_fpn` on the identical dataset class, the identical
   letterbox, the identical evaluator and the identical `maxDets`, with the
   internal resize/normalise disabled so the reference sees WeedDet's exact
   tensor. **No baseline has been run yet.** Until at least one has, a WeedDet AP
   still has nothing to be measured against. See
   [`docs/baselines.md`](baselines.md).

8. **Instrumentation for the pilot** — added 2026-07-31, observation only. Every
   epoch now records to `metrics.jsonl`: pre-clip gradient-norm quantiles
   (`grad_norm/p50|p90|p99|max|clipped_fraction`), the positive vs negative halves
   of the classification loss (`train/cls_loss_pos`, `train/cls_loss_neg`,
   summing exactly to `train/cls_loss`), observed BN state (`bn/eval_mode`,
   `bn/grad_off`, …), and train-mode vs eval-mode peak confidence on 8 fixed
   unaugmented images (`parity/*`). The probe snapshots and restores every BN
   buffer, so running it cannot alter training. The gradient-norm distribution is
   the missing evidence for choosing a real `grad_clip` — the 2026-07-30 run
   clipped **3150 of 3150** steps at 0.5, meaning the effective step size was set
   by the clip and not by the LR schedule, and only the count survived.

## The 2-epoch pilot — how to run it

Needs a GPU; there is none on the dev box, so this runs on the A100.

**Use [`notebooks/weeddet_pilot_colab.ipynb`](../notebooks/weeddet_pilot_colab.ipynb).**
It pins the checkout, gates on the required trainer flags before spending GPU
time, runs both arms, and prints the readout. The raw commands below are what it
drives, recorded here so the pilot can be reproduced without the notebook.

Run both arms, same seed, same data, differing only in what the backbone was
loaded from. `--bn-freeze-scope` is the fail-closed assertion: a partial load
aborts here rather than quietly freezing fewer layers.

RiceSEG arm (57 of 58 BN frozen):

```bash
python -m agrinav.training.weeddet_train --epochs 2 --bn-policy freeze_pretrained --bn-freeze-scope backbone --riceseg-backbone /content/drive/MyDrive/agrinav_data/out/riceseg_backbone.pth --dump-grad-norms --checkpoint-dir checkpoints/pilot_riceseg
```

ImageNet control (48 of 58 BN frozen — the arms differ because they loaded
different things; that is correct, not a confound to be equalised):

```bash
python -m agrinav.training.weeddet_train --epochs 2 --bn-policy freeze_pretrained --bn-freeze-scope imagenet --dump-grad-norms --checkpoint-dir checkpoints/pilot_imagenet
```

**What to read out of the two `metrics.jsonl` files, and the decision each one
settles:**

| Read | Decision |
|---|---|
| `grad_norm/p50`, `p90`, `p99`, `clipped_fraction` | Set `grad_clip` at roughly p99 so it catches outliers only. If `clipped_fraction` is again ~1.0 at 0.5, the clip — not the LR schedule — is setting the step size, and the honest fix is to raise it or drop it, not to keep it |
| `parity/max_conf_ratio` per epoch | Whether the train/eval BN gap is opening, and from which epoch. If it stays near 1.0, the remaining trainable head BN is not the culprit and GroupNorm is unnecessary |
| `train/cls_loss_pos` vs `train/cls_loss_neg` | Whether the classifier is learning objects or just learning to say "background". A total that falls while `cls_loss_pos` does not is the failure that a falling loss curve hides |
| `bn/eval_mode`, `bn/grad_off` | That the freeze actually held for both epochs — observed state, not the value that was requested |

The readout is not left to whoever is reading the file. It exits non-zero if
either arm warns:

```bash
python -m agrinav.training.pilot_report checkpoints/pilot_riceseg checkpoints/pilot_imagenet
```

Do not start the full 18-epoch run until these four have been read.

## Standing engineering debt that does not block a shakedown

EXIF normalization is applied at dataset build time, so the loader's missing
`exif_transpose` no longer corrupts *this* dataset — but the loader is still
wrong for any other source. Also open: silent CUDA-to-CPU downgrade, `assert`-based
split guards, `unpad_boxes` without clipping, unaugmented negatives, stale
`CLASS_NAMES = ['Rice']`, `PyYAML` undeclared, Black version split between CI and
`[dev]`, no coverage threshold, `weeddet_v6b.py` excluded from lint. Full list and
evidence: `docs/audits/2026-07-29/AUDIT_VERIFICATION_2026-07-29.md`.
