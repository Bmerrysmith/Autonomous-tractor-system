# Gate status — the one authoritative answer to "can I run this?"

**Last updated: 2026-09-09.** The current campaign table below supersedes the
historical July/August verdicts retained later in this file.

This file exists because three documents used to answer the same question three
different ways: `START_HERE.md` said "do not train the detector yet",
`docs/HANDOFF.md` reported four completed phase-2 training runs, and `README.md`
carried a status table dated 2026-07-20. A reader could follow any of them and
get a different answer.

**Rule: this file is the only place a go/no-go decision is stated.** Every other
document links here instead of restating it. `docs/HANDOFF.md` remains the
narrative session log; this is the standing verdict.

---

## Current campaign — 2026-09-09

| Activity | Verdict | Evidence or remaining gate |
|---|---|---|
| CPU verification | **PASS** | 689 tests and 16 subtests; subsequent focused runs: 64 and 80 passed. Ruff and Black pass. Default-model state, loss, gradients, one update, and evaluation outputs match HEAD exactly in the bounded CPU check. |
| Rebuilt-data structural preflight | **PASS** | Rechecked without writing the dataset: 2,579 images / 81,201 boxes. |
| Dataset independence claim | **UNRESOLVED** | 24 re-derived groups cross partitions; three merge filename forms. No exact byte/pixel duplicates across splits, but capture-session provenance is missing. Treat this as historical development data. |
| Resource and throughput pilots | **PASS, LIMITED** | Nine configurations completed without OOM. Full training epochs: WeedDet 512/b8 95.00 s, 512/b4 99.15 s, 640/b4 142.97 s; Faster R-CNN 512/b4 91.38 s, 640/b4 106.65 s. Approx. 28.14 training hours for comparable 512 confirmation or 34.88 hours for the 640 finalist design. Validation/overhead remain unmeasured; confirmation cap stays 40 hours. |
| Historical sixteen-image memorization gate | **MIXED; RICE ONLY** | Incumbent fails (AP50 0.3478 / AR100 0.1782); reference Varifocal passes (0.9753 / 0.7643); head GroupNorm passes (0.9780 / 0.6778). All 409 GT annotations are rice. Same seed/subset and unchanged thresholds; no both-class claim. |
| Incumbent normalization diagnosis | **DEFECT LOCALIZED** | Current checkpoint reaches AP50 0.99084 / AR100 0.77139 when only head BN uses batch statistics. Earlier model shows the same failure pattern. Batch-statistics evaluation is diagnostic only; incumbent's original failed gate remains failed. |
| Both-class memorization diagnostic | **VARIFOCAL PASS; GROUPNORM FAIL** | Sixteen images / 407 rice / 105 weed annotations. Varifocal AP50 0.84854 / AR100 0.53329 passes unchanged thresholds; GroupNorm 0.69358 / 0.44957 misses recall. Checkpoints and reconstructed metrics verified. |
| Bounded diagnostics | **GO; CURRENT CHECKS FINISHED** | Local RTX 4070; 46.02 minutes charged to the 5-hour allocation, including failures. Preserve failed evidence and the original acceptance thresholds. No GPU job is running. |
| Seven-arm screening | **PENDING HUMAN REVIEW, SOURCE FREEZE, AND RUNTIME FIT** | 100-object version-2 packet remains unreviewed: class, box/merged plants, and full-image missed labels. Seven configurations are prepared, not launched. Incumbent failure is diagnosed and remains explicit as a research control; the strong reference stays required. A smoke-test pass does not select a finalist. |
| Historical standalone AP for anchor-4 checkpoints | **RE-EVALUATE BEFORE COMPARISON** | Loader previously reconstructed anchor scale 3. In-training AP uses the live model and is not invalidated by that defect alone. |
| Headline detector or generalization claim | **NO-GO** | Baselines now exist; the old “no baseline has been run” blocker is stale. Matched confirmation, uncertainty, and external validity remain unresolved. |
| Final test evaluation | **DEFER UNTIL SELECTION IS FROZEN** | No test-based tuning; historical contamination remains specific to the affected weights. |
| IEEE manuscript package | **PREPARE STRUCTURE; RESULTS PENDING** | Use the existing four paper repositories after a bounded detector study. No venue selected. |
| New data/package release | **RIGHTS CHECK PENDING** | The main GitHub repository is already public; project and source-data licensing still need resolution. |
| Actuation or spray deployment | **NO-GO** | Perception research scope only. |

Campaign details and decision rules:
[detector campaign](research/DETECTOR_CAMPAIGN_2026-09-09.md).
The 60 GPU-hour cap includes failed runs: 15 screening, 40 confirmation, 5 diagnostics.
The [analysis protocol](research/DETECTOR_ANALYSIS_PROTOCOL_2026-09-09.md) specifies
seed-level confidence intervals and inference endpoints. The
[IEEE preparation note](research/IEEE_PREPARATION_2026-09-09.md) lists simpler tasks,
outline corrections, and three preliminary venue candidates.

## Historical verdicts — July/August 2026

| Activity | Verdict | Why |
|---|---|---|
| Phase-1 RiceSEG segmentation pretraining | **DONE, closed** | best mIoU 0.5827 @ ep30, reproduced within 0.001 mIoU. `docs/research/RICESEG_PRETRAIN_RESULTS.md` |
| Phase-2 detector: pipeline shakedown on the rebuilt dataset | **GO FOR A FRESH RERUN** | the validation-loader batch-size crash is fixed and regression-tested; the two 2026-07-30 attempts stopped before recording epoch 1 and are not usable runs |
| Phase-2 detector: a run reporting validation AP on the rebuilt split | **GO FOR A FRESH RERUN** | decode is class-aware, the model-to-COCO adapter is wired, validation AP selects the checkpoint (`val_ap_interval`), and validation now uses an explicit positive batch size |
| Phase-2 detector: a **2-epoch pilot** with the corrected BN freeze + instrumentation | **GO** | freeze scope is fixed and fail-closed (57 of 58 under an injected backbone), and the run now records the gradient-norm distribution, the positive/negative loss split and per-epoch train-vs-eval BN parity. This is the run that produces the evidence for the two open questions below |
| Phase-2 detector: another **full 18-epoch run** | **GO** | the pilot reported on 2026-08-19 (`pilot_20260819_180855`). Both unknowns are resolved: `grad_clip` is raised from 0.5 to 40.0 on measured evidence, and the train/eval BN gap did not open (parity 0.84→0.88 RiceSEG, 0.71→0.75 ImageNet). `notebooks/weeddet_rice_phase2_colab.ipynb` pins `d14c9a2`, which carries both that config and the explicit BN policy. See the pilot section below |
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

## The 2-epoch pilot reported — 2026-08-19

`pilot_20260819_180855` (A100, torch 2.11.0+cu128, both arms exit 0, 2 epochs,
450 steps each, seed 42, archive sha256 `40eb6370…`). Readout:
`rice_phase2_v2/pilots/pilot_20260819_180855/pilot_readout.txt`.

**1. `grad_clip` was the defect.** It fired on **100% of steps in both arms**.
Pre-clip norms, pooled over all 450 steps per arm:

| Arm | p50 | p90 | p99 | p99.9 | max | non-finite |
|---|---:|---:|---:|---:|---:|---:|
| riceseg | 5.377 | 6.872 | 10.538 | 13.837 | 13.996 | 1 |
| imagenet | 8.207 | 18.079 | 32.907 | 39.271 | 39.667 | 3 |

At `grad_clip: 0.5` that is a 10-16x truncation at the median and up to 80x at
the tail: the clip, not the cosine schedule, set every step size in every run
this project has recorded. **Raised to 40.0** in
`configs/training/detector_rice_phase2.yaml`, just above the largest of the 900
measured norms, so it binds on nothing observed while still catching an
explosion. Deliberately *not* set to the production arm's p99 (10.5) — that
would clip ~1% of RiceSEG steps but ~25% of ImageNet steps and make the control's
effective step size a function of the clip.

`configs/training/baseline_det_control.yaml` still carries `grad_clip: 0.5` and is
**unmeasured** — torchvision's reference recipe does not clip at all. Measure it
before the first baseline run rather than copying 40.0 across; tuning a baseline
to the candidate's recipe is the failure this harness exists to avoid.

**2. BatchNorm is cleared.** The freeze held exactly as specified — RiceSEG
57 of 58 (`backbone_eval_mode=57/57`), ImageNet 48 of 58 — and the train/eval
parity ratio did not open a gap (0.84→0.88 and 0.71→0.75, both inside the
[0.33, 3.00] bound and trending toward 1.0). **The GroupNorm swap is not
indicated** and item 6's "BatchNorm is currently the top defect" no longer holds:
those measurements were taken on weights from a run that was itself throttled
~25x, so the running statistics described a network that had barely trained. BN
was a symptom, not the cause.

**3. The classifier is learning objects, not background.** `cls_loss_pos` fell in
both arms while `cls_loss_neg` rose modestly, which is the correct signature.

**4. The RiceSEG warm-start is doing measurable work.** `cls_loss_pos` fell
3.1069 → 1.5294 (−50.8%) on the RiceSEG arm against 3.5442 → 2.6234 (−26.0%) on
the ImageNet control, and the RiceSEG arm's gradient distribution is both lower
and tighter (max 13.996 vs 39.667). Two epochs, one seed — **directional only,
not a result.**

**Instrumentation bug found and fixed by this pilot.** The ImageNet arm's
recommendation printed `nan`. Under AMP, `scaler.unscale_` yields inf/NaN norms
on the steps `GradScaler` then skips (1 and 3 steps here, normal calibration);
those reached `torch.quantile`, and `_series` did not filter them because NaN is
a float. `max()` ordering then decided whether an arm got a number or `nan` —
the printed 12.74 was the RiceSEG arm's worst-epoch p99, and the ImageNet arm's
true p99 of 32.907 was silently dropped, understating the correct clip by ~3x.
Both call sites now filter non-finite values, and `grad_norm/non_finite_steps` is
recorded per epoch.

## The overfit gate, and what `grad_clip` was really worth — 2026-08-19

The gate now **PASSES**, at `--overfit 16 --epochs 150 --grad-clip 100`:

```
AP 0.5224  AP50 0.8750  AR100 0.6315  conf_ratio 1.01  loss 4.5681 -> 0.2309
grad_norm p50 3.804  p90 5.060  p99 23.908  max 69.106  clipped 0.0%
```

Holding one seed and one dataset fixed and changing only the clip:

| `--overfit` | epochs | grad_clip | AP | AP50 | AR@100 | clipped |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 60 | 0.5 | 0.0012 | 0.0064 | 0.0286 | 100.0% |
| 8 | 60 | 120 | 0.0628 | 0.3331 | 0.2563 | 0.0% |
| 8 | 150 | 120 | 0.2440 | 0.7190 | 0.4131 | 0.0% |
| 16 | 150 | 100 | **0.5224** | **0.8750** | **0.6315** | 0.0% |

**A 52x AP50 improvement from one flag.** The gradient norms are the tell: under
the 0.5 clip they sat at p50 65.5 for all 60 epochs, because the weights never
moved and the loss surface never flattened. Unthrottled they fall to 3.8-5.6 as
the model actually converges. The `p50 65` figure was never a property of this
model — it was the fingerprint of a model held still.

**The thresholds were never wrong; the image count was.** `--overfit 8` gives 240
optimizer steps at 60 epochs and 600 at 150, against the ~1200 that the AP50 0.50
and AR@100 0.50 floors were calibrated on at `--overfit 16`. Lowering the AR floor
to 0.35 was proposed on the overfit-8 evidence and **rejected** — correctly. The
floor is reachable; the gate was simply pointed at the wrong configuration. Do not
lower these to fit a smaller gate.

**Correction to the entry above.** The AP-vs-AP50 gap was called a localization
weakness and "the leading suspect" on a single overfit-8 measurement. The AP/AP50
ratio is 0.339 at overfit-8 and 0.597 at overfit-16 — most of that gap was
under-training, not architecture. It is not currently a suspect. The resolution
study still stands on the dataset-card evidence (56% of train boxes COCO-small at
512 px), not on this.

**`grad_clip` raised 40.0 -> 100.0.** 40 was set just above the pilot's observed
max of 39.667, from two epochs. The real run is 18 with a cosine schedule, and a
clip sitting on the observed maximum starts binding the moment a later epoch is
noisier than the pilot — silently, and straight back into this trap. 100 clears
every unthrottled measurement in any regime (max 77.745) and still catches a
divergence, which produces norms in the hundreds.

**Still armed:** `_HARD_DEFAULTS['grad_clip'] = 0.5` remains the fallback for any
invocation without `--config`. The phase-2 YAML was fixed first and this gate
still failed, because the gate passes no config. The gate now sets `--grad-clip`
explicitly; the default itself is unchanged and will catch the next config-less
caller.

## Standing engineering debt that does not block a shakedown

EXIF normalization is applied at dataset build time, so the loader's missing
`exif_transpose` no longer corrupts *this* dataset — but the loader is still
wrong for any other source. Also open: silent CUDA-to-CPU downgrade, `assert`-based
split guards, `unpad_boxes` without clipping, unaugmented negatives, stale
`CLASS_NAMES = ['Rice']`, `PyYAML` undeclared, Black version split between CI and
`[dev]`, no coverage threshold, `weeddet_v6b.py` excluded from lint. Full list and
evidence: `docs/audits/2026-07-29/AUDIT_VERIFICATION_2026-07-29.md`.
