# Detector diagnostics — 2026-09-09

These are plumbing/resource diagnostics on training images, not screening or
generalization results. Original manifests, code archives, checkpoints, commands,
logs, and the GPU budget ledger remain outside Git in
`../audit_artifacts/detector_campaign_2026-09-09` relative to the repository root.

## Resource feasibility

Four dense batches per configuration, up to 67 ground-truth objects per image,
through the actual trainer and optimizer. Nine pilots completed without OOM.
Total charged subprocess wall time: **1.985 minutes**, including startup and I/O.

| Configuration | Input / batch | Peak live allocation, GiB | Peak reserved, GiB |
|---|---|---:|---:|
| WeedDet incumbent | 512 / 8 | 7.866 | 10.732 |
| Predicted IoU, legacy loss | 512 / 8 | 7.866 | 10.732 |
| Predicted IoU, reference Varifocal | 512 / 8 | 7.848 | 10.617 |
| Anchor 3 | 512 / 8 | 7.866 | 10.732 |
| Head GroupNorm | 512 / 8 | 7.948 | 10.859 |
| Batch control | 512 / 4 | 4.469 | 6.604 |
| Resolution arm | 640 / 4 | 6.788 | 10.148 |
| ImageNet Faster R-CNN | 512 / 4 | 2.024 | 2.193 |
| ImageNet Faster R-CNN | 640 / 4 | 2.442 | 2.689 |

These are training-memory measurements, not the inference-memory endpoint in the
useful-benefit gate. Four batches do not guarantee full-run memory behavior or
provide a reliable campaign-duration estimate.

The initial incumbent completed but crossed an overly conservative **reserved**
memory cutoff of 10.5 GiB. Its live allocation was 7.87 GiB, four-batch epoch took
4.53 seconds, and it had no skipped steps or OOM. That threshold was replaced
with a 9.5 GiB live-allocation guard plus a bounded subprocess timeout. The original
record is preserved; the correction is explicit in `gpu_budget.json`.

## Eight-image warm-start stress test

Selected eight distinct training images, balanced by the sampled object's class
and size. ImageNet initialization; anchor 4; hard-target legacy loss; pretrained
BN frozen; batch 2; no augmentation, AMP, or EMA; 150 epochs; constant LR 0.01;
hard NMS; evaluate on the same eight images. This differs from both the screening
recipe and the documented sixteen-image scratch-initialized wiring gate.

The run completed but **failed memorization**: final AP50 and AR100 were both zero,
despite falling training loss and a final peak-confidence ratio of 0.735. All 800
retained detections were wrong at the COCO IoU thresholds. The peak-confidence
ratio alone is therefore insufficient evidence of successful decoding.

Live final-model detections and reloaded-checkpoint detections were **exactly
equal**; the standalone AP also matched the logged AP. This directly verifies the
checkpoint reconstruction fix on a real training run, including its anchor scale.

### Fixed-checkpoint normalization probe

Kept weights, images, batch size, and postprocessor fixed. Each alternative starts
from a fresh copy of the same saved checkpoint. Hooks temporarily use batch
statistics only in the named BatchNorm layers; the source checkpoint is unchanged.

| Statistics used | AP | AP50 | AR100 |
|---|---:|---:|---:|
| Saved running statistics | 0.0000 | 0.0000 | 0.0000 |
| Head batch statistics only | 0.0434 | 0.1416 | 0.1254 |
| Nine unfrozen backbone BN layers use batch statistics | 0.0000 | 0.0000 | 0.0000 |
| Head and those nine backbone layers use batch statistics | 0.0416 | 0.1392 | 0.1199 |

The head's running statistics contribute to this checkpoint's failure, but
switching statistics does not recover a passing detector. Batch-statistics
evaluation depends on batch composition and is **not a deployment protocol**.
This supports testing the already-planned head GroupNorm arm; it does not prove
that GroupNorm improves generalization or that normalization is the only issue.

## Documented acceptance gate

Reproducing the notebook's existing command: first 16 training images, scratch
initialization, 512 pixels, batch 2, 150 epochs, clip 100, seed 42, and the existing
Soft-NMS overfit evaluator. Thresholds remain AP50 >= 0.5, AR100 >= 0.5, confidence
ratio within [1/3, 3], and falling loss. No thresholds are lowered to accommodate
the smaller stress test.

| Same sixteen-image recipe | AP | AP50 | AR100 | Confidence ratio | Gate |
|---|---:|---:|---:|---:|---|
| Incumbent hard-target legacy loss, head BatchNorm | 0.0893 | 0.3478 | 0.1782 | 1.04 | FAIL |
| Predicted-IoU targets and reference Varifocal loss | 0.6637 | 0.9753 | 0.7643 | 1.00 | PASS |
| Hard-target legacy loss, head GroupNorm only | 0.5818 | 0.9780 | 0.6778 | 0.99 | PASS |

Both alternatives meet every existing gate, including falling loss. This is one
seed on sixteen training images / 409 boxes. **All 409 annotations are rice;
weed has no ground truth in this subset.** These passes do not establish
both-class memorization. The Varifocal comparison changes both
the target definition and objective; it does not isolate the formula. The planned
predicted-IoU legacy-loss screening control supplies that distinction.

The historical AP50 0.875 / AR100 0.6315 result is not reproduced by the current
incumbent. Historical code, data, and environment differences need reconciliation;
these measurements alone do not identify the cause. In particular, improved
memorization is not evidence of improved validation AP or a selected finalist.

Each attempt has a launch manifest, source archive, command, log hash, and explicit
gate result in `documented_overfit_gate/<arm>/`. The legacy overfit CLI does not
save weights: no sixteen-image checkpoint is available for a later decoder probe.
Eight-image stress-test checkpoints do exist separately.

### Saved-checkpoint reconstruction and normalization diagnosis

A fresh current-code repetition reproduced all 150 printed epoch losses and the
final AP/AR measurements. This repetition saves its final weights. Holding those
weights, images, and postprocessing fixed gives:

| Model source and statistics | AP | AP50 | AR100 |
|---|---:|---:|---:|
| Current model, saved running statistics | 0.08930 | 0.34780 | 0.17824 |
| Current model, head batch statistics only | 0.64724 | 0.99084 | 0.77139 |
| Current model, all BN batch statistics | 0.65273 | 0.99388 | 0.77433 |
| Earlier model `d14c9a2`, saved running statistics | 0.000002 | 0.000021 | 0.00073 |
| Earlier model, head batch statistics only | 0.69535 | 0.99401 | 0.78729 |
| Earlier model, all BN batch statistics | 0.74034 | 1.00000 | 0.82103 |

Head running statistics are sufficient to explain the current checkpoint's
failed decoded gate: changing only their source recovers passing AP50/AR100.
The head BN is shared across three feature levels with different distributions;
mixing these statistics is a plausible mechanism, not separately isolated here.
Using batch statistics at evaluation is not a deployable fix. Head GroupNorm
remains an already-planned alternative, not an automatically selected winner.

The earlier model runs through the current dataset, overfit runner, and software
environment, with its own source bytes preserved. This is a model counterfactual,
not a complete historical-environment reproduction. Its failure shows that the
problem is not introduced solely by the new campaign options or August 28 ATSS/
classifier-initialization changes. The August AP50 0.875 report remains
unreproduced; its exact historical data/order/environment are not established by
this test. Runner source is unchanged between `d14c9a2` and current HEAD.

The first reconstruction's report writer rejected undefined weed AP (no weed GT).
A recovery then hit the canonical trainer finalizer's required `metrics.jsonl`;
the diagnostic uses a different artifact contract. Both execution failures remain
charged in the ledger. Saved weights were retained, undefined category AP is
explicitly null, and the diagnostic was finalized without another training run.
The original logs, checkpoint/source hashes, recovery history, and results are in
`gate_reconstruction/`. Canonical trainer/aggregation requirements were not relaxed.

### Both-class gate preparation

The additional fixed subset contains sixteen distinct training images selected
from the four class/size strata, retaining every annotation in each image:
**407 rice + 105 weed boxes**. Original annotation and image hashes are checked.
This is a diagnostic sample, not completed human review. It preserves the
historical first-sixteen gate rather than silently changing its data. Its scores
must not be compared directly to that different subset. Results are recorded in
`balanced_gate/`. Both attempts have finished:

| Both-class subset, 150 epochs | AP | AP50 | AR100 | Rice AP | Weed AP | Gate |
|---|---:|---:|---:|---:|---:|---|
| Head GroupNorm | 0.25894 | 0.69358 | 0.44957 | 0.31251 | 0.20538 | FAIL: recall |
| Reference Varifocal | 0.41907 | 0.84854 | 0.53329 | 0.50413 | 0.33400 | PASS |

Class-specific AP is COCO AP over IoU 0.50:0.95, not class-specific AP50. Varifocal
meets the unchanged global AP50/AR100, confidence-ratio, and falling-loss gates.
GroupNorm misses recall; its failed outcome is retained. This demonstrates a
both-class learning/decoding path with a planned alternative, not comparative
validation benefit or a reason to select a finalist before screening.

`gate_evidence_audit.json` verifies four reconstructed/balanced diagnostics:
manifest/source archives, checkpoint and result hashes, original logs, available
wrapper source, and equal live-versus-reconstructed metrics. It also verifies
that only the intended README changed in each paper repository. The diagnostic
finalizer now hashes logs after child exit, avoiding a hash invalidated by the
child's final print. These helper corrections did not change model training.

## Default-behavior regression check

Compared the current model against Git HEAD `9304278` using identical seed-42
initialization, synthetic CPU inputs of shape 2 x 3 x 128 x 128, and one SGD step.
For both anchor scales 3 and 4, initial states, training losses, gradients, updated
states, and evaluation logits/deltas/anchors are bit-for-bit equal. This supports
preservation of legacy model defaults; it is not a complete GPU trajectory test.
Source hashes, the extracted previous module, and results are retained in
`default_model_parity.py` and `default_model_parity.json`.

## Training throughput and budget fit

A representative 64-image, three-epoch probe measured WeedDet epochs at
9.65 / 9.26 / 9.22 seconds, and Faster R-CNN at 32.59 / 7.39 / 7.83 seconds.
This exposed substantial startup/worker overhead; directly scaling those small
epochs would exaggerate the full training cost.

One full epoch on all 1,800 training images then measured **95.00 seconds** for
ImageNet WeedDet at 512 / batch 8 and **91.38 seconds** for ImageNet Faster R-CNN
at 512 / batch 4. Both used two data workers and completed without OOM. Peak live
allocation was 7.842 and 2.024 GiB, respectively. No validation or test AP was
computed. Fresh manifests and source/checkpoint evidence identify these attempts
as diagnostics, not screening runs.

Additional full epochs measured WeedDet 512/b4 at **99.15 s**, WeedDet 640/b4 at
**142.97 s**, and Faster R-CNN 640/b4 at **106.65 s**. Peak live allocations were
4.434, 6.731, and 2.442 GiB; all completed without OOM and without validation AP.

At these rates, six seeds x 60 epochs for each of two comparable 512/b8 WeedDet
arms and one 512/b4 Faster R-CNN reference require about **28.14 training hours**.
A 640/b4 finalist, its matched 512/b4 WeedDet control, and the 640/b4 reference
project to **34.88 training hours**, leaving about **5.12 hours** of the fixed
40-hour confirmation allocation for validation, checkpoints, and timing work.
These projections exclude validation and potential slowdowns; they do not
guarantee completion or establish each loss/normalization arm's duration. Recheck
remaining cost using measured run times before launching each phase and stop
with explicit incomplete evidence if it cannot fit. Do not silently shorten arms
or extend the cap.

## Budget and next gate

After these attempts, the ledger charges **2,760.96 seconds / 46.02 minutes** of
the five-hour diagnostics allocation, including failed attempts and startup/I/O.
No screening, confirmation, or test evaluation has started. Human review remains
pending. See [current gate status](../../docs/GATE_STATUS.md) for launch decisions.

The human-review packet was revised to version 2 to include missed-label
inspection over the full image, with all existing annotations overlaid. It
preserves the same 100 selected objects and original image bytes. Twelve
validator self-checks passed; browser checks verified that coverage assessment
requires full-image display and leaves all judgments unreviewed. This is
qualitative coverage triage, not an estimate of population annotation recall.
