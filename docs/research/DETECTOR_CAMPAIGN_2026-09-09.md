# Detector campaign and publication transition — 2026-09-09

## Objective and scope

Finish a bounded detector study, then move to figures, methods, manuscript drafting,
and an IEEE advisor packet. Preserve the broader four-paper portfolio, with the
monorepo as the implementation source. Test existing detector ideas before any
framework migration. Navigation remains a separate correctness audit.

This records the agreed campaign. A useful detector benefit is a decision gate,
not proof of novelty, field readiness, or likely publication acceptance.

## Evidence and prerequisite work

- [x] Inspect local Git, Claude project memory, Google Drive run artifacts, and GitHub.
  GitHub report: `../../../GITHUB_REVIEW_2026-09-09.md` outside the checkout.
- [x] Confirm local branch `fix/assigner-and-measurement` at `9304278` starts five
  commits ahead of its upstream and seventeen ahead of public master `ed93be5`.
  Preserve the user's baseline-note edits and untracked Claude skills.
- [x] Implement immutable launch manifests, source archives, stable declarative
  config hashes, annotation/split hashes, completion status, and checkpoint hashes
  in both trainers. Preserve legacy checkpoint loading.
- [x] Read historical baseline `run.json`; refuse incomplete, nonfinite, sentinel,
  duplicate-seed, or incompatible research aggregates. Terminal epochs are
  correlated observations, not independent replicates or an unbiased estimator.
- [x] Add separate reference Varifocal loss and head-only GroupNorm options,
  preserving legacy defaults. Fix standalone checkpoint reconstruction to honor
  saved anchor scale, kernel size, normalization, and loss configuration.
- [x] Full CPU suite: 689 tests and 16 subtests passed; Ruff and Black passed.
  Saved-configuration round-trip reproduces logits, box deltas, and anchors exactly.
- [x] Audit 24 re-derived cross-split groups: three collapse distinct filename
  forms (`frame` and `frame_`); capture membership remains unresolved for all 24.
  No cross-split duplicates in source bytes, rebuilt bytes, or decoded RGB pixels.
  This does not rule out adjacent frames or other correlated scenes.
- [ ] Human review of 100 train/validation objects, balanced by class and object
  size: class correctness, box tightness, merged plants, and full-image missed
  labels. Version 2 of the local packet preserves the selected objects and adds
  all existing boxes for coverage inspection; twelve validator checks pass.
  No human judgments exist yet. Expand SAM/CVAT only if defects prevent this campaign.
- [x] Nine short resource pilots completed without OOM; seven screening configs
  prepared outside Git. Resource feasibility is not an inference-efficiency result.
- [x] Run the documented sixteen-image gate: incumbent fails; reference Varifocal
  and head-only GroupNorm pass, on a rice-only subset. A separate sixteen-image
  both-class diagnostic passes with Varifocal; GroupNorm misses recall.
  An eight-image ImageNet stress test also fails.
  These are diagnostic results, not validation gains or finalist selection.
  See [diagnostics](../../reports/summaries/detector_diagnostics_2026-09-09.md).
- [x] Localize the incumbent gate failure to head running statistics using fixed
  checkpoints. Earlier model source shows the same failure pattern; the old
  August metric remains unreproduced and excluded from new comparisons.
- [ ] Finish campaign runtime fit and freeze code/config evidence before
  screening. CPU default-model comparisons
  against HEAD match exactly through one optimizer step at anchor scales 3 and 4.
- [x] Commit reviewed infrastructure as `979627a` on the active branch; reconcile
  all four local paper provenance files to that development source, preserving
  the original snapshot revision and each historical result's own source.
  Paper changes remain uncommitted. The active branch is pushed in
  [draft PR #6](https://github.com/Bmerrysmith/Autonomous-tractor-system/pull/6);
  no merge has occurred. Source freeze still requires final launch checks.
- [x] Correct factual/scope claims in all four existing paper READMEs locally;
  preserve original bytes. The later pin update records development source, not
  a retrospective claim that earlier experiments used that committed revision.

Dataset of record is the local `01_rice_phase2_rebuild_TRAIN_CANDIDATE` root:
1,800 train images / 59,691 boxes; 518 validation / 15,226; 261 historical test /
6,284. A fresh no-write preflight passed 2,579 images / 81,201 boxes during this
audit; repeat if source/data identities change before launch. Existing partitioning is
historical development evidence; capture-group provenance remains incomplete.

RiceSEG's 3,078 tiles were all used during phase-1 training. They cannot become an
unseen test set for a detector initialized with that all-site backbone. Dataset
licensing and semantic-component instance validity also remain unresolved.

## Locked experiment

Use only the local RTX 4070. Total budget: **60 GPU-hours**, including failed runs:
15 screening, 40 confirmation, 5 diagnostics. Record actual elapsed GPU work in a
ledger. Never extend this cap silently. No gradient accumulation or framework
migration in this campaign; do not combine screening winners.

Keep the incumbent initialization and recipe explicit and identical across
WeedDet comparisons: **ImageNet**, confirmed from the retained September 3
`run_ablation.py` launch command, which does not pass `--riceseg-backbone`.
Keep the pretrained BN freeze explicit. RiceSEG transfer remains a separate study.
ATSS stays on with the corrected `cells_best_shape` candidate pool.

| Screening arm | Changed factor | Size / batch |
|---|---|---|
| Incumbent | hard target, legacy loss, anchor 4, head BatchNorm | 512 / 8 |
| Predicted IoU | predicted-box IoU target, legacy loss | 512 / 8 |
| Reference Varifocal | predicted IoU target and reference Varifocal objective | 512 / 8 |
| Anchor 3 | former anchor scale 3, not 2 | 512 / 8 |
| Head GroupNorm | replace only `head.shared.2.seq.3`, 32 groups | 512 / 8 |
| Batch control | batch 4 | 512 / 4 |
| Resolution | compare against the batch-4 control | 640 / 4 |

Screening: seed 42, 36 epochs, evaluate every 4; score mean AP at epochs
28/32/36. Compare reference Varifocal against the predicted-IoU legacy-loss arm
to isolate the loss formula. Keep the incumbent comparison separately labeled.

Select one finalist: accuracy first; otherwise efficiency; ties use lower latency
then lower peak inference memory. Confirmation: finalist, matched WeedDet control,
and ImageNet-initialized Faster R-CNN reference; six fresh seeds 101–106 per arm,
60 epochs, evaluate every 4; terminal mean at 52/56/60. Training budget and
initialization differences in the reference must remain explicit.

Lock hard NMS at IoU 0.5, score floor 0.05, maxDets 100. Preserve native proposal
limits and report them: WeedDet pre-NMS top-k is 2,000; torchvision has internal
proposal filtering. Validation checkpoint selection remains separate from the
terminal estimator. Test data cannot guide thresholds, architecture, or selection.

## Decision and simpler publication work

A useful benefit is either:

1. At least **+0.01 absolute AP**, with the paired 95% CI above zero; or
2. At least **20% lower latency or peak inference memory**, with the paired AP
   difference's lower confidence bound above **−0.005**.

Pair by seed and compare the same terminal statistic. Freeze the confidence
interval procedure before confirmation; the [analysis protocol](DETECTOR_ANALYSIS_PROTOCOL_2026-09-09.md)
specifies the paired t interval and inference endpoints before any confirmation
results exist. A wide interval is inconclusive, not
equivalence. Failure to reach the gate within budget produces a negative result
and a bounded next proposal. It does not authorize more training automatically.

After the gate, freeze model/configs; do the planned final test evaluation once;
generate reproducible figures/tables, methods, limitations, and an advisor packet
comparing three appropriate IEEE venues. Start writing in the existing paper
repositories. No venue has been chosen, and no submission is authorized here.
The [preparation note](IEEE_PREPARATION_2026-09-09.md) records simpler work items,
corrections to existing paper outlines, and three preliminary venue fit assessments.

## Implementation notes and remaining limits

`manifest-<id>.json` and `source-<id>.zip` preserve each launch. `status.json` is a
mutable pointer written atomically, initially incomplete; completed runs include
metric and checkpoint checksums. Resumes preserve old manifests and remain
excluded from automatic aggregation until continuity is audited. An untracked
injected model/callable is also excluded. Fresh runs refuse existing artifacts.

Annotation and split-manifest hashes do not themselves verify current image bytes;
the dataset preflight supplies that check. The aggregation checks establish
internal consistency, not annotation truth, group independence, or novelty.

The reference Varifocal positive weight is target IoU alone, unlike the legacy
objective's additional focal modulation. Source: the authors'
[implementation](https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py).

**New audit correction:** standalone evaluation previously reconstructed anchor
scale 3 regardless of checkpoint configuration. Historical standalone AP for
anchor-4 checkpoints requires re-evaluation before comparison. In-training AP
used the live model and is not invalidated by this loader defect alone.
