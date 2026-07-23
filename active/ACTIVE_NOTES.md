# Active Notes — What To Do Next

**Last updated:** 2026-07-20 (repository recovery baseline)

## Current evidence

The historical T7 recipe recorded val AP@50 0.1040 / AP@75 0.0116 with a frozen RiceSEG-initialized
backbone. The July 20 deep audit found correctness defects, capture-series split leakage, mixed
evaluation protocols, and missing controls. Treat the result as exploratory evidence only—not a
deployable metric or a causal RiceSEG-pretraining result.

## Stop conditions

- Do not connect any detector or visualization in this repository to an actuator.
- Do not interpret “not detected as rice” as weed or permission to treat.
- Do not spend more GPU time tuning T7/T8 until the Phase 0–2 audit gates are closed.
- Do not use the current validation/test split for generalization or paper claims.

## Progress (2026-07-20)

- **RiceSEG transfer-learning correctness: complete.** All 8 Gate-4 "RiceSEG-specific
  repairs" landed in `../training/riceseg_pretrain.py`, test-first (52-test suite green).
  Country/site parsing, mask validation, strict backbone loading, stratified overfit
  gate with enforced threshold + isolated output, fixed-class mIoU with explicit
  absent-class reporting, ImageNet-coverage fail-closed, and a hashed backbone export +
  full resumable checkpoint + run manifest. The RiceSEG backbone pretraining (the first
  batch of training) can now run reproducibly. See
  `../docs/audits/2026-07-20/PHASE1_TRANSFER_LEARNING_FIXLOG_2026-07-20.md` and log results
  in `../docs/research/RICESEG_PRETRAIN_RESULTS.md`.
- **Still open:** detector correctness in `../models/weeddet_v6b.py` (translation aug,
  ATSS, VFL, letterbox, NMS, evaluator), the contaminated COCO split rebuild, and the
  controlled factorial. Do next items 2–5 below remain for the detector track.

## Do next

1. Read `../docs/audits/2026-07-20/AGRINAV_FULL_DEEP_AUDIT_2026-07-20.md` and the paired deployment roadmap.
2. Write failing synthetic tests for translation direction, box/label keep masks, letterbox mapping,
   fail-closed decisions, and evaluator edge cases.
3. Fix the confirmed geometry/assignment defects and build one canonical preprocessing,
   postprocessing, and evaluation path.
4. Rebuild detector splits by capture series/session/site with zero group overlap and a sealed test set.
5. Establish maintained baselines and a controlled multi-seed RiceSEG factorial study before
   revisiting custom T10/T11 experiments.

## Historical run discipline

- Keep one immutable run ID, committed config, code/data/split hashes, evaluator protocol, and seed.
- Record selected and final epochs separately; distinguish raw and EMA checkpoints.
- Never use mutable notebook names, Drive paths, or notebook output as the run identity.

## Data note

`../data/rice_training_curated/` contains a local-only 245-image RiceSEG-derived curation set for
future annotation work. The public Git tree keeps only its manifest, audit, and provenance notes.
Do not mix it into training until source licensing and a cross-dataset near-duplicate check are recorded.
