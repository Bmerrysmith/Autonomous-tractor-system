# Project evidence register — 2026-09-09

This distinguishes inspected artifacts from proposed studies. Immutable hashes for
the current dataset are in the local campaign `group_audit.json`; new training
attempts bind code/config/data identities and checkpoint hashes in launch/status
artifacts. Historical sources without these records remain historical evidence.

| ID | Finding / source | Evidence type | Interpretation |
|---|---|---|---|
| E01 | GitHub main repository public master `ed93be5`; local detector branch starts at `9304278`, five commits beyond upstream | Direct Git/GitHub inspection | Default branch and paper source pins lag active detector work. |
| E02 | Drive Faster R-CNN COCO-init run: 60 epochs, best validation AP 0.1806157 at epoch 50; final 0.1800978 | Training/evaluation on real images; inspected `run.json` | Baseline exists. Initialization/protocol must be matched or named before comparison. |
| E03 | Drive WeedDet 2026-08-25: 60 epochs, best/final AP 0.0924329 using Soft-NMS | Training/evaluation on real images; inspected run artifacts | Not interchangeable with hard-NMS AP. In-training evaluation differs from standalone reconstruction. |
| E04 | September 3 ablation report and retained launch driver | Training/evaluation on real images; historical metadata incomplete | ImageNet initialization confirmed. Several apparent single-seed effects did not replicate. Old checkpoints were deleted by the driver; do not reconstruct missing hashes. |
| E05 | Current no-write preflight: 2,579 images / 81,201 boxes | Direct dataset validation | Structural consistency passes; label quality and capture independence are separate questions. |
| E06 | 24 cross-partition derived group IDs; three merge `frame` and `frame_` families | Direct manifest/parser inspection | All 24 retain unresolved capture provenance. No cross-split exact byte or RGB-pixel duplicates found; this does not exclude correlated scenes. |
| E07 | 100 sampled objects: 50 train, 50 valid; 50 rice, 50 weed; 50 small at 512, 50 larger | Prepared human-review sample; v2 adds all original boxes and full-image coverage review without changing selection | No human judgments have been supplied yet. No test images included. Twelve validator self-checks are software checks, not review judgments. |
| E08 | `bilal/lane-detection`: generated EKF trajectories and hard-coded planner timings; separate MLP trained on synthetic formulas | Generated illustration for trajectory/timing outputs; trained synthetic model for MLP | These are not field validation or measured planner benchmarks. See GitHub audit for exact paths. |
| E09 | `krish/lane-detection-ekf`: simulator motion independent of commands; recorder lacks ground-truth error; quality monitor not coupled to EKF rejection | Source inspection of simulation/algorithm | Navigation/system-paper claims need a separate correctness and evaluation campaign. |
| E10 | Four private paper repositories exist and are synchronized locally; no manuscript sources found | Direct local/GitHub inspection | Existing scaffolds should receive the eventual manuscript work; avoid another package of duplicate drafts. |
| E11 | Standalone loader ignored saved anchor scale; now restored with kernel/normalization/loss settings | Reproduced code defect and regression test | Re-evaluate historical standalone measurements for anchor-4 checkpoints before comparison. |
| E12 | Full CPU suite 689 tests and 16 subtests passed; later focused runs 64 and 80 passed | Executed automated checks | Supports implementation plumbing, not detector accuracy or publication readiness. |
| E13 | Nine four-batch resource pilots complete without OOM | Measured training-memory diagnostics | Does not establish inference efficiency or full-campaign runtime. |
| E14 | Sixteen-image scratch gate: incumbent fails; reference Varifocal and head GroupNorm pass | Actual training-subset memorization, one seed; immutable launch/log evidence, no saved weights from this CLI | Supports investigating planned alternatives. No generalization or finalist claim. |
| E15 | Eight-image ImageNet stress test fails; head batch statistics partly recover AP; loaded/live predictions exactly match | Actual training and fixed-checkpoint diagnostic | Head running statistics contribute to failure; do not fully explain it. Saved checkpoint reconstruction is verified. |
| E16 | Current model vs HEAD matches states, losses, gradients, one SGD update, and evaluation tensors at anchor scales 3 and 4 | Bounded synthetic CPU regression check | Supports default behavior preservation; does not reproduce a complete GPU training trajectory. |
| E17 | Full 1,800-image training epochs: WeedDet 512/b8 95.00 s, 512/b4 99.15 s, 640/b4 142.97 s; Faster R-CNN 512/b4 91.38 s, 640/b4 106.65 s | Actual timed training, no AP scoring | Preliminary training-only confirmation projections: 28.14 hours for comparable 512/b8 arms or 34.88 hours for the 640 finalist design. Validation/overhead remain unmeasured; the 40-hour cap is unchanged. |
| E18 | Frozen 100-image validation timing sample, seed 20260909 | Prespecified file identities and hashes; no predictions | Makes later inference timing reproducible. Does not constitute an inference measurement. |
| E19 | Current sixteen-image repeat reproduces 150 logged epoch losses; head batch statistics change AP50 0.34780 to 0.99084 on fixed weights | Saved-checkpoint diagnostic, same data/postprocessing | Head running statistics explain that checkpoint's failed gate. Earlier model shows the same pattern; the historical August report is still unreproduced. |
| E20 | Historical first-sixteen subset has 409 rice / 0 weed annotations | Direct annotation count | Its passing variants do not establish both-class memorization; undefined weed AP is explicitly null. |
| E21 | Separate sixteen-image subset has 407 rice / 105 weed boxes; Varifocal passes, GroupNorm misses recall | Actual both-class memorization; saved/checksummed weights and logs | Validates a both-class learning/decoding path, without establishing validation improvement or selecting a finalist. |
| E22 | All four existing paper READMEs corrected locally; subsequent PROVENANCE.md updates pin reviewed infrastructure `979627a` | Applied documentation corrections with separate before/after hashes and diffs for each stage | Eight document changes remain uncommitted in paper repositories; remotes are unchanged. Original snapshot/result provenance is retained; copied historical documents still require reconciliation. |
| E23 | Reviewed infrastructure committed as `979627a` on `fix/assigner-and-measurement` | Source commit; 15 explicitly staged implementation/config/test files | User baseline-note edits and Claude skills are excluded. This pins development source, not a comparative detector result. |

Raw local audit folder: `../audit_artifacts/detector_campaign_2026-09-09` relative
to the repository root. Full GitHub report: `../GITHUB_REVIEW_2026-09-09.md`.
Full CPU log: `../audit_artifacts/pytest_campaign_2026-09-09.txt`.
Preflight log: `../audit_artifacts/preflight_campaign_2026-09-09.txt`.

Paper drafting starts from bounded, reproducible results with explicit limits.
Generated figures, simulations, trained synthetic models, and real-data evaluation
must keep separate evidence labels throughout the IEEE package.
