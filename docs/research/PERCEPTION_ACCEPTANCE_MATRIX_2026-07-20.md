# AgriNav Perception Acceptance Matrix

**Date:** 2026-07-20  
**Scope:** rice/weed classification and image-space localization research only  
**Source of requirements:** July 20 audit/roadmap, the supplied dream scorecard, and the current data audits

Status terms: **implemented** means a local artifact or automated check exists; **defined** means the requirement and gate are written but evidence has not been produced; **owner decision** means the experiment cannot be frozen safely without project-owner input.

| ID | Requirement / acceptance gate | Status | Current evidence or blocker |
|---|---|---|---|
| SCP-01 | Outputs are proposals only; no treatment, ground-coordinate, navigation, or actuator command | implemented | ontology, annotation guide, operating-domain boundary, disabled legacy inference |
| SCP-02 | Unknown, ambiguous, missing, invalid, or OOD evidence never becomes an affirmative weed target | implemented in data contract; unmeasured in model | ontology forbidden inferences and annotation validator |
| ODD-01 | Fix region/field, growth stage, target weeds, camera, minimum target, hardware/resolution, crop-risk budget, and reserved sites | owner decision | eight unresolved fields in `docs/operating_domain.md` |
| DAT-01 | Raw archives are immutable and identified by SHA-256 | implemented for received archives | RiceSEG `0071d9…be96`; detector `9da9ce…f74a` |
| DAT-02 | Every image has source/version/license/hash and explicit-null capture lineage | partial | annotation contract exists; detector source/license/session provenance is incomplete |
| DAT-03 | Highest-risk site/session/pass/source-photo groups are assigned before splitting | partial | 773 RiceSEG source-photo groups and eight conservative detector filename families recovered; detector families need owner/source confirmation |
| DAT-04 | Zero source-group/duplicate overlap across train/validation/test/challenge | validator implemented; release missing | group-overlap validator exists; no new sealed release is assigned yet |
| DAT-05 | Development/challenge data contain 20–30% human-verified negative or nonactionable images across independent groups | missing | current archives cannot establish verified negatives; collect and review new sessions |
| DAT-06 | Sealed target test and separate OOD/challenge sets contain no external training image or derivative | defined | membership must be created only after ODD and group recovery |
| ANN-01 | Versioned biological-class/decision-role ontology and boundary guide exist | implemented draft; approval required | `data/ontology.v1.json`, `docs/annotation_guide.md` |
| ANN-02 | Pilot contains 150–300 images, 40 frozen gold images, two independent gold annotators, and adjudication | intake implemented; human work missing | deterministic 200-image intake and 40-image gold membership built; gold labels/review not yet produced |
| ANN-03 | Routine double labeling 10–20%; locked high-risk test/challenge review 100% | defined | requires annotation campaign |
| ANN-04 | Class/status agreement ≥0.98, median box IoU ≥0.85, median mask IoU ≥0.80 | defined/unmeasured | measure after independent labels |
| ANN-05 | Invalid, duplicate, zero-area, or out-of-bounds annotations <0.001 | validator implemented; release unmeasured | source geometry smoke audit passed; human export not yet available |
| ANN-06 | Model/revision/prompt/threshold/raw proposal and every human edit are recoverable | contract implemented; revisions unpinned | proposal methods still contain `PIN_BEFORE_RUN` |
| COR-01 | Joint image/box/mask transforms pass inverse/fixture checks within one pixel | missing | known translation-direction defect remains in historical training path |
| COR-02 | Assignment/loss/evaluator semantics match official definitions and one standard path | missing | historical custom loss/assignment and mixed maxDets protocols remain barred from new claims |
| COR-03 | Detector tiny overfit reaches AP50 ≥0.95 and AP75 ≥0.80 | defined/unmeasured | run only after correctness repairs |
| COR-04 | Segmentation tiny overfit reaches pixel accuracy ≥0.98 and every present-class IoU ≥0.90 | defined/unmeasured | run only after RiceSEG parser/metric repairs |
| EXP-01 | Conditions A scratch, B ImageNet, C scratch→RiceSEG, D ImageNet→RiceSEG use identical downstream budgets | defined | preregister before GPU work |
| EXP-02 | Three paired screening seeds and five paired final seeds; primary contrast `D-B` | defined | no controlled run completed |
| EXP-03 | RiceSEG transfer adds ≥0.05 absolute COCO AP and paired 95% CI excludes zero | dream result/unmeasured | primary novelty gate |
| EXP-04 | FCOS versus Faster R-CNN is a fair, equal-budget stage ablation; mask model is evaluated separately | defined | one-stage remains default baseline; no architecture winner selected yet |
| EVAL-01 | Sealed grouped test: AP@[.50:.95] ≥0.45, AP50 ≥0.75, AP75 ≥0.50 | dream result/unmeasured | test is not yet sealed |
| EVAL-02 | Weed recall ≥0.90 under preregistered crop-overlap and negative-proposal constraint | owner decision/unmeasured | risk constraint must be numerically fixed before threshold selection |
| EVAL-03 | Unseen-site/session rice IoU ≥0.90 and weed IoU ≥0.75 | dream result/unmeasured | requires masks and independent groups |
| EVAL-04 | Zero weed-target proposals on a large negative/OOD set with useful one-sided bound | defined; data missing | use independent capture groups; correlated frames are clustered, not counted as independent |
| RUN-01 | Warmed batch-one p99 model-pipeline latency <50 ms on named hardware/resolution/precision/export | owner decision/unmeasured | hardware and resolution are unresolved |
| REP-01 | Code, environment, configs, seeds, manifests, predictions, checkpoints, and reports are hash-recoverable | defined/partial | archive hashes and deterministic pilot exist; no new training release exists |
| SYS-01 | Physical 95th-percentile ground-target error ≤ one-third of smallest target/nozzle radius | future system gate | explicitly outside current perception claim |
| SYS-02 | Field: weed hit ≥90%, crop contact ≤0.5%, treated-area reduction ≥70% | future system gate | explicitly outside current perception claim |
| SYS-03 | 100% fail-closed response across enumerated software/hardware fault injection | future system gate | explicitly outside current perception claim |

## Release decision

The current deliverable is a **research design and validated annotation-intake scaffold**, not a successful model or deployable perception release. GPU training should remain stopped until `DAT-03` through `DAT-05` and `COR-01` through `COR-04` pass. The first honest go decision is approval of the ontology/ODD plus collection of independent weed-rich and verified-negative sessions.
