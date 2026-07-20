# AgriNav Perception Research Package

**Date:** 2026-07-20  
**Scope:** rice/weed classification and image-space localization only  
**Scientific question:** does rice-domain semantic pretraining improve leakage-free target-domain generalization beyond ImageNet alone?

## 1. What an amazing result actually is

The strongest paper is not “a new detector got a high score.” It is a controlled causal result plus a useful perception release:

> Under identical target data, group splits, tuning budgets, augmentations, selection rules, and seeds, ImageNet -> RiceSEG pretraining improves a maintained rice/weed localizer over ImageNet alone by at least five absolute COCO AP points, with a 95% confidence interval excluding zero, and the result holds on unseen source groups rather than neighboring tiles or frames.

For the current perception-only phase, the dream-but-credible scorecard is:

| Area | Amazing result | Claim gate |
|---|---:|---|
| RiceSEG transfer | `D - B >= +0.05` absolute COCO AP across five paired seeds; 95% CI excludes zero | primary novelty claim |
| Detector quality | AP@[.50:.95] >= 0.45, AP50 >= 0.75, AP75 >= 0.50 | sealed source-grouped target test |
| Weed operating point | >= 90% weed recall under a preregistered protected-crop-overlap and negative-proposal constraint | threshold fixed on validation |
| Protected-rice segmentation | IoU >= 0.90 | unseen sessions/sites |
| Weed-target segmentation | IoU >= 0.75 | unseen sessions/sites |
| Negative/OOD behavior | zero observed positive weed-target proposals with a useful one-sided upper bound | large independent challenge set |
| Runtime | warmed batch-one p99 model-pipeline latency < 50 ms | named hardware, resolution, precision, and export |
| Reproducibility | every condition, seed, prediction, split, environment, and artifact is recoverable by hash | independent rerun |

The “zero observed” negative result needs scale. With zero events, the approximate 95% upper failure bound is `3/n`; about 3,000 reasonably independent negative opportunities are needed to bound the rate near 0.1%. Before evaluation, define one opportunity as one full, human-verified image from an independent capture group/session at the frozen operating threshold. Treat adjacent frames and repeated views as one cluster, report both image- and cluster-level counts, and use a group/session-clustered interval rather than pretending correlated frames are independent.

Physical and field outcomes remain later system-level gates and cannot be claimed by an image classifier/localizer. Preserve the requested numerical targets for that later work: 95th-percentile total ground-target error no greater than one-third of the smallest acceptable target/nozzle radius; at least 90% weed hit, at most 0.5% crop contact, and at least 70% treated-area reduction in a preregistered field trial; and 100% fail-closed behavior across the enumerated software/hardware fault suite. These are downstream acceptance requirements, not results of the current package.

## 2. How this compares with modern papers

Direct cross-dataset score comparisons are usually invalid. The defensible comparison has two columns: an official same-protocol benchmark and a harder local grouped test.

### Same-protocol public benchmark

The [RiceSEG paper](https://arxiv.org/pdf/2504.02880) reports an 80/20 image-random split. Its strongest listed model, Mask2Former, reaches 74.69 mIoU and 83.85 mAcc; the best reported weed IoU is 65.73. A meaningful public-benchmark result would:

1. reproduce the authors' split/protocol for context;
2. exceed 74.69 mIoU and 65.73 weed IoU under exactly that protocol;
3. separately publish source-photo/country-held-out results, without pretending they are numerically interchangeable with the image-random benchmark.

The grouped result is scientifically more valuable even if its raw score is lower.

### Context-only rice-weed detectors

- [PHRF-RTDETR](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1556275/full) reports 76.6% mAP50:95 and 88.2% mAP50 on its 986-image, five-weed upland-rice dataset. The data are request-only, so AgriNav cannot currently reproduce the comparison.
- [GE-YOLO](https://www.mdpi.com/2076-3417/15/5/2823) reports 93.1% mAP on 995 images from two Hunan sites under its own protocol. Its paper says the data cannot be shared.

These numbers are context, not thresholds AgriNav may claim to beat. Different weed taxonomies, image geometry, grouping, negatives, and evaluators can change AP dramatically. AgriNav's publishable advantage should be the controlled transfer effect, source-grouped generalization, mask-based protection semantics, negative/OOD evaluation, and reproducibility.

## 3. Requirements extracted from the July 20 roadmap

The authoritative requirements are now narrowed to the perception scope.

### P0 - scope and semantics

- Protected class currently means cultivated rice only. Additional plant classes require new data and independent results.
- The output is an image-space proposal with uncertainty and provenance, never a treatment command.
- Biological identity and decision role are separate.
- Canonical classes are defined in [`data/ontology.v1.json`](../../data/ontology.v1.json).
- Unknown, ambiguous, missing, or OOD evidence produces no affirmative target.

### P1 - data foundation

- Preserve immutable raw, human annotation, split/manifest, and reproducible derived layers.
- Record image hash, dimensions, source/version/license, site/field/session/pass/frame/source photo/camera, conditions, class state, verified-empty state, annotators, review, and versions. Unknown values remain explicit `null`.
- Group by site/session/pass/video/source photo before splitting. No sibling tile, burst, or frame family crosses partitions.
- Retire the historical detector test. Seal a new in-domain test and a separate hard/OOD challenge set.
- Begin development/challenge collection with 20-30% verified negative or nonactionable frames distributed across independent groups.

### P2 - annotation evidence

- Masks/polygons are canonical for `rice_protect` and `weed_target`; boxes are derived.
- Pilot 100-300 images with two independent annotators and adjudication.
- Double-label 10-20% of routine data and 100% of locked high-risk test/challenge labels.
- Starting gates: class/status agreement >= 98%, median box IoU >= 0.85, median mask IoU >= 0.80, and invalid/duplicate/out-of-bounds rate < 0.1%.
- Preserve proposal model/revision/prompt/threshold/original output and every human edit.

### P3 - software correctness

- Fix the known translation-direction defect and verify every joint image/box/mask transform to within one pixel.
- Match assignment/loss names to official semantics or rename them honestly.
- Keep one preprocessing/postprocessing/evaluator path across train, validation, inference, and export.
- Pass detector tiny-overfit AP50 >= 0.95 and AP75 >= 0.80; pass segmentation pixel accuracy >= 0.98 and present-class IoU >= 0.90 before full runs.
- Standard COCO AP@[.50:.95] at maxDets100 is the primary detector metric.

### P4 - experiment governance

- Preregister the hypothesis, primary contrast, metrics, slices, seeds, split hashes, compute/tuning budget, stopping rule, raw/EMA choice, threshold selection, statistics, and invalidation rules.
- Tune on grouped validation only. Lock the conclusion and one checkpoint before opening the sealed test once.
- Report all conditions and seeds, not only the best run.

The full acceptance matrix remains in the [July 20 roadmap](../audits/2026-07-20/AGRINAV_DEPLOYMENT_ROADMAP_2026-07-20.md); this document is the perception-only execution layer.

## 4. Comparable data ranked by value

The machine-readable, license-aware registry is [`data/dataset_registry.v1.json`](../../data/dataset_registry.v1.json). Do not concatenate these sources into one anonymous training pool.

| Priority | Dataset | Why it fits | Correct role | Main risk |
|---:|---|---|---|---|
| 1 | [Ma/Deng rice seedlings + weeds](https://figshare.com/articles/dataset/rice_seedlings_and_weeds/7488830) | closest early-paddy, low-camera view; rice/weed/water masks | RiceSEG companion segmentation bridge | only 28 parents -> 224 tiles; parent leakage |
| 2 | [RiceS](https://github.com/aipal-nchu/RiceSeedlingDataset) | 22,438 rice boxes in 600 early-stage UAV samples | rice-protect localizer warm-up | one field/repeated regions/dates; archive terms require review |
| 3 | [WeedyRice-RGBMS-DB](https://data.mendeley.com/datasets/vt4s83pxx6/1) | 734 expert-corrected masks in Vietnam rice fields | hard phenotype-confusion/OOD set | UAV domain gap; weedy rice is not ordinary weed truth |
| 4 | [MH-Weed16](https://data.mendeley.com/datasets/d3n3mgjjbv/2) | broad 16-species boxed-weed vocabulary in India | generic weed head/feature pretraining | soybean crop must never become protected rice |
| 5 | [Sorghum weed segmentation](https://data.mendeley.com/datasets/y9bmtf4xmr/1) | monocot crop, grass and broadleaf masks | auxiliary segmentation transfer control | only 252 source images |
| 6 | [Bangladesh rice-field weeds](https://data.mendeley.com/datasets/mt72bmxz73/4) | 3,632 rice-field weed images over 11 species | appearance pretraining or annotate a curated subset | classification labels only; grouping metadata must be recovered |
| 7 | [WE3DS](https://zenodo.org/records/7457983) | 2,568 ground-robot RGB-D crop/weed masks | generic segmentation/depth ablation | no rice; domain and ontology gap |
| 8 | [CropAndWeed](https://github.com/cropandweed/cropandweed-dataset) | >8k images, ~112k instances, boxes/masks/stems/session metadata | large research-only localization auxiliary | non-commercial license and no rice |
| 9 | [Paddy panicle](https://zenodo.org/records/4444741) | 400 4K paddy images with panicle masks | optional late-stage rice-protect auxiliary | no weeds; 32.8 GB full release |

Contact-only high-match sources:

- [GE-YOLO](https://www.mdpi.com/2076-3417/15/5/2823): 995 boxed images; authors report no sharing permission.
- [PHRF-RTDETR](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2025.1556275/full): 986 boxed upland-rice images; inquire with authors.
- [Huang et al. rice/weed UAV segmentation](https://www.mdpi.com/2072-4292/15/23/5615): 91 original images but the published Figshare path is not a usable dataset record; resolve with authors before use.
- WeedDet's >6k source set has no public licensed archive located and contains internet-sourced imagery; it is a provenance/overlap risk, not an ingest target.

### Recommended staged bundle

1. **Semantic calibration:** RiceSEG + Ma/Deng, with dataset-specific label adapters and balanced sampling.
2. **Rice localization:** RiceS, grouped by mission and spatial region.
3. **Weed visual vocabulary:** MH-Weed16 boxes plus a newly mask-annotated, diverse Bangladesh subset.
4. **Hard challenge:** WeedyRice, kept separate and treated as phenotype ambiguity/OOD.
5. **Target adaptation:** local COCO plus new local sessions, masks, verified negatives, and source metadata.
6. **Evaluation:** sealed local site/session test only; no external training image or derivative enters it.

The Hugging Face Dataset Viewer verified the public mirrors' sizes/schemas, but an important trap emerged: the generic Data Studio/Parquet rows for Voxel51 CropAndWeed, WE3DS, and WeedsGalore expose image media rows rather than the complete FiftyOne label objects. Use the original release or the documented FiftyOne loader when annotations are required.

## 5. Fast annotation system

### Recommended production-of-labels flow

```text
FiftyOne group/diversity selection
        -> proposal method bakeoff
        -> CVAT masks + independent review/adjudication
        -> canonical COCO-instance + semantic export
        -> automated schema/geometry/group validation
        -> FiftyOne error/duplicate/slice audit
        -> next active-learning batch
```

The pilot is pinned in [`configs/annotation/pilot_v1.json`](../../configs/annotation/pilot_v1.json), and the boundary/review rules are in [`docs/annotation_guide.md`](../annotation_guide.md).

### Tool choice

| Tool | Use here | Why | Constraint |
|---|---|---|---|
| [CVAT](https://docs.cvat.ai/) | canonical label/review store | masks/polygons, COCO instance export, consensus and quality workflows, custom auto-annotation SDK | setup heavier than a desktop labeler |
| [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) | fastest Windows proposal pilot | native GUI, batch inference, COCO import/export, LocateAnything/Grounding DINO/SAM integrations | GPL-3.0; weaker governance than CVAT |
| [FiftyOne](https://docs.voxel51.com/) | curate, deduplicate, sample, inspect, evaluate | uniqueness/similarity/error slices and CVAT round trip | not the primary dense-mask editor |
| [LocateAnything](https://huggingface.co/nvidia/LocateAnything-3B) | proposal challenger | open-vocabulary dense boxes/points and a released batch runtime | Linux/NVIDIA-oriented, no confidence scores in the current FiftyOne wrapper, no masks, non-commercial research weights |
| [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) | default open local proposal path | text boxes -> SAM2.1 masks, JSON/RLE output, slicing support for dense small objects | prompt/threshold misses must be measured |
| [SAM 3](https://github.com/facebookresearch/sam3) | direct text-to-mask challenger | category/text plus mask proposals | gated/heavy; record model license and hardware |
| [Label Studio](https://labelstud.io/) | fallback/custom workflow | flexible ML backend and interactive SAM | more plumbing; its brush-mask tutorial does not directly export COCO |

LocateAnything is worth testing, not adopting by reputation. Its official model is 3B, about 7.8 GB of weights, officially Linux-oriented, and licensed for non-commercial research. It emits boxes/points rather than the required masks. The quickest Windows route is X-AnyLabeling on Windows with the LocateAnything server in WSL2; its boxes then prompt SAM2.1, followed by human correction.

### What Codex can do internally

Within the current workspace, Codex can:

- inventory ZIPs without modifying raw data;
- compute hashes, mask-class distributions, source-photo groups, and detector family imbalance;
- build a source-diverse annotation manifest and optionally materialize only selected media;
- normalize proposal provenance, validate COCO/mask geometry, detect duplicate/group overlap, and generate QA reports;
- inspect sampled overlays and failure cases with the local image viewer;
- maintain the research protocol, dataset registry, and machine-readable acceptance evidence.

There is no installed LocateAnything/CVAT annotation connector in this Codex session, and generative image tools are not ground-truth annotators. Heavy proposal inference still needs a local/WSL/Colab GPU environment and accepted model licenses.

### Verified local pilot intake

`scripts/build_annotation_pilot.py` was run against the two received archives on 2026-07-20 with a 200-image target. The read-only source audit and deterministic selection passed:

- RiceSEG: 3,078 RGB/mask pairs, 773 inferred source-photo groups, and 1,295 proposal weed-positive tiles; every opened RGB/mask pair had matching dimensions and valid mask IDs.
- detector export: 1,347 images, 39,556 in-bounds boxes, 144 proposal weed-positive images, and eight provisional filename capture families; every image opened successfully and matched its COCO dimensions.
- selected pilot: 100 images from each source, 102 proposal weed-positive, 179 proposal-hard flagged, 108 represented provisional groups, and 200 unique per-image SHA-256 values.
- frozen gold intake: 40 images across 28 provisional groups (20 per source; 23 proposal weed-positive), assigned before any new proposal bakeoff.
- safety state: all 200 `verified_empty` and `positive_weeds` values remain `null`; zero samples are eligible for training or evaluation before human review.

The generated local manifest and schema-valid manual-first JSONL intake are `artifacts/annotation_pilot_v1/manifest.json` and `artifacts/annotation_pilot_v1/annotation_intake.jsonl` (ignored by Git because the manifest contains machine-local archive paths). The required 25% verified-negative/nonactionable quota is **not yet met or measurable**: absence of an upstream weed proposal is not a human-verified negative. Add and review independent negative sessions before the proposal bakeoff.

### Proposal bakeoff

Use 200 source-group-diverse images and freeze 40 independently annotated gold images before model inference. Include at least 35% weed-positive, at least 25% verified-negative/nonactionable, and at least 30% hard conditions. Compare:

1. LocateAnything -> SAM2.1;
2. Grounding DINO -> SAM2.1;
3. SAM3 text-to-mask.

Primary selection metric: median human seconds per accepted image. A method is eligible only if it attains >=95% weed recall at IoU 0.50, reports recall at 0.75, meets median mask IoU >=0.80, and passes schema/geometry QA. Count deleted false proposals and human-added missed weeds explicitly.

## 6. One-stage or two-stage?

Do not switch the whole project to a two-stage detector by assumption. Make the stage choice an ablation.

### Recommended ladder

1. **SegFormer or DeepLabV3** - primary RiceSEG-aligned semantic baseline for `rice_protect` and `weed_target` masks.
2. **FCOS** - maintained one-stage, anchor-free detector for the cleanest transfer-learning measurement.
3. **Faster R-CNN** - two-stage accuracy/control baseline with the same backbone/FPN/input and equal budget as FCOS.
4. **Mask R-CNN** - first instance-mask control after instance annotations exist.
5. **RT-DETRv2** - real-time/NMS-free challenger after data and correctness gates pass.
6. **Repaired WeedDet** - only after its transforms, assignment, loss semantics, evaluator, and tiny-overfit gates are corrected.

The likely final perception system is mask-first, not a single box detector. RiceSEG supplies semantic supervision, and physical protection needs plant boundaries. A one-stage FCOS/RT-DETR branch may still be the best runtime localizer, while a segmentation head supplies the protection/target masks. The novelty should be the transfer/generalization evidence, not merely choosing one-stage or two-stage.

### Fair stage ablation

Keep fixed:

- backbone and FPN where the frameworks permit it;
- input size and preprocessing;
- grouped manifests and augmentations;
- optimizer/search budget, seeds, stopping, and evaluator;
- operating threshold constraint;
- parameter, memory, and latency reporting.

Then compare FCOS versus Faster R-CNN by overall/per-class AP, AP75, small-weed recall, negative false proposals, calibration, and p99 latency. Pick the smallest model that meets the mask/recall/risk requirements; do not choose by AP50 alone.

## 7. Controlled RiceSEG transfer study

| Condition | Initialization path | Causal question |
|---|---|---|
| A | random -> target | pure scratch baseline |
| B | ImageNet -> target | generic-pretraining baseline |
| C | random -> RiceSEG -> target | RiceSEG effect without ImageNet |
| D | ImageNet -> RiceSEG -> target | incremental RiceSEG effect beyond ImageNet |

Use three paired seeds for screening and five paired seeds for the frozen comparison. Equalize downstream data, augmentations, steps/epochs, hyperparameter trials, early stopping, and model-selection rules. Record exact weight-loading coverage; a partial or mismatched load invalidates a run.

Primary contrast: `D - B` in standard COCO AP on the locked grouped evaluation. Supporting contrasts:

- `C - A` for domain pretraining without generic pretraining;
- RiceSEG effect on weed-target and rice-protect IoU;
- interaction with FCOS versus Faster/Mask R-CNN;
- low-label curves at 10%, 25%, 50%, and 100% of target groups;
- unseen country/site/session and tiny-weed slices.

Use paired seed differences and cluster bootstrap by source group/session. Report all individual seeds, mean/spread, effect CI, learning curves, and failures. The strongest novelty would be a positive transfer effect replicated across a one-stage detector and a mask model, especially in the low-label and unseen-site regimes.

## 8. Immediate stop/go sequence

1. Approve the ontology and pilot guide; fill unknown operating-domain fields.
2. Run the annotation-pilot builder against the two received ZIPs; inspect its group and quota report.
3. Add genuinely independent local negative and weed-rich sessions; the current detector archive cannot supply them.
4. Import the 40-image gold subset into CVAT and double-label it.
5. Run the three proposal methods, time corrections, and choose one under the weed-recall gate.
6. Repair transforms/RiceSEG grouping and pass tiny-overfit tests.
7. Run FCOS/Faster R-CNN plus a semantic model with three screening seeds.
8. Freeze the A-D five-seed protocol, select on validation, and open the new sealed test once.

Do not spend another full GPU run on the historical split, use the historical test for a new claim, convert empty model output into a negative, train on unreviewed proposals, or compare one best RiceSEG run with one ImageNet run.
