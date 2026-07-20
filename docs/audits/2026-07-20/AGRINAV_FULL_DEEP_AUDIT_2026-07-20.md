# AgriNav / WeedDet Deep Technical Audit

**Audit date:** 2026-07-20  
**Revision:** Complete-data update after RiceSEG and WeedDet v2 archives were supplied  
**Repository snapshot:** commit 2a056ce on local master  
**Additional evidence:** executed T7d notebook and checkpoint; complete RiceSEG image/mask archive and pretraining notebook; complete COCO detector split; exact Drive copies of the training source files  
**Review scope:** source code, notebooks, documentation, Git history, curated images, manifests, archives, checkpoint structure, recorded run outputs, and relevant current research and safety guidance  
**Review mode:** diagnostic only; no project source files were changed

---

## 1. Executive verdict

AgriNav contains a serious and potentially useful research effort, but the folder is not yet a complete autonomous-tractor project and the WeedDet subsystem is not ready to control a sprayer. The strongest interpretation of the project is:

1. an academic investigation of whether in-domain RiceSEG pretraining improves rice detection; and
2. an eventual perception component for selective treatment in flooded paddy fields.

The current repository is an early, custom, one-class object-detection experiment supporting those goals. It does not contain navigation, localization, motion planning, obstacle detection, actuation, nozzle timing, safety supervision, or a verified system interface. Its present inference script turns every pixel outside predicted rice boxes into a “weed zone.” That inversion is unsafe in principle and especially unsafe with the attached checkpoint, whose visualizations produce 300 detections per image, many at displayed confidence 1.00, while achieving only 0.0884 validation AP50 and 0.0010 AP75 under the project’s custom AP50/AP75 protocol.

Two confirmed current-code defects must be addressed before interpreting another training run:

- The translation augmentation moves image content and boxes in opposite directions.
- The deployable inference script is architecturally and geometrically incompatible with the current model and implements an unsafe spray-by-default policy.

Several additional defects and design mismatches materially weaken the experimental conclusions:

- The loss named “Varifocal Loss” does not implement the published Varifocal Loss weighting, and the active hard-positive configuration is not IoU-aware classification.
- ATSS top-k selection is performed across 12 co-located anchor shapes, which can select shapes at one cell instead of diverse spatial candidates.
- The “one positive per ground-truth object” fallback can silently lose a ground-truth assignment when two objects choose the same anchor.
- Evaluation mixes standard AP at maxDets 100 with AP50/AP75 at maxDets 300, while model selection is performed on the customized AP50.
- The actual T7d notebook ran 100 epochs even though the repository’s current notebook specifies 32.
- The archived weeddet_v6b.py from the exact imported Drive directory is byte-for-byte identical to the repository file, directly linking T7d to the translation-label bug.
- The claimed “dHash leakage-safe” COCO split is not source-independent: seven identifiable capture series span all three splits, including at least 73/134 validation images and 71/134 test images from series also represented in training.
- The saved checkpoint is valid for inference but is not a complete resumable or reproducible training artifact.
- Dataset provenance, the split-generation script/manifest, a standalone RiceSEG backbone artifact, and the final paper source are absent.

The project should therefore be treated as **research in progress with invalidated or unverified causal claims**, not as a finished detector and not as an autonomous treatment system.

### Overall readiness

| Area | Assessment | Principal reason |
|---|---:|---|
| Archive recoverability | Conditional pass | Committed Git snapshot was reconstructed, but the uploaded outer ZIP is truncated |
| Source syntax | Pass | All 14 Python files compile; all 7 repository notebooks and the added T7d notebook are valid JSON with syntax-valid code cells |
| Training-code correctness | Fail | Confirmed image/box translation mismatch |
| Model/evaluation correctness | High risk | Custom loss semantics, ATSS candidate geometry, collision bug, mixed maxDets protocol |
| Reproducibility | Fail | No locked environment, data hashes, complete run configuration, full-state checkpoint, or automated experiment record |
| Data file integrity | Pass | All supplied RGB/mask and COCO image/annotation pairs are present, readable, and structurally valid |
| Data governance | Fail | Missing source licenses/versioned provenance; COCO split has confirmed capture-series leakage |
| Scientific validity | Not established | One seed, tuned comparisons, no untouched test result, no confidence intervals, missing reference baselines |
| Inference compatibility | Fail | Stale model import, stale input geometry, stale anchor assumptions, incompatible checkpoint expectations |
| Spray safety | Stop condition | “Not detected as rice” is treated as “spray,” with no uncertainty or independent safety layer |
| Full AgriNav integration | Not present | No navigation, sensing fusion, planning, control, or actuation implementation |
| Paper readiness | Fail | Paper source, final figures, reproducible tables, and defensible locked protocol are absent |

### Immediate decision

Do not connect inference/inference_rice.py or its red “weed zone” output to an actuator. Preserve the current work as exploratory evidence, fix and test correctness first, then rerun the core ablation campaign under a locked protocol.

---

## 2. Evidence labels and severity

Findings use the following labels:

- **Confirmed defect:** directly demonstrated from current source or attached outputs.
- **Confirmed discrepancy:** two project artifacts make incompatible claims.
- **Evidence gap:** the project makes a claim that cannot be verified from supplied artifacts.
- **Design risk:** code can run as written, but its semantics conflict with the intended goal or established method.
- **Hypothesis:** plausible explanation that requires measurement or ablation.
- **Historical-only defect:** present in archived code but not necessarily in the active pipeline.

Severity:

- **P0:** unsafe output, experiment-invalidating defect, or blocker to meaningful use.
- **P1:** major correctness, scientific-validity, or reproducibility defect.
- **P2:** maintainability, performance, or workflow problem that will predictably slow progress.
- **P3:** cleanup or documentation issue with limited immediate technical effect.

---

## 3. Audit scope and archive recovery

### 3.1 Uploaded project archive

The uploaded file agrinav_full (2).zip is 104,550,400 bytes with SHA-256:

**fc61ee37bb93f29619a6d51712caa2a4b882d3036745811f11812f138959b752**

The archive has no end-of-central-directory record and is physically truncated. A sequential local-header recovery found:

- 854 complete archive entries
- 637 recovered files
- 94,166,530 recovered uncompressed bytes
- truncation inside agrinav_full/data/rice_training_curated_source.zip
- the incomplete nested member was discarded rather than treated as valid

The embedded .git object database is intact. Git object validation found only one dangling tree, not a corrupt reachable commit. A complete committed HEAD snapshot was reconstructed from Git objects:

- HEAD: 2a056ce
- branch: local master
- tracked files: 288
- tracked content: approximately 39.8 MB
- local master: three commits ahead of the locally recorded origin/master
- change from recorded origin/master: 281 files changed, 96,450 insertions, 121 deletions

Because the truncation occurs in an untracked duplicate data archive, all content committed at HEAD was recoverable. It remains impossible to know whether additional untracked files originally followed the truncated member. This audit can be exhaustive for the committed snapshot and recovered pre-truncation files, but not for unknowable post-truncation untracked content.

The outer ZIP truncation is itself a workflow warning. It is consistent with an interrupted or size-limited transfer, but the exact cause cannot be proven from the bytes alone.

### 3.2 Git state

Reachable history:

| Commit | Date | Summary |
|---|---|---|
| 2a056ce | 2026-07-15 | Handoff correction |
| a95ceeb | 2026-07-15 | T2–T7c campaign cleanup and reorganization |
| 77e0f58 | 2026-07-09 | V7 era, COCO split, RiceSEG pretraining, hard target and ATSS changes |
| f9c7cc3 | 2026-04-09 | Recorded origin/master initial commit |
| aba64a4 | 2026-04-09 | Recorded origin/main upload commit |

The configured remote URL names Autonomous-tractor-system, but remote state could not be verified. Statements in this report about “origin” refer only to the locally stored remote-tracking references, not the current live GitHub repository.

The working Git repository was stored under OneDrive. The archive contains multiple signs of file-transfer or synchronization damage: the missing ZIP directory, an index that cannot be trusted as a normal working index, trailing NUL bytes in a committed Markdown file, and physically truncated documentation. Active Git work should be moved out of OneDrive.

### 3.3 Additional T7d evidence

The separately supplied checkpoint bundle is valid:

- bundle SHA-256: 4373337af0214ce822992f69a3deeee96e2334c69cd6254eb0fea313005d4638
- checkpoint SHA-256: 43e47cf234d857f463b1d4e8edc6e89b68e08f29707f5af1914933d223988e1f
- curve image SHA-256: fc6cd66e810fcfacbe02696b5fe7c575309d592d85a22abaec0d2a2fc2fc085a
- checkpoint size: 211,756,791 bytes
- internal PyTorch archive: structurally valid, 742 members

The separately supplied T7d notebook is valid JSON and all code cells parse. Its code is almost the same as the repository active notebook. The meaningful source difference is that the supplied executed notebook sets NUM_EPOCHS to 100, while the repository notebook sets 32. The supplied notebook has outputs but all execution_count fields are null; this may result from export cleanup, but it removes a normal provenance signal.

### 3.4 Complete-data archive evidence

The two later archives are structurally valid.

#### RiceSEG pretraining bundle

- outer archive SHA-256: 2b08eabe9c4011fdb7b4cfab2f3b6fbe8536172b61b263f1c18a1c00db40883a
- nested RiceSEG.zip SHA-256: 0071d9f941508afd9a86aa5ef740433dee938e1c7a9508a191d3e46a2909be96
- contents: run_pretrain.ipynb, riceseg_pretrain.py, and the full 3,078-pair RiceSEG archive
- riceseg_pretrain.py SHA-256: 066c58437ef22b1b52f16546a39b6aa38c47f2aa864552d6f1172cbc97150b48
- the supplied script is byte-for-byte identical to training/riceseg_pretrain.py at repository HEAD
- the bundle does not contain the exported riceseg_backbone.pth

#### WeedDet v2 Drive bundle

- outer archive SHA-256: e774c6e4bb4de3ecfa8c771b708a32f795e1e3b0e794e41a6dd000c0816e01f8
- two copies of rice_detection_coco_split.zip are byte-for-byte identical, each with SHA-256 c5a57ef79473801b2d4a7aa5d843f6d352f408c306fc3a1a6647d6f69586f6ec
- the duplicated archives waste approximately 116.4 MB uncompressed in the Drive folder
- weeddet_v6b.py SHA-256: 51b89ce44842256f8d4bc1a3b2f30c13a88b5ef5c76ecb1471a848a6b0e5dd64
- the supplied weeddet_v6b.py is byte-for-byte identical to models/weeddet_v6b.py at repository HEAD
- six CPython 3.12 cache files are present for historical model variants, but their source files are absent
- despite the folder name weeddet_v2_checkpoints, this bundle contains no .pth, .pt, or .ckpt model checkpoint

The T7d notebook says it imports /content/drive/MyDrive/weeddet_v2_checkpoints/weeddet_v6b.py. The supplied file comes from that exact directory, predates the run, and matches repository HEAD. This closes the previous model-source provenance gap for the current defect analysis.

---

## 4. What the project actually contains

### 4.1 Tracked-content inventory

The committed snapshot contains:

| Type | Count |
|---|---:|
| Markdown | 19 |
| Python | 14 |
| Jupyter notebooks | 7 |
| JPEG images | 211 |
| PNG images | 34 |
| CSV manifests | 1 |
| requirements file | 1 |
| .gitignore | 1 |
| **Total** | **288** |

Top-level concentration:

- data: 249 tracked files, predominantly curated images
- archive: 17 files
- active: 2 files
- models: 2 files
- training: 2 files
- inference: 2 files
- paper: 2 files
- drive_links: 1 file
- root documentation/configuration: 13 files

### 4.2 Reconstructed project goal

The documents describe a CEN 4930 / FGCU AgriNav autonomous paddy-field tractor. Within that larger idea, this repository is Benny’s WeedDet perception research:

- identify rice plants;
- protect detected rice;
- eventually enable selective treatment of non-crop areas;
- investigate RiceSEG in-domain pretraining;
- produce an IEEE-style research paper.

That is not the same as a complete autonomous tractor. The folder contains no operational implementation of:

- ROS or another robotics middleware;
- GNSS, RTK, LiDAR, wheel odometry, IMU, or sensor fusion;
- camera calibration or camera-to-ground projection;
- localization or mapping;
- row following, path planning, coverage planning, or obstacle avoidance;
- vehicle control, steering, speed control, or watchdogs;
- pump/nozzle drivers, spray timing, flow control, or treatment logging;
- human/animal/vehicle detection;
- safety PLC, emergency stop, geofence, or supervisory state machine;
- simulator or hardware-in-the-loop validation;
- cross-team interface definitions.

The name agrinav_full therefore overstates the supplied implementation. A defensible title today would be “AgriNav WeedDet Perception Research.”

### 4.3 Current model path

The active model in models/weeddet_v6b.py is a custom dense, anchor-based detector:

- custom DetResNet50-style backbone stem;
- enhanced FPN at strides 4, 8, and 16;
- ERetinaHead;
- 12 anchor shapes per feature-map cell;
- four aspect ratios: 0.2, 0.33, 0.5, and 1.0;
- three scale multipliers;
- custom ATSS-style assignment;
- custom focal-like classification loss labeled Varifocal Loss;
- Smooth L1 plus CIoU regression;
- hard target 1.0 for positive classification in the active configuration;
- RiceSEG initialization with the backbone frozen in T7/T7d.

At 512 by 512 input resolution, the three maps contain:

- 128 by 128 cells at stride 4;
- 64 by 64 cells at stride 8;
- 32 by 32 cells at stride 16.

With 12 anchors per cell this produces:

**(128² + 64² + 32²) × 12 = 258,048 anchors per image**

That anchor volume is central to both the assignment behavior and the inference bottleneck.

---

## 5. P0 findings

### P0-1 — Translation augmentation corrupts labels

**Label:** Confirmed defect  
**Location:** models/weeddet_v6b.py, lines 1076–1088  
**Impact:** Current training samples can contain images and boxes translated in opposite directions.

The code calls Pillow affine transformation with:

    (1, 0, dx, 0, 1, dy)

Pillow’s affine tuple maps output coordinates back into input coordinates. A positive dx therefore samples from a position to the right and makes visible content move left. The code then adds positive dx to every box coordinate, moving labels right.

A direct synthetic pixel test confirmed:

- dx = +2 moved a bright pixel from x = 5 to x = 3;
- dx = -2 moved it from x = 5 to x = 7.

The boxes are adjusted in the opposite direction. Translation is applied to approximately half of augmented samples that contain boxes, with offsets up to 10% of image width and height. Image-to-label displacement can therefore approach 20% of a dimension when considering the relative separation between content and its box.

This is not augmentation noise; it is systematic target corruption.

Correct alternatives:

1. To move visible image content by positive dx and dy, use negative dx and dy in Pillow’s inverse affine tuple, while continuing to add positive offsets to boxes.
2. Keep the current affine tuple and subtract dx and dy from boxes.
3. Prefer a tested detection transform library that applies one sampled transform jointly to image, boxes, labels, masks, and keypoints.

Required unit test:

- create a small image with a known rectangle and identical box;
- apply fixed positive and negative translations;
- verify the transformed non-background bounding rectangle and target box coincide within one pixel;
- verify clipping removes or retains the same object and its label together.

Historical impact is now high-confidence rather than hypothetical. The attached T7d notebook imported weeddet_v6b.py from /content/drive/MyDrive/weeddet_v2_checkpoints. The later archive of that exact Drive directory contains a source file that predates the run and is byte-for-byte identical to repository HEAD, including the faulty transform. T7d therefore trained with systematically corrupted translated samples. Earlier T1–T7 runs using the same Drive source must be treated as invalid unless a different exact source snapshot proves otherwise.

Reference: [Pillow affine transform documentation](https://pillow.readthedocs.io/en/stable/reference/ImageTransform.html).

### P0-2 — Inference is incompatible with the active model

**Label:** Confirmed defect  
**Location:** inference/inference_rice.py  
**Impact:** The shipped inference entry point cannot be trusted to load or correctly preprocess the active model.

Conflicts include:

| Concern | Active training | inference/inference_rice.py |
|---|---|---|
| Model module | models/weeddet_v6b.py | imports weeddet_for_VSCode from /content |
| Architecture generation | current v6b/V7 | historical V5-era module |
| Anchors | 12 per cell | historical head assumptions around 9 |
| Feature strides | 4, 8, 16 | historical 8, 16, 32 design |
| Input | 512-square letterbox | direct resize to 600 by 1000 |
| Coordinate restoration | remove padding and divide by scale | independent x/y scaling from stretched image |
| Evaluation threshold | 0.01 | 0.5 |
| NMS protocol | hard NMS 0.50 in active notebook | delegated to old model defaults |
| Checkpoint metadata | best checkpoint has epoch, AP, states, partial config | script requires ckpt loss |

Possible outcomes include immediate import failure, shape mismatch while loading, silent semantic mismatch, distorted geometry, or incorrect coordinates. Even if a historical checkpoint loads, it is not evidence that current V7/T7d inference works.

The repository needs one canonical model builder and one canonical preprocessing/postprocessing pipeline used by training evaluation, offline inference, deployment export, and tests. Checkpoint config must be validated before model construction.

### P0-3 — Spray-by-default logic is unsafe

**Label:** Confirmed design hazard  
**Location:** inference/inference_rice.py, lines 6–8, 103–137, 152–172  
**Impact:** A detector miss, unknown object, water, soil, shadow, person, animal, equipment, or out-of-distribution scene becomes a spray target.

The script:

- paints the full image red by default;
- clears only detected rice rectangles;
- labels all remaining pixels “weed zone” and “spray target”;
- declares the entire image a weed zone if there are no rice detections.

This reverses the burden of proof. “Not detected as rice” does not imply “weed,” and “weed” does not automatically imply “safe treatment location.”

The area calculation is also invalid:

- it sums rectangle areas, not plant masks;
- overlapping rice rectangles are double-counted;
- boxes include background;
- the complement of box area is not weed area;
- perspective makes image-area percentages unrelated to ground-area percentages.

The attached T7d evidence makes the hazard concrete:

- best checkpoint: AP50 0.0884 and AP75 0.0010;
- 300 detections per validation image at the evaluator cap;
- visualization labels show many scores as 1.00;
- dense duplicate boxes do not define protected crop geometry reliably;
- validation average recall at maxDets 100 is only 0.083 over IoU 0.50:0.95.

Safe minimum semantics:

- a rice detector may provide a crop-protection veto;
- a separate positive weed/treatment-zone model must provide affirmative evidence;
- uncertainty, out-of-distribution input, sensor failure, no detection, and stale data must all resolve to no spray;
- human/animal/vehicle detection, geofence, speed, pose quality, and actuator health must independently veto spraying;
- no image-space command should directly drive a nozzle without calibrated ground projection and timing.

This is a stop condition, not a metric-tuning issue.

### P0-4 — The core experimental campaign cannot yet support causal claims

**Label:** Confirmed scientific-validity failure  
**Impact:** The project cannot currently claim that RiceSEG pretraining improves WeedDet.

Reasons:

- confirmed current augmentation corruption;
- the exact T7d model source is now available and confirms that the run used the faulty augmentation;
- no independent test-set result;
- one seed per condition;
- no confidence intervals;
- RiceSEG and scratch conditions were not shown to receive equal tuning budgets;
- ImageNet factorial control is missing;
- training/evaluation protocol changed across runs;
- AP50/AP75 use a custom maxDets 300 setting;
- reference implementations and modern baselines are absent;
- the supplied split is now directly shown to leak related capture series across train, validation, and test;
- the split-generation script and immutable source-group manifest are still absent.

The pretraining result remains a useful hypothesis. It is not yet a defensible causal conclusion.

### P0-5 — The detector validation and test splits are capture-series contaminated

**Label:** Confirmed scientific-validity defect  
**Evidence:** complete supplied rice_detection_coco_split.zip  
**Impact:** Existing validation results are not independent, and the current test split cannot serve as a defensible untouched holdout.

The three JSON files and all 1,347 images are present and structurally valid. There are no exact image duplicates across splits. The problem is higher-level source leakage: related consecutive frames and images from the same named acquisition series are divided among train, validation, and test.

Seven unambiguous filename families occur in all three splits:

| Capture family | Train | Validation | Test |
|---|---:|---:|---:|
| 1a_image | 82 | 11 | 6 |
| 1b_image | 73 | 12 | 11 |
| 2-series | 72 | 4 | 9 |
| frame | 28 | 1 | 3 |
| seedlingCol_03 | 42 | 7 | 4 |
| seedlingCol_04 | 183 | 21 | 22 |
| weeds_seq | 130 | 17 | 16 |
| **Identifiable-series total** | **610** | **73** | **71** |

These 754 images are 56.0% of the whole dataset. At minimum, 54.5% of validation and 53.0% of test come from named series also represented in training. The remaining 593 numerically named images may contain additional sequences that cannot be safely grouped without source metadata.

The split does satisfy a narrow near-duplicate check:

- no exact image hashes cross splits;
- no cross-split dHash distance is below 6;
- 108 cross-split pairs have dHash distance 6–8.

That does not make it leakage-safe. For example, frame_0000 is in train, frame_0001 in test, and frame_0002 in train. They visibly show the same camera/platform and field in consecutive views, but dHash distances are 16–24. Likewise, seedlingCol_04_0004 and 0006 are in train while 0005 is in validation. dHash clustering catches near-identical pixels, not shared acquisition context.

Required action:

1. recover source dataset/session/video identifiers;
2. group all frames from one capture, field, camera pass, source dataset, and augmentation lineage;
3. create a new train/validation/test split at the highest available source-group level;
4. reserve geographically or temporally distinct sites for challenge/test evaluation;
5. archive the current split as historical and do not report its “test” result as independent;
6. rerun every model comparison on the new split.

---

## 6. P1 model and loss findings

### P1-1 — “Varifocal Loss” is not the published Varifocal Loss

**Label:** Confirmed discrepancy  
**Location:** models/weeddet_v6b.py, lines 643–662 and 826–845

The current positive weight is:

    target × absolute(target − probability)^gamma

In the official VarifocalNet implementation, positive examples are weighted by the target score itself; focal modulation is applied to negatives. More importantly, VarifocalNet trains an IoU-aware classification score, while the active project sets positive targets to 1.0 through cls_hard_target.

The active behavior is therefore a custom focal-style binary classification loss with hard positives, not published Varifocal Loss and not IoU-aware classification score learning. The code comments themselves show that predicted-IoU targets were disabled after poor results.

Why this matters:

- classification confidence is not trained to rank boxes by final localization quality;
- NMS receives many high-confidence boxes with weak localization;
- AP75 can remain near zero even when AP50 briefly rises;
- the attached visualizations show many 1.00 scores and 300 retained boxes.

Recommendation:

- rename the current loss honestly, for example HardTargetFocalLikeLoss;
- implement official VFL exactly in a separate tested class if it is a research condition;
- compare hard focal, official VFL/IACS, Quality Focal Loss, and a maintained baseline under one locked assigner and evaluator;
- log score-versus-IoU calibration and ranking correlation, not just AP.

References:

- [VarifocalNet paper, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html)
- [Official VarifocalNet loss implementation](https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py)
- [Generalized Focal Loss, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html)

### P1-2 — ATSS candidate selection collapses spatial diversity

**Label:** Confirmed design mismatch  
**Location:** models/weeddet_v6b.py, lines 712–759

ATSS selects the k nearest candidate anchors per level by center distance. The project has 12 anchor shapes at each cell. All 12 at a given cell share the same center and therefore have the same center distance to a ground-truth box.

With top-k equal to 9, the selection can be dominated by nine shapes from one spatial cell rather than nine nearby cells. Tie ordering can make the exact chosen shapes implementation-dependent. The candidate sample used to compute mean-plus-standard-deviation IoU is then not the intended spatial neighborhood.

This is particularly problematic because the ATSS paper argues that the positive/negative sample definition accounts for much of the apparent gap between anchor-based and anchor-free detectors, and its standard design does not require a large set of anchor shapes at every point.

T10’s proposed “per-cell candidates” is directionally sound, but it must be treated as a new assigner implementation:

- deduplicate candidates by center before top-k;
- select top-k cells per level;
- decide whether to evaluate one canonical anchor or all shapes at selected cells;
- compare against an official maintained ATSS implementation;
- test permutation invariance across anchor-shape ordering;
- log candidates and positives per level and per ground-truth object.

References:

- [ATSS paper](https://arxiv.org/abs/1912.02424)
- [Official ATSS repository](https://github.com/sfzhang15/ATSS)

### P1-3 — Forced-positive assignment can orphan a ground-truth object

**Label:** Confirmed defect  
**Location:** models/weeddet_v6b.py, lines 770–774

The fallback finds each ground-truth object’s best anchor and performs vectorized writes:

    pos[gt_best_anchor] = True
    best[gt_best_anchor] = gt_arange

If two ground-truth objects select the same best anchor, duplicate advanced indices are overwritten. Only one object remains associated with that anchor. The comment promises at least one positive anchor per ground-truth object, but the code does not guarantee it.

Required fix:

- resolve duplicate claims explicitly;
- assign the contested anchor by highest IoU;
- give losing ground-truth objects their next-best unclaimed anchor, or use a matching procedure;
- assert after assignment that every ground-truth index appears at least once;
- cover dense and overlapping synthetic cases.

### P1-4 — Anchor volume creates avoidable memory and latency pressure

**Label:** Design and performance roadblock

258,048 anchors per image imply:

- an anchor-by-ground-truth IoU matrix;
- an anchor-by-ground-truth distance matrix;
- large classification and regression outputs;
- many low-threshold candidates;
- expensive Python-level NMS.

The active checkpoint retains 300 detections per image at the cap. The underlying pre-NMS candidate count is not logged, so the evaluator cap does not reveal the actual overload.

Measure:

- candidates above threshold before NMS by level;
- candidates after top-k and after NMS;
- GPU memory by batch size;
- assignment time, forward time, decode time, NMS time, image I/O, and end-to-end latency;
- latency on intended hardware, not only Colab T4.

Then compare against:

- a maintained RetinaNet or FCOS baseline;
- RT-DETR, which avoids NMS;
- a smaller feature pyramid or anchor-free head.

Reference: [RT-DETR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html).

### P1-5 — Python NMS is a real-time blocker

**Label:** Confirmed implementation roadblock  
**Location:** models/weeddet_v6b.py, NMS and Soft-NMS helpers

The project performs iterative NMS in Python with repeated tensor operations and synchronization. With dense, low-threshold output, hundreds or thousands of loop iterations can dominate runtime. No end-to-end latency is reported.

Use a compiled/vectorized operation from a maintained framework or an end-to-end detector. Validate that exported runtime behavior matches evaluation behavior.

### P1-6 — Three conflicting postprocessing protocols exist

**Label:** Confirmed discrepancy

1. Model defaults use Soft-NMS, a low score threshold, and up to 1,000 detections.
2. Active notebook evaluation uses hard NMS at 0.50, score threshold 0.01, and max 300.
3. Stale inference uses a 0.50 score threshold and historical model defaults.

Reported accuracy, visual output, and deployment output are therefore different systems. Postprocessing must be a versioned configuration included in every checkpoint and evaluation record.

### P1-7 — Regression objectives are combined without measured balance

**Label:** Design risk / hypothesis  
**Location:** models/weeddet_v6b.py, lines 817–824

Smooth L1 is summed over four encoded components and CIoU is multiplied by the number of positives, then both are added with implicit equal coefficients. This is not automatically wrong, but their scales and gradients are not logged or ablated.

Given AP75 near zero, inspect:

- normalized Smooth L1 and CIoU contributions;
- gradient norms per head;
- decoded width/height distributions;
- IoU histograms for matched positives;
- localization errors by object size and aspect ratio.

Project documentation also describes Peng’s WeedDet as Smooth L1 plus GIoU, while current code uses CIoU. The project is a derivative, not a faithful reproduction, and should be described that way.

### P1-8 — Letterbox scale is slightly inconsistent with integer resize

**Label:** Confirmed low-magnitude defect  
**Location:** models/weeddet_v6b.py, letterbox_pil

The helper computes one nominal floating scale, rounds the resized width and height to integers, then uses the nominal scale for boxes. The actual x and y scales after rounding may differ slightly from one another and from the returned value.

This normally causes subpixel error, occasionally approaching roughly one pixel. It is not sufficient to explain the observed AP collapse, but it is unnecessary error for tiny objects and AP75 evaluation.

All supplied COCO detector images are already 512 by 512, so this rounding issue did not affect T7d training or validation. It remains a defect for arbitrary-resolution inference and future datasets.

Return actual_scale_x and actual_scale_y from the integer dimensions, and use the same values for forward and inverse transforms.

### P1-9 — ImageNet loading can silently fall back to random initialization

**Label:** Confirmed defect  
**Location:** models/weeddet_v6b.py, load_imagenet_backbone

An import or download error returns zero matched keys instead of raising. Callers do not consistently enforce the returned match count. An “ImageNet” experiment could therefore become a mislabeled scratch run.

Fail closed:

- require an explicit local weights artifact and hash;
- require exact expected backbone-key coverage;
- record matched, missing, and unexpected keys;
- abort if the condition is not satisfied.

### P1-10 — VOC label filtering can misalign labels and boxes

**Label:** Confirmed latent defect  
**Location:** models/weeddet_v6b.py, WeedDataset

Augmentation filters boxes but does not carry labels through the same keep mask. Later code uses either labels[keep] or a prefix fallback. If an interior box was removed during augmentation, the prefix fallback can associate the wrong class with remaining boxes.

The active one-class COCO path stores the label alongside the box and is less exposed. The generic VOC path remains executable and unsafe for multi-class use.

---

## 7. Attached T7d run audit

### 7.1 What actually ran

The supplied notebook reports:

- Google Colab T4 GPU;
- 1,079 train, 134 validation, and 134 test images;
- train boxes: category 1 = 31,013; category 2 = 285;
- validation boxes: category 1 = 3,669; category 2 = 45;
- test boxes: category 1 = 3,706; category 2 = 28;
- 512 by 512 letterboxed input;
- batch size 2;
- 100 epochs;
- SGD, base learning rate 0.0005, cosine schedule, 5% warmup;
- RiceSEG initialization;
- frozen backbone and frozen backbone batch normalization;
- 26.38 million parameters total, 2.88 million trainable;
- 54,000 total steps, 2,700 warmup steps;
- hard NMS 0.50, score threshold 0.01, max 300;
- EMA decay 0.999;
- seed 42.

The test split was not used; final evaluation remained on valid.

The exact imported Drive source is now available and matches repository HEAD, so this run can be tied to the current implementation. Because training used CocoWeedDataset with augment=True and every training image contains rice boxes, approximately half of sampled training images were eligible for the faulty translation on each epoch. T7d’s metrics are therefore measurements of a detector trained with corrupted image/box alignment. They should be preserved as debugging history, not used as a baseline for a paper.

The validation split is also capture-series contaminated as documented in P0-5. Leakage normally biases results upward, but its exact effect here cannot be isolated from the augmentation and model defects. The low AP must not be “corrected” analytically; the run must be repeated on a new grouped split.

### 7.2 T7d result

| Evidence | Result |
|---|---:|
| Best validation epoch by custom AP50 | 14 |
| Best AP50, maxDets 300 | 0.0884 |
| AP75 at selected epoch, maxDets 300 | 0.0010 |
| AP 0.50:0.95, maxDets 100 | 0.0166 |
| Small AP, custom maxDets 300 | 0.033 |
| Medium AP, custom maxDets 300 | 0.017 |
| Large AP, custom maxDets 300 | 0.000 |
| AR 0.50:0.95, maxDets 100 | 0.083 |
| AR 0.50:0.95, maxDets 300 | 0.126 |
| Detections per image | 300.0, exactly at cap |
| Raw model AP50 at best-checkpoint time | 0.0110 |
| EMA model AP50 | 0.0884 |
| Final epoch AP50 | 0.0181 |
| Final epoch AP75 | 0.0005 |

The final AP50 is approximately 79.5% below the selected best value. Training loss keeps declining from 4.2465 to 2.2627 while validation AP collapses after early peaks. This is a strong generalization or score-ranking failure signal, not evidence that longer training solves the problem.

The 100-epoch run also contradicts its own comment: “20-epoch gate first; extend only if AP is moving.” AP had already peaked and degraded by epoch 20, but training continued to 100.

### 7.3 Visual evidence

The notebook’s six prediction plots show:

- exactly 300 red predictions in every displayed sample;
- extensive duplicate and overlapping boxes;
- score labels rendered as 1.00 across much of the image;
- predictions on rice clusters, water, and background;
- very poor one-to-one correspondence with green ground-truth boxes.

This is stronger evidence than the scalar det/img value alone. det/img equal to 300 proves output censoring at the cap; the plots confirm severe duplicate/saturation behavior in the sampled cases.

The plots contain another bug: the visualization loads every annotation into gt_by_img without filtering category_id. The validation JSON reports 45 category-2 weed boxes, so some green “GT” boxes may be weeds even though the legend and model semantics say rice. Visualization cannot be used for class-specific qualitative claims until category filtering is fixed.

### 7.4 Metric protocol is internally mixed

The evaluator sets maxDets to [10, 100, 300]. pycocotools summary output shows:

- AP 0.50:0.95 at maxDets 100;
- AP50 and AP75 at maxDets 300;
- size-specific AP at maxDets 300.

The returned tuple and printed one-line summary place these values side by side without making the differing caps prominent. This is not a standard single COCO protocol.

The official COCO headline convention uses maxDets 100. Keep a locked standard table at maxDets 100, then report a separately named high-density diagnostic table if maxDets 300 is scientifically justified.

Reference: [official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

### 7.5 Checkpoint content

The best checkpoint contains:

- epoch = 14;
- AP50 and AP75 scalars;
- EMA state_dict;
- raw_state_dict;
- a small config dictionary.

The config records version, run tag, backbone initialization label, ATSS and hard-target flags, anchor base scale, base learning rate, class name, seed, and a free-text dataset name.

It does not contain:

- optimizer state;
- scheduler state;
- gradient scaler state;
- global step;
- RNG states for Python, NumPy, CPU Torch, and CUDA;
- epoch/batch sampler position;
- training and AP histories;
- weight decay, momentum, warmup, minimum LR, batch size, image size, epoch target, clipping, EMA decay, freeze setting, NMS, score threshold, or max detections;
- Git commit or source SHA;
- exact imported model-file SHA;
- dataset archive/JSON/image hashes;
- Python, PyTorch, torchvision, CUDA, cuDNN, driver, or GPU versions;
- command or notebook hash;
- license/provenance;
- optimizer-ready trainable/frozen parameter map.

It is an inference snapshot, not a reproducible experiment or resumable training checkpoint.

The notebook explicitly loads with weights_only=False. Only load trusted checkpoint files, and prefer a constrained state-dict format and current safe-loading defaults where possible.

### 7.6 EMA dependence

Raw AP50 0.0110 versus EMA AP50 0.0884 is an eightfold difference. EMA is not inherently a bug, but this degree of dependence needs investigation:

- verify update timing begins after a sensible warmup;
- compare EMA decay against total steps and effective half-life;
- confirm buffers and frozen batch-normalization state are copied as intended;
- evaluate raw and EMA throughout training, not only once;
- ensure the saved “best” model is always identified as EMA;
- do not describe the raw training trajectory as equivalent to deployed behavior.

### 7.7 Scores of 1.00

The visual score labels are rounded to two decimals, so they do not prove exact floating value 1.0. They do prove widespread scores of at least approximately 0.995. Combined with the loss semantics and 300-cap saturation, this supports a score-calibration and duplicate-suppression failure.

Record:

- full pre-sigmoid logit distribution;
- predicted score quantiles before and after NMS;
- score versus matched IoU;
- expected calibration error and reliability plots;
- precision-recall curves at operational thresholds;
- duplicate count per ground-truth object.

---

## 8. Notebook and experiment-workflow findings

### P1-11 — Data extraction is not validated

The active notebook extracts the dataset if one train annotation file is missing. It does not:

- verify archive SHA-256;
- validate every expected file;
- validate split image counts and annotation schema as a gate;
- prevent path traversal before extractall;
- compare manifest and JSON hashes;
- detect a half-extracted existing directory.

Use a manifest with expected hashes and extract into a temporary directory before an atomic rename.

### P1-12 — Runtime dependency installation is unpinned

The notebook installs pycocotools at runtime without a version pin. requirements.txt omits pycocotools entirely. A future Colab image can change behavior without a repository change.

### P1-13 — Reproducibility configuration is internally contradictory

Seeds are set, but cudnn.benchmark is enabled and deterministic algorithms are not requested. One seed is insufficient for a scientific comparison even with deterministic kernels.

Use:

- a locked software/hardware environment record;
- deterministic settings for the controlled comparison where practical;
- explicit documentation of nondeterministic operations;
- at least three, preferably five, independent seeds;
- confidence intervals over seeds and over bootstrapped test images.

Reference: [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

### P1-14 — Model selection optimizes the wrong objective

The best checkpoint is selected only by AP50 at maxDets 300. For crop protection and treatment:

- false-negative rice detections are safety-critical;
- localization quality affects protected-zone geometry;
- duplicate boxes affect timing and downstream logic;
- false-positive weed or treatment detections affect herbicide use;
- latency affects whether a command reaches the correct ground location.

At minimum, preregister a selection metric and safety constraints. A plausible research selection rule is standard AP 0.50:0.95 at maxDets 100 subject to minimum rice recall, then evaluate treatment metrics once on test.

### P1-15 — “Detections per image” is censored

When det/img equals max_dets, it reports a cap, not the underlying number of outputs. It cannot distinguish 301 from 30,000 candidates. It also cannot prove whether assignment, classification, NMS, or threshold is the root cause.

### P1-16 — No independent baseline

There is no executed, current:

- torchvision RetinaNet or FCOS baseline;
- YOLO-family baseline;
- RT-DETR baseline;
- official ATSS baseline;
- faithfully reproduced WeedDet baseline.

A custom architecture with many simultaneous changes cannot reveal which change matters. Establish a maintained reference baseline before additional custom components.

### P1-17 — No test result

The notebook correctly leaves EVAL_SPLIT as valid, but this means all current headline numbers are validation results. Documentation and paper claims must not present them as test performance.

### P1-18 — No statistical uncertainty or subgroup analysis

Missing analyses include:

- multiple seeds;
- bootstrap confidence intervals;
- per-site, country, camera angle, growth stage, illumination, occlusion, object-size, and density results;
- precision-recall and calibration curves;
- error taxonomy;
- per-image failure review;
- comparison on a locked hard-case challenge set.

### P1-19 — Notebook outputs and source are not self-authenticating

The notebook imported weeddet_v6b.py from Google Drive. It records the path, not the file contents or hash. The notebook source can also be edited after execution without invalidating existing outputs. Null execution counts further weaken provenance.

Every run should emit a machine-readable manifest containing:

- repository commit;
- dirty-tree diff hash;
- notebook or command hash;
- imported module file hashes;
- dataset JSON and image-manifest hashes;
- environment;
- full configuration;
- start/end timestamps;
- hardware;
- output artifact hashes.

---

## 9. RiceSEG pretraining audit

### 9.1 What is promising

RiceSEG is directly relevant in-domain visual material. The project correctly recognizes that a rice-specific representation may transfer better than generic ImageNet features. The active notebook also improved backbone loading by checking missing backbone keys rather than relying only on strict=False.

The RiceSEG dataset paper reports broad country, genotype, growth-stage, and class coverage, making it a reasonable source for a carefully controlled transfer-learning study.

Reference: [RiceSEG dataset paper, 2025](https://www.sciencedirect.com/science/article/pii/S2643651525001050).

### 9.2 Complete RiceSEG data integrity

The supplied RiceSEG archive resolves the earlier mask uncertainty:

- 3,078 RGB images and 3,078 matching masks;
- no unmatched images or masks;
- every image and mask is 512 by 512;
- all RGB inputs decode as RGB;
- all masks decode as single-channel L;
- every mask pixel is one of 0, 1, 2, 3, 4, or 5;
- no void value 255 or other out-of-range value occurs;
- no exact duplicate RGB images;
- 773 source-photo groups after removing tile suffixes.

Country coverage:

| Country | Tiles |
|---|---:|
| China | 1,120 |
| India | 600 |
| Japan | 704 |
| Philippines | 600 |
| Tanzania | 54 |

Pixel distribution confirms substantial class imbalance:

| Class | Pixel share | Images containing class |
|---|---:|---:|
| Background | 48.325% | 3,059 |
| Green vegetation | 43.418% | 3,036 |
| Senescent | 2.481% | 2,138 |
| Panicle | 3.380% | 565 |
| Weeds | 1.568% | 1,295 |
| Duckweed | 0.828% | 432 |

Six masks are byte-identical all-background masks attached to different RGB images. This is not duplicate-image leakage; it is a legitimate consequence of empty segmentation tiles, though those pairs should be visually sampled for annotation quality.

No exact RGB overlap exists between the 3,078 RiceSEG inputs and the 1,347 COCO detector images. The minimum cross-dataset dHash distance is 8; the closest pair was visually unrelated. There is no evidence of direct image reuse from detector validation/test into RiceSEG pretraining.

### 9.3 What the pretraining notebook actually did

The executed run_pretrain.ipynb has six executed code cells and retained outputs. It:

1. copied the exact repository WeedDet and RiceSEG scripts from Drive;
2. passed a 342-tensor backbone key and export/load round-trip self-test;
3. ran an eight-tile overfit exercise for 60 epochs;
4. ran a fresh 30-epoch full pretraining job on a Tesla T4.

The full run reports:

- 2,768 training tiles;
- 310 validation tiles;
- zero source-photo overlap under the implemented basename grouping;
- 288 of 342 backbone tensors initialized from ImageNet;
- best validation mIoU 0.5749 at epoch 27;
- final validation mIoU 0.5651 at epoch 30.

At the selected epoch, class IoUs were approximately:

- background 0.90;
- green vegetation 0.86;
- senescent 0.36;
- panicle 0.75;
- weeds 0.23;
- duckweed 0.35.

This is valid evidence that the segmentation task learned nontrivial features. It is not a held-out test result, and it does not by itself prove downstream detection benefit.

The condition called “RiceSEG initialization” is actually **ImageNet → RiceSEG → frozen WeedDet backbone**. Because 288 backbone tensors were first loaded from ImageNet and no completed no-ImageNet RiceSEG run is supplied, any downstream difference cannot be attributed specifically to RiceSEG rather than the combined initialization path.

### 9.4 Confirmed riceseg_pretrain.py and notebook issues

#### Missing referenced instructions

The script refers to training/USING_RICESEG_BACKBONE.md, which is absent.

#### Country parsing is broken for the supplied archive layout

scan_riceseg sets country to the first path component relative to data_root. The actual archive has a wrapper directory:

    global rice segmentation / Country / Site / rgb / image

Every tile is therefore labeled with the country value “global rice segmentation.” The notebook output confirms exactly that. Consequences:

- holdout_country='Tanzania' or any real country finds zero tiles and fails;
- country-level reporting is false;
- domain-gap experiments advertised in the notebook cannot run correctly.

The random group split still keeps basename-defined source photos together because source basenames are unique across the supplied sites. The reproduced 2,768/310 split has no source-photo or exact-hash overlap. It is nevertheless geographically incomplete: all 54 Tanzanian tiles are in training, and China/HB and Japan/TKO_3 also contribute no validation tile.

Parse country and site from a dataset manifest or locate the component immediately above known site/rgb structure. Add a unit test using every supported path layout.

#### The “overfit gate” is not a gate and overwrites the real output path

The notebook says the eight-tile run does not write the real backbone. In fact it passes the production BACKBONE_OUT path into riceseg_pretrain.py, which exports to that path on every mIoU improvement. The retained output proves repeated writes to:

    /content/drive/MyDrive/weeddet_v6_checkpoints/riceseg_backbone.pth

The subsequent full run deletes and replaces the file, so the final completed notebook likely left the correct full-run export. If a user runs only the gate or the full run fails, the production path can contain an eight-image-overfit backbone.

The gate also has no assertion or threshold. It always continues after 60 epochs. Its best printed mIoU is 0.6147, but the first eight sorted tiles contain no duckweed. When the model predicts no duckweed, that class becomes NaN and np.nanmean excludes it, artificially lifting the headline mIoU by roughly one-sixth relative to epochs where false duckweed predictions produce IoU zero. The “best” gate score is therefore not a stable six-class pass criterion.

Use a temporary output path, choose a stratified tiny set containing all classes, and assert a preregistered metric after a fixed budget.

#### Mask clipping is safe on this archive but unsafe as a general loader

All supplied masks use only 0–5, so clipping did not corrupt this pretraining run. The unconditional clip would still silently convert any future 255/void pixel to duckweed. Replace it with explicit validation and ignore_index handling.

#### Class weights are approximate

Weights are computed once from a deterministic stride sample, not dynamically per batch. That is stable within a run, but the function may inspect more than its declared max_samples and can produce a biased approximation when files are ordered by country/site. Record exact sampled files or compute weights once from the complete training masks.

#### Partial validation of loaded weights

load_riceseg_backbone accepts any shape-compatible set with more than 150 matches. The active detector notebook later checks all missing backbone keys, but the reusable function should itself require the exact expected key set.

#### Validation and checkpoint limitations

- best selection uses mean IoU without a formally locked absent-class policy;
- no held-out country/test result or confidence interval;
- one seed;
- augmentation is only horizontal flip;
- the exported file contains weights only, without optimizer, scheduler, RNG, configuration, data hashes, or metric history;
- the standalone exported backbone was not supplied in the complete-data bundle;
- deprecated CUDA AMP interfaces are used.

### 9.5 Required pretraining experiment

Use a balanced factorial design:

| Initialization | Backbone frozen phase | Full fine-tune phase | Equal tuning budget |
|---|---:|---:|---:|
| Scratch | yes/no as preregistered | yes | yes |
| ImageNet | same | yes | yes |
| RiceSEG | same | yes | yes |

Run all conditions over the same grouped splits, seeds, schedule search budget, evaluator, and selection rule. Add both ImageNet→RiceSEG and random→RiceSEG conditions so that RiceSEG’s incremental effect is identifiable. Report effect sizes and uncertainty, not only best-run AP.

---

## 10. Data audit

### 10.1 Curated RiceSEG image subset

The recovered curated directory contains:

- 245 images;
- 245 manifest rows;
- exact filename agreement between manifest and directory;
- 150 front-view images;
- 95 aerial images;
- all images 512 by 512 RGB;
- 211 JPEG and 34 PNG files;
- no exact duplicate image hashes;
- no near-duplicate pair at dHash distance 4 or less in the audit;
- minimum observed pairwise dHash distance 10.

Those are positive integrity signals.

### 10.2 Source-photo grouping

Filename analysis identifies 147 source-photo groups among 245 tiles:

| Tiles from same source | Number of source groups |
|---:|---:|
| 1 | 89 |
| 2 | 31 |
| 3 | 15 |
| 4 | 11 |
| 5 | 1 |

Overlapping crops from the same original photo must stay in the same split. Image-level hashes alone do not prevent source-photo leakage.

### 10.3 Curated set excludes difficult conditions

data/rice_training_curated/AUDIT_REPORT.md says glare, reflection, blur, occlusion, and other hard scenes were intentionally excluded. That can be appropriate for initial mask annotation, but it creates a clean-biased training distribution if used as the final deployment dataset.

Maintain two explicit sets:

- a curated clean set for controlled pretraining/annotation;
- a locked hard/OOD challenge set covering glare, water reflection, mud, partial submergence, motion blur, shadows, dense overlap, varied growth stages, rain, equipment, people, animals, and non-rice vegetation.

### 10.4 Missing provenance and rights information

The curated data lacks a complete dataset card containing:

- source dataset version and retrieval date;
- original stable URL/DOI;
- license and redistribution terms;
- author/owner;
- geographic and temporal coverage;
- intended and prohibited uses;
- preprocessing and selection rules;
- image and manifest hashes;
- consent/privacy review where relevant;
- known biases and exclusions;
- mapping from tile to original image and split group.

This is both a reproducibility and publication roadblock.

### 10.5 Complete COCO detector dataset integrity

The supplied rice_detection_coco_split.zip confirms the notebook’s counts and permits direct validation:

| Split | Images | Rice boxes | Weed boxes | Total boxes |
|---|---:|---:|---:|---:|
| Train | 1,079 | 31,013 | 285 | 31,298 |
| Validation | 134 | 3,669 | 45 | 3,714 |
| Test | 134 | 3,706 | 28 | 3,734 |
| **Total** | **1,347** | **38,388** | **358** | **38,746** |

Positive structural findings:

- every JSON image record has exactly one matching file;
- no extra image files;
- all images decode as 512 by 512 RGB;
- recorded dimensions match actual dimensions;
- all annotation image/category references are valid;
- every box has positive finite width/height and lies inside its image;
- stored areas agree with width × height;
- no exact duplicate annotations;
- no same-category box pair has IoU at least 0.9;
- no exact duplicate image hashes within or across splits;
- categories are consistently id 1 rice and id 2 weed;
- iscrowd is zero for all sampled/validated annotations.

The dataset is dense: median annotations per image are 27 in train and validation and 26 in test; the maximum is 103. Rice is strongly elongated, with training median width 20.8 pixels, median height 54.4 pixels, and median square-root area 34.1 pixels. This validates the need for small-object and tall-aspect-ratio handling, but it does not validate the current 12-anchor/ATSS design.

### 10.6 COCO split leakage and distribution limitations

The split-generation script and immutable source manifest are still absent, but the supplied files are sufficient to disprove the “leakage-safe” label. P0-5 gives the full counts: seven named capture series cross all splits, covering at least 73/134 validation and 71/134 test images.

The dHash procedure prevented only extremely close pixel duplicates:

- cross-split minimum dHash distance: 6;
- 29 pairs at distance 6;
- 36 pairs at distance 7;
- 43 pairs at distance 8;
- 108 pairs total at distance 6–8.

Consecutive frames from one source can have dHash distance well above that threshold because the camera moves. Source/session grouping must precede perceptual-hash clustering.

Additional distribution limitations:

- all 1,347 images contain at least one rice annotation;
- there are no no-rice/negative-scene images;
- only 132 images contain a weed annotation;
- weed boxes are 0.924% of all boxes;
- identifiable video/image series dominate more than half of the data;
- the split is not grouped by site, session, camera pass, or source dataset;
- validation and test are therefore neither field-independent nor deployment-like.

The absence of negative scenes is particularly important for a model that already saturates its detection cap. Background anchors exist within positive images, but the detector never sees a scene-level no-rice case resembling an empty field region, equipment view, transit frame, or OOD input.

### 10.7 Category-semantic inconsistency

The JSON outputs contain weed annotations, but the active model defines only one class, rice, and filters category 1 for training. That can be legitimate for a crop-protection detector, but every analysis tool must make the same choice. The visualization does not.

Document whether category-2 images are:

- ignored regions;
- true negatives;
- a future second class;
- quality-control annotations;
- or an independent weed detector target.

Ignoring weed boxes during one-class rice training can teach the model that weeds are background, which may be intended, but the downstream spray logic then cannot infer weed presence from that background label.

### 10.8 Missing COCO provenance

The JSONs have no info or licenses table even though every image declares license id 1. All image records use the same 2026-07-07 export timestamp rather than original capture times. Images have no EXIF. Roboflow-style names and extra.name preserve fragments of original filenames, but not:

- source dataset/revision;
- source URL or DOI;
- license;
- field/site/session;
- capture device;
- date/weather/growth stage;
- original sequence/video;
- annotation author/review status;
- augmentation lineage.

This missing metadata is the reason the split cannot be repaired reliably from hashes alone.

### 10.9 Cross-dataset overlap

No exact image hash is shared between RiceSEG pretraining and the detector dataset. The closest cross-dataset dHash distance is 8, and visual review of that closest pair shows unrelated scenes. Direct image leakage from detector validation/test into RiceSEG pretraining was not found.

### 10.10 Data in Git and duplicate archives

The repository tracks approximately 26.5 MB of curated image data and also contains a valid rice_training_curated.zip that duplicates the directory. The outer uploaded archive was truncated in another rice_training_curated_source.zip copy. The WeedDet v2 Drive bundle adds two byte-identical 116.35 MB copies of rice_detection_coco_split.zip.

Use:

- Git for source, manifests, small representative fixtures, and dataset metadata;
- a versioned artifact/data store for full datasets and checkpoints;
- content hashes and immutable version IDs to connect them.

---

## 11. Annotation-pipeline audit

### data/auto_annotate.py

**Status:** executable but not suitable for trusted ground truth.

Risks:

- broad prompts such as plant, crop, and green plant are all normalized to Rice;
- weeds and unrelated green material can become rice labels;
- synonym prompts can create duplicate boxes;
- no prompt-level deduplication or NMS;
- no score, model revision, prompt, or provenance stored in VOC XML;
- empty predictions produce an XML that may be mistaken for a reviewed negative;
- no mandatory human-review state;
- no sampled precision/recall quality audit;
- model and dependency revisions are unpinned;
- device fallback claims are not consistently implemented;
- one inference failure can interrupt a batch.

If retained:

- mark all generated labels as proposals, not truth;
- store model ID/revision, prompts, thresholds, scores, date, and source hash;
- deduplicate proposals;
- require review status and reviewer identity;
- estimate proposal precision and missed-object rate on a random audited sample;
- never convert empty model output directly into an accepted negative.

---

## 12. Historical and archived-code findings

Archived defects do not necessarily affect T7d, but they create serious workflow risk because several files remain runnable and names are similar.

### 12.1 archive/agrinav_github_FULL.zip

This valid but untracked archive is an old April snapshot. It duplicates old inference, annotation, VOC conversion, baseline notebook, model, and training files already represented elsewhere. A ZIP of source inside the source repository obscures provenance and can reintroduce fixed bugs.

Disposition: move to immutable release storage with a manifest or delete after verifying Git history contains the required material.

### 12.2 V4 notebook

archive/v4_weeddet_60epoch_no_val/weeddet_trainingV4.ipynb is approximately 12.8 MB and contains about 10.8 MB of embedded output. It is the largest tracked file.

Historical defects:

- two cells are about 98.5% duplicated;
- checkpoint path mismatch: one cell expects weeddet_v4_best.pth while training writes weeddet_best.pth;
- imports historical weeddet_for_VSCode;
- no reliable validation protocol.

Strip outputs for source history and store an executed report separately.

### 12.3 V5 phase-2 notebook

archive/v5_era/weeddet_phase2_neckhead.ipynb has two serious historical issues:

1. It pools test images into a random 80/20 train/validation split, contaminating the intended holdout.
2. It computes loss as the sum of cls_loss, reg_loss, and total_loss even though total_loss already equals classification plus regression. This exactly doubles the loss.

It then compares that doubled phase-2 validation loss against an undoubled V5 value, making checkpoint selection invalid.

### 12.4 Paper-evaluation notebook

The archived paper figure notebook uses score threshold 0.05, NMS 0.25, and a first-200-image selection described as representative. This is not a locked, unbiased test protocol.

### 12.5 Baseline notebook

The historical baseline uses a random image split that can leak related source photos and collapses categories into rice. It is not an acceptable modern baseline result.

### 12.6 VOC conversion and splitting

Historical VOC scripts contain:

- basename-only matching, which can collide across directories;
- conversion of all categories to Rice;
- omission of negative images;
- an unexplained area-at-least-95 filter;
- ignored iscrowd semantics;
- unsafe ZIP extractall;
- selection of an arbitrary first ZIP;
- substring-based split inference;
- swallowed exceptions;
- random image-level splitting;
- no test split;
- no image/box/coordinate validation;
- unused counters and partial error reporting.

These scripts should be clearly marked historical/non-runnable or moved to a tagged archive branch.

### 12.7 Dead code in the current model module

models/weeddet_v6b.py still contains:

- the generic VOC WeedDataset;
- ModelEMA;
- WarmupMultiStepLR;
- train_with_progress;
- duplicated decode pathways;
- historical defaults such as ERetinaHead num_anchors = 9.

train_with_progress is especially misleading:

- it uses the VOC path;
- initializes from ImageNet rather than the active RiceSEG protocol;
- selects “best” by training loss;
- has no proper validation/test evaluation;
- uses generic checkpoint names;
- can pass None gradients to clipping logic;
- uses deprecated AMP style;
- saves incomplete state.

This creates two competing training systems in one module. Extract tested reusable model code, delete or quarantine historical trainers, and expose one current CLI.

---

## 13. Documentation audit

### 13.1 Physically damaged files

- README.md ends mid-table at line 63.
- HANDOFF_2026-07-16.md ends mid-command near line 196 and leaves a code fence unclosed.
- active/ACTIVE_NOTES.md in committed HEAD contains 67 trailing NUL bytes.

The recovered upload copy of ACTIVE_NOTES lacks those NULs, which is further evidence of synchronization or file corruption.

### 13.2 Stale Git state

HANDOFF says the branch is at 77e0f58 with pending work, but HEAD is 2a056ce. It also says a commit is pending even though the described fix is committed.

### 13.3 Notebook naming mismatch

The active file is named weeddet_trainingV7_1_T1.ipynb and its title says T1, while configuration says T7d. The newly supplied notebook is named T7 but runs T7d. The run identity should be a generated manifest, not a mutable filename/comment combination.

### 13.4 TEST_LOG contradictions

TEST_LOG.md contains:

- a “fixed protocol” using anchor base scale 3 although later decisions use 6;
- experiments still marked CURRENT or PLANNED after actuals and verdicts are recorded;
- duplicated blank Actual/Verdict fields;
- a test identifier T10 reused for different ideas;
- T7d described as planned in one place while the handoff says it was superseded;
- queue ordering that no longer matches the attached 100-epoch T7d run.

### 13.5 Metric discrepancy

The project records T7 best AP50 as 0.1040, but AP75 appears as 0.0051 in one table and 0.0116 elsewhere. The likely explanation is selected-epoch versus final-epoch reporting, but neither table makes that distinction clear.

Every result row must include:

- run ID;
- selected epoch;
- checkpoint hash;
- raw or EMA;
- split;
- AP evaluator settings;
- maxDets;
- score/NMS settings;
- seed;
- code/data hashes.

### 13.6 Project tracker is stale

AgriNav_Project_Tracker.md predates later changes and describes:

- IACS assignment rather than the current ATSS-style implementation;
- GIoU rather than CIoU;
- scale 4 rather than 6;
- different NMS and gradient clipping;
- V7 scratch as pending though later documents say it ran.

It also claims changing anchor scale changes head weight tensor shapes. With a fixed number of anchors, scale changes anchor geometry but not head tensor shapes. Loading weights may be semantically questionable, but it is not a shape mismatch.

### 13.7 Master history and Drive inventory are stale

MASTER_PROJECT_HISTORY.md and drive_links/GOOGLE_DRIVE_INVENTORY.md still present older V5-era architecture and workflow as current.

### 13.8 Paper artifacts are absent

The paper folder contains revision/feedback notes, but not:

- paper source;
- bibliography;
- final figures;
- architecture diagram;
- generated PDF;
- table-generation code;
- experiment manifest.

The research paper cannot be audited for claim-evidence consistency because it is not supplied.

---

## 14. Repository engineering and security audit

### 14.1 Missing engineering controls

The repository has no:

- automated tests;
- CI workflow;
- linter or formatter configuration;
- type checker;
- pre-commit hooks;
- packaging metadata;
- structured experiment configuration;
- current training CLI;
- environment lock;
- container or Conda environment;
- model/data artifact registry;
- release process;
- license file.

All Python files parsing successfully is useful, but syntax is not behavioral verification.

### 14.2 requirements.txt is insufficient

It contains loose lower bounds for a small subset of packages. It omits pycocotools and gives no:

- upper bounds;
- known-good Python version;
- CUDA/cuDNN compatibility;
- platform matrix;
- hashes;
- transitive lock;
- Colab image identifier.

### 14.3 Runtime test limitation

The audit environment did not contain Torch/torchvision, so the full detector could not be instantiated for forward/backward, gradient, GPU memory, or checkpoint inference tests. Static analysis, archive validation, notebook-output inspection, and a direct Pillow augmentation test were completed. Runtime model behavior beyond recorded outputs remains an evidence gap.

### 14.4 Unsafe archive extraction

Several scripts use extractall without checking member paths. A malicious or malformed archive can write outside the intended directory. Validate resolved member paths before extraction.

### 14.5 Checkpoint loading

PyTorch checkpoints use pickle-compatible serialization. Loading untrusted checkpoints can execute code depending on loading mode and contents. The active notebook explicitly uses weights_only=False for the supplied checkpoint. Restrict sources, hash artifacts, and prefer safe state-dict loading.

### 14.6 Secrets and external identifiers

No obvious private keys, tokens, or passwords were found. Documents contain Google Drive identifiers and discuss classic personal access tokens. Before publishing:

- verify Drive links expose only intended data;
- remove token instructions that encourage broad or long-lived credentials;
- use least-privilege, short-lived authentication;
- run automated secret scanning in CI.

### 14.7 OneDrive is the wrong active Git workspace

Use a normal local development directory for Git. Synchronize through the remote repository, not filesystem mirroring. Store large datasets/checkpoints in an artifact system with hashes. Back up the remote and release artifacts independently.

---

## 15. Diagnosis against current literature

Literature was reviewed through 2026-07-20. The point of comparison is methodological direction, not direct metric ranking; datasets, class definitions, image geometry, and evaluation protocols differ.

### 15.1 WeedDet reference

Peng et al. report a WeedDet system for rice and weed recognition with a nine-class dataset and reported 94.1% mAP and 24.3 FPS. The project’s one-class rice detector, customized evaluator, and different data cannot be compared numerically to that headline.

Reference: [WeedDet, Computers and Electronics in Agriculture, 2022](https://www.sciencedirect.com/science/article/pii/S0168169922004963).

Required paper wording:

- “inspired by” or “derived from,” not a faithful reproduction;
- enumerate deviations: backbone/stem, FPN strides, anchors, assignment, classification targets, regression loss, data, classes, input, NMS, and evaluator.

### 15.2 RiceSEG transfer hypothesis

RiceSEG’s cross-country and genotype diversity makes it a rational pretraining source. It also contains challenging illumination and reproductive-stage imagery, while the project’s curated subset intentionally removes several hard conditions. Transfer benefit must therefore be tested on both clean and hard target distributions.

### 15.3 Practical spraying metrics

Field spraying research emphasizes that generic detector mAP is not sufficient. Operational metrics include weed/crop hit rate, area sprayed, chemical savings, false treatment, missed treatment, and inference speed.

Reference: [Frontiers in Plant Science field spraying evaluation, 2023](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1183277/full).

For AgriNav, add:

- rice crop-hit probability;
- weed control rate;
- false-spray area;
- untreated weed area;
- total treated ground area;
- command localization error at nozzle;
- latency distribution and deadline misses;
- no-spray rate under uncertainty/OOD;
- safety-veto counts;
- calibration and risk curves.

### 15.4 Data diversity

Recent rice/weed work such as GE-YOLO collects across sites, sessions, weather, and growth stages and explicitly examines occlusion and lighting. Multi-site crop/weed literature similarly emphasizes environmental variation and data-centric generalization.

References:

- [GE-YOLO rice-field weed detection, 2025](https://www.mdpi.com/2076-3417/15/5/2823)
- [Multi-site crop/weed field study, 2024](https://www.sciencedirect.com/science/article/pii/S2772375524001436)

AgriNav’s clean-curated subset and confirmed capture-series-contaminated split are not enough for field-generalization claims.

The methodological literature is explicit that cross-validation must respect temporal, spatial, and hierarchical dependence. Near-duplicate removal is useful, but it is narrower than source independence: two consecutive frames can be visually different enough to escape a hash threshold while sharing the same field, camera, lighting, plants, and annotation process.

References:

- [Roberts et al., cross-validation for temporal, spatial, hierarchical, and phylogenetic structure](https://onlinelibrary.wiley.com/doi/10.1111/ecog.02881)
- [Barz and Denzler, purging image benchmarks of near-duplicates](https://arxiv.org/abs/2008.12952)
- [Kapoor and Narayanan, leakage and reproducibility in ML-based science](https://arxiv.org/abs/2207.07048)

### 15.5 Modern detector baselines

Before adding more custom loss/assigner combinations, compare against maintained baselines:

- standard RetinaNet or FCOS;
- official RT-DETR;
- optionally DINO for a higher-capacity transformer reference;
- official ATSS;
- a current YOLO-family implementation if licensing and reproducibility are acceptable.

References:

- [RT-DETR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html)
- [Official RT-DETR code](https://github.com/lyuwenyu/RT-DETR)
- [DINO, ICLR 2023](https://openreview.net/forum?id=3mRwyG5one)
- [TOOD task-aligned detection](https://arxiv.org/abs/2108.07755)

The purpose is not architecture shopping. A maintained baseline localizes whether the problem is data, labels, transforms, objective, assignment, or custom implementation.

### 15.6 Safety and validation

ISO 18497-1:2024 addresses principles for partially automated, semi-autonomous, and autonomous agricultural machinery. The ISO 18497 series covers machine design, obstacle-protection systems, autonomous operating zones, and verification/validation.

References:

- [ISO 18497-1:2024](https://www.iso.org/standard/82684.html)
- [ISO agricultural machinery standards index](https://www.iso.org/ics/65.060.01.html)

This audit does not claim formal compliance, and these standards do not replace a project-specific hazard analysis. They do establish that perception accuracy alone is not a safety case.

### 15.7 Risk-aware calibration

Conformal risk control may later help place statistical bounds on selected operational risks under stated exchangeability assumptions. It cannot repair mislabeled data, distribution shift, unsafe semantics, or missing system interlocks.

Reference: [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

---

## 16. Prioritized recovery and development plan

### Phase 0 — Safety containment

1. Retire “everything not rice is weed/spray” semantics.
2. Rename current visualization output to “unclassified / no treatment.”
3. Prevent any inference output from reaching an actuator.
4. Define treatment command schema with explicit no-spray default.
5. Create a preliminary hazard analysis:
   - rice miss;
   - weed false positive;
   - person/animal/equipment;
   - stale frame;
   - dropped frame;
   - poor pose;
   - calibration drift;
   - latency;
   - stuck valve;
   - communication loss;
   - OOD weather/lighting.
6. Define independent vetoes: emergency stop, human/obstacle, geofence, pose quality, sensor health, speed, actuator health, and uncertainty.

**Exit gate:** zero code path can interpret detector absence as permission to spray.

### Phase 1 — Preserve a trustworthy project state

1. Move the active repository outside OneDrive.
2. Push the authoritative 2a056ce-plus-fixes branch to a verified remote.
3. Repair truncated/NUL-corrupted documents.
4. Choose one default branch and remove stale remote-tracking ambiguity.
5. Move large images, ZIPs, notebooks with outputs, and checkpoints to versioned artifact storage.
6. Keep hashes and small fixtures in Git.
7. Add a project license and dataset/model license inventory.
8. Preserve the attached T7d notebook, curve, and checkpoint as immutable historical artifacts with their hashes.

**Exit gate:** a fresh clone plus artifact manifest resolves every required file unambiguously.

### Phase 2 — Correctness before training

1. Fix translation direction.
2. Carry labels through every box keep mask.
3. return exact x/y letterbox scales.
4. Make ImageNet/RiceSEG loading fail on incomplete coverage.
5. Fix ATSS duplicate forced assignments.
6. Either implement spatial per-cell ATSS correctly or adopt a tested assigner.
7. Rename current classification loss; implement official VFL separately if needed.
8. Build one canonical inference/evaluation pipeline.
9. Remove unsafe spray visualization from inference.
10. Filter visualization ground truth by target category.
11. Validate masks and ignore labels.
12. Validate every COCO image, box, category, and split group.
13. Fix RiceSEG country/site parsing and make the overfit gate use a temporary artifact plus an enforced threshold.

**Exit gate:** all unit/integration tests in Section 17 pass.

### Phase 3 — Reproducible experiment system

1. Create a normal Python package or stable module layout.
2. Add a command-line trainer driven by versioned YAML/TOML configuration.
3. Pin Python, Torch, torchvision, pycocotools, CUDA compatibility, and all dependencies.
4. Log run manifests in JSON.
5. Save complete checkpoints:
   - raw and EMA states;
   - optimizer;
   - scheduler;
   - scaler;
   - RNG;
   - global step;
   - configuration;
   - code/data/environment hashes;
   - histories.
6. Add exact resume testing.
7. Add CI for CPU unit tests and a documented GPU smoke test.

**Exit gate:** rerunning a fixture experiment reproduces metrics within declared tolerance and an interrupted run resumes identically.

### Phase 4 — Scientific redesign

1. Retire the current detector split and reconstruct grouped train/validation/test manifests by source dataset, video/series, camera pass, field/session, and site.
2. Never tune on test.
3. Preregister:
   - primary metric;
   - secondary metrics;
   - selection rule;
   - seeds;
   - tuning budget;
   - stopping rule;
   - subgroup analyses.
4. Run scratch, ImageNet, random→RiceSEG, and ImageNet→RiceSEG under equal budgets so the incremental effect of RiceSEG is identifiable.
5. Run three to five seeds.
6. Keep standard COCO AP 0.50:0.95 at maxDets 100 as the primary detector metric.
7. Report AP50, AP75, recall, size-specific metrics, PR curves, calibration, and bootstrap confidence intervals.
8. Add hard/OOD challenge sets.
9. Record per-level positive counts, assignment coverage, candidate counts, and loss components.

**Exit gate:** a locked validation conclusion exists before the test set is opened.

### Phase 5 — Reference baselines

1. Train a maintained RetinaNet/FCOS baseline.
2. Train RT-DETR or another maintained real-time reference.
3. If claiming WeedDet reproduction, implement the paper’s method faithfully and list unavoidable deviations.
4. Compare:
   - AP and recall;
   - parameters and FLOPs;
   - peak memory;
   - end-to-end latency;
   - calibration;
   - target-hardware throughput.
5. Retain the custom architecture only if it adds measured value or answers a specific research question.

### Phase 6 — Treatment perception and integration

1. Add affirmative weed evidence:
   - weed detector;
   - crop/weed semantic or instance segmentation;
   - or treatment-zone segmentation.
2. Keep rice detection as an independent veto, not the sole classifier.
3. Calibrate the camera to ground coordinates.
4. Estimate vehicle pose and uncertainty.
5. Track targets over time.
6. compensate for sensor-to-nozzle distance, vehicle speed, pipeline latency, and valve dynamics.
7. Integrate human/obstacle and system-health vetoes.
8. Log every proposed, vetoed, and executed treatment with evidence.

### Phase 7 — Staged validation

1. Offline replay on locked data.
2. Simulator tests.
3. Bench tests with water and fixed targets.
4. Stationary field tests with water.
5. Slow supervised field tests with water.
6. Supervised treatment trials only after safety review.
7. Document failure rates, intervention criteria, and rollback.

### Phase 8 — Paper

1. State the actual scope: perception subsystem unless integration evidence exists.
2. Include source, bibliography, figures, and table-generation code.
3. Separate validation from test.
4. Report all conditions, seeds, uncertainty, and negative results.
5. Include threats to validity:
   - source-photo grouping;
   - geography;
   - clean-data bias;
   - annotation uncertainty;
   - evaluator customization;
   - compute/tuning budget;
   - domain shift;
   - lack of field actuation.
6. Avoid “RiceSEG improves” until the controlled test supports it.

---

## 17. Required test suite and acceptance gates

### 17.1 Geometry and data tests

- fixed horizontal flip aligns pixels, boxes, and labels;
- fixed positive/negative translation aligns pixels, boxes, and labels;
- clipped objects retain the correct label;
- letterbox forward/inverse round trip is within one pixel;
- encode/decode round trip recovers boxes within tolerance;
- invalid and degenerate COCO boxes are rejected;
- category filtering is identical in training, evaluation, and visualization;
- source-photo groups never cross splits;
- no video, numbered capture series, field/session, or source dataset crosses detector splits;
- RiceSEG country/site parsing returns China, India, Japan, Philippines, and Tanzania for the supplied archive;
- the RiceSEG overfit gate cannot write the production backbone path and fails when its threshold is missed;
- archive extraction cannot escape target directory;
- unknown mask values fail rather than clip.

### 17.2 Anchor and assignment tests

- exact anchors per level, count, order, shape, and centers;
- anchor permutation does not change spatial candidate set;
- top-k candidates cover intended cells;
- every ground-truth object receives at least one positive;
- no ground-truth object is orphaned by collisions;
- each positive has one valid assigned ground truth;
- empty-ground-truth images produce finite classification loss;
- crowded, overlapping, tiny, large, and edge-touching boxes remain finite;
- assignment output matches an official reference on shared fixtures where semantics are intended to match.

### 17.3 Loss tests

- published VFL equation matches official values on fixed tensors;
- current custom loss has a distinct truthful name;
- loss is finite at extreme logits;
- positive/negative gradients have expected signs;
- normalization is invariant to batch padding;
- regression component scales and gradient norms are logged;
- mixed precision agrees with float32 within tolerance on a fixture.

### 17.4 Evaluation tests

- a perfect synthetic detector returns AP near 1;
- an empty detector returns zero without crashing;
- duplicated boxes degrade precision as expected;
- maxDets 100 and 300 are separate named protocols;
- standard COCO metrics remain standard;
- image/category IDs map correctly;
- raw and EMA identity is explicit;
- visualization counts only target-class GT;
- no test split can be selected without an explicit release gate.

### 17.5 Checkpoint and reproducibility tests

- strict model loading from a checkpoint built by the same config;
- architecture mismatch fails with a clear error;
- exact imported source and data hashes are present;
- interrupted training resumes to the same next-step loss and weights within tolerance;
- optimizer, scheduler, scaler, RNG, and global step restore;
- inference artifact loads in a clean environment;
- unsafe or untrusted checkpoint loading is prohibited by default.

### 17.6 Safety-interface tests

- no rice detection yields no spray;
- no positive weed/treatment evidence yields no spray;
- low confidence or OOD yields no spray;
- stale frame yields no spray;
- invalid calibration or pose yields no spray;
- person/animal/obstacle yields no spray;
- geofence violation yields no spray;
- actuator/watchdog fault yields no spray;
- all vetoes are logged;
- only calibrated ground-space targets inside the treatment zone can become commands.

---

## 18. File and directory disposition

| Path | Audit disposition |
|---|---|
| models/weeddet_v6b.py | Active but requires correctness fixes, decomposition, tests, and truthful loss naming |
| active/weeddet_trainingV7_1_T1.ipynb | Active template; rename, remove mutable run identity, replace with config-driven trainer |
| active/ACTIVE_NOTES.md | Repair NUL corruption; reconcile with current run history |
| training/riceseg_pretrain.py | Retain after fixing country parsing, overfit output isolation/gating, strict loading, full checkpointing, and controlled experiment redesign |
| inference/inference_rice.py | Do not use; replace completely with compatible, no-spray-default inference |
| data/auto_annotate.py | Proposal-generation only; add provenance, dedupe, review state, and quality audit |
| data/rice_training_curated | Move bulk images to versioned data storage; keep manifest, hashes, dataset card, and fixtures in Git |
| data/rice_training_curated.zip | Redundant; remove from source tree after artifact registration |
| supplied rice_detection_coco_split.zip | Structurally sound images/annotations but scientifically invalid split; preserve as historical, then rebuild by capture series/session/site |
| duplicate rice_detection_coco_split(1).zip | Byte-identical redundancy; remove after hash-verified artifact registration |
| supplied RiceSEG.zip | Structurally valid 3,078-pair source dataset; register version/license/hash and build explicit country/site/source manifest |
| supplied run_pretrain.ipynb | Preserve as executed evidence; mark overfit gate/output-path and country-parsing defects |
| supplied weeddet_v2 __pycache__ files | Remove; opaque stale CPython 3.12 caches are not source or checkpoints |
| archive/voc_era | Historical-only; tag/quarantine, do not expose as current |
| archive/v4_weeddet_60epoch_no_val | Historical; strip notebook output from source history or externalize artifact |
| archive/v5_era | Historical and contains invalid evaluation/training logic; clearly mark non-authoritative |
| archive/agrinav_github_FULL.zip | Redundant nested snapshot; externalize or remove after provenance check |
| README.md | Rebuild from current authoritative state; file is truncated |
| HANDOFF_2026-07-16.md | Repair truncation and replace with generated status |
| TEST_LOG.md | Normalize IDs, immutable run rows, selected/final epoch distinction, hashes, protocol columns |
| RESEARCH_PLAN_DETECTION_ACCURACY.md | Lock protocol and correct metric discrepancies |
| AgriNav_Project_Tracker.md | Reconcile architecture/config and split perception from full-system roadmap |
| MASTER_PROJECT_HISTORY.md | Mark historical states clearly and update “current” pointers |
| drive_links/GOOGLE_DRIVE_INVENTORY.md | Replace mutable path inventory with hashed artifact manifest |
| paper | Add actual paper source, bibliography, generated PDF, figures, and reproducible tables |
| requirements.txt | Replace loose subset with locked, documented environment |
| supplied T7d notebook/checkpoint/curve | Preserve as immutable historical evidence with supplied hashes; do not call it a reproducible run |

---

## 19. Recommended experiment record schema

Every run should produce one immutable JSON record with:

### Identity

- run UUID and human label;
- parent run, if resumed;
- start/end timestamps;
- operator;
- purpose/hypothesis.

### Code

- Git remote;
- commit;
- branch;
- dirty diff hash;
- imported module hashes;
- entry-point/notebook hash.

### Data

- dataset version;
- archive hash;
- annotation hash;
- split-manifest hash;
- image-manifest hash;
- group strategy;
- category policy.

### Environment

- OS;
- Python;
- Torch/torchvision/pycocotools;
- CUDA/cuDNN/driver;
- GPU model;
- deterministic settings.

### Model and objective

- exact architecture config;
- anchor generator;
- assigner;
- loss equations and weights;
- initialization artifact/hash;
- frozen parameters;
- EMA settings.

### Optimization

- optimizer and all parameters;
- schedule and warmup;
- epochs/steps;
- batch/accumulation;
- AMP;
- clipping;
- seed.

### Evaluation

- split;
- checkpoint identity;
- raw/EMA;
- score/NMS/maxDets;
- standard/custom protocol name;
- all metrics and curves;
- latency measurement method.

### Outputs

- checkpoint hash;
- logs;
- predictions;
- figures;
- failures;
- selected epoch and rule;
- test-release authorization.

---

## 20. What can and cannot be concluded today

### Supported conclusions

- The committed WeedDet research snapshot was recoverable despite the outer ZIP truncation.
- The supplied T7d run and checkpoint are structurally intact.
- T7d’s best recorded validation performance is AP50 0.0884 at maxDets 300 and AP75 0.0010 at maxDets 300, selected at epoch 14.
- Its standard-cap AP 0.50:0.95 is 0.0166 at maxDets 100.
- The run saturates the 300-detection cap and qualitative outputs show dense high-score duplicates.
- EMA materially outperforms the corresponding raw weights in the supplied evaluation.
- The exact T7d model source matches the current repository, and its translation augmentation is wrong.
- All supplied RiceSEG RGB/mask pairs and COCO image/annotation records are structurally valid.
- The RiceSEG pretraining run reached best validation mIoU 0.5749, but country parsing and its overfit gate are defective.
- The detector validation/test split is capture-series contaminated and cannot support independent model selection or final evaluation.
- Current inference is incompatible and unsafe for spraying.
- The project still lacks the split-generation source manifest, full resumable run state, original data provenance/licenses, and paper artifacts needed for reproducibility and publication.

### Unsupported conclusions

- That RiceSEG pretraining causally improves detection.
- That T7 is field-ready or materially accurate.
- That validation AP predicts treatment safety.
- That all non-rice pixels are weeds.
- That bounding-box complement estimates weed coverage.
- That the COCO split is leakage-free.
- That recorded validation results generalize across sites, seasons, countries, lighting, or growth stages.
- That the repository is a full autonomous-tractor implementation.
- That the system meets ISO 18497 or any formal safety standard.
- That the current checkpoint can be resumed or exactly reproduced.

---

## 21. Final diagnosis

The project’s main roadblock is not a need for another 100-epoch hyperparameter run. It is a chain-of-evidence problem:

1. training geometry is currently wrong;
2. assignment and classification semantics differ from their published names;
3. evaluation is customized and internally mixed;
4. run provenance is incomplete;
5. qualitative output is saturated and poorly localized;
6. the supplied split is confirmed to leak related capture series across train, validation, and test;
7. inference does not match training;
8. downstream spray semantics are unsafe;
9. the rest of the autonomous system is absent.

The positive news is that the project has enough structure, history, and domain motivation to recover. The fastest credible path is:

**contain unsafe inference → fix and unit-test geometry/assignment → establish a maintained baseline → lock data and evaluator → rerun controlled pretraining ablations → add positive weed evidence and safety vetoes → validate on target hardware and staged field conditions.**

Until those gates are met, AgriNav should be presented as an exploratory perception research prototype.

---

## 22. Primary references

1. [Peng et al., WeedDet, Computers and Electronics in Agriculture, 2022](https://www.sciencedirect.com/science/article/pii/S0168169922004963)
2. [Zhang et al., ATSS, 2019/2020](https://arxiv.org/abs/1912.02424)
3. [Official ATSS implementation](https://github.com/sfzhang15/ATSS)
4. [Zhang et al., VarifocalNet, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html)
5. [Official VarifocalNet loss implementation](https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py)
6. [Li et al., Generalized Focal Loss, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html)
7. [Feng et al., TOOD, 2021](https://arxiv.org/abs/2108.07755)
8. [RiceSEG dataset paper, 2025](https://www.sciencedirect.com/science/article/pii/S2643651525001050)
9. [GE-YOLO rice-field weed detection, 2025](https://www.mdpi.com/2076-3417/15/5/2823)
10. [Multi-site crop/weed field study, 2024](https://www.sciencedirect.com/science/article/pii/S2772375524001436)
11. [Field spraying metrics, Frontiers in Plant Science, 2023](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1183277/full)
12. [RT-DETR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html)
13. [Official RT-DETR implementation](https://github.com/lyuwenyu/RT-DETR)
14. [DINO, ICLR 2023](https://openreview.net/forum?id=3mRwyG5one)
15. [Official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
16. [Pillow affine transform documentation](https://pillow.readthedocs.io/en/stable/reference/ImageTransform.html)
17. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
18. [ISO 18497-1:2024](https://www.iso.org/standard/82684.html)
19. [ISO agricultural machinery standards index](https://www.iso.org/ics/65.060.01.html)
20. [Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
21. [Roberts et al., structured cross-validation, Ecography](https://onlinelibrary.wiley.com/doi/10.1111/ecog.02881)
22. [Barz and Denzler, near-duplicates in image benchmarks](https://arxiv.org/abs/2008.12952)
23. [Kapoor and Narayanan, ML leakage and reproducibility](https://arxiv.org/abs/2207.07048)

---

## 23. Audit limitations

- The uploaded outer project ZIP is truncated, so post-truncation untracked content is unknowable.
- Current live remote state was not verified; locally stored Git refs were used.
- Torch/torchvision runtime tests were unavailable in the audit environment.
- The complete detector COCO data was supplied, but its split-generation script and original source/session manifest were not.
- The complete RiceSEG RGB/mask archive was supplied, but its standalone exported backbone, original license record, and full resumable pretraining state were not.
- The T7d notebook did not record a model-file hash; the exact Drive-directory source was supplied later and independently matched to repository HEAD.
- No navigation/control/actuation repositories or interface specifications were supplied.
- No paper source or final paper PDF was supplied.
- Literature comparisons are methodological; reported metrics across different datasets are not directly comparable.

These limitations are incorporated into the severity and wording of every relevant conclusion.
