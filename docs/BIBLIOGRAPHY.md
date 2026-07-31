# Bibliography

Works this project has drawn on, grouped by the role they play here rather than
alphabetically, because the useful question is usually "what backs this design
decision?" rather than "what did Zhang write?".

## How to read the status column

| Status | Meaning |
|---|---|
| **repo** | The URL was already recorded in this repository before this file existed — in a module docstring, an ADR, or an audit. The project demonstrably consulted it. |
| **recalled** | Added from discussion. Title, authors and venue are stated with confidence; **the identifier has not been checked against the source in this repository**. Verify before citing. |

**Nothing here has been machine-verified.** No entry has had its DOI resolved, its
author list completed, or its page numbers confirmed. That is why this is a
reading list and not a `.bib` file: a BibTeX entry with a guessed author list or
a wrong page range produces a wrong citation in a paper silently. Verify each
entry against the publisher's page, then generate the `.bib`.

Per `CLAUDE.md` §14.5 and §40: do not cite these from memory as exact references,
and do not represent an unverified identifier as checked.

---

## 1. The detector this project implements

| Work | Why it is here | Status |
|---|---|---|
| Peng et al., *Weed Detection in Paddy Field Using an Improved RetinaNet Network*, Computers and Electronics in Agriculture 199:107179, 2022. [link](https://www.sciencedirect.com/science/article/pii/S0168169922004963) | The architecture `src/agrinav/models/weeddet_v6b.py` is based on. Its headline (94.1% mAP, 24.3 FPS, nine classes) is **not comparable** to anything here — different data, different class count, different evaluator. | repo |
| Zhang et al., *Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection* (ATSS), CVPR 2020. [arXiv:1912.02424](https://arxiv.org/abs/1912.02424) | The positive/negative assignment used when `use_atss: true`. Also the source of the argument that assignment, not anchor density, explains most of the anchor-based/anchor-free gap. | repo |
| [Official ATSS implementation](https://github.com/sfzhang15/ATSS) | Reference for what standard ATSS does — relevant to the `atss_all_neg` decision recorded in `WeedDetLoss`. | repo |
| Zhang et al., *VarifocalNet: An IoU-aware Dense Object Detector*, CVPR 2021. [link](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html) | The published VFL objective. **This repo does not implement it** — see `HardTargetFocalLikeLoss`, renamed for exactly that reason. Cited to mark the difference, not the lineage. | repo |
| [Official VarifocalNet loss implementation](https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py) | The reference implementation the local loss deviates from. | repo |
| Li et al., *Generalized Focal Loss*, NeurIPS 2020. [link](https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html) | Alternative quality-aware classification target; a candidate if the hard-target loss stays stuck at low confidence. | repo |
| Feng et al., *TOOD: Task-aligned One-stage Object Detection*, 2021. [arXiv:2108.07755](https://arxiv.org/abs/2108.07755) | Task alignment between classification and localisation — relevant to the near-zero AP75 observed. | repo |
| Lin et al., *Focal Loss for Dense Object Detection* (RetinaNet), 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | The loss family and the prior-bias initialisation trick the classification head depends on. | recalled |
| Lin et al., *Feature Pyramid Networks for Object Detection*, 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144) | The neck `eFPN` modifies. | recalled |
| He et al., *Deep Residual Learning for Image Recognition*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | The backbone `DetResNet50` modifies (stem only). | recalled |
| Zheng et al., *Distance-IoU Loss*, 2020. [arXiv:1911.08287](https://arxiv.org/abs/1911.08287) | The CIoU regression term in `CIoULoss`. | recalled |

## 2. Normalisation and batch size

Added 2026-07-31 after the phase-2 run scored AP 0.0054 and the cause was traced
to BatchNorm eval-mode statistics. See `docs/GATE_STATUS.md`.

| Work | Why it is here | Status |
|---|---|---|
| Ioffe & Szegedy, *Batch Normalization*, 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) | The train/eval duality at the centre of the defect: batch statistics while training, running statistics at inference. | recalled |
| Wu & He, *Group Normalization*, ECCV 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494) | Documents BN degrading sharply below batch ~16 and proposes a batch-independent alternative. The candidate fix for the FPN and head. **Verify the exact error figures before quoting them.** | recalled |
| Ioffe, *Batch Renormalization*, 2017. [arXiv:1702.03275](https://arxiv.org/abs/1702.03275) | Written specifically about the train/inference mismatch growing at small or non-i.i.d. batches — the failure measured here. | recalled |
| Peng et al., *MegDet: A Large Mini-Batch Object Detector*, CVPR 2018. [arXiv:1711.07240](https://arxiv.org/abs/1711.07240) | Argues detection is held back by tiny batches and cross-GPU SyncBN helps. Relevant, and not available on a single Colab T4. | recalled |
| Keskar et al., *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*, ICLR 2017. [arXiv:1609.04836](https://arxiv.org/abs/1609.04836) | The main argument *for* small batches. Note it concerns SGD generalisation, which is orthogonal to the BN statistics problem. | recalled |
| Masters & Luschi, *Revisiting Small Batch Training for Deep Neural Networks*, 2018. [arXiv:1804.07612](https://arxiv.org/abs/1804.07612) | Reports batch 2–32 giving the best test performance across architectures. | recalled |
| Goyal et al., *Accurate, Large Minibatch SGD*, 2017. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677) | The counterweight: large batch works with linear LR scaling and warmup. | recalled |
| Smith et al., *Don't Decay the Learning Rate, Increase the Batch Size*, 2017. [arXiv:1711.00489](https://arxiv.org/abs/1711.00489) | Batch size and LR are one knob (noise scale), not two. | recalled |

## 3. Baselines and reference detectors

Nothing in this section has been run yet. See `docs/baselines.md`.

| Work | Why it is here | Status |
|---|---|---|
| Ren et al., *Faster R-CNN*, 2015. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497) | `fasterrcnn_resnet50_fpn_v2`, the two-stage reference arm. | recalled |
| Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection*, 2019. [arXiv:1904.01355](https://arxiv.org/abs/1904.01355) | `fcos_resnet50_fpn`, the anchor-free arm — tests whether the anchor design carries anything. | recalled |
| Zhao et al., *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR), CVPR 2024. [link](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html) · [code](https://github.com/lyuwenyu/RT-DETR) | The modern transformer detector the gate asks for. Not yet wired. | repo |
| Zhang et al., *DINO*, ICLR 2023. [link](https://openreview.net/forum?id=3mRwyG5one) | Alternative transformer baseline. | repo |
| [Torchvision detection models](https://docs.pytorch.org/vision/main/models.html) · [finetuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) | The implementations `agrinav baseline-detector` actually calls, including their FrozenBN default. | repo |

## 4. Data, splits, and leakage

| Work | Why it is here | Status |
|---|---|---|
| RiceSEG dataset, 2025. [paper](https://www.sciencedirect.com/science/article/pii/S2643651525001050) · [arXiv:2504.02880](https://arxiv.org/abs/2504.02880) · [dataset](https://huggingface.co/datasets/PheniX-Lab/RiceSEG) | Source of the phase-1 segmentation pretraining and its `weeds` class. Cited two different ways inside this repo (ScienceDirect in the audits, arXiv in `riceseg_pretrain.py`) — reconcile to one before publishing. | repo |
| Lin et al., *Microsoft COCO*, 2014. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312) | The annotation format and the AP definition every number here uses. | repo |
| [COCO evaluator source](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) | The authority on `maxDets` behaviour, including why a non-100 value returns the `-1.0` sentinel. | repo |
| Roberts et al., *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*, Ecography. [doi:10.1111/ecog.02881](https://onlinelibrary.wiley.com/doi/10.1111/ecog.02881) | Why grouped splitting is required when adjacent video frames share a capture family. | repo |
| Barz & Denzler, *Do We Train on Test Data? Purging CIFAR of Near-Duplicates*, 2020. [arXiv:2008.12952](https://arxiv.org/abs/2008.12952) | Near-duplicate leakage — the failure mode `grouped_split.json` exists to prevent. | repo |
| Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science*, 2022. [arXiv:2207.07048](https://arxiv.org/abs/2207.07048) | The taxonomy that the 2026-07-27 archive incident is a textbook instance of. | repo |
| Bodla et al., *Soft-NMS*, 2017. [arXiv:1704.04503](https://arxiv.org/abs/1704.04503) | The default suppression in `agrinav.inference.postprocess`, and the one difference the torchvision baselines cannot match. | repo |
| Kirillov et al., *Segment Anything*, 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) · Ravi et al., *SAM 2*, 2024. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) | Source of the box-prompted polygons in the intake. **Marked `review_status: unreviewed` and dropped from the phase-2 build** — do not describe them as ground truth. The model revision is recorded as `UNPINNED_UPSTREAM`; pin it before any release use. | recalled |

## 5. Agricultural application context

| Work | Why it is here | Status |
|---|---|---|
| GE-YOLO rice-field weed detection, 2025. [link](https://www.mdpi.com/2076-3417/15/5/2823) | Recent comparable task; a candidate external reference point. | repo |
| Multi-site crop/weed field study, 2024. [link](https://www.sciencedirect.com/science/article/pii/S2772375524001436) | Multi-site evaluation — the kind of external validity this project does not yet have. | repo |
| Field spraying metrics, Frontiers in Plant Science, 2023. [link](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2023.1183277/full) | What a deposition/efficacy metric looks like. Out of scope: this project is perception-only. | repo |
| [Field smart-sprayer performance example](https://elibrary.asabe.org/azdez.asp?AID=55717&CID=ja0000&JID=3&T=2&i=0&search=0&v=0) | As above. | repo |

## 6. Protocol, reproducibility, and standards

| Work | Why it is here | Status |
|---|---|---|
| Angelopoulos et al., *Conformal Risk Control*, ICLR 2024. [link](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf) | A principled route to an operating threshold with a guarantee, rather than one picked by eye. Not implemented. | repo |
| [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) | The basis for the seeding policy and for not promising bitwise determinism on GPU. | repo |
| [Pillow affine transform documentation](https://pillow.readthedocs.io/en/stable/reference/ImageTransform.html) | Letterbox geometry; the source of the per-axis scale correction. | repo |
| [ISO 18497-1:2024](https://www.iso.org/standard/82684.html) · [ISO 65.060.01 index](https://www.iso.org/ics/65.060.01.html) | Agricultural machinery safety. Relevant only if an actuation path is ever proposed — currently **NO-GO, permanently**. | repo |
| [CVAT](https://docs.cvat.ai/docs/) · [Label Studio labeling](https://labelstud.io/guide/labeling) · [COCO export](https://labelstud.io/guide/export) | Annotation tooling evaluated during the intake. | repo |

---

## Gaps worth closing

Honest list of what a reviewer would ask for and this bibliography does not have:

- **No small-object detection literature.** 56% of train boxes are COCO-small at
  512 px and 32.6% have a minimum side under 16 px, yet nothing here justifies
  512 as the operating resolution or considers tiled inference.
- **No EMA reference.** `ModelEMA` averages BN buffers as well as weights, which
  is a real design choice with consequences (it is currently the only reason the
  2026-07-30 checkpoint loads at all) and nothing is cited for it.
- **No dataset/model card methodology reference.** Both artifacts exist; neither
  cites the convention it follows.
- **No licence for the Roboflow RICE export.** Not a citation gap — a legal one.
  It blocks release. See `docs/GATE_STATUS.md`.
- **The RiceSEG citation is inconsistent** between the arXiv preprint and the
  journal version across files in this repo. Pick one.

## Before this becomes a `.bib`

1. Resolve every `recalled` identifier against the publisher or arXiv page.
2. Complete the author lists — most entries above are truncated to first author.
3. Confirm year and venue; several of these have a preprint year and a
   publication year that differ.
4. Re-check any numeric claim attributed to a paper against the paper itself.
   The Group Normalization error figures in particular are quoted here from
   memory and are not to be reproduced without checking.
