# Google Drive File Inventory
## All AgriNav-related files

**Account:** bdmerrymansmith@gmail.com
**Indexed:** 2026-07-09 (V7-era update; 2026-06-01 entries kept below)

---

## ACTIVE — V7 / COCO era

| File | Type | Modified | Drive ID | Notes |
|---|---|---|---|---|
| **weeddet_trainingV7_coco.ipynb** | Colab | 2026-07-09 | 1sKq1EGt0eo9Ey3yyka3wabOoe25NNaBb | **CURRENT training notebook** |
| **weeddet_v6b.py** | Python | 2026-07-09 | 1hwy3Kne8UBQF20zi2llj3TFeahjz1NM4 | **CURRENT model** (cls_hard_target fix) — in weeddet_v2_checkpoints/ |
| riceseg_backbone.pth | ckpt 94 MB | 2026-07-08 | 1LZuLdT08ShKnNTpNtCGtSZiRM2aGYOW5 | in weeddet_v6_checkpoints/ |
| riceseg_pretrain.py | Python | 2026-07-08 | 1GjNAGYd3nO4cCai50YnKkvCfsTw4SukO | in riceseg_pretraining/ |
| RiceSEG.zip | 231 MB | 2026-07-07 | 1pXtQtVwC5nj85aYIIc4Dv-uka4CucZaX | canonical copy in riceseg_pretraining/ |
| RiceSEG.zip (DUPLICATE) | 231 MB | 2026-07-07 | 1wVSISfhCv867so9vjAv5G8HKxcBweihg | at Drive root — safe to delete |
| weeddet_v7_riceseg_best.pth | ckpt 212 MB | 2026-07-09 | 1g6lVg-EZoaqcp6AUkGVCsvUIb5XX1D2k | run #2 best (AP 0.0010); has raw_state_dict too |
| curves_v7_riceseg.png | PNG | 2026-07-09 | 1bdzuffvrPJ_LxHfGFu2nWmHmjsyE9SIT | run #1 curves |
| weeddet_v7_riceseg_epoch4..60.pth | 15 ckpts ~1.6 GB | 2026-07-09 | (in v7 folder) | failed run #2 — delete after triage |

## Checkpoint folders

| Folder | Drive ID | Contents |
|---|---|---|
| weeddet_v2_checkpoints/ | 1yRIIemlDkhSAsAM0KKAz_2_SoiVjy3RH | weeddet_v6b.py + rice_detection_coco_split.zip + datasets |
| weeddet_v6_checkpoints/ | 1F1As5GPJzlsyh9659OuGbz9c42fWjNk8 | riceseg_backbone.pth |
| **weeddet_v7_checkpoints/** | 1d9YrtI7z_noVmUa-i6mlsGZXOh42KqT6 | **ACTIVE** V7 checkpoints + curves |
| riceseg_pretraining/ | 1p6u6pRK0LZYXuqcBMBCQREcvfyzD1iRn | riceseg_pretrain.py, RiceSEG.zip, run_pretrain.ipynb |
| weeddet_checkpoints/ (v1) | 1MmcPXlO1wt7Y6HTvGfs5o3ag4A-pIf6F | early runs — archive |
| weeddet_v4_checkpoints/ | 1Uk2mVQtuP2TxUn93miumAPchLL0m7j39 | v4 best — archive |
| weeddet_v5_checkpoints/ | 1sYMaWVemAACZjGc2KGX8ouU-m-TaTB1q | v5 era — archive |

## Notebooks (V6/V7 era, chronological)

| Notebook | Modified | Drive ID | Status |
|---|---|---|---|
| weeddet_trainingV5GPT_fixed.ipynb | 2026-06-10 | 18t1dU_6UgBDdvhkpIooWHNfePDYLgtH0 | 0.166 baseline reference (VOC-era split) |
| weeddet_trainingV6.ipynb | 2026-07-03 | 1ZkMwjqRxZklXThQUYn8hzSvreJUTKFYT | archive |
| weeddet_trainingV6rd.ipynb | 2026-07-04 | 1TYr2eJIMIhVB3WR_VaW-1G5FTYS9UEWf | archive |
| weeddet_trainingV6b.ipynb | 2026-07-05 | 1HDKZ6-NNo0HwhoCJNkTO8E0MvZLMFS-Q | archive (overfit-10 gate) |
| weeddet_trainingV6b.ipynb (2nd) | 2026-07-07 | 111-bG376oi2mIBM2z1GrdhpHplv8ipfK | archive |
| **weeddet_trainingV7_coco.ipynb** | **2026-07-09** | **1sKq1EGt0eo9Ey3yyka3wabOoe25NNaBb** | **ACTIVE** |

## Datasets

| File | Images | Format | Drive ID |
|---|---|---|---|
| rice_detection_coco_split.zip | 1347 (1079/134/134) | COCO, pre-split | in weeddet_v2_checkpoints/ |
| rice_detection_for_export.v1i.voc.zip | ~1,347 | VOC XML | 1KHdX2tduaeLDC9QEHTHMGUOc374Ys8Ft |
| Rice_Classification.v1i.voc.zip | ~4,041 | VOC XML | 1HztyoHqdmP08ZaFsRwszY-DfMwKEgASJ |
| rice-dataset-two.zip | Mixed | Mixed | 1tkQeHmqaNS5PFzVQXjR4-Tb7sLabynvS |

---

# Archive — inventory as of 2026-06-01 (superseded)

## Key Model Scripts (legacy)

| File | Drive ID | Notes |
|---|---|---|
| weeddet_Latest.py | 1WPJ0ocfCtjrBxfv4udwt2caLtZ7ck-IV | v5-era model |
| weeddet_for_VSCode.py (v2_ckpts) | 1vxf29yeHuZiP75U-r0uTw9EDee-sJnTa | older |
| weeddet_for_VSCode.py (early) | 1mbqgO3Tu4osqLxf2qjwcNpFrA1TatLl2 | oldest |

## Legacy Notebooks

| Notebook | Modified | Drive ID |
|---|---|---|
| weeddet_trainingV5.ipynb | 2026-06-01 | 1r5gjFAPcICGxCddr174-x9tAvYs3HyOj |
| weeddet_trainingV5_fixed.ipynb | 2026-06-04 | 1hk4wN3enM9-cPQbPqlnGyS15WiKA-EV3 |
| weeddet_QUICK_FIX_NMS_eval.ipynb | 2026-07-02 | 15jdexrWpjkazJlXBb7XmzDzMFNwqEZqo |
| weeddet_trainingV4.ipynb | 2026-06-01 | 1ao9PE4UI7peRnTAHWbo2LX-EP6Ww6FJq |
| Weed_Det_Training_v3.ipynb | 2026-04-24 | 1FXrpnFOtDspk0Wy0BlpxvsS8YwVVYh1e |
| (earlier notebooks unchanged — see git history of this file) |

## Presentations and Documents

| File | Type | Modified | Drive ID |
|---|---|---|---|
| Architecture for Autonomous Tractor | GDoc | 2026-03-31 | 1Yg8q1w8hS2qNdyguzjAz5AyPUeOkurTHT35IgcscuGc |
| Outline technical analysis of paper | GDoc | 2026-03-25 | 1iSt948yNEh1AxczXNAhOuzUb0RRoe7xkFTR8kyv_CzQ |
| presentation transcript | GDoc | 2026-03-25 | 1rQIpLzCNERHf4WZNlUI79mZOKVhGHYEmpnrvMNhZBfc |
| Tractor_Team_Proposal.docx | Docx | 2026-04-12 | 1QIl38F7H7_aCjaWbzasZ12NLLF8UpmWe |
| AgriNav_Progress_Report.pptx | PPTX | 2026-04-29 | 1NTHncDo4bmZvMHzqgv-J5RQzPlnkbVdi |
| AgriNav_Final_Report.pptx | PPTX | 2026-04-29 | 1KoYsrhCWTcASQf9DW2iOENuOi0SA4bQM |
| Changes to research paper | GDoc | 2026-05-19 | 1-xBkJVoiux7mu0zZbvLK5dY4mXtVzON5BhPG3buk48w |
| Assignment_3_CEN4930_completed | GDoc | 2026-05-02 | 1PS7bcbUgz1VTUHJLoZgl3CB4FASLmUR2cb5pTgAV4HY |
| rice_detection_demo.mp4 | Video | 2026-04-28 | 1hFilZGUM7MrvN13kmSNpKnpsE7DGlC_q |

## Personal (not project-critical)

| File | Drive ID |
|---|---|
| Benjamin_Merryman_Resume_Updated | 1Puoz2Ac0KtomNxsb_OGQroAJPqQ-NVMQ |
| Copy of Interview Story Portfolio | 1MoY_x309Ww_386WZhGeYLMGCQu2vi9I6dQnf2LOEjel0 |
