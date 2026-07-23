# Deep Dive: Previous Training Outcomes & How the Data Affected Them

*AgriNav rice/weed perception — design-review-style root-cause analysis.*
*Generated 2026-07-22. Evidence: internal iteration summaries, phase fix-logs, dataset cards, and first-hand re-analysis of the `detector_v1` COCO splits.*

---

## Executive Summary

Across seven training iterations (v1 RetinaNet → v6b/T7d), **the binding constraint moved over time from data to code and back to a data *ceiling***:

1. **Early runs (v1–v4) failed silently for data/label reasons** — unrescaled boxes after resize (v2: loss → 0), cross-dataset domain gap (v3: mAP ≈ 0), and *no validation set at all* (v4). None produced a usable metric.
2. **Mid runs (v5–T7d) were throttled by training-code bugs** — double-counted loss, an ATSS ignore-band that flooded 300 detections/image at confidence 1.00, and label-corrupting translation augmentation. Several earlier "the data is bad" conclusions were later **retracted** once these code bugs were found (the RiceSEG backbone was explicitly *exonerated*).
3. **What remains is a genuine data ceiling, not a crash:** severe class imbalance on the hard classes (weed / duckweed / senescent), a clean-biased curated set with **no verified empty/negative scenes**, and no aerial-weed source. Best honest detector result is **val AP@50 ≈ 0.088** (EMA; raw 0.011), and segmentation tops out at **mIoU ≈ 0.62 in-distribution** vs a ≥0.75 unseen-site target.

The good news, confirmed first-hand: the **current `split_v1` is remediated** — 0 exact and ~1% residual leakage, imbalance pulled from ~107:1 down to ~2:1. The trap is the **`_with_paddy` variant**, which swings imbalance back to **9.2:1**.

---

## 1. Timeline of Training Outcomes (what actually happened)

| Ver | Setup | Reported result | Primary failure | Root cause class |
|---|---|---|---|---|
| **v1** | RetinaNet R50-FPN, 12 ep, 80/20 | no mAP retained (POC) | — | — |
| **v2** | WeedDet initial, 1000×600 | **loss → 0.0000 from ep2** (silent) | VOC boxes not rescaled after resize → 0 positive anchors → 0 gradient | **DATA/label geometry** |
| **v3** | 94 ep, anchor scale 4, merged COCO ~5.5k | best val_loss 0.7437; **mAP ≈ 0** | cross-dataset domain gap ("explicitly NOT a code bug") | **DATA/domain** |
| **v4** | 60 ep, "**no_val**", IMG 512 | loss 1.04→0.79; **mAP not evaluable** | `valid/` folder empty → no held-out set | **DATA/split hygiene** |
| **v5** | Det-R50 from scratch, eFPN, VariFocal | val_loss 0.616; **AP@50 0.166, AP@75 0.004, 48.8 det/img** *(later superseded)* | loss double-counted `2×(cls+reg)`; NMS too aggressive; **test mixed into train/val** | **CODE + DATA/leakage** |
| **v7 / T-series** (TEST_LOG) | COCO split, ATSS, riceseg-pretrained backbone | T5 **AP@50 0.141**; T7 frozen-backbone **AP@50 0.104 (stable)** | ATSS ignore-band → FP flood; clip starves positives; **catastrophic forgetting of riceseg backbone confirmed** | **CODE / training dynamics** |
| **T7d** (deep audit) | 100 ep, frozen RiceSEG init | **val AP@50 0.0884 EMA / 0.0110 raw**; AP75 0.001; **AP-large 0.000**; 300 det/img @ conf 1.00 | translation aug corrupts ~½ of samples → **T7d & earlier baselines invalid** | **CODE bug corrupting DATA** |
| **RiceSEG pretrain** | segmentation backbone | **mIoU 0.575**; per-class weeds **0.23**, duckweed 0.35, senescent 0.36 | hard classes starved | **DATA/imbalance** |
| **DeepLabV3 control** | stock model, same split | **mIoU 0.617**; weeds IoU **0.493 (~2× custom)** | still in-distribution only | arch + **DATA ceiling** |

**Cross-version pattern:** the recurring failure mode was *silent invalidity* — a run that "completed" while its objective or its metric was wrong. Data/label defects were the true cause in v2–v4; code bugs dominated v5–T7d; and even after fixes, the **hard-class + domain ceiling** persists.

---

## 2. How the Data Specifically Affected Outcomes

### 2.1 Train/val leakage — severe in the past, fixed in the current split
- **Past (invalidated metrics):** an earlier "dHash leakage-safe" split was **not source-independent** — *"seven identifiable capture series span all three splits, including at least 73/134 validation and 71/134 test images from series also in training."* The external In-Field Panicle set was worse: **903/908 source photos in both splits (99.4% of val leaked)**; its paper AP (96.8/93.7/82.4%) was *"measured across this leak"* and is not a comparable baseline.
- **Now (re-verified first-hand on `artifacts/detector_v1`):**
  - `split_v1`: **0** exact train∩val overlap; **4/510 (1%)** val images share a capture series with train.
  - `split_v1_with_paddy`: **0** exact; **6/445 (1%)** residual.
  - Residual is a handful of shared plot-series (e.g., `20js0805xian2plot`). **Effectively remediated.**
- **Effect:** leakage inflates val metrics and hides poor generalization. Because it co-occurred with the augmentation and loss bugs, the audit is honest that *"its exact effect cannot be isolated"* — so **all pre-fix numbers should be treated as unreliable, not as evidence the model works.**

### 2.2 Class imbalance — the hard-class ceiling (re-verified first-hand)
- **Historical detector truth:** weed boxes were **0.924%** of all boxes (358 weed vs 38,388 rice) ≈ **107:1**. RiceSEG weed *masks* were imported specifically to fix this (RiceSEG internal rice:weed ≈ 2.8:1).
- **Current `split_v1` boxes (my count):** rice_protect **10,965** : weed_target **5,511** : non_target_aquatic **5,525** → **~2:1**. The weed-supply fix worked.
- **`split_v1_with_paddy` (my count):** rice_protect **49,310** : weed 5,347 : aquatic 5,613 → **9.2:1**. Adding paddy panicles (all "rice") **re-creates the imbalance**.
- **Effect:** segmentation confirms the ceiling — weed IoU **0.23 (custom) → 0.49 (DeepLabV3)** on identical data, but duckweed/senescent stay ~0.36 for *both* architectures. That residual is **data, not architecture**. Training on `_with_paddy` would predictably depress weed/aquatic recall further.

### 2.3 Domain gap & missing negatives — the safety problem
- The curated set **intentionally excludes glare/blur/occlusion** (clean-biased) and contains **no verified no-rice / empty scenes**. v3's mAP ≈ 0 was attributed directly to a cross-dataset domain gap.
- In-distribution mIoU **0.62** is far below the **≥0.75 unseen-site** target; the detector saturates at **300 detections/image at confidence 1.00**.
- **Effect:** with a spray-by-default control policy, an over-triggering detector with no "empty field" training signal is an **unsafe stop condition**, not just a low score. This is the most consequential data gap for deployment.

### 2.4 Label noise & auto-labeling
- `auto_annotate.py` mapped broad prompts (*plant/crop/green plant*) all to **"Rice"** → not trustworthy ground truth (parallels this session's finding that **SAM3 text auto-label produced 0 usable rice/weed annotations**).
- A superseded audit flagged **4.5% out-of-bounds** and **10.6% ultra-tight** Roboflow boxes; those Roboflow boxes were subsequently **demoted to unreviewed proposals and excluded from training truth**.

### 2.5 Duplicate & unusable source datasets (leak/waste risk if used naively)
- **BD weed V4 vs V3: 3,630/3,632 identical (99.9%)** — loading both would place the same photo in train and test.
- **WeedDataset-main: 1,761 LFS pointer stubs, 0 real images.**
- Grain-cultivar set (5.56 GB): wrong subject, 0 annotations.
- **Effect:** none of these reached training truth (correctly rejected), but they are the reason a naive "throw all datasets in" split would have leaked badly.

---

## 3. Data vs. Code — an honest attribution

A recurring theme in the audits is **retraction of earlier data-blame**:
- The AP *collapse* (raw AP 0.0003 / EMA 0.0010; yet overfit-16 reached AP@50 0.60) was attributed by the 2026-07-09 audit to a **training-supervision code bug** (classification ignore band, `neg_iou_threshold=0.4`) — **explicitly not a data problem**, and the RiceSEG init was *"exonerated."*
- `COMPLETE_ANALYSIS_FINDINGS.md` metrics (AP@50 0.166, etc.) are **superseded**; the real causes were *no pretrained backbone* and *BN frozen at random init*.
- The single largest baseline-invalidator (T7d) is the **Pillow translation augmentation** moving pixels and boxes in opposite directions — a *code* bug that *corrupts the data* at train time.

**Takeaway:** don't over-attribute to raw data. The pattern is: (a) genuine data defects early, (b) code/training bugs in the middle that were sometimes *misdiagnosed* as data problems, and (c) a real, measured data ceiling on hard classes + domain coverage that survives every code fix.

---

## 4. Does the current direction fix it?

- **Leakage:** ✅ largely fixed — `split_v1` is source-independent (verified 1% residual). The leak-safe **grouped split** used for this session's RICE work applies the same principle.
- **Weed supply:** ✅ fixed for `split_v1` (2:1). ⚠️ **Avoid `_with_paddy` for weed-sensitive training** (9.2:1).
- **Hard-class ceiling (duckweed/senescent/weed):** ❌ still open — needs *more real minority-class instances*, not more rice or more augmentation.
- **Domain gap / negatives:** ❌ still open — needs glare/blur/occluded frames and **verified empty-field scenes** for a safe stop condition.
- **This session's `Training_DataV7` pool** (6,291 images, clear/dense, annotations stripped) is a **labeling pool**, useful for expanding *coverage/diversity*, but note it is (a) image-classification + paddy imagery, mostly *rice*-side, and (b) still clean-biased — it does **not** by itself add the missing weed instances or negative scenes.

---

## 5. Recommendations (priority order)

1. **Freeze `split_v1` (not `_with_paddy`) as the balanced detector split**; keep `_with_paddy` only for a rice-recall ablation.
2. **Fix the code bugs before trusting any metric:** disable/repair the translation augmentation, land the honest loss + NMS/postprocess fixes (P1-5/6/7 still open in Phase 2), then re-baseline. Every pre-fix AP is uninterpretable.
3. **Attack the hard-class ceiling with data, not tricks:** source/label additional weed, duckweed, senescent instances; measure per-class recall, not just mAP.
4. **Add negative/empty-field and degraded (glare/blur/occlusion) scenes** — required for a defensible generalization *and* a safe spray stop-condition.
5. **Keep an unseen-site holdout** (`holdout_country`/site) and report on it; in-distribution 0.62 mIoU is not the ≥0.75 target.

---

## Open Questions / Caveats

- **No saved weights or metric CSVs** are in the repo, so reported AP/mIoU numbers could not be independently *recomputed* — they are taken from the iteration/audit docs (several of which supersede each other; superseded figures are flagged above).
- The exact magnitude of past leakage on reported metrics **cannot be isolated** from the concurrent augmentation/loss bugs — treat pre-fix numbers as invalid rather than quantifiably biased.
- Capture-series grouping in §2.1 uses a filename-prefix heuristic; it confirms the *order of magnitude* (≈1% residual) but is not the project's canonical `group_id` logic.

## Sources (internal, read-only)
- `_archive/legacy_code/archive/v{1..5}*/ITERATION_SUMMARY.md`, `v5_era/weeddet_latest_v5_optimization_report.md`
- `_archive/historical_docs/{TRAINING_OPTIMIZATION_GUIDE,TEST_LOG,AUDIT_2026-07-09_V7,COMPLETE_ANALYSIS_FINDINGS}.md`
- `docs/audits/2026-07-20/{AGRINAV_FULL_DEEP_AUDIT,PHASE1_TRANSFER_LEARNING_FIXLOG,PHASE2_DETECTOR_FIXLOG,AGRINAV_DEPLOYMENT_ROADMAP}*.md`
- `docs/research/RICESEG_PRETRAIN_RESULTS.md`, `docs/detector_dataset_card.md`, `docs/EXTERNAL_DATASETS_AUDIT.md`, `docs/DATA_ORGANIZATION.md`
- `data/rice_training_curated/AUDIT_REPORT.md`, `active/ACTIVE_NOTES.md`
- First-hand re-analysis: `artifacts/detector_v1/split_v1/{train,val}.coco.json` and `split_v1_with_paddy/` (this report, §2.1–2.2)
