# Plan Audit — split / pretrain / train (2026-07-07)

## Headline
The strategy is sound. The **execution artifacts are not there.** Your repo's git is a single
April "Initial commit," and *none* of the files the plan tells you to "drag in" actually exist
in the connected folder. Both Colab notebooks will stop on their first real cell.

Verified MISSING from `agrinav_full/`:
- `training/riceseg_pretrain.py`  ← run_pretrain.ipynb depends on this
- `training/USING_RICESEG_BACKBONE.md`
- `data/build_coco_splits.py`, `data/split_coco_on_drive.py`  ← run_split.ipynb depends on this
- `paper/references.bib`, `paper/references_IEEE.tex`, `SESSION_CONTEXT_2026-07-08.md`

They were produced in a previous ephemeral session and never persisted here. Everything below
assumes you either regenerate them or I rebuild them.

---

## 1. Is `split.py` (the Colab split-on-Drive pattern) necessary? — No.

It is both avoidable and partly redundant.

**Redundant:** your training notebook (`active/weeddet_trainingV6b.ipynb`, Cell 2) already does its
own 80/10/10 split in-notebook straight from the VOC zip. The output of `run_split.ipynb` is never
consumed by the path you actually train on. You are maintaining two split mechanisms that don't
feed each other.

**Can I just split it and hand you a zip?** Yes — with one real constraint to plan around:
- The Google Drive connector only moves files **≤10 MB**, so I cannot pull the 67 MB VOC zip
  (or the 645 MB COCO zip) through it, and the sandbox can't reach Drive directly.
- The 645 MB size was the *only* reason the Colab-on-Drive pattern was invented. The sandbox
  processes that size locally without issue — it's just the *connector transfer* that's capped.
- **Fix:** drop the source zip into this "Autonomous driving tractor" folder. I'll produce a
  frozen, leakage-safe `train/valid/test` zip in one pass — no Colab, no Drive round-trip.

**Two broken links in the current split plan (it will not run today):**
- `run_split.ipynb` expects `Rice Classification.coco.zip` in My Drive root. It is **not on your
  Drive** — search finds only the two VOC zips and `RiceSEG.zip`. Step 1 fails.
- It says to drag in `split_coco_on_drive.py`, which **doesn't exist** in the repo.

**Recommendation:** pick ONE split path and delete the other.
- (a) Simplest: let the training notebook keep splitting internally. Delete `run_split.ipynb`
  and the two COCO-split scripts from the plan entirely.
- (b) Better for a clean ablation: I produce one frozen dedup-safe split zip locally, and every
  experiment reads that same split. Reproducible, and no leakage across train/val/test.
  Don't do both.

---

## 2. Fresh-run check — nothing resumes from an old checkpoint (good), but step 4 isn't wired up.

**Pretraining (`run_pretrain.ipynb`): genuinely fresh. ✓**
Assemble → self-test → overfit-8 → full pretrain (`--epochs 30`) → export `riceseg_backbone.pth`.
No checkpoint resume. The optional ImageNet warm-start is a backbone *init*, not a project
checkpoint; pass `--no-imagenet` if you want fully from-scratch.

**WeedDet training (`weeddet_trainingV6b.ipynb`): fresh, but WRONG config for this plan.**
Each run builds a fresh `WeedDet`, calls `load_imagenet_backbone`, and runs a fresh epoch loop —
it never resumes from `weeddet_v6b_best.pth`. So "no old checkpoint" is satisfied. **However** it
uses `load_imagenet_backbone` + `apply_bn_policy` — the exact F1+F2 combo your own audit found
crushes AP (~0.017 vs v5GPT's 0.166). It does **not** load `riceseg_backbone.pth` and does **not**
keep BN trainable.

**So the plan's Step 4 notebook effectively does not exist.** You need one new fine-tune notebook =
copy of V6b with three edits: `load_imagenet_backbone` → `load_riceseg_backbone`, delete every
`apply_bn_policy` call, keep the fresh epoch loop. That is the only training notebook you need.

---

## 3. Efficiency / simplicity

Over-built areas:
- **Three overlapping split mechanisms** (`data/step2_split.py`, the missing COCO-split pair, and
  the notebook's own in-cell split). Collapse to one.
- **COCO conversion adds hops for no accuracy gain** — your own notes concede format ≠ accuracy.
  The only real reason to touch COCO is the *eval metric* (AP@[.5:.95], AP@75). You can compute
  those with pycocotools on GT built from the VOC boxes at eval time, without converting the
  training data or maintaining a COCO split on Drive. V6b already trains from VOC and reports
  AP@50/75.

### Minimal executable plan
1. **Freeze one split.** Drop the VOC zip in this folder → I return a leakage-safe train/valid/test
   zip. (Or keep the notebook's internal split and drop the COCO-split pipeline.)
2. **Pretrain:** `run_pretrain.ipynb` is clean — but first confirm `riceseg_pretrain.py` actually
   exists and gets uploaded; it's currently missing.
3. **Fine-tune WeedDet:** one new notebook (V6b copy) → `load_riceseg_backbone` → BN trainable →
   fresh epoch loop. No `apply_bn_policy`.
4. **Eval:** pycocotools AP@50 / AP@75 on the held-out test split.

### Gate first
Your open item #1 still stands: run the Overfit-10 A/B (pretrained+BN-freeze vs scratch+all-BN)
before investing in the full RiceSEG pipeline. That verdict is what justifies the whole backbone
strategy.
