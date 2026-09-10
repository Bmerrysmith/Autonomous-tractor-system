# Phase-2 RICE dataset — measured comparison, 2026-08-31

Independent re-measurement of the rebuilt detector dataset against the two older
arrangements of the same imagery. Every number below was computed from the files
on disk today, not carried over from a prior report. Where a measurement
disagrees with what `docs/HANDOFF.md` records, that is called out rather than
smoothed over.

Datasets compared:

| Tag | What it is | Location |
| --- | --- | --- |
| **NEW** | rebuild with `grouped_split.json` actually applied, EXIF normalized | `Downloads/RICE_phase2_rebuild_2026-07-29/` |
| **NATIVE** | the intake deliverable in Roboflow's own folder split | `Downloads/agrinav_intake_2026-07-21/deliverable/detection/RICE/` |
| **OLD ARCHIVE** | the zip the two voided 2026-07-28 runs trained on | `Downloads/RICE_curated_phase2.zip` |

---

## 1. Headline counts

### NEW — grouped split applied

| split | images | boxes | rice | weed | ratio | small (<32²) | empty | degenerate | out-of-bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,800 | 59,691 | 52,194 | 7,497 | 7.0:1 | 22,778 | 0 | 0 | 0 |
| valid | 518 | 15,226 | 13,201 | 2,025 | 6.5:1 | 5,793 | 0 | 0 | 0 |
| test | 261 | 6,284 | 5,355 | 929 | 5.8:1 | 2,106 | 0 | 0 | 0 |
| **total** | **2,579** | **81,201** | **70,750** | **10,451** | **6.8:1** | **30,677** | 0 | **0** | **0** |

### NATIVE — Roboflow folder split

| split | images | boxes | rice | weed | ratio | small | empty | degenerate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,798 | 56,502 | 49,226 | 7,276 | 6.8:1 | 21,344 | 0 | **2** |
| valid | 518 | 16,587 | 14,467 | 2,120 | 6.8:1 | 6,327 | 0 | 0 |
| test | 263 | 8,115 | 7,059 | 1,056 | 6.7:1 | 3,007 | 0 | 0 |
| **total** | **2,579** | **81,204** | **70,752** | **10,452** | **6.8:1** | **30,678** | 0 | **2** |

### OLD ARCHIVE — what actually trained

| split | image files in archive | boxes |
| --- | ---: | ---: |
| train | 1,798 | 56,502 |
| valid | 518 | 16,587 |
| test | **0** | — |
| **total** | **2,316** | **73,089** |

The archive has no `test/` directory at all — it was produced by deleting that
folder from the native split, which is the root of everything in §3.

---

## 2. Geometry and integrity

| Metric | NEW | NATIVE / OLD |
| --- | --- | --- |
| Degenerate boxes (w≤0 or h≤0) | **0** | **2** (native/old train) |
| Boxes outside image bounds | 0 | 0 |
| Images with zero annotations | 0 | 0 |
| Boxes total | 81,201 | 81,204 |
| Exact-duplicate images (same SHA-256) | **0 of 2,579** | — |
| Identical pixels spanning >1 split | **0** | — |
| EXIF re-oriented during rebuild | 214 of 2,579 | n/a |
| Distinct image resolutions | 9 | 9 |

The 3-box difference (81,204 → 81,201) is the rebuild's documented rejection of
3 boxes it could not repair under its one stated rule; 115 more were clipped by
≤1 px. The 2 degenerate boxes present in the native/old train split are gone.

**Zero duplicate-pixel leakage.** All 2,579 images hash distinctly, and no hash
appears in more than one split. Filename-level overlap is also zero within every
dataset. Whatever leakage risk remains is *not* exact duplication.

Object scale is the dominant property of this data: **30,677 of 81,201 boxes
(37.8%) are COCO-small (<32×32 px)**. Any detector result that reports only
aggregate AP is hiding where the difficulty actually is.

### Trap: annotation IDs are not unique across the split files

Each `instances_<split>.coco.json` numbers its annotations from 1, so every ID in
`valid` and `test` also exists in `train`. Measured in **both** datasets:

| dataset | train↔valid | test↔train | test↔valid |
| --- | ---: | ---: | ---: |
| legacy deliverable | 16,587 | 8,115 | 8,115 |
| **phase-2 rebuild** | **15,226** | **6,284** | **6,284** |

This is legal COCO — an `id` need only be unique within one document — but it
means **any lookup keyed on a bare annotation id across the three files returns
the wrong object, on the wrong image, silently**. It is not a hypothetical: it
was hit while building the polygon review sheet, where 332 of 1,238 paired
objects resolve from a non-train file and rendered unrelated polygons until the
key became `(split, id)`.

Anything joining these files must key on `(split, id)`, and should assert the
resolved box against the expected one rather than trusting the join.

Class imbalance is 6.8:1 rice-to-weed overall. Note the grouped split does *not*
equalize it — train 7.0:1, valid 6.5:1, test 5.8:1. The test split being the
weed-richest is convenient for evaluation, but it means test-set weed recall is
not directly comparable to train-time class frequency.

---

## 3. Contamination — reproduced exactly

The claim in `docs/HANDOFF.md` was re-derived from the files, not cited:

| Measurement | Result |
| --- | ---: |
| Intended sealed test images (NEW/test) | 261 |
| ...of those, present in OLD ARCHIVE train+valid | **231 (88.5%)** |
| ...still absent from the old archive entirely | 30 |
| Intended train+valid images (NEW) | 2,318 |
| ...missing from the old archive altogether | **233** |

Confirmed. The archive could not have been repaired by re-sorting its own
contents, because 233 of the images it needed were in the folder that was
deleted.

### How far apart the two arrangements are

`native_split` → applied grouped `split`, per image:

| from → to | images | |
| --- | ---: | --- |
| train → train | 1,261 | same |
| valid → valid | 115 | same |
| test → test | 30 | same |
| train → valid | 358 | moved |
| valid → train | 351 | moved |
| test → train | 188 | moved |
| train → test | 179 | moved |
| valid → test | 52 | moved |
| test → valid | 45 | moved |

**1,173 of 2,579 images (45.5%) sit in a different split than Roboflow put them
in.** Using the native split is not a minor deviation from the grouped one.

### Two contamination numbers, both correct

`trained_on_legacy` marks 179 test images, while §3 above says 231. They measure
different things and both are right: **179** test images landed in the old
archive's `train/` folder (gradient updates), **231** were in the archive at all
(train + valid, so 52 more influenced checkpoint selection). Per split:

| split | trained_on_legacy | never in old train |
| --- | ---: | ---: |
| train | 1,261 | 539 |
| valid | 358 | 160 |
| test | 179 | 82 |
| **total** | **1,798** | **781** |

The 781 figure in HANDOFF is confirmed, and it means "never in the old *train*
folder" — not "never seen by any model".

---

## 4. Discrepancy found: the group-straddle count

`docs/HANDOFF.md` states that **3** re-derived capture-family/frame-block groups
straddle a split boundary. Measured from `manifests/split_membership.json`:

- `group_id` yields **115 distinct groups**
- **24 of them (20.9%) straddle a split boundary**, several spanning all three
  splits (e.g. `1a_image (#0` appears in train, valid and test)
- `grouped_split.json` separately records `num_groups: 68`, `block_size: 40`

So three numbers disagree: 68 recorded groups, 115 re-derived groups, and 3 vs
24 straddles.

**This is not proof of leakage in the applied split**, and the dataset says so
itself. `manifests/provenance.json` carries an explicit `group_id_note`:

> `group_id` is RE-DERIVED by `derive_group_id()` (capture-family stem +
> `frame // block_size`). The source manifest does not store per-file group ids,
> so this is not the provenance of the train/valid assignment.

The same file records the actual method: *"grouped by capture-series family,
40-frame contiguous blocks; greedy weed-balanced 70/20/10"* — which also
explains the per-split class ratios in §1 (the split was balanced on weed share,
not on rice:weed ratio).

So the 24 straddles are a property of the re-derivation, not a defect in the
split. The spanning groups are still-image families, where a `frame // 40` rule
has no meaning to begin with. Two things nonetheless remain true:

1. The manifest **cannot be used to prove group integrity** on non-sequence
   families. "Leakage-free" is still not a claim this data can support — it is
   simply unproven rather than disproven.
2. HANDOFF's specific figure of **3** straddling groups is not reproducible from
   any artifact the dataset ships, and should be corrected or sourced.

Resolving (1) means deciding a grouping rule and recording per-file group ids —
already flagged in HANDOFF as ADR-level work, deliberately not done.

---

## 4b. Roboflow adds nothing — the clean-split lead is local

A parallel read-only enumeration of the Roboflow workspace
(`reports/summaries/roboflow_inventory_2026-08-31.md`) found **no genuinely new
imagery in any of the 7 projects / 20,489 images**. Everything resolves to files
already on local disk. There is no export worth doing.

The lead is `agrinav_data/incoming/extracted/riceseg`: per-pixel expert masks
that convert to boxes mechanically, spanning 5 countries and 19 sites, and never
trained on. That is real domain shift with labels attached — the thing the
phase-2 set cannot supply.

**It carries a grouping hazard that must be handled before it is split.**
Verified here: its 6,156 files collapse to ~1,521 distinct source names once the
`_subset_overlap_<row>_<col>` tile suffix is stripped, **1,361 of which have more
than one tile, and a single photograph yields up to 52 overlapping crops**. A
tile-level random split therefore puts *the same pixels* in train and test. Any
split of this set must group by source photograph, and preferably by site.

**The rebuild is sound and strictly better than both older arrangements.** It
applies the intended split, removes 2 degenerate boxes, normalizes 214
EXIF-rotated images so pixels and boxes agree, carries per-image SHA-256, and
has zero duplicate-pixel or filename leakage.

**Do not use NATIVE or OLD ARCHIVE for training.** 45.5% of images sit in the
wrong split relative to the intended grouping, and the archive is missing 233
images outright.

**Open, and blocking any quotable test number:**

1. The group-straddle discrepancy in §4. A test split whose grouping cannot be
   reproduced cannot be called leakage-free.
2. Only 82 of the 261 test images were never in the old train folder. A model
   trained from scratch on the rebuild has seen none of them, so the split is
   usable for *that* model — but the 2026-07-28 checkpoints may never be scored
   on it.

## Reproducing this

Script: `scratchpad/compare_datasets.py` (session scratchpad), plus two inline
manifest analyses. All inputs are read-only; nothing was modified.
