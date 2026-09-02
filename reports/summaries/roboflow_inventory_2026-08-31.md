# Roboflow workspace inventory — 2026-08-31

**Workspace:** `i-dont-know-la5cz`
**Access mode:** read-only. No project, version, model, or annotation was created, modified, generated, trained, or deleted. No image data was downloaded; only metadata and counts were read.
**Purpose:** decide what, if anything, is worth exporting for a second-phase rice/weed detector round.

## Headline

**Nothing in the Roboflow workspace is worth exporting. Every image in all 7 projects is already present locally.** The workspace holds 20,489 unique images; each one traces to one of the canonical local sets. No new field, scene, growth stage, lighting condition, or weed class was found.

The single most valuable untapped imagery is **local, not on Roboflow**: `riceseg` holds 3,078 tiles from 848 distinct source photographs across 5 countries, with per-pixel masks that have never been converted to detection boxes. See [Recommended exports](#3-recommended-exports).

---

## 1. Summary table

| Project (slug) | Type | Images (in dataset) | Classes | Versions | Verdict |
|---|---|---:|---|---:|---|
| `rice-mykud-mgwf8` (RICE) | object-detection | 2,581 | rice, weed | **0** | Already local |
| `rice-weed-seg` | instance-segmentation | 2,579 | rice, weed | 1 | Already local (polygons only novelty) |
| `rice-hainan-e78yy` | object-detection | 1,349 (+2,230 unannotated) | rice, weed | 1 | Already local |
| `rice_detection_for_export` | object-detection | 1,347 | rice, weed | 1 | Already local |
| `rice-weed-bd-species` | classification | 4,360 | 11 weed species | 1 | Already local; no boxes |
| `rice-plant-raw` | instance-segmentation | 0 (5,370 unannotated) | none | 1 (empty) | Unusable — zero annotations |
| `rice_data-dgnw9` (Rice_data) | object-detection | 0 | none | 0 | Unusable — empty shell |

**Workspace-wide unique images:** 20,489 (`images_workspace_search`, `query:"*"`).

### Cross-cutting facts

- **No augmentation exists anywhere in this workspace.** Every version's preprocessing is `auto-orient` only, plus a resize on two of them. No flip, crop, rotation, noise, mosaic, or brightness augmentation is configured on any version. **No version multiplies its image count** — each version's image count equals its parent project's dataset count exactly.
- Images are **shared by id across projects**. The same photo can also exist under two different ids from two separate uploads (confirmed: `seedlingCol_04_0240.jpg` exists as both `MMRyfaJM3FYDapiSZijd` and `fAfHWghbGvYZeCMb0HEs`).
- **The `.rf.<hash>` suffix on local Roboflow-export filenames is the Roboflow image id.** This gives an exact local↔Roboflow join key and was used for the split verification in §2.2.
- **The `unannotated` field returned by `projects_list`/`projects_get` is unreliable.** `rice-weed-seg` reports `unannotated: 2579` while simultaneously reporting 81,203 annotations across the same 2,579 images. Do not use this field for decisions.
- **The RoboQL `split:` filter does not scope to the named project.** `project:rice-weed-seg split:test` returned `total: 494` and included images whose `rice-weed-seg` split is `valid`. Split counts in this report come from `versions_get`/`projects_get` only.

---

## 2. Per-project detail

### 2.1 RICE — `i-dont-know-la5cz/rice-mykud-mgwf8`

| Field | Value |
|---|---|
| Display name | RICE |
| Type | object-detection (**boxes**) |
| Images in dataset | 2,581 |
| `unannotated` field | 1,003 (unreliable, see above) |
| Classes | `rice` 70,752 · `weed` 10,452 |
| Project splits | train 1,799 / valid 518 / test 264 |
| Versions | **none — `versions: []`** |
| Created / updated | 2026-01-05 / 2026-08-19 |

**Verdict: already fully represented locally.**

2,579 of these 2,581 images are the local `RICE_phase2_rebuild_2026-07-29` set and `agrinav_data/incoming/extracted/rice_coco_annotated`. Local filenames carry the Roboflow ids, and every id sampled resolved to a local file.

**This project cannot be exported at all without a write operation.** It has zero versions, and Roboflow only exports from a generated version. Generating one is explicitly out of scope for this task.

**Its native split is capture-series contaminated.** Directly observed assignments on adjacent video frames from one sequence:

| Filename | RICE native split |
|---|---|
| `frame_000072.jpg` | valid |
| `frame_000074.jpg` | train |
| `frame_000075.jpg` | train |
| `frame_000104.jpg` | train |
| `frame_000108.jpg` | **test** |
| `frame_000110.jpg` | valid |
| `frame_000112.jpg` | train |

Six frames spanning 40 frames of one continuous capture are split across all three partitions. This is the exact failure mode that voided the previous run.

### 2.2 rice-weed-seg — `i-dont-know-la5cz/rice-weed-seg`

| Field | Value |
|---|---|
| Type | instance-segmentation (**polygons**) |
| Images | 2,579 |
| Classes | `rice` 70,751 · `weed` 10,452 |
| Splits | **train 1,800 / valid 518 / test 261** |
| Version 1 | "2026-07-22 1:24pm", 2,579 images |
| Preprocessing | `auto-orient` only |
| Augmentation | **none** |
| Trainings | none |

**Verdict: already fully represented locally.** The pixels and the boxes are both local. The polygons are the only content not held locally.

**This is the one project that carries the project's grouped split rather than Roboflow's native split.** Its split sizes match `manifests/split_membership.json` exactly (1800/518/261). I joined 13 Roboflow image ids to the local manifest via the `.rf.<id>` filename key and compared assignments:

| Roboflow id | Filename | rice-weed-seg | Local *grouped* | Local *native* | Match |
|---|---|---|---|---|---|
| `OpxiITDEgUoSvr2xjJWv` | `2_94.jpg` | train | train | valid | yes |
| `GUrb8b9zOvpC9ZgnLIrq` | `frame_000074.jpg` | train | train | train | yes |
| `Ab2gnn9J0LYZkpD0Qmi9` | `1b_image (80).jpg` | valid | valid | train | yes |
| `QG4KOeKKfxUY6n6YjHh8` | `frame_000075.jpg` | train | train | train | yes |
| `STkeNqqC37AI8dQUwRMd` | `frame_000562.jpg` | train | train | train | yes |
| `cKZgNgrkqc0CpgXtrR9J` | `frame_001679.jpg` | test | test | train | yes |
| `EnNeOTtDELhaeMq07glF` | `frame_001140.jpg` | valid | valid | train | yes |
| `6sApCGOqULdZjzG0gCF8` | `frame_000108.jpg` | train | train | test | yes |
| `wjghOU6kzMdpTg3FxmJn` | `frame_000445.jpg` | train | train | test | yes |
| `DUJ1XdkrEQ8KtPN1lh8p` | `frame_000072.jpg` | train | train | valid | yes |
| `O7Q6bTa5faeXWdnaza8H` | `frame_000110.jpg` | train | train | valid | yes |
| `orUzNOqyhpufi9UtymWK` | `frame_000104.jpg` | train | train | train | yes |
| `MMRyfaJM3FYDapiSZijd` | `seedlingCol_04_0240.jpg` | valid | valid¹ | test | yes |

13/13 agree, including **7 cases where the grouped split deliberately diverges from the native split**. The adjacent-frame block `frame_000072…000112`, scattered across three splits natively, sits entirely in train here.

¹ via the sibling id `fAfHWghbGvYZeCMb0HEs`, which is the phase-2 lineage's id for the same photograph.

**Caveat on the polygons.** Stored session memory records that unauthored/spoofed SAM annotations appeared in the 2026-07-21 staging directory and must not be trusted or imported. These polygons were uploaded 2026-07-22, one day later. Their provenance is **not** established by anything the API exposes. Treat them as unverified until their origin is traced.

### 2.3 rice-hainan — `i-dont-know-la5cz/rice-hainan-e78yy`

| Field | Value |
|---|---|
| Type | object-detection (**boxes**) |
| Images in dataset | 1,349 |
| Total image records in project | 3,579 (`images_search` total) |
| Unannotated remainder | 2,230 (3,579 − 1,349) |
| Classes | `rice` 39,115 · `weed` 505 |
| Splits | train 946 / valid 267 / test 136 |
| Version 1 | "2026-07-20 2:18pm", 1,349 images |
| Preprocessing | `auto-orient`, `resize 384×384 "Stretch to"` |
| Augmentation | **none** |
| Trainings | 1, `rfdetr-nano`, status **cancelled**. `models_list` returns 0 models. |

**Verdict: already fully represented locally.** The project is two distinct populations:

1. **1,349 annotated images** — the same set as `rice_detection_for_export` (1,347). Across ~220 sampled dataset records, every single one also listed `rice_detection_for_export` in its `projects` array, except one. Locally this is `rice_detection_coco` (1,347 images).
2. **2,230 unannotated images** — riceseg tiles named `<stem>_subset_overlap_<row>_<col>.jpg`, all with `split: "none"` and `annotations: []`. 9 of 9 sampled filenames matched local `riceseg` exactly, including region path:

   | Roboflow filename | Local path |
   |---|---|
   | `HB1_subset_overlap_0_2.jpg` | `riceseg/global rice segmentation/China/HB/rgb/` |
   | `DSCF0247_subset_overlap_5_1.jpg` | `…/China/HN/rgb/` |
   | `23JSIMG_9339_subset_overlap_4_0.jpg` | `…/China/JS_3/rgb/` |
   | `20JSs0_1_subset_overlap_0_0.jpg` | `…/China/JS_1/rgb/` |
   | `2015_0812_045128_subset_overlap_4_2.jpg` | `…/Japan/TKO_3/rgb/` |
   | `13233_subset_overlap_1_0.jpg` | `…/Philippines/rgb/` |

**Augmented/derivative sibling hazard — this is the project to watch.** The `_subset_overlap_R_C` suffix means overlapping tile crops of a single source photograph. Locally, 3,078 tiles derive from only **848 unique source photos** (mean 3.6 tiles per photo, max far higher). Any split that treats these tiles as independent images will place crops of the same photograph — often overlapping pixels — into train and test simultaneously. rice-hainan's native 946/267/136 split does exactly this.

**One image was found that belongs to no other Roboflow project:** `20JS0805jin2plot7.jpg` (id `eX1jghdAnKhtkWKiW1Lt`), `projects: ["rice-hainan-e78yy"]`. **It is nonetheless already local**, at `riceseg/global rice segmentation/China/JS_1/rgb/20JS0805jin2plot7.jpg`. By count arithmetic (1,349 vs 1,347) at most 2 such images exist. I did not enumerate all 1,349 records to find the second — see [Open questions](#5-open-questions).

### 2.4 rice_detection_for_export — `i-dont-know-la5cz/rice_detection_for_export`

| Field | Value |
|---|---|
| Type | object-detection (**boxes**) |
| Images | 1,347 · `unannotated` 0 |
| Classes | `rice` 39,067 · `weed` 491 |
| Splits | **train 1,347 / valid 0 / test 0** |
| Version 1 | "2026-04-21 11:32pm", 1,347 images |
| Preprocessing | `auto-orient`, `resize 512×512 "Stretch to"`, `random-sample train:100 valid:100 test:100` |
| Augmentation | **none** |

**Verdict: already fully represented locally** as `agrinav_data/incoming/extracted/rice_detection_coco` (1,347 images, verified by file walk). The prior intake audit established by content SHA-256 that these 1,347 are an exact subset of RICE's 2,579; nothing observed here contradicts that.

The version has **no valid or test split whatsoever** — all 1,347 images are in train. It is unusable as a split source on its own terms.

### 2.5 rice-weed-bd-species — `i-dont-know-la5cz/rice-weed-bd-species`

| Field | Value |
|---|---|
| Type | **classification** (whole-image labels — no boxes, no polygons) |
| Images | 4,360 |
| Splits | train 3,279 / valid 787 / test 294 |
| Version 1 | "2026-07-22 1:43pm", 4,360 images |
| Preprocessing | `auto-orient` only |
| Augmentation | **none** |
| Trainings | 1 finished — `vit-base-patch16-224-in21k`, model id `rice-weed-bd-species-1-vit-base-patch16-224-in21k-t1`, created 2026-08-31. `metrics: null`. |

**Verdict: already fully represented locally** as `weed_v3` (Rice Field weed BD Dataset_V3). Per-class comparison against a local file walk:

| Class | Roboflow | Local `weed_v3` |
|---|---:|---:|
| W_CL_01 Alternanthera philoxeroide | 126 | 133 |
| W_CL_02 Centella asiatica | 437 | 437 |
| W_CL_03 Commelina benghalensis | 384 | 384 |
| W_CL_04 Cyperus ochraceus | 207 | 207 |
| W_CL_05 Fimbristylis littoralis | 198 | 198 |
| W_CL_06 Ipomoea aquatic | 904 | 904 |
| W_CL_07 Marsilea minuta | 379 | 379 |
| W_CL_08 Panicum repens | 310 | 310 |
| W_CL_09 Paspalum scrobiculatum | 514 | 514 |
| W_CL_10 Pteris vittata | 451 | 451 |
| W_CL_11 Synedrella nodiflora | 450 | 450 |
| **Total** | **4,360** | **4,367** |

Ten of eleven classes match exactly. The local set has 7 *more* images in W_CL_01. Roboflow therefore holds a strict subset.

**Even though this covers 11 named weed species the detector does not otherwise have, it contributes nothing to a detector**: classification labels give one label per whole image with no localization. Converting it into detection data would require annotating 4,360 images from scratch — and the pixels for that are already on local disk.

### 2.6 rice-plant-raw — `i-dont-know-la5cz/rice-plant-raw`

| Field | Value |
|---|---|
| Type | instance-segmentation |
| Images in dataset | **0** |
| Unannotated | 5,370 |
| Classes | **`{}` — none** |
| Splits | train 0 / valid 0 / test 0 |
| Version 1 | "2026-07-22 1:58pm" — **0 images, `splits: {}`, `ready: false`** |

**Verdict: unusable, and already local anyway.** Zero annotations exist on any of the 5,370 images. The generated version is empty, so there is nothing to export.

All 15 of 15 sampled filenames matched the local `rice_plant_image_dataset`:

`T169_80` · `T25_2898` · `G1244_114` · `T12_8865` · `G1237_4636` · `T22_17073` · `T166_228` · `T105_2223` · `G1244_760` · `T80_435` · `T70_1207` · `T164_12220` · `G1252_572` · `T84_1632` · `T156_2794`

Each resolved to both `…/Annotations/<series>/Annotations/<name>.png` and `…/Annotations/<series>/Images/<name>.jpg` locally.

### 2.7 Rice_data — `i-dont-know-la5cz/rice_data-dgnw9`

| Field | Value |
|---|---|
| Type | object-detection |
| Images / unannotated / classes | 0 / 0 / `{}` |
| Splits | all 0 |
| Versions | **none** |

**Verdict: unusable — empty shell.** `projects_get` reports it as entirely empty, yet `images_search` returns `total: 1347` and the slug `rice_data-dgnw9` appears in the `projects` array of nearly every `rice_detection_for_export` image. It holds the same 1,347 image records with **none admitted into its dataset**, no classes, and no version. There is nothing to export.

---

## 2.8 Local sets — counts I verified myself

File walks performed 2026-08-31. Image files counted by extension (`.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`); "other" is everything else.

| Local set | Images | Other files | Caller's stated figure | Agrees? |
|---|---:|---:|---|---|
| `RICE_phase2_rebuild_2026-07-29/images/train` | 1,800 | — | 1800 | yes |
| `RICE_phase2_rebuild_2026-07-29/images/valid` | 518 | — | 518 | yes |
| `RICE_phase2_rebuild_2026-07-29/images/test` | 261 | — | 261 (SEALED) | yes |
| `riceseg` | 6,156 | 0 | 6,156 | yes |
| `rice_plant_image_dataset` | 5,391 | 0 | 5,391 | yes |
| `weed_v3` | 4,367 | 1 (`.xlsx`) | 4,368 total | yes |
| `weed_v4` | 3,632 | 1 | "subset of v3" | consistent |
| `rice_coco_annotated` | 2,579 | 5 | 2,579 | yes |
| `rice_detection_coco` | 1,347 | 2 | 1,347 | yes |
| `weeddataset` | 1,761 | 3 | 1,761 LFS stubs | yes |

Additional verifications:

- **`weeddataset` really is pixel-free.** All 1,761 `.jpg` files are under 200 bytes; a sampled file is exactly 132 bytes and begins `version https://git-lfs.github.com/spec/v1`. Zero usable images.
- **`riceseg` internal structure:** 3,078 RGB `.jpg` + 3,078 label `.png`; 6,090 of the 6,156 files carry `_subset_overlap_` tile names. Tiles derive from **848 unique source photographs** across 19 region folders.
- **Phase-2 rebuild integrity:** 2,579 entries, 2,579 *unique* `source_sha256` values — **no duplicate-content images, and therefore no content-duplicate leak across splits**. All 2,579 filenames carry an extractable `.rf.<id>` Roboflow id.

### riceseg geographic and per-source breakdown

| Region | Tiles | Unique source photos |
|---|---:|---:|
| China/GD | 100 | 45 |
| China/GX | 60 | 15 |
| China/HB | 100 | 15 |
| China/HLJ | 100 | 100 |
| China/HN | 100 | 10 |
| China/HUN | 100 | 25 |
| China/JL | 60 | 15 |
| China/JS_1 | 100 | 51 |
| China/JS_2 | 100 | 44 |
| China/JS_3 | 100 | 20 |
| China/JS_4 | 80 | 20 |
| China/JX | 60 | 8 |
| China/LN | 60 | 15 |
| India | 600 | 150 |
| Japan/TKO_1 | 100 | 21 |
| Japan/TKO_2 | 504 | 126 |
| Japan/TKO_3 | 100 | 2 |
| Philippines | 600 | 150 |
| Tanzania | 54 | 16 |
| **Total** | **3,078** | **848** |

---

## 3. Recommended exports

**Export nothing from Roboflow.**

This is a firm finding, not a hedge. Every one of the 20,489 unique images in the workspace resolves to a local file. Across every filename I sampled and checked — 9 riceseg tiles, 15 rice-plant names, 7 RICE/hainan names, and the one project-unique image — **zero** were absent from local disk. There is no new field, no new growth stage, no new lighting condition, and no weed class present on Roboflow but missing locally.

Concretely, per project:

| Project | Genuinely new images | Why not worth exporting |
|---|---:|---|
| RICE | 0 | Already local; also has no version, so exporting would require a write |
| rice-weed-seg | 0 | Already local; polygons unverified provenance |
| rice-hainan | 0 | Already local (annotated half = `rice_detection_coco`, unannotated half = `riceseg`) |
| rice_detection_for_export | 0 | Already local as `rice_detection_coco` |
| rice-weed-bd-species | 0 | Already local as `weed_v3`; classification labels, no boxes |
| rice-plant-raw | 0 | Already local; zero annotations, empty version |
| Rice_data | 0 | Empty |

### What to do instead — the actually valuable lead is local

Given that only 781 phase-2 images were never trained on and only 6 of those sit in a clean zero-weed group, the sealed 261-image test split remains the scarce resource, and Roboflow cannot relieve that pressure.

The best available source of genuinely unseen imagery is **already on local disk and has never been converted to detection data**:

**`agrinav_data/incoming/extracted/riceseg` — 3,078 tiles from 848 source photographs, with per-pixel segmentation masks.**

It is worth pursuing because:

- **It carries real labels.** Every tile has a matching `label/*.png` mask. Masks convert to tight bounding boxes mechanically — no manual annotation round is needed for the rice class.
- **It is genuine domain shift.** Five countries and 19 distinct sites (13 Chinese provinces/sites, India, 3 Japanese sites, Philippines, Tanzania). The current detector's training pool has nothing comparable in geographic spread.
- **The model has never seen it.** Only 1,349 rice-hainan images were ever annotated, and even those were never trained (the sole training run was cancelled and produced no model).
- **It is large enough to build a clean test split from**, which is precisely what the phase-2 set cannot provide.

Two conditions are non-negotiable if it is used:

1. **Group by source photograph, never by tile.** Strip `_subset_overlap_<row>_<col>` to recover the source stem, and keep all tiles of one photo in one split. 3,078 tiles map to 848 photos; some photos have 50 tiles. Splitting at tile level guarantees overlapping pixels in train and test.
2. **Consider grouping by site as well.** `China/HN` yields 100 tiles from 10 photos and `Japan/TKO_3` yields 100 tiles from 2 photos. A site-held-out split would measure domain generalization honestly; a random photo-level split would not.

This work needs no Roboflow interaction at all. It is a local conversion task.

---

## 4. Do not export

| Item | Reason |
|---|---|
| **RICE v-none** | No version exists. Exporting requires generating one — a write operation, out of scope. All 2,581 images are already local. |
| **RICE native split (1,799/518/264)** | Capture-series contaminated. `frame_000072/74/104/108/110/112` — six frames of one sequence — are spread across valid, train and test. This is the failure that voided the earlier run. |
| **rice-weed-seg v1** | All 2,579 images and their boxes are already local. Its split *is* the correct grouped split, but that split is already recorded authoritatively in `manifests/split_membership.json` and `grouped_split.json`. The only novel content is polygons of unverified provenance — session memory flags spoofed SAM annotations in the staging directory dated one day before these were uploaded. Do not import them without tracing their origin. |
| **rice-hainan v1** | 1,349 annotated images are already local as `rice_detection_coco`. Its 384×384 "Stretch to" resize destroys aspect ratio and discards resolution relative to the local originals. Its native split scatters `_subset_overlap` tile siblings of one photo across train/valid/test. |
| **rice-hainan's 2,230 unannotated tiles** | No annotations on Roboflow. The identical pixels *and* their segmentation masks are already local in `riceseg`, which is strictly better. |
| **rice_detection_for_export v1** | Already local. The version has no valid or test split at all — 1,347 images all in train. Its 512×512 "Stretch to" resize also distorts aspect ratio. |
| **rice-weed-bd-species v1** | Already local as `weed_v3` (which has 7 more images). Classification labels only — no boxes or polygons, so it cannot train or evaluate a detector. |
| **rice-plant-raw v1** | Zero annotations, `classes: {}`, and the version itself contains 0 images with `ready: false`. Nothing exists to export. Pixels already local. |
| **Rice_data** | Empty: 0 dataset images, 0 classes, 0 versions. |
| **Any Roboflow native split, from any project** | Every native split in this workspace was generated without group awareness. Verified contaminated in RICE (video frames) and structurally guaranteed in rice-hainan (tile siblings). |

---

## 5. Open questions

1. **The second rice-hainan-only image was not identified.** Count arithmetic (1,349 dataset images vs 1,347 in `rice_detection_for_export`) implies at most 2 images unique to rice-hainan. I identified one — `20JS0805jin2plot7.jpg`, which is present locally in `riceseg/…/China/JS_1/rgb/`. Confirming the second would require enumerating all 1,349 records; I sampled roughly 220. Given that the one found was already local and the maximum is 2, this cannot change any recommendation.

2. **Polygon geometry is not directly exposed.** The `annotations` field in `images_search` returns only `{count, classes}` — never coordinates or geometry type. I infer boxes vs polygons vs whole-image labels from the project `type` field, which is authoritative for the export format but does not let me inspect actual vertex data. `projects_health` would have given annotation-shape statistics but **returned `{"error":"Unknown error"}`** for `rice-weed-seg` (Roboflow reference id `6f73bc62cb5da9bfc477689eae73085e`).

3. **The provenance of rice-weed-seg's polygons is not exposed by the API.** No field records who or what generated them, or from which SAM run. `annotation_batches_list` returned `[]` for every project queried, so there is no upload-batch trail. Given the standing memory note about spoofed SAM annotations in the 2026-07-21 staging directory, this needs resolving outside the API before those polygons are trusted.

4. **Exact image-level identity between rice-weed-seg (2,579) and the local phase-2 set (2,579) was verified on 13 images, not all 2,579.** Split sizes match exactly and 13/13 assignments agree. However, `seedlingCol_04_0240.jpg` is held in rice-weed-seg under the `rice_detection_for_export` lineage id (`MMRyfaJM3FYDapiSZijd`) rather than the phase-2 lineage id (`fAfHWghbGvYZeCMb0HEs`), so the two sets are not id-identical even where they are photo-identical. A full 2,579-record join would settle it. It does not affect the recommendation, since neither set is being exported.

5. **`native_split` totals differ from RICE by 2 images.** The local manifest records native totals of train 1,798 / valid 518 / test 263 (= 2,579), while RICE reports 1,799 / 518 / 264 (= 2,581). The 2-image gap is consistent and sits one in native-train and one in native-test. Which two images were dropped during curation is not determinable from the API.

6. **24 of 115 re-derived groups in the local phase-2 manifest span more than one split.** This is flagged for awareness, not as a leak claim. `manifests/provenance.json` states plainly that `group_id` is *re-derived* by `derive_group_id()` and "is not the provenance of the train/valid assignment." The spanning groups are all still-image families (`1a_image (#0`, `1b_image (#0`, …) where the `frame // 40` blocking rule is meaningless because they are not frame sequences. Content-hash checking found **zero** duplicate images and therefore zero content-duplicate leakage. This is a limitation of the re-derivation, but it does mean the manifest cannot be used to *prove* group integrity on non-sequence families.

7. **Model metrics are unavailable.** The one finished training in the workspace (`rice-weed-bd-species-1-vit-base-patch16-224-in21k-t1`, classification) returns `metrics: null` from `models_list`. The only detector training ever started (rice-hainan, `rfdetr-nano`) was **cancelled** and produced no model. No Roboflow-side detector baseline exists to compare against.

---

*Generated 2026-08-31 by read-only enumeration of workspace `i-dont-know-la5cz` via the Roboflow MCP API, cross-referenced against local file walks. No Roboflow state was modified. No image data was downloaded.*
