# AgriNav Detector Dataset — Card v1

**Date:** 2026-07-20
**Purpose:** the data foundation for the **detector training that follows RiceSEG
pretraining**, built to fix the two problems the audit flagged: weed scarcity and
a capture-series-contaminated split.
**Governing plans:** deployment roadmap Gate 2 (grouped provenance split) and
Gate 3 (annotation ontology / proposals-not-truth); `data/ontology.v1.json`.

> **Integrity rule enforced here:** only *human* annotations are training truth.
> RiceSEG and paddy ship human masks → converted to polygons (valid). COCO ships
> only boxes with a contaminated split → emitted as **unreviewed proposals**, never
> mixed into the training split until reviewed.

---

## 1. Sources and what was done

| Source | Native annotation | Action | Role |
|---|---|---|---|
| **RiceSEG.zip** (3,078 tiles, 5 countries, 773 source photos) | human **semantic masks**, 6 classes | mask → instance **polygons** (geometry transform of human labels) | **training truth**; the weed supply |
| **paddy-rice-imagery-DatasetNinja.tar** (403 UAV frames, 8 sessions) | human **panicle polygons** | polygon → COCO `rice_protect` | **aux aerial** rice context (separate file; panicle-only, no weeds) |
| **rice_detection…coco.zip** (1,347 imgs) | human **boxes** (rice/weed), contaminated split | boxes → **unreviewed proposals** | pending SAM box→mask + human review; **not** training truth |

## 2. Ontology mapping (`data/ontology.v1.json`)

| RiceSEG semantic value | Canonical label (id) |
|---|---|
| 1 green_rice, 2 senescent_rice, 3 rice_panicle | `rice_protect` (1) |
| 4 weed | `weed_target` (2) |
| 5 duckweed | `non_target_aquatic` (4) |
| 0 background | ignored |

Paddy `panicle` → `rice_protect` (1), tagged `panicle_only_partial_rice_coverage`.
COCO `rice`→`rice_protect` (1), `weed`→`weed_target` (2), all `unreviewed_proposal`.

## 3. Outputs (regenerable; under gitignored `artifacts/detector_v1/`)

| File | Content |
|---|---|
| `riceseg_instances.coco.json` | 3,072 imgs, **31,224 polygons**, **1,295 weed imgs / 8,200 weed polygons**, 772 groups |
| `paddy_panicle.coco.json` | 400 imgs, 50,239 panicle polygons, 8 sessions (aerial aux) |
| `coco_proposals_unreviewed.coco.json` | 1,347 imgs, 39,556 box proposals (491 weed) — **unreviewed** |
| `split_v1/{train,val,test}.coco.json` | grouped split (RiceSEG) |
| `split_v1/split-v1.json` | immutable group→split manifest (also tracked at `data/manifests/detector_split_v1.json`) |

**The weed fix, quantified:** RiceSEG yields **8,200 real weed polygons vs. 491 COCO
weed boxes — a 16× increase**, all from human masks.

## 4. Split policy and stats (roadmap Gate 2)

- Assignment at the **group** level (`group_id` = RiceSEG source photo or paddy
  session), **never** the tile/frame level.
- **Zero group overlap** — independently verified: 0 source photos cross splits.
- Stratified by `source_dataset | country | has_weed`; test marked **sealed**.
- Ratios 70/15/15 (seed 20260720).

**RiceSEG-only split (`data/manifests/detector_split_v1.json`):**

| Split | Images | Groups | Weed-positive imgs | Weed polygons |
|---|---:|---:|---:|---:|
| train | 2,135 | 540 | 890 | 5,511 |
| val | 510 | 114 | 199 | 1,543 |
| test (sealed) | 427 | 118 | 194 | 1,146 |

## 5. Known limitations

- **External contours only** — interior holes in RiceSEG masks are not RLE-encoded; components < 16 px are dropped.
- **Paddy is panicle-only** partial rice coverage and aerial; keep it auxiliary, do not naively merge with full-plant RiceSEG rice.
- **COCO proposals are not truth** and the original COCO split is discarded (contaminated). Do not train on the COCO file until reviewed.
- **Aerial + weed barely coexists** in available data; the weed supply is RiceSEG (mostly ground/canopy). There is no public aerial-weed source to add.

## 6. Regenerate

```bash
python scripts/riceseg_masks_to_coco.py --riceseg-zip <RiceSEG.zip> \
    --out-json artifacts/detector_v1/riceseg_instances.coco.json
python scripts/paddy_supervisely_to_coco.py --tar <paddy...tar> \
    --out-json artifacts/detector_v1/paddy_panicle.coco.json
python scripts/coco_boxes_to_proposals.py --coco-zip <rice_detection...coco.zip> \
    --out-json artifacts/detector_v1/coco_proposals_unreviewed.coco.json
python scripts/build_detector_split.py \
    --coco artifacts/detector_v1/riceseg_instances.coco.json \
    --out-dir artifacts/detector_v1/split_v1
```

## 7. What you do next (before detector training)

1. **(Optional) SAM box→mask on COCO proposals** — GPU step using
   `facebook/sam2.1-hiera-large` (see `configs/annotation/pilot_v1.json`), then
   human-review in CVAT. Only reviewed masks may join a `split_v2`.
2. **Train the detector** on `split_v1/train.coco.json`, select on `val`, and keep
   `test.coco.json` **sealed** until the final locked evaluation (roadmap Gate 7).
3. The detector correctness fixes in `models/weeddet_v6b.py` (translation aug,
   ATSS, VFL, evaluator) are still open — see the audit; they gate a defensible
   result, independent of this data.
