# External Dataset Audit — 2026-07-21

Five archives were staged in `Downloads/` (and `MyDrive/agrinav_data/data zips/`).
Each was inspected by reading its archive index and annotation files directly — no
archive was extracted, and no claim here comes from the filename or the publishing
paper alone. Reproduce with the scripts noted per section.

**Headline: one archive is empty, one is the wrong subject entirely, two are
duplicates of each other, and the one with real polygons has a broken split.**

| Archive | Size | Images | Annotations | Weeds? | Verdict |
|---|---|---|---|---|---|
| An Image Dataset for Automated Identification and… | 5.56 GB | 4,922 | none | no | **Reject** — rice *grain*, not fields |
| In-Field Rice Panicles Detection… | 3.52 GB | 7,047 | COCO polygons, 144,837 | no | **Conditional** — rice only, split broken |
| Rice Field weed BD Dataset_V3 | 0.26 GB | 4,367 | none (folder=class) | yes, 11 species | **Accept V3 only**, not as detection truth |
| Rice Field weed BD Dataset_V4 | 0.23 GB | 3,632 | none (folder=class) | yes, 11 species | **Reject** — 99.9% subset of V3 |
| WeedDataset-main | 0.5 MB | 0 real | none | n/a | **Reject** — LFS pointers, no images |

---

## 1. "An Image Dataset for Automated Identification and…" — REJECT

5.56 GB, 4,922 JPEGs (mean 1.18 MB), **zero annotation files**. Layout is
`Data/<variety>/<New|Old>/`, 15 classes:

`Amon (Less Fiber)`, `Amon Manikganj (Fiber)`, `BRRI28`, `Banglamoti (Bashmoti-Kushtia)`,
`Biroi Mymensingh`, `Biroi Sylhet`, `Black Rice Sylhet`, `Durga28`, `Gutishorna Dinajpur`,
`Haski 29`, `Lal Gutishorna`, `Miniket`, `Najirshail`, `Paijam`, `Utshob Najir`.

These are **Bangladeshi rice grain cultivars sold as milled/husked grain**, with
`New`/`Old` denoting grain age — this is the rice *kernel variety classification*
dataset. It contains no field imagery, no plants, no weeds, and no geometry.

**Nothing in AgriNav can use it.** It is 5.56 GB — the largest archive — and the
least relevant. Do not upload it to Drive or Colab.

## 2. "In-Field Rice Panicles Detection of Different Growth Stages" — CONDITIONAL

3.52 GB, 7,047 JPEGs at 1066×800, plus COCO `train.json` / `val.json` and four
`.pth` model checkpoints. Guangdong, China; smartphone on a tripod at 1.6 m,
pointing straight down.

**Annotations are real and dense:**

| Split | Images | Annotations | With segmentation |
|---|---|---|---|
| train | 3,793 | 101,167 | 101,167 (100%) |
| val | 1,627 | 43,670 | 43,670 (100%) |

Categories are `0`, `1`, `2` (supercategory `mark`) = booting / heading / filling
growth stages. **Every object is a rice panicle. There are no weeds.**

### Defect: their published split leaks 99.4%

Images are crops of larger source photographs, with a `_N` suffix. Splitting was
done per-crop, not per-photo:

```
IMG_20210628_084358    train: _.jpg, _1.jpg, _2.jpg    val: _3.jpg, _4.jpg
IMG_20210628_084438    train: _2.jpg, _4.jpg           val: _.jpg, _1.jpg, _3.jpg
```

- train covers 1,079 distinct source photos, val covers 908
- **903 source photos appear in both — 99.4% of val**
- 4,336 of 5,420 images are crop-suffixed

This is the same capture-series contamination already quarantined in this project
for the Roboflow COCO export. The paper's reported AP (96.8 / 93.7 / 82.4%) is
measured across this leak and should not be treated as a comparable baseline.

**If used:** rebuild the split by source-photo ID (regex `^(IMG_\d{8}_\d{6})`),
never by filename. It would then add ~144k human panicle polygons and Chinese
cultivar/density/fertilizer diversity — but it duplicates the *role* of the paddy
panicle set already integrated (50,239 panicle polygons), and panicles are
`rice_protect` context, not the weed bottleneck. **Low priority.**

## 3. Rice Field weed BD V3 / V4 — ACCEPT V3, REJECT V4

Bangladesh (Narsingdi, Munshiganj), smartphone. Folder name is the only label —
**neither archive contains a single annotation file.** Published descriptions
claiming YOLO bounding boxes do **not** apply to these archives; verified by
extension census (V3: 4,367 `.jpg` + 1 `.xlsx`; V4: 3,632 `.jpg` + 1 `.xlsx`).

11 species, with real binomials — the first genuine species vocabulary this
project has:

`Alternanthera philoxeroides`, `Centella asiatica`, `Commelina benghalensis`,
`Cyperus ochraceus`, `Fimbristylis littoralis`, `Ipomoea aquatica`,
`Marsilea minuta`, `Panicum repens`, `Paspalum scrobiculatum`, `Pteris vittata`,
`Synedrella nodiflora`.

### V4 is a subset of V3 — use exactly one

Comparing `class/filename` pairs:

- V3: 4,367 images, V4: 3,632
- **3,630 identical in both — 99.9% of V4**
- only in V3: 737 · only in V4: **2**

They are the same Mendeley dataset at two versions. **Loading both would inject
3,630 duplicate images**, and a naive split would place the same photograph in
train and test. Use V3 (superset) or V4 (curated), never both. V3 is recommended
for coverage; V4 if the maintainers' filtering is trusted.

### What it is actually good for

Not detection truth — there is no geometry, and one species per folder means the
label describes the *photo*, not any object in it. It is valuable as:

1. **The independent signal the triage design was missing.** A species/rice-vs-weed
   classifier trained here is genuinely independent of SAM's self-reported score,
   which the design phase rejected as circular. This is the strongest use.
2. **Species vocabulary** for the ontology's `object_attributes.species` field,
   currently unpopulated.
3. **Hard negatives** — `Ipomoea aquatica` (water spinach) and `Marsilea minuta`
   are aquatic and will co-occur with the `non_target_aquatic` / duckweed class
   that the backbone currently scores worst on.

**Caution:** these are Bangladeshi species under smartphone capture. Treating a
classifier trained here as authoritative over Chinese/Indian/Japanese paddy imagery
is a distribution-shift assumption that must be measured, not assumed.

## 4. WeedDataset-main — REJECT (contains no images)

498 KB archive listing 1,761 `.JPG` entries across `Goosegrass` (521), `Rice` (439),
`Sedge` (405), `Barnyard_Grass` (396). Nanjing, China; CC BY-NC 4.0.

**Every image file is 132 bytes** (min = median = max) and begins with `vers`, not
the JPEG magic `\xff\xd8`:

```
version https://git-lfs.github.com/spec/v1
oid sha256:566c55a140467b3a4409996c4138bdc628e93637c8ff2c48a3189f3f007a9a2f
size 3085395
```

`.gitattributes` confirms `*.jpg filter=lfs`. These are **Git LFS pointer stubs** —
the GitHub "Download ZIP" button does not resolve LFS. The real images (~3 MB each,
~5 GB total) were never downloaded.

To obtain them: `git lfs install && git clone <repo>` — a plain zip download will
never work. **Worth re-fetching:** `Barnyard_Grass` (*Echinochloa*) is the single
most important paddy weed genus and the hardest rice look-alike at seedling stage,
and this set has a matched `Rice` class from the same fields. That is precisely the
rice-vs-grass-weed discrimination the safety case depends on.

---

## Recommended actions

1. **Delete** the 5.56 GB grain archive from Drive — it is pure storage cost.
2. **Delete V4**, keep V3. Record the 99.9% overlap so a future contributor does
   not re-add it.
3. **Re-clone WeedDataset via `git lfs`** — highest value per GB of anything here.
4. **Defer the panicle set.** If used, rebuild the split by source photo first.
5. **Do not** merge any of this into `detector_split_v1.json`. None of it carries
   instance geometry, so none of it can become detection truth without annotation.
6. Register whichever archives are kept in `data/dataset_registry.v1.json` with
   SHA-256 (not yet computed — the two large archives take a few minutes to hash).

## Reproducing

Inspection scripts are in the session scratchpad and are not committed
(one-shot forensics, not pipeline code):
`audit_zips.py` (index + type census), `peek_json.py` / `deep2.py` (COCO schema,
class folders, V3/V4 overlap), `lfscheck.py` (LFS pointer detection).
