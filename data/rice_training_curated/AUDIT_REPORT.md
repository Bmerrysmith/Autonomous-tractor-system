# RiceSEG Curation Audit — Training Images for Bounding-Box Annotation

**Goal:** Curate high-quality rice-plant images (low occlusion, aerial + front angles, minimal water reflection) for object-detection training.

## Source
`RiceSEG.zip` → *global rice segmentation* dataset. **3,078 RGB tiles** (512×512), each with a matching semantic label mask, across 5 countries / 19 sites: China (14 provinces), India, Japan, Philippines, Tanzania.

## Method (automated + visual QA)
Every one of the 3,078 images was scored on four measurable traits, then top candidates were visually verified:

- **Sharpness** — variance of the Laplacian (rejects blur). Aerial imagery is inherently softer, so sharpness floors were angle-specific.
- **Rice coverage** — fraction of pixels labeled rice (class 1, confirmed by mask overlay). Kept to a **0.15–0.80** window: enough plant to box, but not a solid wall of leaves (a wall = high self-occlusion, no boundable individuals).
- **Specular glare** — bright, low-saturation pixels (sun glint on water).
- **Water reflection** — bright grayish non-vegetation pixels (sky sheen on paddy water). This metric cleanly flagged the reflection-heavy sites.

**Filters applied:** front angle needed Laplacian ≥ 300, reflection ≤ 0.05, glare ≤ 0.02; aerial needed Laplacian ≥ 45, reflection ≤ 0.06, glare ≤ 0.015. Both required rice coverage 0.15–0.80 and vegetation < 0.92 (occlusion cap). Survivors were ranked by a composite quality score and capped per site for diversity.

## Result: 245 images
Organized into `front/` (150) and `aerial/` (95). Filenames are prefixed with the source site (e.g. `India__IMG_0561_...jpg`) and every image is logged in `manifest.csv` with its scores.

| Angle | Site | Count | Character |
|---|---|---|---|
| aerial | China/HLJ | 34 | Top-down seedlings, separated, matte water |
| aerial | Philippines | 35 | Overhead young plants, low glare |
| aerial | China/HN | 13 | Individual plants over dark water |
| aerial | China/GX | 8 | Well-spaced seedlings, near-zero reflection |
| aerial | China/JX | 5 | Top-down seedlings |
| front | India | 40 | Panicles over dry soil, no water — cleanest set |
| front | Japan/TKO_2 | 35 | Oblique canopy |
| front | Tanzania | 25 | Sharp mature panicles |
| front | China/HUN | 15 | Mature panicles |
| front | China/GD | 10 | Mature panicles |
| front | China/JS_4 | 10 | Individual plants, minimal reflection |
| front | China/JL | 6 | Panicles, dense green |
| front | China/JS_1, JS_2, HB, LN | 9 | Assorted clean tiles |

## What was excluded and why
- **China/HN (most), JS_1, JS_2, JS_4 (most)** — heavy water-reflection sites (reflection scores 0.19–0.40). Only the small clean subset survived.
- **Japan/TKO_1, TKO_3** — too blurry (Laplacian ~40–52) or a solid wall of leaves (rice coverage 0.85).
- **China/LN, HUN, HB (most)** — extreme close-up walls of leaves = high self-occlusion.
- **Near-empty tiles** — mostly bare soil or open water with negligible rice.

## Notes for annotation
The 512×512 tiles are overlapping crops of larger field photos, so some plants are cut at tile edges — normal for detection training. The `aerial/` set gives cleanly separated individuals (easy tight boxes); the `front/` set gives panicle- and canopy-level views (box the whole plant/tiller). Matching segmentation masks were **not** copied since you're drawing boxes fresh, but they remain in the extracted dataset if you ever want them as a reference.
