# SAM 2.1 polygon re-seed — re-pilot after fixing the candidate set

**Date:** 2026-09-01
**Supersedes the recommendation in:** [`sam_reseed_pilot_2026-08-31.md`](sam_reseed_pilot_2026-08-31.md)
(that report's measurements stand; its "do not run" conclusion applied to the
old candidate set and has now been acted on)
**Status:** re-pilot only. **The full 2,318-image run was not started.**

Everything here is an **unreviewed proposal**. No annotation record, no
`review_status: accepted`, no `verified_empty`, no `treatment_eligible`, no
`annotator_id`/`reviewer_id` was written. The 261-image test split was not read.

---

## 1. What changed

### 1.1 The missing candidate (`sam_box_to_mask.py`)

The stage emitted 8 candidates per box — 3 from one `multimask_output=True`
pass on the exact human box, 5 from `multimask_output=False` passes on jittered
boxes — and therefore could not express `multimask_output=False` on the
*unjittered* box, which is the configuration SAM is trained for on an
unambiguous prompt. A ninth candidate now supplies it.

- **Appended, not inserted.** `candidate_index` is positional, so the new
  candidate is last (index 8). Every pre-existing index is untouched. Verified
  empirically in §3.1, not just by inspection.
- **Own `kind`:** `"single_mask"`, alongside `"multimask"` and `"jitter"`, with
  `multimask_index: None` and `jitter_index: None`. The three kinds are now a
  named closed vocabulary `CANDIDATE_KINDS`.
- **Counts are derived, never literals:**
  `CANDIDATES_PER_BOX = MULTIMASK_CANDIDATES + JITTER_CANDIDATES + SINGLE_MASK_CANDIDATES`
  = 9 and `DECODER_PASSES_PER_BOX = 1 + JITTER_CANDIDATES + SINGLE_MASK_CANDIDATES`
  = 7. `run_thresholds()` records both, plus `single_mask_candidates` and
  `candidate_kinds`, so provenance claiming N passes while the loop emits M is
  structurally impossible.
- **The docstring's COST MODEL was itself wrong before this.** It said "8
  decoder-only passes" for what were **6** passes producing 8 candidates — one
  `multimask_output=True` pass returns three masks, so passes and candidates are
  different numbers. The pilot's measured `decoder_calls_per_box: 6.0` confirmed
  the old text was inaccurate. Both counts are now named constants and the
  discrepancy is called out in the docstring.
- Existing `"multimask"`/`"jitter"` string literals replaced by the constants.

Invariants preserved and re-tested: label still copied from the human box
through the closed dict with a counted drop and no `else`/`.get(default)`; no
text prompts; raw candidates only, no annotation records; nothing writable as
truth; a degenerate box still emits zero candidates and a reason, and now also
makes **zero** decoder passes.

### 1.2 The pin defeat (`default_predictor_factory`)

Was:

```python
from sam2.sam2_image_predictor import SAM2ImagePredictor
return SAM2ImagePredictor.from_pretrained(model_id, revision=revision)
```

Upstream, `from_pretrained` forwards `**kwargs` to `build_sam2_hf` →
`build_sam2` → `SAM2ImagePredictor.__init__`; none passes `revision` to
`hf_hub_download`, each absorbs it. The download resolves the repo's current
`main` while `validate_model_revision` has already written the requested sha
into provenance — an unpinned run whose provenance validates clean and is
unrecoverable. As the coordinator noted, the bug was **latent, not active**:
`sam2` is not installed here, so that path died at import.

It now routes through `agrinav.data.sam2_predictor.build_predictor`, which uses
`transformers` (which honours `revision` on `from_pretrained`) and re-asserts
the resolved sha against the HuggingFace API before any GPU allocation. The
revision is validated at *factory-construction* time, so a placeholder fails
before anything is loaded. The import is function-local because
`sam2_predictor` imports from `sam_box_to_mask`.

`--device` and `--dtype` were added to the CLI rather than hard-coded
(CLAUDE.md §3.3 lists both as things not to hard-code; `dtype` also changes
mask geometry).

### 1.3 Incidental

`process()`'s `stats` dict is now annotated `dict[str, Any]`. mypy reported 6
errors on this file; **all 6 are present at HEAD** (verified against
`git show HEAD:...`) and none were introduced here. Zero runtime effect.

---

## 2. Tests

`tests/test_sam_box_to_mask.py` is new — **the file the module docstring has
always advertised** (`--predictor-factory tests.test_sam_box_to_mask:make_stub_predictor`)
but which did not exist, so the documented CPU path could not be run and the
stage's own safety invariants had no test at all. It supplies
`make_stub_predictor` and 17 tests. All CPU, no GPU, no network.

Candidate set:

- the single-mask candidate is present exactly once, and its count equals `SINGLE_MASK_CANDIDATES`
- it uses the unjittered clipped box — same prompt as the multimask group, and provably not any jittered box
- **it was produced with `multimask_output=False`**, asserted on the recorded
  *call sequence* (exactly one `(exact_box, False)` pass, and it is the last),
  which is what distinguishes a real extra pass from a relabelled candidate
- it carries null `multimask_index`/`jitter_index` and a well-formed RLE
- the original 8 are unchanged: indices `[0,1,2]` / `[3..7]`, jitter prompt
  boxes equal `jitter_box()` recomputed independently, single-mask at index 8
- `run_thresholds()` reports the count actually emitted (asserted against the
  emitted list and the recorded call count, not against a literal)
- a degenerate box makes zero passes and emits zero candidates

Pin enforcement:

- placeholder (`PIN_BEFORE_RUN`) and branch names refused at factory construction
- the factory routes to the pinning backend and **`sam2` is not in `sys.modules` afterwards**
- `build_predictor` forwards `revision` to *both* `Sam2Model.from_pretrained`
  and `Sam2Processor.from_pretrained` (mocked; no network, no GPU)
- `resolve_pinned_revision` queries the hub with the requested sha and refuses
  when the hub resolves something else

> Note for whoever maintains these: the obvious way to write the forwarding
> test — `monkeypatch.setattr(transformers, "Sam2Model", Fake)` — **passes
> vacuously**. `transformers` is a `_LazyModule` and that patch does not survive
> a later `from transformers import Sam2Model`; the first version of this test
> "passed" while loading real weights. It patches `from_pretrained` on the real
> classes instead. Verified empirically, and documented in the test docstring.

---

## 3. Re-pilot: same 40 images

Input rebuilt from the same `--sample-seed 20260831 --sample-size 40`:
identical image sha256 set (40) and identical box list (1,280) — confirmed, not
assumed. 29 train / 11 valid, 1,075 `rice_protect` / 205 `weed_target`.

```bash
.venv/Scripts/python.exe -m agrinav.data.build_sam_reseed_input \
  --rebuild-root "C:/Users/Benny Merr/Downloads/RICE_phase2_rebuild_2026-07-29" \
  --splits train,valid --sample-size 40 --sample-seed 20260831 \
  --out-zip artifacts/sam_reseed_2026-09-01/pilot_images.zip \
  --out-json artifacts/sam_reseed_2026-09-01/pilot_proposals_unreviewed.coco.json \
  --out-manifest artifacts/sam_reseed_2026-09-01/pilot_input_manifest.json

.venv/Scripts/python.exe -m agrinav.data.sam_reseed_pilot \
  --coco-zip artifacts/sam_reseed_2026-09-01/pilot_images.zip \
  --proposals-json artifacts/sam_reseed_2026-09-01/pilot_proposals_unreviewed.coco.json \
  --model-revision 665f8e2ad61cf5f53d65644ff27c8ee525124610 \
  --out-shard artifacts/sam_reseed_2026-09-01/sam_raw/pilot_shard_000.jsonl \
  --out-manifest artifacts/sam_reseed_2026-09-01/pilot_run_manifest.json

# once per selection rule
.venv/Scripts/python.exe -m agrinav.data.compare_sam_polygons \
  --raw-shard artifacts/sam_reseed_2026-09-01/sam_raw/pilot_shard_000.jsonl \
  --proposals-json artifacts/sam_reseed_2026-09-01/pilot_proposals_unreviewed.coco.json \
  --legacy-annotations-dir ".../agrinav_intake_2026-07-21/deliverable/detection/RICE/annotations" \
  --legacy-splits train,valid,test \
  --selection-rule {multimask_argmax_pred_iou|single_mask_exact_box} \
  --out-json reports/metrics/sam_reseed_iou_2026-09-01_<rule>.json
```

Stage counters:

```
images_processed: 40      boxes_emitted: 1280     candidates_emitted: 11520
boxes_degenerate: 0       unmapped_dropped: 0     labels: {rice_protect:1075, weed_target:205}
decoder_calls_per_box: 7.0
```

11,520 = 1,280 × 9 exactly. 7.0 passes/box exactly. Zero drift in labels or counts.

### 3.1 Regression: the original 8 candidates are bit-identical

Compared candidate-for-candidate against the 2026-08-31 shard on all 1,280
objects, on `(candidate_index, kind, multimask_index, jitter_index, prompt_box,
area_px, rle.counts, sam_pred_iou)`:

```
original-8 candidates identical : 10240
original-8 candidates differing : 0
rows without exactly +1 candidate: 0
kind of the appended candidate  : {'single_mask'}
appended candidate_index values : {8}
```

Nothing was renumbered or perturbed, so the downstream jitter-agreement
statistic consumes exactly the set it consumed before. This also demonstrates
run-to-run determinism of the inference path.

### 3.2 Corrected ETA and peak GPU

| | 8-candidate (2026-08-31) | **9-candidate (2026-09-01)** |
| --- | ---: | ---: |
| Wall, 40 images | 69.20 s | **74.92 s** |
| Steady state (less model load) | 68.40 s | **74.08 s** |
| Images/sec | 0.585 | **0.540** |
| Boxes/sec | 18.71 | **17.28** |
| Encode s/image | 0.10664 | 0.11237 |
| Decode s/box | 0.03542 | 0.03851 |
| Non-inference s/box | 0.01469 | 0.01585 |
| **Peak GPU reserved** | 0.910 GiB | **0.910 GiB (unchanged)** |
| Headroom vs 11.994 GiB visible | 11.08 GiB | **11.08 GiB** |

```
2,318 × 0.11237  +  74,917 × (0.03851 + 0.01585)  =  260 s + 4,072 s = 4,333 s
```

**Corrected ETA: 72.2 minutes** (box-rate cross-check 72.3 min; pixel-mix
corrected 71.5 min, since the pilot sample is 3.0% heavier in megapixels than
the corpus). **Practical range 68–75 minutes.** The added candidate costs
**+8.3%** wall time.

**Peak GPU is unchanged at 0.910 GiB — 7.6% of the card, 11.08 GiB headroom**
(~9.9 GiB with the desktop resident). It does not move because the encoder
input is a fixed 1024×1024 and this backend upsamples masks on CPU; the extra
candidate is one more batch-1 decoder pass.

---

## 4. The three-column comparison

Same 37 images (3 EXIF-reoriented excluded), same **1,238 paired objects** at box
IoU ≥ 0.99, same legacy set. The `new:MM` column reproduces 2026-08-31 exactly
(0.711 / 0.759 / 0.815), which confirms the comparison itself is stable and the
only thing that changed is which candidate is scored.

- **legacy** — existing polygons at `agrinav_intake_2026-07-21/.../RICE/annotations/`
- **new:MM** — new shard, `argmax(sam_pred_iou)` over multimask (the only rule
  expressible before this change)
- **new:SM** — new shard, the added `single_mask` candidate

There is still **no mask ground truth**; all three columns are proxies against
the human box, which is the only reviewed geometry in the pipeline.

### Overall (n = 1,238)

| metric (median unless noted) | legacy | new:MM | **new:SM** |
|---|---:|---:|---:|
| box IoU vs human box | 0.759 | 0.711 | **0.774** |
| containment (mask inside box) | **0.985** | 0.939 | 0.978 |
| fill (mask area / box area) | 0.517 | 0.605 | 0.561 |
| outside-box area fraction | **0.0153** | 0.0611 | 0.0225 |
| multi-component rate (≥2) | **2.0%** | 22.5% | 13.6% |
| 5+ component rate | **0.0%** | 5.0% | 2.9% |
| agreement IoU with legacy | — | 0.815 | 0.858 |
| fraction where new beats legacy | — | 36.3% | **56.9%** |
| fraction where legacy beats new | — | 62.9% | 41.4% |
| median paired delta (new − legacy) | — | **−0.0268** | **+0.0128** |

### rice_protect (n = 1,043)

| metric | legacy | new:MM | **new:SM** |
|---|---:|---:|---:|
| box IoU | 0.768 | 0.713 | **0.778** |
| containment | **0.984** | 0.932 | 0.977 |
| fill | 0.544 | 0.644 | 0.584 |
| outside-box fraction | **0.0163** | 0.0682 | 0.0233 |
| multi-component rate | **1.6%** | 19.9% | 10.6% |
| new beats legacy | — | 33.8% | **55.6%** |
| median paired delta | — | −0.0342 | **+0.0114** |

### weed_target (n = 195) — the safety-critical class

| metric | legacy | new:MM | **new:SM** |
|---|---:|---:|---:|
| box IoU | 0.699 | 0.704 | **0.741** |
| containment | **0.992** | 0.969 | 0.983 |
| fill | 0.393 | 0.415 | 0.420 |
| outside-box fraction | **0.0084** | 0.0306 | 0.0175 |
| multi-component rate | **4.1%** | 36.9% | 29.2% |
| new beats legacy | — | 49.2% | **63.6%** |
| median paired delta | — | −0.0009 | **+0.0187** |

### Small objects, box < 32×32 px (n = 509)

| metric | legacy | new:MM | **new:SM** |
|---|---:|---:|---:|
| box IoU | 0.736 | 0.690 | **0.766** |
| containment | **0.984** | 0.919 | 0.978 |
| outside-box fraction | **0.0157** | 0.0809 | 0.0216 |
| multi-component rate | **0.2%** | 10.8% | 4.1% |
| new beats legacy | — | 37.5% | **60.1%** |
| median paired delta | — | −0.0311 | **+0.0235** |

### What the table says

**The sign flipped.** With the added candidate the re-seed goes from losing
(−0.027 median, worse on 62.9% of objects) to winning (+0.013 median, better on
56.9%). The gain is largest exactly where it matters most and where the old
configuration was weakest: `weed_target` +0.019 (63.6% of objects) and small
objects +0.024 (60.1%).

**But legacy still wins on shape cleanliness, on every slice.** Containment
0.985 vs 0.978, outside-box area 1.5% vs 2.3%, and above all **multi-component
rate 2.0% vs 13.6%** — 29.2% vs 4.1% on weeds. Mask pixels outside the human box
are geometry no human asserted, and a fragmented mask is a redraw for a
reviewer. The added candidate roughly halves both defects relative to `new:MM`
(22.5% → 13.6% fragmentation, 6.1% → 2.3% leakage) without closing the gap.

So the honest summary is: **better localisation, still messier shapes.** Whether
that is a net gain depends on what the polygons are for. For a reviewer in CVAT
correcting geometry, a fragmented mask probably costs more than a slightly
loose one — and the pilot cannot measure that, because the pilot's own primary
metric (`configs/annotation/pilot_v1.json`) is
`median_human_seconds_per_accepted_image`, which needs a human.

---

## 5. Effect on `optimize_proposals` (noted, not acted on)

Out of scope as instructed; confirmed still absent (referenced from
`triage_proposals.py` ×2, `compare_sam_polygons.py`, and recorded in
`sam_box_to_mask.py` as `candidate_selection: "deferred_to_optimize_proposals"`).

**These changes do alter what that stage would need to do, and reduce its
difficulty:**

1. It must handle **9 candidates, not 8**, and must not assume `candidate_index`
   ∈ [0,8) — read `thresholds.candidates_per_box`, which shards now carry.
2. Its selection rule now has an obvious strong default. On this sample the
   `single_mask` candidate beats `argmax(sam_pred_iou)` over the multimask heads
   on 68.4% of objects; a rule of "prefer `single_mask`" is defensible from
   measurement rather than needing a fitted selector.
3. `SELECTION_RULES` in `compare_sam_polygons.py` is a closed dict with exactly
   the two rules measured here, and is a reasonable starting vocabulary — but it
   is a *comparison* device and asserts no decision. Selection remains that
   stage's job.
4. Old shards (8 candidates) and new shards (9) will coexist.
   `select_single_mask` returns `None` for an old shard and the caller counts it
   as `objects_skipped_no_candidate_for_rule` rather than silently substituting
   another candidate; `optimize_proposals` needs the same discipline.

Nothing here removes the blocker: the raw candidate sidecar still cannot reach
CVAT, which imports COCO 1.0 polygon JSON. That needs `optimize_proposals` plus
an RLE → polygon → COCO writer, neither of which exists, and neither of which
needs a GPU.

---

## 6. Recommendation

**The full run is now defensible on quality, where before it was not. It remains
the user's call, and I have not started it.**

If the goal is **better polygons than the ones on disk**: yes, but modestly, and
with a caveat — +0.013 median box IoU overall (+0.019 on weeds) bought at the
price of a 6.8× higher fragmentation rate. I would not call that a clear win on
its own.

If the goal is **trustworthy provenance**: this is the stronger case and it is
now unblocked. The existing set records `facebook/sam2-hiera-large (upstream
agent pipeline)` with **no revision at all** and `verified_by_claude_against_gt`
but "NOT visually reviewed". The re-seed produces SAM 2.1 at a verified,
enforced commit, with the pin now actually reaching the download call. Combined
with a small quality gain, ~72 minutes is a reasonable price.

**Before committing the GPU hours, the cheapest next step is not a GPU step.**
Two things would settle the remaining doubt for far less:

1. **A human spot-check of 20 objects** across the three columns. The whole
   comparison rests on a proxy; 20 minutes of a person's attention would say
   whether "better box IoU, more fragments" is actually better to review. This
   is also the only way to touch the pilot's declared primary metric.
2. **Build `optimize_proposals`.** Without it neither set reaches CVAT, so a
   full re-seed produces an artifact that cannot be reviewed regardless of its
   quality. It is on the critical path and needs no GPU.

If the user wants to proceed anyway, the run is ready: change `--sample-size 40`
to `--sample-size 0`. The stage is resumable and shardable, and peak memory
leaves room for 2–4 concurrent shards (~18–36 min wall).

---

## 7. Validation

| Check | Result |
| --- | --- |
| `pytest tests/test_sam_box_to_mask.py tests/test_sam_reseed_input.py` | **43 passed** (17 new + 26) |
| `black --check` (8 files) | pass |
| `ruff check` (8 files) | pass |
| `mypy` (6 modules incl. the modified stage) | **Success: no issues found** |
| mypy errors at HEAD on the stage, for comparison | 6 (all pre-existing, now fixed) |
| Original 8 candidates vs previous shard | **10,240 / 10,240 bit-identical** |
| Candidate count vs `run_thresholds()` | 11,520 = 1,280 × 9, `candidates_per_box: 9` |
| Decoder passes | 7.0/box = `decoder_passes_per_box: 7` |
| `new:MM` column vs 2026-08-31 | reproduces exactly |
| Re-pilot input vs original input | identical 40 shas, identical 1,280 boxes |
| Test-split leakage in any artifact | none |
| Truth markers in any artifact | none |
| Line endings on all touched files | LF, matching the repo |

**Not run:** the full repository test suite (~23 GB temp burn). `sam_box_to_mask.py`
is imported by `agrinav.cli` (`data-sam-mask`) and by three of my modules; all
19 `agrinav.data` modules still import cleanly. No other existing file was
modified.

---

## 8. Risks

- **The proxy is still a proxy.** Fragmentation and containment now point the
  opposite way from box IoU, and no metric here resolves that trade-off. Only a
  human can.
- **`new:SM` fragmentation is 13.6% overall and 29.2% on weeds.** That is the
  clearest remaining quality defect in the new set and it is *not* fixed by this
  change, only halved. A morphological cleanup or largest-component rule would
  be a *selection/post-processing* decision belonging to `optimize_proposals`,
  not to this stage — the stage must keep emitting raw candidates.
- **n = 40 images / 1,238 objects, one seed, 29 of 103 capture groups**;
  `weed_target` n = 195.
- **`dtype` (bfloat16) still is not in the stage's provenance.** It affects mask
  geometry and appears only in the pilot run manifest. `run_thresholds()` would
  be the right home; I did not add it because it is a separate change to a
  safety-critical file and was not in scope.
- **The `sam2`-package path is gone rather than fixed.** If someone genuinely
  needs the upstream predictor, they must pass `--predictor-factory`, and that
  path would have the original pin defect. A guard there would be worth adding.
- **`default_predictor_factory` is now GPU-default** (`--device cuda`) and
  refuses to fall back to CPU silently. On a CPU-only machine the documented CLI
  requires `--device cpu` explicitly. That is deliberate, but it is a
  behavioural change to a documented entry point.

---

## 9. Artifacts

| Path | Contents |
| --- | --- |
| `artifacts/sam_reseed_2026-09-01/pilot_images.zip` | same 40 verified images (train+valid only) |
| `artifacts/sam_reseed_2026-09-01/pilot_proposals_unreviewed.coco.json` | staged human boxes |
| `artifacts/sam_reseed_2026-09-01/pilot_input_manifest.json` | input provenance and hashes |
| `artifacts/sam_reseed_2026-09-01/sam_raw/pilot_shard_000.jsonl` | **9-candidate raw sidecar — unreviewed proposals** |
| `artifacts/sam_reseed_2026-09-01/pilot_run_manifest.json` | pin, environment, timings, GPU memory |
| `artifacts/sam_reseed_2026-09-01/iou_objects_*.jsonl` | per-object rows, one file per rule |
| `reports/metrics/sam_reseed_iou_2026-09-01_multimask_argmax_pred_iou.json` | new:MM column |
| `reports/metrics/sam_reseed_iou_2026-09-01_single_mask_exact_box.json` | new:SM column |

Code modified: `src/agrinav/data/sam_box_to_mask.py` (candidate set, constants,
`run_thresholds`, `default_predictor_factory`, CLI `--device`/`--dtype`,
docstring), `src/agrinav/data/compare_sam_polygons.py` (selection rules,
components, outside-box fraction).
Code added: `tests/test_sam_box_to_mask.py`.

Nothing committed or pushed.

---

# Addendum, 2026-09-01: the human review sheet

The trade-off in §4 (better localisation, messier shapes) is a judgement about
reviewer effort, so it needs a reviewer. This addendum records the instrument
built for that, one measurement that shrinks the question before anyone spends
attention on it, and a bug found while building it.

## A.1 A measurement that removes most of the fragmentation worry

Before asking a human, it is worth asking what the extra components actually
*are*. Measured over all 1,280 `single_mask` masks in the re-pilot shard:

| quantity | value |
| --- | ---: |
| masks with >= 2 connected components | 207 of 1,280 (16.2%) |
| extra (non-largest) components, total | 945 |
| ...that are < 1% of their mask area | **850 (89.9%)** |
| ...that are >= 1% of their mask area | 95 (10.1%) |
| mask area outside the largest component, median | **0.57%** |
| ...mean | 5.09% |
| ...p90 / p99 / max | 22.1% / 43.9% / 72.0% |

**Nine out of ten extra components are specks.** For the median fragmented mask,
a largest-component-only rule would discard 0.57% of its area — that is a
post-processing decision costing a reviewer nothing, not a correction burden.

But the tail is real: p90 is 22% and the maximum is 72%, so a blanket
largest-component rule would delete substantial structure from a minority of
masks. Only **27 of 1,238** paired objects have >= 5% of mask area outside the
largest component.

This reframes the question usefully. It is no longer "is 13.6% fragmentation
worth 0.013 box IoU"; it is **"for the ~2% of objects with substantial extra
structure, is that structure real?"** If it is real (a second leaf blade of the
same weed), the new masks are better and fragmentation was never the problem. If
it is noise, a largest-component rule fixes the other 98% for free.

That is a much smaller question, and it is what the sheet is now weighted to
answer: all 27 substantial-fragment objects are included, none sampled away.

## A.2 The sheet

```text
reports/figures/sam_reseed_review_2026-09-01.html          (11.5 MB, standalone)
reports/figures/sam_reseed_review_2026-09-01.BLINDING_KEY.json
reports/figures/sam_reseed_review_2026-09-01_panels/       (178 PNGs)
```

Open the HTML in any browser. No server, no network, no missing images: every
crop is inlined as a base64 data URI, and the same PNGs are on disk for reuse.

Built by `agrinav.data.build_review_sheet`:

```bash
.venv/Scripts/python.exe -m agrinav.data.build_review_sheet \
  --objects-jsonl artifacts/sam_reseed_2026-09-01/iou_objects_single_mask_exact_box.jsonl \
  --raw-shard artifacts/sam_reseed_2026-09-01/sam_raw/pilot_shard_000.jsonl \
  --coco-zip artifacts/sam_reseed_2026-09-01/pilot_images.zip \
  --proposals-json artifacts/sam_reseed_2026-09-01/pilot_proposals_unreviewed.coco.json \
  --legacy-annotations-dir ".../agrinav_intake_2026-07-21/deliverable/detection/RICE/annotations" \
  --legacy-splits train,valid,test \
  --out-html reports/figures/sam_reseed_review_2026-09-01.html \
  --out-key reports/figures/sam_reseed_review_2026-09-01.BLINDING_KEY.json \
  --out-png-dir reports/figures/sam_reseed_review_2026-09-01_panels \
  --sample-seed 20260901 --blind-seed 20260901
```

Each card shows the same crop and the same human box twice, with only the
polygon differing and identical styling on both sides. Per panel: box IoU,
component count, and share of area outside the largest component. Per object:
class, box pixel size, and the IoU between the two polygons. The reviewer picks
*A easier to correct*, *B easier*, *about the same*, or *both need a redraw*, and
a button collects the answers as JSON for copy-paste (no network).

**Blinding.** Left/right is randomised per object from `blind_seed 20260901`;
48 objects have legacy on the left, 41 on the right. The key is in the separate
`.BLINDING_KEY.json`, which the sheet names but does not embed. Verified: the
strings `new_sm` and `legacy` appear nowhere inside any card.

The stratum name is deliberately **not** shown per card. Names like
`weed_legacy_wins` state which set wins, and next to the per-panel box IoU that
hands over the answer; exported verdicts therefore carry only the object number
and join to the stratum through the key after unblinding. The full stratum table
*is* on the sheet, above the cards, as disclosure.

Disclosed residual leak: component counts are printed because the reviewer needs
them to judge effort, and a panel with 7 components is weak evidence of being
the new set. The sheet says so.

**Strata** (first match wins, so disjoint; drawn from the 1,238 paired objects):

| stratum | pool | drawn |
| --- | ---: | ---: |
| `weed_fragment_substantial` | 17 | **17 (all)** |
| `weed_new_wins` | 79 | 8 |
| `weed_legacy_wins` | 44 | 8 |
| `weed_agree` | 55 | 5 |
| `rice_fragment_substantial` | 10 | **10 (all)** |
| `rice_fragment_specks` | 101 | 8 |
| `small_new_wins` | 237 | 8 |
| `small_legacy_wins` | 123 | 8 |
| `large_new_wins` | 178 | 6 |
| `large_legacy_wins` | 126 | 6 |
| `rice_agree` | 268 | 5 |
| **total** | **1,238** | **89** |

Composition of the drawn set: 38 `weed_target` (42.7% of the sheet against 15.8%
of the pool — deliberately over-represented), 27 objects under 32x32 px, 44 where
at least one panel is multi-component. Both substantial-fragment strata are taken
whole because they are the question. Selection inside a stratum is
content-addressed with `sample_seed 20260901`, so the same seed reproduces the
sheet exactly.

**Objects dropped because a crop could not be rendered: 0.** None substituted.
The 3 EXIF-reoriented images were already excluded upstream at pairing time and
were never in the 1,238-object pool.

For scale: 89 objects from 1,238 paired objects on 37 images, which are
themselves a 40-image pilot of the 2,318-image train+valid corpus. The sheet
says this in its own header.

## A.3 A bug found while building it

The first generated sheet had blank legacy panels. The cause was mine and worth
recording because it will bite anything else that touches this data:

**Legacy COCO annotation ids are not unique across split files.**
`instances_train` numbers 1..56502, `instances_valid` restarts at 1 (1..16587),
`instances_test` restarts again (1..8115). Every valid id and every test id
collides with a train id. Looking an annotation up by bare id across the three
files silently returns a *different object on a different image*.

332 of the 1,238 paired objects (254 valid + 78 test) resolve their legacy
annotation from a non-train file, so roughly a quarter of the sheet was showing a
polygon belonging to an unrelated object.

Fixed by keying on `(split, id)`; `compare_sam_polygons` now emits `legacy_split`
alongside `legacy_annotation_id`. **The comparison metrics in §4 were never
affected** — `compare()` pairs within an image record and never re-looked-up by
id — and re-running confirms every figure reproduces unchanged.

A guard was added rather than trusting the fix: before rendering, the legacy
annotation box must agree with the human box at IoU >= 0.99, the same threshold
the pairing uses. A mismatch is now a hard, counted drop with a reason. The old
bug scored 0.000 against that guard. A blank or wrong panel is worse than a loud
failure, because the reviewer would have judged it without knowing.

Verified after the fix: all 178 panels contain both a drawn polygon and the human
box; zero blank panels.

## A.4 On the framing, since it was asked

Two things would settle this faster than a bigger sheet, and neither needs more
GPU:

1. **A.1 already removes most of the question.** 89.9% of fragments are specks
   worth a median 0.57% of mask area. If the answer to "is the substantial
   structure real?" is yes on the 27 objects that have it, then the new set wins
   on both axes once a largest-component rule is applied, and the trade-off
   dissolves. The sheet is weighted to answer exactly that.
2. **The metric that actually decides this is not on the sheet.**
   `configs/annotation/pilot_v1.json` names
   `median_human_seconds_per_accepted_image` as the primary selection metric. A
   preference vote is a proxy for it. If CVAT were standing (it is not — see
   `docs/cvat.md` section 1, Docker is not installed), loading the same 89
   objects as two small tasks and timing the corrections would measure the real
   quantity instead of a stand-in. That is more setup than 20 minutes, but it is
   the honest instrument, and it is the same work the pilot needs anyway.

An overlay view — both contours on one crop in different colours — was
considered and rejected: it is faster to scan but cannot be blinded, since two
colours in one image announce which is which.

## A.5 Validation for this addendum

| Check | Result |
| --- | --- |
| `pytest tests/test_review_sheet.py` | **12 passed** |
| `pytest` all three new test files | **55 passed** |
| `black --check` (10 files) | pass |
| `ruff check` (10 files) | pass |
| `mypy` (7 modules) | Success: no issues found |
| Comparison metrics after the `legacy_split` change | reproduce exactly |
| Panels containing a polygon and a box | 178 / 178 |
| Cards leaking source or stratum | 0 |
| Objects dropped | 0 |
