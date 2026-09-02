# SAM 2.1 polygon re-seed — pilot report

**Date:** 2026-08-31
**Scope:** 40-image pilot of a box→mask re-seed over the curated RICE detection set.
**Status:** pilot only. **The full 2,318-image run was not started.**
**Recommendation:** **do not run the full re-seed as currently configured.** See §7.

> **SUPERSEDED 2026-09-01.** Recommendation §7.1 was acted on: the missing
> multimask_output=False candidate was added to the stage and the same 40
> images were re-piloted. The measurements below stand as the record of the
> old 8-candidate configuration, but the conclusion no longer reflects the
> code. Current numbers, the three-column comparison and the corrected ETA
> (72.2 min) are in
> [sam_reseed_repilot_2026-09-01.md](sam_reseed_repilot_2026-09-01.md).

Everything produced by this pilot is an **unreviewed proposal**. No annotation
record, no `review_status: accepted`, no `verified_empty`, no
`treatment_eligible`, no `annotator_id`/`reviewer_id` was written by any stage
described here. Human review in CVAT remains the only step that can produce truth.

---

## 1. Verification (CUDA, pin, input tree)

### 1.1 CUDA

```
torch 2.13.0+cu130   torch.cuda.is_available() -> True
torchvision 0.28.0+cu130
CUDA runtime 13.0    NVIDIA GeForce RTX 4070, 12,282 MiB, driver 595.71
Python 3.12.13       transformers 5.14.1
```

The parallel CUDA install had completed before any GPU work began. No torch
package was installed, reinstalled or modified by this work.

### 1.2 Model pin

Re-verified independently against the HuggingFace API on 2026-08-31 (UTC),
not taken on trust from the task brief:

| Field | Value |
| --- | --- |
| Model | `facebook/sam2.1-hiera-large` |
| Requested revision | `665f8e2ad61cf5f53d65644ff27c8ee525124610` |
| Resolved sha | `665f8e2ad61cf5f53d65644ff27c8ee525124610` (identical) |
| Repo `lastModified` | `2025-08-15 21:19:57+00:00` |
| Resolved via | `huggingface_hub.HfApi().model_info` |
| Resolved at | `2026-09-01T00:43:50Z` |
| Gated | no; licence apache-2.0 |

Reproduce with:

```bash
.venv/Scripts/python.exe -m agrinav.data.sam2_predictor --verify-revision \
  --model-id facebook/sam2.1-hiera-large \
  --revision 665f8e2ad61cf5f53d65644ff27c8ee525124610
```

The negative control also passes: `--revision PIN_BEFORE_RUN` is refused with a
non-zero exit before any GPU allocation or model download.

**The pin is now actually enforced, which it would not have been before.**
See §6.1 — this is the most important engineering finding of the pilot.

Weight-load integrity was checked explicitly, because `transformers` emits a
`sam2_video` → `sam2` architecture warning on this repo:

```
missing_keys: 0   unexpected_keys: 0   mismatched_keys: 0   error_msgs: 0
parameters: 216,924,865
```

Nothing was randomly initialised. The warning is cosmetic (the repo's
`config.json` declares the video model; `Sam2Model` consumes the image subset in full).

### 1.3 Input tree

`C:/Users/Benny Merr/Downloads/RICE_phase2_rebuild_2026-07-29/`

| Split | Images | Annotations | rice_protect | weed_target | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 1,800 | 59,691 | 52,194 | 7,497 | in scope |
| valid | 518 | 15,226 | 13,201 | 2,025 | in scope |
| **train+valid** | **2,318** | **74,917** | **65,395** | **9,522** | **re-seed scope** |
| test | 261 | 6,284 | — | — | **SEALED — not read** |

- Boxes/image: mean 32.32, median 33, min 2, max 103, p90 45. No zero-box images.
- 9 distinct resolutions; 640×640 (45.1%) and 640×480 (39.8%) dominate.
- Category ids `{1: rice_protect, 2: weed_target}` — inside the closed map in
  `sam_box_to_mask.COCO_CATEGORY_TO_LABEL`, so zero counted drops.
- Annotation-file sha256 verified against `manifests/provenance.json`: **match**.
- **All 2,318 train+valid image files re-hashed against their COCO records:
  0 mismatches** (645.1 MB read in 1.5 s). The re-seed input is verifiably intact.
- No duplicate image sha256 across train and valid.
- 115 of 2,318 carry `exif_reoriented: true` (the rebuild applied
  `ImageOps.exif_transpose` and re-encoded those files).

The 261-image test split was never opened. Its annotation JSON was not parsed,
its images were not hashed, and no proposal references it. This is enforced
structurally, not by convention — see §5.

---

## 2. Configuration and exact commands

Environment: `.venv/Scripts/python.exe` (uv-managed 3.12). No new environment,
no package installs.

**Step 1 — stage the input** (2,318-image corpus, deterministic 40-image sample):

```bash
.venv/Scripts/python.exe -m agrinav.data.build_sam_reseed_input \
  --rebuild-root "C:/Users/Benny Merr/Downloads/RICE_phase2_rebuild_2026-07-29" \
  --splits train,valid \
  --sample-size 40 --sample-seed 20260831 \
  --out-zip artifacts/sam_reseed_2026-08-31/pilot_images.zip \
  --out-json artifacts/sam_reseed_2026-08-31/pilot_proposals_unreviewed.coco.json \
  --out-manifest artifacts/sam_reseed_2026-08-31/pilot_input_manifest.json
```

Result: 40 images (29 train / 11 valid), 1,280 boxes (1,075 rice_protect /
205 weed_target), 3 EXIF-reoriented, 10.4 MB.

**Step 2 — measured pilot run:**

```bash
.venv/Scripts/python.exe -m agrinav.data.sam_reseed_pilot \
  --coco-zip artifacts/sam_reseed_2026-08-31/pilot_images.zip \
  --proposals-json artifacts/sam_reseed_2026-08-31/pilot_proposals_unreviewed.coco.json \
  --model-revision 665f8e2ad61cf5f53d65644ff27c8ee525124610 \
  --out-shard artifacts/sam_reseed_2026-08-31/sam_raw/pilot_shard_000.jsonl \
  --out-manifest artifacts/sam_reseed_2026-08-31/pilot_run_manifest.json
```

**Step 3 — comparison against the existing polygon set:**

```bash
.venv/Scripts/python.exe -m agrinav.data.compare_sam_polygons \
  --raw-shard artifacts/sam_reseed_2026-08-31/sam_raw/pilot_shard_000.jsonl \
  --proposals-json artifacts/sam_reseed_2026-08-31/pilot_proposals_unreviewed.coco.json \
  --legacy-annotations-dir "C:/Users/Benny Merr/Downloads/agrinav_intake_2026-07-21/deliverable/detection/RICE/annotations" \
  --legacy-splits train,valid,test \
  --out-json reports/metrics/sam_reseed_pilot_iou_2026-08-31.json \
  --out-objects-jsonl artifacts/sam_reseed_2026-08-31/pilot_iou_objects.jsonl
```

**Step 4 — selection-rule probe:**

```bash
.venv/Scripts/python.exe -m agrinav.data.sam_selection_probe \
  --coco-zip artifacts/sam_reseed_2026-08-31/pilot_images.zip \
  --proposals-json artifacts/sam_reseed_2026-08-31/pilot_proposals_unreviewed.coco.json \
  --model-revision 665f8e2ad61cf5f53d65644ff27c8ee525124610 \
  --out-json reports/metrics/sam_selection_probe_2026-08-31.json
```

Run parameters (from `sam_box_to_mask.run_thresholds`, unchanged): 8 candidates
per box — 3 `multimask_output=True` on the exact human box plus 5
`multimask_output=False` on deterministic ±3% box jitters — giving 6 decoder
calls per box. Compute dtype `bfloat16`; device `cuda`; batch size 1;
`candidate_selection: deferred_to_optimize_proposals`.

---

## 3. Measurements (verbatim)

From `artifacts/sam_reseed_2026-08-31/pilot_run_manifest.json`:

```
stat.images_in_shard: 40
stat.images_processed: 40
stat.images_skipped_resume: 0
stat.boxes_emitted: 1280
stat.boxes_degenerate: 0
stat.candidates_emitted: 10240
stat.unmapped_dropped: 0
stat.labels: {'rice_protect': 1075, 'weed_target': 205}
rate.wall_seconds: 69.203
rate.inference_seconds: 49.599
rate.non_inference_seconds: 18.804
rate.images_per_second: 0.578
rate.boxes_per_second: 18.4964
rate.boxes_per_image: 32.0
rate.encode_seconds_per_image: 0.1066
rate.decode_seconds_per_box: 0.03542
rate.overhead_seconds_per_box: 0.01469
rate.decoder_calls_per_box: 6.0
gpu.peak_allocated_bytes: 742075904
gpu.peak_reserved_bytes: 977272832
gpu.device_total_bytes: 12878086144
gpu.peak_reserved_gib: 0.91
gpu.device_total_gib: 11.994
gpu.headroom_gib: 11.084
predictor_counters.model_load_seconds: 0.7997
predictor_counters.set_image_seconds: 4.2656
predictor_counters.predict_seconds: 45.3339
predictor_counters.predict_calls: 7680
```

Derived, excluding the 0.80 s warm model load (steady state 68.40 s):

- **0.585 images/sec**, **18.71 boxes/sec**
- encode 0.10664 s/image; decode 0.03542 s/box; non-inference 0.01469 s/box

### 3.1 Acceptance / demotion counts

The task asked for "the mask-acceptance/demotion counts the stage reports".
**The stage deliberately reports none, and that is correct behaviour, not a
gap in the measurement.** `sam_box_to_mask` emits all 8 candidates and defers
every acceptance predicate to CPU-only downstream code so it can be unit-tested
without a GPU. The only demotion signal it owns is `degenerate_reason`, for a
box that cannot be clipped to positive area:

| Counter | Value | Meaning |
| --- | ---: | --- |
| `boxes_emitted` | 1,280 | one row per input box, always — no box was dropped |
| `boxes_degenerate` | **0** | no box demoted to bbox-only geometry |
| `unmapped_dropped` | **0** | no category outside the closed map |
| `candidates_emitted` | 10,240 | exactly 8.0 per box, no shortfall |
| `labels` | 1075 / 205 | identical to the input; **zero label drift** |

Every emitted mask was non-empty (0 empty masks in 10,240 candidates), and the
label of every row equals the human COCO box's label.

### 3.2 GPU memory and headroom against 12 GB

| Quantity | Value |
| --- | ---: |
| Peak allocated | 0.691 GiB |
| **Peak reserved** | **0.910 GiB** |
| Torch-visible device total | 11.994 GiB (12,282 MiB) |
| **Headroom vs. an idle card** | **11.08 GiB (92.4% free)** |
| Other processes resident at check time | 1,174 MiB |
| Practical headroom with desktop running | ≈ 9.9 GiB |

Peak reserved is **7.6% of the card**. This will not change with corpus size or
image resolution: the processor resizes every input to a fixed 1024×1024
(verified from `preprocessor_config.json`: `size {height: 1024, width: 1024}`,
bilinear), and this backend moves masks to CPU before upsampling to the original
resolution. Memory is therefore resolution-independent by construction, and the
pilot already spanned 320×240 to 2048×3648. **Memory is not a constraint.**

---

## 4. ETA for the full 2,318-image run

Cost model: per-image encode + per-box decode, extrapolated on their own totals
(2,318 images and 74,917 boxes are different multipliers).

```
2,318 × 0.10664 s/image  +  74,917 × (0.03542 + 0.01469) s/box
=            247 s       +            3,754 s
=          4,001 s  =  66.7 minutes
```

Cross-check on the box rate alone: 74,917 / 18.71 = 4,004 s = 66.7 min. The two
methods agree to within 0.1%.

**ETA: ≈ 67 minutes (1.1 h)** of GPU wall time, single process, batch 1, warm
HF cache. Add ~5 s to stage the full input zip (measured: 645 MB read+hashed in
1.5 s at 424 MB/s) and ~1 s for a warm model load.

Sampling adequacy: the pilot's mean image size is 1.056 MP against the corpus's
1.025 MP — the sample is **3.0% heavier** than the corpus, so the estimate is
marginally conservative. Pixel-mix-corrected ETA: **66.0 min**. The pilot's
boxes-per-image (32.0) matches the corpus (32.32) to within 1%.

**Practical range: 63–70 minutes.** It is not multi-hour. Peak memory leaves
room for 2–4 concurrent shards (the stage already supports
`--shard-index`/`--shard-count`), which would cut wall time proportionally if it
ever mattered — but see §7: wall time is not the binding constraint here.

---

## 5. Sealed-split handling

The 261-image test split was excluded **structurally**, through a closed
allow-list rather than a deny-list, at three independent layers:

1. `build_sam_reseed_input.parse_splits` validates `--splits` against
   `SELECTABLE_SPLITS = ("train", "valid")`. `test` is refused with an explicit
   error; so is a typo such as `vaild`, which would otherwise silently select nothing.
2. After path resolution, `assert_not_sealed` re-checks that no selected file
   lies under a sealed split directory, and that no path escapes `--rebuild-root`.
3. `sam_reseed_pilot.assert_no_sealed_split` refuses at the run boundary any
   proposal document containing a sealed split — or any image with a missing
   `split` field, since absence cannot be proven then.

All three are covered by tests (§8). The comparison in §6 reads legacy
annotation files whose *own* `train/valid/test` folders are a **different,
superseded partition** — 3 pilot images sit in the legacy `test` folder while
being train/valid in the current build. The sealed set is defined by the current
build's membership; no current-build test image was read, hashed, or emitted.

---

## 6. Comparison against the existing polygon set

Compared on the same images against
`agrinav_intake_2026-07-21/deliverable/detection/RICE/annotations/`.

**There is no mask ground truth in this project.** Both sets are unreviewed
machine output, so "better" can only be measured by proxy against the human box —
the only reviewed geometry in the pipeline. Three proxies are used together
because each alone is gameable:

- `box_iou` — IoU of the mask's tight bbox with the human box. Higher is better,
  but a mask that *is* the rectangle also scores 1.0.
- `fill` — mask area / box area. Distinguishes "tracks the object" from
  "returned the box".
- `containment` — fraction of mask pixels inside the human box. Pixels outside
  are geometry no human asserted.

Matching: 37 of 40 images compared (3 excluded as EXIF-reoriented — for
orientation 3 the dimensions are unchanged while every pixel moves, so a
dimension check would not catch it and the comparison would silently score a
rotated mask against an unrotated one). All 37 matched by sha256; 1,238 of 1,239
objects paired at box IoU ≥ 0.99.

**Decoding cross-check:** the recomputed legacy `box_iou` reproduces the legacy
pipeline's own recorded `attributes.mask_box_iou` to a median difference of
0.0000 (mean +0.0012; only 0.3% differ by more than 0.05). The legacy numbers
below are not an artefact of mis-decoding its polygons.

### 6.1 Headline result — the new masks are worse as configured

| Metric (median) | Legacy (existing) | New, stage as configured | Verdict |
| --- | ---: | ---: | --- |
| `box_iou` vs human box | **0.759** | 0.711 | legacy better |
| `containment` | **0.985** | 0.939 | legacy better |
| `fill` | 0.517 | 0.605 | new masks are larger |
| `iou_new_old` | — | 0.815 | substantial agreement |
| jitter agreement | n/a | 0.869 | new set is stable |

Paired per object: the new mask is worse on `box_iou` for **62.9%** of objects,
better for 36.3%, tied within 0.01 for 11.6%. Median paired delta **−0.027**.

The legacy set wins on alignment *and* tightness *simultaneously* — it is not
winning by degenerating into the bounding rectangle. Two concrete defects in the
new set explain the gap:

- **Box leakage.** A median of **6.1%** of each new mask's area falls outside the
  human box; 30.6% of objects leak more than 10% of their mask, 3.2% more than
  25%. Legacy containment is 0.985 against the new set's 0.939.
- **Fragmentation.** **24.9%** of new masks have ≥ 2 connected components and
  7.6% have ≥ 5. Every legacy annotation is a single polygon part (measured:
  1,239/1,239), median 52 vertices — detailed, not over-simplified, so the
  raster-vs-polygon storage difference is a minor confound at most.

On the safety-critical class the two sets are **tied**: `weed_target` box_iou
0.699 legacy vs 0.704 new (median paired delta −0.0009; 49.2% better, 50.3%
worse). The re-seed buys nothing on weeds as configured.

### 6.2 Why — the candidate set is missing the right configuration

The stage picks nothing, so the comparison had to pick: `argmax(sam_pred_iou)`
over the 3 `multimask_output=True` candidates, which is SAM's own default. That
rule is the problem.

`multimask_output=True` exists to resolve *ambiguity* (whole / part / subpart)
and is aimed at single-point prompts. A box is an unambiguous prompt, for which
SAM's single-mask head is the recommended output. The stage's 8 candidates
contain 3 multimask outputs on the exact box and 5 single-mask outputs on
*jittered* boxes — and therefore **never contain the single-mask output on the
unjittered human box**. Measured on the same 1,280 boxes:

| Configuration | box_iou (median) | containment | fill |
| --- | ---: | ---: | ---: |
| New — `multimask` argmax(pred_iou) *(what the stage emits)* | 0.710 | 0.939 | 0.598 |
| **Legacy — existing polygons** | **0.759** | **0.985** | 0.517 |
| New — **`multimask_output=False` on the exact box** *(absent from the candidate set)* | **0.772** | 0.977 | 0.556 |
| New — oracle best of the 3 multimask | 0.787 | — | — |
| New — oracle best of all 8 candidates | 0.819 | — | — |
| New — mean over the 5 single-mask jitters | 0.768 | — | — |

The single-mask configuration beats the stage's own default on 68.4% of objects
(mean +0.056 box_iou), and on `weed_target` it reaches 0.741 against legacy's
0.699 — a real but modest gain on the class that matters.

Two conclusions follow, and they point in opposite directions:

1. **As configured, a full re-seed would replace the existing polygons with
   measurably worse ones.** That is the finding.
2. **Even fixed, the geometric gain over the existing set is small** — +0.013
   median box_iou overall, with containment still slightly worse (0.977 vs
   0.985). The 0.815 median agreement between the two sets says the same thing:
   re-seeding mostly reproduces what is already there.

### 6.3 The engineering finding that does justify work

`sam_box_to_mask.default_predictor_factory` calls
`SAM2ImagePredictor.from_pretrained(model_id, revision=revision)` on the
upstream `sam2` package. That call **does not pin anything**: upstream,
`from_pretrained` forwards `**kwargs` to `build_sam2_hf` → `build_sam2` →
`SAM2ImagePredictor.__init__`, none of which pass `revision` to
`hf_hub_download`; each absorbs it into `**kwargs`. The download resolves the
repo's current `main`, and the pinned sha is written into provenance having had
no effect on the weights that drew the masks.

That is the exact failure mode `validate_model_revision` exists to prevent, one
layer further down: provenance that validates perfectly clean and is
unrecoverable. This pilot used `transformers`, which honours `revision` on
`from_pretrained` and whose resolution is re-asserted against the HF API before
any GPU allocation, so the pin here is *verified*, not merely recorded.

For context on why provenance is the real motive: the existing polygon set's own
provenance block reads `proposal_model_id: "facebook/sam2-hiera-large (upstream
agent pipeline)"` with **no revision at all**, `generated_by_upstream: true`, and
`verification_note: "... NOT visually reviewed."` It is SAM 2.0, from an
unverified upstream pipeline. A re-seed's genuine value is that lineage, not
geometry. But a re-seed that produces *worse* geometry is not a good trade even
for better provenance — and one that produces geometry nobody can load into CVAT
(§6.4) is not a trade at all.

### 6.4 Blocking gap: the pilot output cannot reach CVAT

`sam_box_to_mask` emits a raw candidate sidecar (8 RLE candidates per box, no
selection). `docs/cvat.md` §7 imports **COCO 1.0 polygon JSON**. The two stages
that would bridge them do not exist:

- **`optimize_proposals` is absent from the repository.** Both
  `sam_box_to_mask.py` (`candidate_selection: "deferred_to_optimize_proposals"`)
  and `triage_proposals.py` reference `scripts/optimize_proposals.py`, and there
  is no `scripts/` directory at all. `triage_proposals` requires the
  `proposal_features_v1.jsonl` that stage would emit.
- No RLE → polygon → COCO writer exists.

So the full run's output would sit as a `.jsonl` of raw candidates with no path
to review, whereas the legacy set is already in the exact COCO format
`docs/cvat.md` tells you to upload. **The blocker is missing pipeline stages,
not GPU hours.**

---

## 7. Recommendation

**Do not proceed to the full 2,318-image run.** Not because it is expensive — it
is only ~67 minutes — but because as configured it would spend those minutes
producing polygons that are measurably worse than the ones already on disk, in a
format nothing can currently consume.

Ordered by value:

1. **Fix the candidate set before any full run.** Add a
   `multimask_output=False` candidate on the unjittered human box to
   `candidates_for_box` (a 9th candidate, +1 decoder call per box, ~17% more GPU
   time). This is a change to a safety-critical file and should be a reviewed
   change with its own tests, not something bolted on mid-run. It is the single
   change that moves the new set from worse-than-legacy to better-than-legacy.
2. **Then decide whether the gain is worth it.** Even fixed, the improvement
   over the existing polygons is +0.013 median box_iou overall (+0.042 on
   `weed_target`), with slightly worse containment. If the goal is *better
   polygons*, that is a thin return. If the goal is *trustworthy provenance* —
   replacing SAM 2.0 output from an unverified upstream pipeline with pinned,
   verified SAM 2.1 output — then it is worth the hour, and should be justified
   on those grounds explicitly rather than on quality.
3. **Build `optimize_proposals` first regardless.** Without it neither the new
   nor a re-run set can reach CVAT, and the raw candidate sidecar is not a
   reviewable artifact. This is on the critical path for the stated goal and
   needs no GPU.
4. **Consider the cheaper alternative.** A single-mask-only configuration (1
   decoder call per box instead of 6) has an estimated ETA of **~18 minutes**
   (derived from the probe's measured 0.00621 s/decoder-call, not measured
   end-to-end) and produced the best non-oracle geometry in §6.2. If the 8-way
   candidate set is not actually being used by a downstream selector that does
   not yet exist, generating 8 candidates is paying 6× for optionality nobody
   consumes.
5. **Fix `default_predictor_factory` (§6.3)** or document that the
   `transformers` backend is the supported path. Right now a run through the
   documented CLI without `--predictor-factory` would silently ignore the pin.

If the user decides to proceed anyway, the run is ready: swap
`--sample-size 40` for `--sample-size 0` in step 1 and re-run steps 1–2. The
stage is resumable (`load_done_keys` skips completed keys) and shardable.

---

## 8. Validation performed

| Check | Result |
| --- | --- |
| `pytest tests/test_sam_reseed_input.py` | **26 passed** |
| `black --check` (6 files) | pass |
| `ruff check` (6 files) | pass |
| `mypy` (5 new modules) | **Success: no issues found** |
| All 19 `agrinav.data` modules import | pass, no failures |
| Comparison re-run reproduces identical figures | pass |
| SAM2.1 weight load: missing/unexpected/mismatched keys | 0 / 0 / 0 |
| Legacy polygon decode vs legacy's own recorded IoU | median Δ 0.0000 |
| All 2,318 train+valid image hashes vs COCO records | 0 mismatches |

Tests cover the predicates whose failure is unrecoverable: sealed-split refusal
(4 tests, including a typo and a path-escape), image sha256 mismatch, declared-size
mismatch, annotation-file hash drift, missing image, unmapped category, sampling
determinism and order-independence, zip-member/proposal-filename agreement,
verbatim box copying, absence of any accepted/adjudicated status in emitted
artifacts, and refusal of all 7 forbidden text-style prompt kwargs.

**Not run:** the full repository test suite (a full `pytest` run here is known to
burn ~23 GB of temp and has filled the drive mid-run). No existing file was
modified, and every `agrinav.data` module still imports, so existing tests should
be unaffected — but that is inference, not a measurement.

---

## 9. Risks and open questions

- **The primary metric is a proxy.** With no hand-drawn mask anywhere in the
  project, `box_iou` against the human box is the best available signal and it is
  imperfect: it cannot distinguish a good object mask from a thin shape spanning
  the box. `fill` and `containment` were added to constrain that, and the legacy
  set wins on all three at once, but a 20-image human spot-check would settle it
  far more cheaply than a full re-seed.
- **n = 40 images / 1,238 paired objects, one seed.** Resolution mix and
  boxes-per-image match the corpus within 3%, and the box-level n is large, but
  only 29 of the 103 capture groups in train+valid are represented. Group-level variation is
  unmeasured.
- **weed_target n = 195** in the comparison. The class that decides whether a
  robot cuts a rice seedling is the one with the smallest sample here.
- **dtype is not in the provenance.** The run used `bfloat16`, which affects mask
  geometry, but `run_thresholds()` does not record it; it appears only in this
  pilot's run manifest. A full run should record it in the stage's own
  provenance block.
- **The 3 EXIF-reoriented pilot images were excluded from the comparison**, so
  the ~5% of the corpus that was re-encoded is unmeasured for IoU. They were
  processed normally for throughput.
- **Single-mask ETA (~18 min) is derived, not measured end-to-end.** It combines
  the probe's measured per-call decode time with the pilot's measured encode and
  an inferred per-candidate RLE cost.
- **`--predictor-factory` is documented in `sam_box_to_mask` as a test-only
  injection point.** This pilot drives it programmatically via
  `process(predictor_factory=...)` with a production backend. That is a
  deliberate reinterpretation of that seam and should be documented in the module
  if the transformers backend becomes the supported path.
- **`tests/test_sam_box_to_mask.py` does not exist**, though the module docstring
  references `tests.test_sam_box_to_mask:make_stub_predictor`. The stage's own
  safety invariants are therefore untested in this repository.

---

## 10. Artifacts

| Path | Contents |
| --- | --- |
| `artifacts/sam_reseed_2026-08-31/pilot_images.zip` | 40 verified pilot images (train+valid only) |
| `artifacts/sam_reseed_2026-08-31/pilot_proposals_unreviewed.coco.json` | staged human boxes, no geometry added |
| `artifacts/sam_reseed_2026-08-31/pilot_input_manifest.json` | input provenance, hashes, counts |
| `artifacts/sam_reseed_2026-08-31/sam_raw/pilot_shard_000.jsonl` | **raw SAM candidates — unreviewed proposals** |
| `artifacts/sam_reseed_2026-08-31/pilot_run_manifest.json` | pin, environment, timings, GPU memory |
| `artifacts/sam_reseed_2026-08-31/pilot_iou_objects.jsonl` | per-object comparison rows |
| `reports/metrics/sam_reseed_pilot_iou_2026-08-31.json` | IoU aggregate, overall / per class / small objects |
| `reports/metrics/sam_selection_probe_2026-08-31.json` | selection-rule probe |

Code added (none of it modifies an existing file):
`src/agrinav/data/build_sam_reseed_input.py`, `sam2_predictor.py`,
`sam_reseed_pilot.py`, `compare_sam_polygons.py`, `sam_selection_probe.py`,
`tests/test_sam_reseed_input.py`.

Nothing was committed or pushed. Working tree is on branch
`fix/assigner-and-measurement` at `424e9a09`, which was already dirty from
unrelated in-flight work before this task began.
