# ADR 0005: Select the `single_mask` SAM candidate, and clean fragments by size

## Status

**DRAFT — awaiting approval. Not accepted, not in force.**

Written 2026-09-03 alongside the first implementation of
`src/agrinav/data/optimize_proposals.py`. The code ships the behaviour described
here as its *defaults*; this ADR is the record of why, and it should not be
treated as a settled decision until a reviewer accepts it. Nothing has been
committed. If the decision changes, change the defaults and this file together
before either is merged.

## Context

`sam_box_to_mask` prompts SAM2.1 with each human COCO box and emits every
candidate it computed — currently 9 per box — selecting none of them. It records
that abdication explicitly as `candidate_selection:
"deferred_to_optimize_proposals"`. Selection was deferred because it is a
*safety-relevant* predicate and the project requires those to live in CPU-only,
unit-testable code rather than inside a Colab notebook.

Nothing then implemented the deferred stage, so the raw candidate sidecar could
not reach CVAT at all: CVAT imports COCO 1.0 polygon JSON, and the shard holds
RLE candidate sets. This blocked the entire annotation-review path — not on GPU
hours, but on ~600 lines of CPU code.

Two decisions have to be made before any polygon reaches a reviewer, and both
change the geometry a human will see and correct:

1. **Which candidate becomes the proposal.**
2. **What to do about masks that come back in several disconnected pieces.**

Neither can be deferred further: a proposal has exactly one geometry.

There is no mask ground truth in this project. Nobody has hand-drawn a rice or
weed polygon, so "which candidate is best" cannot be measured directly. What can
be measured is agreement with the human box — the only reviewed geometry in the
pipeline — and that is what the re-pilot measured.

## Decision

### 1. Default selection rule: `single_mask_exact_box`

Select the `multimask_output=False` candidate computed on the **unjittered**
human box (`kind="single_mask"`).

Measured over 1,238 paired objects (`reports/summaries/sam_reseed_repilot_2026-09-01.md`):

| rule | median box IoU vs human box |
|---|---:|
| legacy polygons (what a re-seed would replace) | 0.759 |
| `argmax(sam_pred_iou)` over the multimask heads | 0.711 |
| **`single_mask` on the exact box** | **0.774** |

`single_mask` beats multimask-argmax on **68.4% of objects**. The mechanism is
not subtle: the multimask head exists to resolve whole/part/subpart *ambiguity*,
which is a point-prompt problem. A box has already resolved it, and asking an
ambiguity-resolving head to answer an unambiguous question is how a re-seed
measurably *lost* geometry before this candidate existed.

Consequences of the rule are recorded per object, not inferred: the emitted COCO
annotation and the feature row both carry `selection_rule`, `candidate_index`
and `candidate_kind`.

### 2. The rule may never be silently substituted

Shards written before the `single_mask` candidate existed carry 8 candidates and
no such kind. For those objects the selector returns `None` and the object is
**counted** under `rule_kind_not_in_shard`; the run aborts unless
`--allow-rule-unavailable` is passed, and even then the object is emitted as a
bbox-only proposal rather than being served a different candidate kind.

Serving multimask geometry while the artifact says `single_mask` would misreport
by 6.4 points of median box IoU, in an artifact whose whole purpose is to be
trusted about what a reviewer is looking at. `thresholds.candidates_per_box` is
therefore read from each shard and a shard that does not state it is a hard
error — there is no safe value to assume.

### 3. Default fragment policy: `drop_small_components` at 1% of mask area

Drop each connected component whose area is below
`--fragment-min-component-fraction` (default `0.01`) of the mask's area. Keep
everything else, including a second large component.

Measured over all 1,280 `single_mask` masks in the re-pilot shard:

| quantity | value |
|---|---:|
| masks with >= 2 connected components | 207 of 1,280 (16.2%) |
| extra (non-largest) components, total | 945 |
| ...that are < 1% of their mask area | **850 (89.9%)** |
| mask area outside the largest component, median | **0.57%** |
| ...p90 / p99 / max | 22.1% / 43.9% / 72.0% |
| objects with >= 5% of area outside the largest component | **27 of 1,238 (2.2%)** |

Nine out of ten extra components are specks. For the median fragmented mask a
largest-component rule would discard 0.57% of its area — free cleanup, costing a
reviewer nothing. But the tail is real: at p90 it discards 22% and at the maximum
72%, which for a weed may be a second leaf blade of the same plant. A
size-thresholded rule removes the specks *and* keeps the tail; a
largest-component rule cannot do both.

The policy is a named option with a default, never a silent always-on
transformation. `keep_all` and `largest_component` remain available and are
recorded in the output. Every dropped component is counted per object
(`components_dropped`, `dropped_area_fraction`) and in the run manifest.

## Alternatives considered

**A fitted candidate selector.** Learn a scoring function over the 9 candidates.
Rejected: there is no mask ground truth to fit against, so it would be fitted on
box-agreement proxies and then evaluated on the same proxies. A rule that is
defensible from a single measured comparison is worth more than a selector
fitted on the metric it is judged by.

**`multimask_argmax_pred_iou` as the default** — SAM's own default, and the only
rule expressible before the `single_mask` candidate existed. Rejected on the
measurement: it is worse than the *legacy polygons a re-seed would replace*
(0.711 vs 0.759), so defaulting to it would make a re-seed a regression.

**`largest_component` as the default fragment policy.** Rejected: it deletes
substantial structure from the ~2% tail, and deleting machine geometry that may
be a real part of the object, without any record, is the failure mode this
pipeline is built to avoid. It stays available for anyone who wants it.

**`keep_all` as the default.** Rejected as a default, though it is the only
lossless option. At 16.2% fragmentation it sends a large number of multi-ring
polygons into CVAT for no reviewer benefit, since 89.9% of the extra components
are specks a reviewer would delete by hand.

**Emitting one annotation per connected component.** Rejected outright: it
invents objects no human drew. Two annotations where the human asserted one box
is a fabricated object identity, which is a named forbidden inference.

**Routing fragmented masks through `reason_codes` for human attention.**
Rejected after reading the consumer: `triage_proposals.rule_mask_demoted` sends
any object with a non-empty `reason_codes` to `auto_reject`, which *discards the
mask*. Flagging a fragmented-but-real mask that way would throw away exactly the
geometry the fragment policy was designed to preserve. `reason_codes` is a
mask-rejection channel, so it carries only genuine mask failures (no candidate,
empty mask, no extractable ring); fragmentation travels as ordinary feature
values.

## Consequences

- The CVAT import path is unblocked without GPU work. A raw shard now becomes
  importable COCO polygon JSON plus the `proposal_features_v1.jsonl` that
  `triage_proposals` has always expected.
- **Nothing produced here is truth.** Every annotation is
  `review_status: "unreviewed_proposal"`, every feature row is `"unreviewed"`,
  and no code path writes `verified_empty`, `treatment_eligible`,
  `annotator_id` or `reviewer_id`. Selecting a machine mask is not reviewing it.
- **A multi-ring object cannot round-trip through this repo's own converter.**
  `cvat_export_to_records._polygon_ring` raises on a multi-ring polygon ("the
  wire format holds one ring per object"). Objects the fragment policy leaves
  multi-ring must be split into separate shapes in CVAT before export. They are
  counted as `objects_multi_ring` so the size of that manual task is known
  before the import, not discovered after it.
- Holes are lost. A COCO polygon ring cannot express one; interior contours are
  dropped and counted as `objects_with_holes_flattened`.
- A feature sidecar from this stage alone can **never** promote anything to
  `bulk_confirm` in triage, because promotion requires an independent
  corroborating signal (family B photometric or family C GrabCut) and both need
  image pixels this stage never reads. That is the safe direction and it is
  asserted by a test, not left to chance.
- The 0.01 threshold is a *configured* number, not a fitted one. It is set where
  the measurement shows the speck population sits (89.9% below 1%), and it is a
  CLI option so it can be moved without a code change.
- No checkpoint or model compatibility is affected; this is a data-preparation
  decision only.

## What it would take to revisit

This decision rests on one measurement of 1,238 paired objects from 40 images,
one seed, 29 of 103 capture groups. Specific triggers to reopen it:

1. **The blinded review sheet returns.** The re-pilot built a sheet weighted to
   answer exactly one question: *for the ~2% of objects with substantial extra
   structure, is that structure real?* All 27 such objects are included, none
   sampled away. If a human says the extra components are noise,
   `largest_component` becomes defensible as the default and the threshold
   becomes irrelevant. If they say it is real, the 1% threshold should probably
   drop further.
2. **Per-image reviewer timings exist.** The pilot's own primary metric is
   `median_human_seconds_per_accepted_image`. A fragmented mask costing a
   reviewer more than a loose one would justify trading box IoU for shape
   cleanliness — the `single_mask` candidate localises better but fragments more
   (13.6% vs 2.0% multi-component), and no measurement here settles that trade.
3. **The candidate set changes.** If `sam_box_to_mask` adds or removes a
   candidate kind, `candidates_per_box` changes, the comparison must be re-run,
   and `SELECTION_RULES` gains a member.
4. **A different SAM revision is pinned.** Every number here is from one pinned
   model revision. Mask geometry is revision-sensitive and dtype-sensitive.
5. **Real mask ground truth appears.** The moment any human polygon exists, the
   proxies above stop being the best available evidence and the comparison
   should be redone against truth.

## References

- `reports/summaries/sam_reseed_repilot_2026-09-01.md` — §A.1 fragmentation
  measurement, §5 effect on `optimize_proposals`, overall box-IoU table
- `reports/summaries/sam_reseed_pilot_2026-08-31.md` — superseded pilot,
  retained for the "do not run" conclusion it reached before the fix
- `src/agrinav/data/optimize_proposals.py` — the implementation and its defaults
- `src/agrinav/data/sam_box_to_mask.py` — candidate generation,
  `CANDIDATES_PER_BOX`, `run_thresholds`
- `src/agrinav/data/compare_sam_polygons.py` — `SELECTION_RULES`, the
  comparison-only vocabulary this stage reuses
- `src/agrinav/data/triage_proposals.py` — the feature-row consumer;
  `rule_mask_demoted` is why fragmentation is not a reason code
- `src/agrinav/data/cvat_export_to_records.py` — `_polygon_ring`, the one-ring
  round-trip limit
- `tests/test_optimize_proposals.py` — behavioural guards for every claim above
- Kirillov et al., *Segment Anything*, ICCV 2023 (arXiv:2304.02643) — the
  ambiguity motivation for the multimask head. Note this is SAM v1; the pipeline
  pins a SAM 2.1 checkpoint, which inherits the head but is a different paper.
  Verify the citation before it appears in anything published.
