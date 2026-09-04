# Phase-2 detector ablation — 2026-09-03

**Status: complete.** Part 1 (single seed) is superseded by part 2's seed
replication. Read §"What part 2 established" before quoting anything from the
part 1 table — both of the effects part 1 appeared to show failed to reproduce.

Protocol, frozen across all arms: rebuilt phase-2 data
(`RICE_phase2_rebuild_2026-07-29`), **train + valid only — the 261-image test
split is never referenced**, seed 42, 8 epochs, validation AP every 2 epochs,
batch 8, RTX 4070 12 GB, `torch 2.13.0+cu130`. Exactly one config key differs
per arm, so a difference is attributable to that key.

## Part 1 results (single seed)

| arm | variable | best val AP | vs baseline | wall |
| --- | --- | ---: | ---: | ---: |
| `A_baseline` | as committed (img 512, anchor 4.0, ATSS on) | 0.0391 | — | 24.9 min |
| `B_anchor2` | `anchor_base_scale` 4.0 → 2.0 | 0.0191 | **−51%** | 24.7 min |
| `C_no_atss` | `use_atss` → false | 0.0473 | +21% | 24.2 min |
| `D_img768` | `img_size` 512 → 768 | **terminated** | — | 85.9 min |

**These are a development signal, not a result.** Single seed, 8 epochs, AP still
~0.04. Part 2 adds seeds 43/44 to establish the noise floor, without which
neither the −51% nor the +21% can be interpreted.

### Both part-1 readings were wrong — see §"What part 2 established"

Part 1 appeared to show `anchor_base_scale` halving AP by 51% and `use_atss:
false` gaining 21%. **Neither survived replication.** The anchor claim in
particular was asserted here as "large enough to be unlikely to be seed noise",
and that was wrong.

## `D_img768` — terminated, and why that is itself the finding

Killed after epoch 1. It was not failing; it was succeeding far too slowly, and
its result would not have been comparable anyway.

| | 512 px | 768 px |
| --- | ---: | ---: |
| iteration rate | 2.58 it/s | **17.83 s/it** |
| epoch 1 wall | ~90 s | **71 min** |
| GPU memory | — | **11,860 MiB of 12,282** |

At batch 8, a 768 px letterbox fills the card to 96.6% and the run collapses into
memory thrashing — roughly 46x slower per iteration, not the ~2.25x the pixel
count predicts. Eight epochs would have taken about **10 hours** and starved the
seven queued runs that answer the more important question.

**It also stopped being a one-variable arm.** At that memory pressure the run
differs from the baseline in its memory regime as well as its resolution, so any
AP difference would confound the two.

The useful conclusion: **on this card, testing higher input resolution requires
lowering the batch size**, which then confounds resolution with batch size and
effective learning rate. So resolution cannot be cleanly ablated here at batch 8,
and the small-object question (37.8% of boxes are COCO-small) needs a different
instrument — anchor scale, FPN level assignment, or a gradient-accumulation
design that holds the effective batch constant.

## Fixed along the way

`src/agrinav/models/weeddet_v6b.py` — a `★` (U+2605) in the best-checkpoint log
line raised `UnicodeEncodeError` on the Windows cp1252 console, *after* the
checkpoints were written. Every run on this machine died at its first checkpoint
save with its weights already safely on disk. `_log` now coerces to the stream's
encoding and degrades un-encodable characters rather than raising; the character
is ASCII. Re-running the same seed after the fix reproduced `avg_loss=3.8337,
val/AP=0.0013` exactly, which is also a determinism check.

## What part 2 established

Seeds 43 and 44 on `A_baseline` and `C_no_atss`, seed 43 on `B_anchor2`, and
`A` vs `C` at 18 epochs. All seven runs exited 0.

### The noise floor is larger than either effect

| arm | n | mean best val AP | sd | CV | values |
| --- | ---: | ---: | ---: | ---: | --- |
| `A_baseline` | 3 | 0.0414 | 0.0075 | 18.2% | 0.0391, 0.0353, 0.0498 |
| `C_no_atss` | 3 | 0.0462 | 0.0154 | 33.3% | 0.0473, 0.0303, 0.0610 |
| `B_anchor2` | 2 | 0.0277 | 0.0121 | 43.6% | 0.0191, 0.0362 |

Seed-to-seed variation is 18-44% of the mean. **With n=3 at this variance, only
an effect larger than about 65% relative could be distinguished from noise.**
Both hypotheses under test were far below that, so the experiment was
underpowered before it started.

### ATSS: no evidence of a difference, and the ordering flips

| seed | A | C | C-A | winner |
| ---: | ---: | ---: | ---: | --- |
| 42 | 0.0391 | 0.0473 | +0.0082 | C |
| 43 | 0.0353 | 0.0303 | **-0.0050** | **A** |
| 44 | 0.0498 | 0.0610 | +0.0112 | C |

Mean paired difference +0.0048, sd 0.0086 — a ratio of **0.56**, where n=3 needs
roughly >2.5 to mean anything. The winner changes with the seed. There is no
support here for turning ATSS off, and none for keeping it on either; the
experiment simply cannot see a difference this size.

### `anchor_base_scale`: the -51% did not reproduce at all

| seed | A | B (anchor 2.0) | B-A |
| ---: | ---: | ---: | ---: |
| 42 | 0.0391 | 0.0191 | **-51%** |
| 43 | 0.0353 | 0.0362 | **+3%** |

One seed said halving the anchor scale halves AP; the next said it makes no
difference. **The part-1 claim that this gap was "too large to be seed noise"
was wrong**, and it was wrong in the direction of believing a dramatic result
from a single run. `anchor_base_scale: 4.0` is still unvalidated — `667a9b4`
raised it without a measurement, and this ablation did not supply one.

### 8 epochs is far too early to rank anything

| | 8 epochs (seed 42) | 18 epochs (seed 42) | gain |
| --- | ---: | ---: | ---: |
| `A_baseline` | 0.0391 | 0.0661 | **+69%** |
| `C_no_atss` | 0.0473 | 0.0758 | **+60%** |

AP is still climbing steeply at 18 epochs. A ranking taken at 8 epochs is a
ranking of early-training dynamics, not of converged quality — and the
per-arm gain differs (69% vs 60%), so the ordering is not even guaranteed to be
preserved as training continues.

## The actual result

**The ablation as designed cannot answer the questions it was asked.** That is
the finding, and it is worth more than a false positive would have been:

1. Run-to-run variance at this budget (CV 18-44%) swamps any plausible config
   effect. Detecting a realistic 10-20% improvement needs either many more seeds
   or a budget long enough for variance to shrink relative to signal.
2. Neither `use_atss` nor `anchor_base_scale` has been validated. Both remain
   open, and `anchor_base_scale: 4.0` is still in the committed config on the
   strength of no measurement.
3. Any future comparison on this data must report mean ± sd over at least 3
   seeds and must run long enough to be past the steep part of the curve.
   Single-seed 8-epoch numbers should not be quoted at all.

## What this cannot tell you

No test-split number, by construction. No baseline from any other codebase or
protocol. Validation AP on a single dataset at a short budget ranks *this* set of
config choices against each other and nothing more.

---

# Council review — corrections to this report

Three reviewers audited the above independently. Two have reported. **Their
findings invalidate several conclusions in the sections above**, which are left
in place with this section as the correction of record.

## The design was wrong before it was underpowered

**1. Arm B never tested the change it was supposed to validate.** `667a9b4`
raised `anchor_base_scale` from **3 → 4.0**. Arm B tested **2.0**. Every other
config in the repo (`detector_default`, `detector_gpu`, `detector_smoke`) uses 3.
So the arm does not bracket the value under suspicion, and no result from it
could have validated or refuted `667a9b4`. This is a design error, not a power
problem.

**2. The 8-vs-18 epoch comparison varied two things, not one.**
`weeddet_v6b.py:2168-2173`: `total_steps = num_epochs * len(train_loader) *
repeat`, and `CosineAnnealingLR(T_max = total_steps - warmup_iters)`. The cosine
LR horizon is a function of `num_epochs`, so the two runs had different LR
trajectories from step 1. An 8-epoch run is a *complete, fully annealed short
schedule*, not a snapshot of a longer one. The report's claim that 8 epochs
"ranks early-training dynamics" is therefore wrong as stated — and "AP is still
climbing steeply at 18 epochs" is unsupported, since epoch 18 is exactly where
the cosine has annealed to `min_lr`.

**3. `use_atss` is not a single mechanism.** The non-ATSS path
(`weeddet_v6b.py:1083`) swaps adaptive per-GT thresholds for fixed 0.5/0.4 *and*
changes negative-band semantics. Even a well-powered A-vs-C result could not be
attributed to "ATSS" as one thing.

> One reviewer additionally claimed `_force_one_positive_per_gt` is bypassed in
> the non-ATSS path. **That is false** — line 1087 calls it in the fallback,
> commented "The fallback path must also guarantee per-GT coverage (audit
> P1-3)." Verified directly. Recorded because a confident, code-cited claim from
> review is still only as good as the check.

## The estimator is biased, and the bias points at the reported effect

Every headline number is `max(val/AP)` over logged epochs — max-of-4 at 8 epochs,
max-of-9 at 18. That is a **maximum of a noisy sequence**: upward biased, and the
bias is larger for noisier arms. `sd(C) = 0.01538` is **2.0x** `sd(A) = 0.00752`,
so C wins a maximum contest even with an identical true curve. Bounding the
arm-specific term gives a maximum differential selection bias of **0.00895 AP**
against an observed C−A of **0.00481**. **The entire ATSS "advantage" fits inside
the selection artifact.**

Fixing this is free: report AP at a pre-specified epoch, or the mean of the last
few evaluations. Doing so would have cut sd from ~0.0121 to ~0.0105 at zero
compute.

## Corrections to the statistics

- The **"65% detectable effect" derivation is invalid.** It used `2.5 *
  mean(sd) / mean` — averaging standard deviations instead of pooling variances,
  and an arbitrary multiplier. The correct **unpaired** MDE at n=3, 80% power is
  **84.8%**; the correct **paired** MDE is **68.0%**. The reported 65% was
  approximately right by coincidence, two errors cancelling. This report
  **understated** its own underpowering.
- Actual power of the test as run: **0.081** for a 10% effect, **0.170** for 20%.
- **No exact paired test at n=3 can reach p<0.05 for any data** — minimum
  attainable two-sided p is 2/2³ = 0.25. The design could not produce a
  distribution-free rejection even in principle.
- **"No support for keeping ATSS on either" is wrong.** ATSS-on is the committed
  incumbent; failing to reject does not put incumbent and challenger on equal
  footing. The 95% CI on the paired difference is **[−40.1%, +63.4%]** relative —
  uninformative, not null. Retain ATSS.
- **The B correction over-swung.** "Did not reproduce at all" replaced one
  unsupported claim with its mirror image. B's point estimate is **−33%** with a
  CI spanning roughly −100% to +30%: compatible with substantial harm. The right
  statement is that nothing is knowable from n=2.
- **Part 1 was near-certain to produce a spurious effect a priori.** Under a
  pure-noise null at CV ≈ 29%, P(seeing a ≥21% apparent effect in at least one of
  three arms) = **0.94**; for ≥51%, **0.52**. This kills the part-1 reading
  without needing part 2 at all.
- **Blocking on seed removes 75% of the variance** (r(A,C) = 0.946). The MDE
  above was computed unpaired, making the seed requirement ~3x more pessimistic
  than necessary: a 20% effect needs **11 seed-pairs**, not 35 per arm.

## The finding that reframes the resolution question

**512 px downsamples 91.8% of this dataset below its native resolution.**
Verified from `instances_train.coco.json`: 1,652 of 1,800 train images have a
long side of exactly 640 (885 at 640x640, 718 at 640x480, 49 at 640x312).

| letterbox | COCO-small share of boxes |
| --- | ---: |
| **512 (current)** | **56.1%** |
| 640 (native) | 37.7% |
| 768 | 23.7% |

The small-object problem is substantially **self-inflicted by the resize
target**. The killed 768 arm was chasing *upsampled* pixels on 92% of the data —
interpolation, not information. The real resolution question was never 512 vs
768; it is **512 vs 640**, which is not a tuning knob but "stop discarding the
data you already have."

Note also that the "37.8% COCO-small" figure quoted throughout this project is
the *evaluator's* view (areas in original coordinates). The network at 512 px is
actually seeing 56.1% small. Both numbers are correct; the gap between them is
the letterbox.

## Two more things measured, both contradicting earlier framing

- **`weed_target` is the *larger* class**, not the harder-because-smaller one:
  median box area 1,622 px² vs `rice_protect` 1,325, and 28.0% COCO-small vs
  39.6%. Class imbalance and small-object difficulty are **anti-correlated**
  here, so the minority class is not doubly penalised.
- **The EMA was effectively switched off at these budgets.** `ModelEMA`
  (`weeddet_v6b.py:1722`) ramps as `decay * (1 - exp(-updates/2000))`. At 8
  epochs (1,800 steps) the effective decay is **0.593** — a ~2.5-step averaging
  window, against a config advertising 0.999. Verified. This is a mechanical,
  GPU-free explanation for a large share of the variance the seed replication was
  built to characterise, and it means the 8- and 18-epoch runs were not even
  scored by the same estimator.

## What was collected and thrown away

`val/AP50`, `val/AP75`, `val/AP_small`, `val/AR100` and per-class AP are written
to `metrics.jsonl` on every evaluation (`weeddet_v6b.py:2482-2483`). This report
quoted only aggregate `val/AP` — including in the section naming small objects as
the question that matters. That also violates CLAUDE.md §13.2, which requires
per-class and threshold-resolved metrics.

## Standing conclusion

The original verdict — *this ablation cannot answer the questions it was asked* —
**survives**, but for stronger and different reasons than it gave: the arms did
not test the right values, the estimator is biased toward the reported effect,
two of the comparisons were confounded, and the design was a priori near-certain
to manufacture a false positive. Fix the estimator and the confounds before
buying a single additional seed.

---

# Adversarial audit — third reviewer

Verified against the artefacts rather than the narrative. It **confirmed the
experiment was executed correctly** and found the defects to be in the writing
around the numbers, not the numbers.

## Confirmed correct (verified, not asserted)

- **The design is genuinely one-variable**, established by diffing every
  `config_*.yaml` against the committed base and against each other. Each arm
  differs by exactly its intended key; nothing drifted.
- **Stronger than this report claimed:** after `set_seed(42)`, constructing the
  model under all three arm settings yields **bit-identical initial weights**
  (identical SHA-256 across all 368 state-dict tensors). For a given seed the
  arms share initialisation, data order and augmentation stream — a genuinely
  paired design.
- **The sealed test split was not touched.** No `instances_test` / `images/test`
  string in any driver, config, log or result. Test image files were last read
  2026-09-01, two days before the ablation window.
- **Every AP recomputes exactly** from `metrics.jsonl`, as do all means, sds,
  paired differences and percentages. All seven part-2 runs exited 0 first try.
- **Determinism is stronger than claimed:** the pre-fix and post-fix runs at seed
  42 agree on **60 of 62 metric keys bit-identically**; the only two that differ
  are wall-clock timings.
- **Seeding is complete**, covering Python, NumPy, torch and CUDA before model
  construction; per-seed epoch-1 losses genuinely differ (3.8337 / 3.6583 /
  3.8103) while same-seed cross-process runs are bit-identical.

## Factual errors in this report

**1. "`667a9b4` raised `anchor_base_scale` without a measurement" is false.**
That commit shipped an anchor-assignment audit over 2,034 phase-2 train boxes at
base 3.0/4.0/4.5, four committed `reports/metrics/anchor_audit_rebuild_*.json`
files, and ADR 0004. What it lacks is an **end-to-end AP** measurement, which is
a different and much weaker claim. This report asserted the stronger version
repeatedly and built a standing risk item on it.

**2. "AP is still climbing steeply at 18 epochs" is contradicted by the runs.**
In **both** 18-epoch runs the peak is **epoch 16**, and epoch 18 is lower:

| run | ep14 | ep16 | ep18 | lr@18 |
| --- | ---: | ---: | ---: | ---: |
| `A_baseline_e18` | 0.0575 | **0.0661** | 0.0643 | 1.0e-5 |
| `C_no_atss_e18` | 0.0726 | **0.0758** | 0.0735 | 1.0e-5 |

The numbers reported as "18-epoch" are epoch-16 values, and AP was declining by
epoch 18 with the cosine annealed to `min_lr`.

**3. "`D_img768` stopped being a one-variable arm" was a rationalisation.**
Its config differs from the baseline arm in exactly one key (`img_size`). Memory
pressure is a *consequence* of the manipulated variable, not a second variable,
and an 8-epoch 768px AP would still have been attributable to `img_size`. The
honest ground for the kill is the one also stated in that section: ~10 hours
against seven queued runs. The confounding argument applies to a hypothetical
future run that lowers batch size, not to this one.

**4. The 11,860 MiB memory figure is unretained.** No memory line, OOM or
allocator warning appears in `D_img768.log`, and no `nvidia-smi` capture was
saved. The observation was real but made out-of-band and not kept, so it is not
reproducible from the artefacts. Treat it as an unretained observation, not a
measurement.

**5. "37.8% COCO-small" is a whole-corpus figure and includes the sealed test
split.** Train-only is **38.2%**; train+valid **38.1%**. The difference is
immaterial to the argument, but this report states "the test split is never
referenced" absolutely, and then quotes a number computed over it.

## Two silent failures in the drivers

Both are `.get()`-returns-`None` failures of the kind CLAUDE.md §3.4 forbids:

- `avg_loss` was read, but the JSONL key is `train/total_loss` — so `final_loss`
  is `null` in **all 11** result records and no loss curve was ever captured.
- `val/AP-small` was read (hyphen); the key is `val/AP_small` (underscore) — so
  **the small-object AP was dropped from every record**, in an experiment whose
  stated motivation is small objects.

## The data that was collected and discarded

Recovered from `metrics.jsonl`. This is what aggregate AP was hiding:

| run | AP | AP50 | AP75 | AP_small | AR100 | rice | **weed** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_baseline` | 0.0391 | 0.1608 | 0.0074 | 0.0685 | 0.1548 | 0.0563 | **0.0219** |
| `B_anchor2` | 0.0191 | 0.0851 | 0.0031 | 0.0278 | 0.0942 | 0.0357 | **0.0026** |
| `C_no_atss` | 0.0473 | 0.1916 | 0.0080 | 0.0710 | 0.1396 | 0.0736 | **0.0211** |
| `A_baseline_s43` | 0.0353 | 0.1524 | 0.0070 | 0.0611 | 0.1012 | 0.0477 | **0.0229** |
| `C_no_atss_s43` | 0.0303 | 0.1453 | 0.0019 | 0.0421 | 0.0972 | 0.0420 | **0.0187** |
| `A_baseline_s44` | 0.0498 | 0.1963 | 0.0099 | 0.0708 | 0.1523 | 0.0812 | **0.0183** |
| `C_no_atss_s44` | 0.0610 | 0.2407 | 0.0077 | 0.0701 | 0.1570 | 0.0994 | **0.0226** |
| `B_anchor2_s43` | 0.0362 | 0.1607 | 0.0074 | 0.0377 | 0.1079 | 0.0500 | **0.0224** |
| `A_baseline_e18` | 0.0661 | 0.2572 | 0.0113 | 0.0718 | 0.2072 | 0.0996 | **0.0326** |
| `C_no_atss_e18` | 0.0758 | 0.2868 | 0.0157 | 0.0830 | 0.1903 | 0.1205 | **0.0311** |

Three findings that aggregate AP concealed, all more important than anything the
ablation was designed to test:

1. **Weed detection is 3-5x worse than rice, everywhere.** `weed_target` AP sits
   at 0.018-0.033 against `rice_protect` 0.036-0.121, in every arm and at every
   budget. The decision-relevant class is the one that barely works.
2. **Localisation is the dominant failure, not detection.** AP75 is 0.002-0.016
   against AP50 of 0.085-0.287 — roughly a 20:1 ratio. The model finds objects
   approximately and places boxes badly.
3. **On `AP_small`, the ATSS "advantage" reverses.** Paired by seed:
   **+0.0024, -0.0190, -0.0007** (mean -0.0058). On the metric this dataset is
   actually about, turning ATSS off is if anything worse, and the sign flips
   again.

## Provenance gap

`status.json` records none of: git commit, dirty flag, config hash, data version,
framework version, or seed. And the runs executed against a **dirty working
tree** — `weeddet_v6b.py` carried the uncommitted encoding fix throughout. The
code state of these runs therefore exists only in an uncommitted working tree.
CLAUDE.md §11.2 and §24 both require this metadata. It should be recorded before
any further run, and the encoding fix committed so the runs have a citable commit.

## Correction to the encoding fix itself

The audit found the `LookupError` branch was dead code that re-raised: it retried
`message.encode(encoding, ...)` with the same codec name that had just raised
`LookupError`. Fixed — that branch now falls back to ASCII, and the
`UnicodeEncodeError` branch uses `backslashreplace` rather than `replace`, so a
degraded character stays recoverable (`\u2605`) instead of collapsing to `?` in a
line that carries checkpoint paths. Verified against stub streams for `utf-8`
(unchanged passthrough), `cp1252`, `ascii`, and an unknown codec name.

Also noted: both drivers set `PYTHONIOENCODING=utf-8`, so the guard was never
exercised during the ablation, and the env var alone would have prevented the
original crash. The source fix is still the right one — it does not depend on a
caller remembering to set an environment variable.
