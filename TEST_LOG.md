# TEST_LOG — One change at a time

**Rule:** exactly ONE change per test. Before running, fill in the *Interaction risks* row —
what else in the system this change touches and how it could break. After running, fill in
*Actual* and *Verdict*, then pick the next test. Never stack untested changes.

**Fixed protocol for all tests** (unless the test IS a protocol change): COCO split 1079/134/134
seed 42 · 1-class rice · IMG 512 · anchor_base_scale 3 · LSC k=7 · ATSS · `cls_hard_target=True` ·
eval score_thr 0.01, hard NMS 0.5, maxDets 300 · selection = val AP@50. Tag every checkpoint with
the test ID (e.g. `weeddet_v7_riceseg_T1_best.pth`).

---

## Baseline entries (historical, pre-log)

### T0.1 — V7 riceseg run #1 (2026-07-08)
| | |
|---|---|
| Change | First full run on COCO split, riceseg init, soft VFL targets (anchor-GT IoU) |
| Expected | ≥ scratch-era performance |
| Actual | val AP@50 **0.0088** (peak ep 20, decays); scores capped 0.05–0.21 |
| Verdict | ❌ FAIL — probe showed boxes OK (median best-IoU 0.69) but confidence capped → VFL target bug |

### T0.2 — Overfit-16 gate with VFL hard-target patch (2026-07-08)
| | |
|---|---|
| Change | Monkey-patched VFL: positives → target 1.0. Scratch init, lr 0.005, batch 4, no EMA/AMP/clip, augment=False, eval on same 16 train imgs |
| Expected | AP@50 climbs high if pipeline OK |
| Actual | **0.60** by ep 70 |
| Verdict | ✅ pipeline can learn. ⚠️ Caveat learned later: overfit-on-train can't detect generalization-only bugs (ignore band) |

### T0.3 — V7 riceseg run #2, hard fix baked in (2026-07-09)
| | |
|---|---|
| Change | `cls_hard_target=True` in weeddet_v6b.py; same full-run config |
| Expected | ≥ 0.05 val AP@50 |
| Actual | ≤0.0010 → 0.0000; det/img frozen ~239.6; loss 5.08 → 2.1 plateau. Killed ~ep 61 |
| Verdict | ❌ FAIL — fix necessary but not sufficient |

### T0.4 — Raw vs EMA eval + prediction visualization (2026-07-09)
| | |
|---|---|
| Change | none (diagnostic) |
| Actual | raw 0.0003 / EMA 0.0010 — both dead. Visuals: FPs = unregressed ar-0.2 stride-4 anchors at score exactly 1.00, on plant-like texture |
| Verdict | EMA **exonerated**. Signature = saturated cls + untrained reg on non-positive anchors → ignore-band leak (see AUDIT_2026-07-09_V7.md §2) |

---

## T1 — Remove ATSS classification ignore band  ⟵ CURRENT

| | |
|---|---|
| Status | **IMPLEMENTED 2026-07-09, not yet run** |
| Change | `models/weeddet_v6b.py` `WeedDetLoss._assign_atss`: `neg = ~pos` (all non-positives are negatives, ATSS-paper semantics). Flag-gated: `self.atss_all_neg = True`; set `False` to reproduce old behaviour |
| Hypothesis | Ignore-band anchors (IoU ≥ 0.4, not pos) had zero cls gradient and saturated to 1.0 under hard targets → FP floods. Making them negatives restores suppression |
| Expected | Overfit-16: ~unchanged (≥0.5). Full run: val AP@50 clearly > 0 (0.05–0.2 band plausible); det/img well below 200 and FLUCTUATING between evals; scores no longer pinned at 1.00 |
| Abort criteria | AP still <0.005 by ep 20, or det/img frozen again → T1 insufficient, escalate to T2 without re-running 100 ep |

**Interaction risks (filled before run):**

1. **Loss magnitude ↑** — thousands of former-ignore anchors now contribute negative VFL (α·p²·BCE).
   Watch epoch-1 loss (expect > 5.08). Interacts with **GRAD_CLIP 0.5**: a bigger loss clipped to the
   same norm shrinks the effective step on everything else. If training stalls (loss flat by ep 10),
   test clip separately — do NOT bundle a clip change into T1.
2. **Borderline true positives suppressed** — anchors at IoU 0.4–0.5 that used to be safely ignored
   are now pushed toward 0 while their neighbour is pushed to 1.0. With dense overlapping plants this
   sharpens the decision boundary; possible mild recall drop at AP@50, likely AP@75 gain. Acceptable.
3. **AMP**: more saturated-sigmoid negatives early → BCE on fp16 logits; `binary_cross_entropy_with_logits`
   is autocast-safe. Low risk, but if NaN loss appears, rerun with USE_AMP=False before blaming T1.
4. **EMA**: no interaction (weight-space average; exonerated in T0.4). Keep ON so results stay comparable.
5. **Checkpoints**: no parameter shapes change → old checkpoints still load. But metrics are NOT
   comparable to T0.x numbers except as before/after for this exact bug.
6. **Notebook Cell 1 asserts**: unaffected — `CocoWeedDataset` and `'quality_iou[pos_idx].clamp'`
   (still present in forward) both still pass. Optional stronger gate: `assert wd.WeedDetLoss().atss_all_neg`.
7. **Non-ATSS fallback path** (use_atss=False) intentionally UNCHANGED — still has the classic
   RetinaNet ignore band. Only the ATSS branch changed.
8. **riceseg_pretrain.py**: no interaction (segmentation loss, separate).
9. **Deployment step**: Colab imports from Drive — **re-upload `models/weeddet_v6b.py` to
   `MyDrive/weeddet_v2_checkpoints/` (overwrite)** or the run silently uses the old loss.

| Run order | 1) overfit-16 (sanity, ~10 min) → 2) 20-ep riceseg full → 3) if moving, let it run 100 ep |
| Actual | 2026-07-09, 20 ep: best AP@50 **0.0017** (ep 6) → 0.0002 by ep 12+. Loss 4.55 → **2.43 plateau**. det/img: 299.8 (ep 2–10) → **239.6 frozen** (ep 14–20). Cells 10/11 (raw-vs-EMA, visuals) not run |
| Verdict | ❌ **FAIL per abort criteria** — but informative. The 239.6 det/img is the *collapse-to-prior* fingerprint (identical value in run #2 ep 12+): an init-like classifier at p≈0.01 against score_thr 0.01 yields a deterministic detection count. Dynamics are two-phase: early saturation up (299.8 phase) → negative flood crushes ALL scores back toward prior, dragging positives with them (loss plateau 2.43 ≈ positives stuck at p≈0.05–0.1). T1 (all-neg) is principled — KEEP the flag — but the failure is optimization balance, not supervision coverage alone. The band was masking, not causing, the deeper issue |

---

## T2 — Remove/raise gradient clipping  ⟵ CURRENT

| | |
|---|---|
| Status | PLANNED |
| Change | Notebook Cell 3 only: `GRAD_CLIP = None` (skip `clip_grad_norm_` when None — guard already trivial to add) or `10.0`. NO model-file change |
| Hypothesis | Clip 0.5 rescales the whole gradient; with ~258K negatives dominating the norm, the positives' share of every step is minuscule. Positives never climb before cosine LR decays → collapse to prior. The only run that ever learned (overfit-16, AP 0.60) had **no clip** (plus lr 5×). Clip is the most binding difference |
| Expected | Loss falls **below 2.0** within 10 ep (positives escaping p≈0.1); det/img NOT frozen at 239.6; AP@50 > 0.01 by ep 20. If loss diverges/NaNs → clip was load-bearing, retry with 10.0 then lr 0.0005 |
| Abort | loss plateau ≥2.3 by ep 10, or 239.6 fingerprint returns → clip exonerated, escalate to T3 (lr 0.005) / T4 (ATSS per-location topk) |
| Actual | 2026-07-09, 20 ep, GRAD_CLIP=10.0: **AP@50 0.0412 / AP@75 0.0102 at ep 4** (new best on clean split, 24× T1), then decays: 0.0102 (ep6) → 0.0012 (ep8) → 0.0001 (ep20) while train loss falls 3.36 → 2.02. det/img pinned 300.0 throughout. Raw vs EMA at best ckpt: raw **0.0010** vs EMA **0.0412** — raw weights oscillate hard, EMA is doing the detecting. Size breakdown at ep4: AP small 0.022 / medium 0.016 / **large 0.000**; AR large **0.000**. Visuals: scores ~0.65 (not saturated), preds sized ≤ anchor ceiling, cluster on distant small plants + plant bases; big foreground GT untouched |
| Verdict | ✅/❌ **PARTIAL — clip was a real blocker (keep 10.0), not the root cause.** Learning now happens but the objective degrades val AP after ep 4 (train loss ↓, AP ↓): with anchors capped at 76 px, large/medium GT are unreachable, so continued optimization only suppresses scores. Large-object AP/AR = 0.000 directly confirms the T4 anchor-scale diagnosis. → run T4 next, keep GRAD_CLIP=10.0 |

**Interaction risks:**

1. **AMP + unclipped large early gradients** → possible inf/overflow; GradScaler will skip those steps
   (watch for repeated `scaler` skips = loss stalling at start). The in-loop `torch.isfinite(loss)` assert catches NaN loss.
2. **EMA** smooths over any instability — fine, keep ON.
3. **BN (batch 2)** — larger steps + noisy BN stats can wobble early epochs; judge at ep 10+, not ep 2.
4. Warmup (5%) already softens the first 540 steps — that mitigates most of the no-clip blowup risk.
5. No checkpoint/shape implications; T1 flag stays active; protocol otherwise identical.
6. **Run cells 10 & 11 this time** — raw-vs-EMA + visuals are the discriminating diagnostics if it fails again.

| Actual | *(fill in)* |
| Verdict | *(fill in)* |

---

## T4 — Anchor-scale realignment  ⟵ CURRENT (keep GRAD_CLIP=10.0 from T2)

| | |
|---|---|
| Status | PLANNED — next run |
| Change | Cell 3 only: `ANCHOR_BASE_SCALE = 6` (v4-era value) and `RUN_TAG = 'T4'`. Keeps clip 10.0. One knob |
| Why 6 | anchors become 11–152 px (stride bases 24/48/96). Recomputed best-centered IoUs: 25×60 → 0.51, 40×90 → 0.64, 100×200 → **0.73** (was 0.29). Every GT size band becomes matchable |
| Expected | AP large > 0 for the first time; AP@50 well above 0.04; the ep-4-peak-then-decay pattern weakens or shifts later; preds visibly plant-sized in Cell 11 |
| Abort | AP@50 < 0.04 by ep 20 or large-AP still 0.000 → anchor scale exonerated at this value; escalate to per-level scales or T5 (ATSS per-location topk) |

**Interaction risks:** regression head re-calibrates to new anchor sizes → FRESH run only (done — no resume);
ATSS candidate IoU statistics shift wholesale (adaptive threshold recomputes — that's the point);
stride-4 level becomes useful for the smallest plants instead of pure noise; det/img may drop a lot
(fewer junk candidates above 0.01) — don't compare raw det/img to T1/T2; NMS interactions unchanged;
run BEFORE the GT-size stats cell conclusions if percentiles disagree strongly with 6, note it and
adjust in a T4b. Notebook's new Cell 4b prints GT width/height percentiles — paste them into Actual.

| Actual | 2026-07-09, 20 ep (run tagged 'T3' in Colab — config verified = this test: base_scale 6, clip 10, lr 0.001). **AP@50 0.1327 / AP@75 0.0061 @ ep 4** (3.2× T2; ≈ the retired 0.166 leaky baseline, now on the clean split). Pattern: peak ep 4 → dip to 0.022 (ep 8, lr≈0.0007) → recovery to 0.0465–0.0486 as lr decays → plateau. Size: AP small 0.049 / med 0.028 / **large still 0.000**; AR med 0.014→**0.174** (12×). Raw 0.0033 vs EMA 0.1327 — raw weights still thrashing. Visuals: plant-sized boxes overlapping GT, still 300 det/img flood, localization coarse (AP@75 ~0). Cell 4b GT percentiles not run (older notebook copy) |
| Verdict | ✅ **SUCCESS — anchors were a primary ceiling; keep ANCHOR_BASE_SCALE=6.** Remaining limiters, ranked: (1) LR instability — best model is the ep-4 EMA transient; training at lr>~0.0006 actively destroys it and the cosine-tail recovery never regains the peak → T5 = BASE_LR 0.0005. (2) AP-large exactly 0.000 despite coverage now existing (IoU ceiling 0.73) → consistent with ATSS tie-break excluding ar≥0.5–1.0 candidates → T6. (3) AP@75 ≈ 0 — regression quality; revisit after T5/T6 |

---

## T5 — BASE_LR 0.001 → 0.0005  ⟵ CURRENT (keep clip 10.0 + base_scale 6)

| | |
|---|---|
| Status | PLANNED — next run |
| Change | Cell 3 only: `BASE_LR = 0.0005`, `RUN_TAG = 'T5'`. (MIN_LR stays 1e-5) |
| Hypothesis | Every run peaks early then degrades while lr > ~0.0006; raw-vs-EMA gap (0.0033 vs 0.1327) shows weight thrashing. Halving lr should keep the model near its transient optimum instead of destroying it |
| Expected | No mid-run crash; AP@50 climbs steadily; final ≥ 0.13 (i.e., ≥ the T4 transient) with a shrinking raw-vs-EMA gap. If the curve is still climbing at ep 20 → extend to 100 ep same config (T5b, not a new test) |
| Abort | peak-then-collapse recurs with peak < 0.10 → lr exonerated; go to T6 |
| Interaction risks | slower convergence → judge at ep 20, not ep 8; warmup shortens in absolute lr terms (same ratio); EMA lag matters less at lower lr; nothing else touched |
| Actual | 2026-07-09, 20 ep, config verified (lr 0.0005, clip 10, anchors 6). **AP@50 0.1406 / AP@75 0.0123 @ ep 4** — new best; AP@75 2× T4. But decay persists and is now MONOTONIC: 0.1406 → 0.0237 by ep 20 with NO cosine-tail recovery, even at lr 1e-5. Train loss ↓ steadily (2.78→2.40) while val AP ↓ 6×. Raw vs EMA: 0.0181 vs 0.1406 (gap narrower than T4's 40× but still 8×). AP large still 0.000. Cell 4b percentiles still missing (older copy again) |
| Verdict | ⚠️ **PARTIAL — lr helps the peak (+6%) but is EXONERATED as the root cause of the decay.** Collapse happens at every lr from 9e-4 down to 1e-5 → this is genuine train/val divergence, not step-size instability. Leading hypothesis now: **catastrophic forgetting of the riceseg backbone** — ep-2 AP already 0.100 (very fast = good pretrained features + quick head adaptation), then joint full-lr training destroys those features; EMA preserves a lagged copy of the early good state, raw model degrades. Discriminating test: **scratch control (T6)** — if scratch climbs slowly with NO early peak, forgetting is confirmed → remedy is backbone freeze / 0.1× backbone lr (T7). Scratch is also the mandatory paper baseline, zero code change |

---

## T6 — Scratch control (identical config)  ⟵ CURRENT

| | |
|---|---|
| Status | PLANNED — next run |
| Change | Cell 3 only: `BACKBONE_INIT = 'scratch'`, `RUN_TAG = 'T6'`. Everything else identical to T5 (lr 0.0005, clip 10, anchors 6, 20 ep) |
| Purpose | (a) The paper's real baseline on the clean split. (b) Discriminates the forgetting hypothesis: scratch has nothing to forget |
| Expected if forgetting is real | slow monotonic climb, no ep-4 peak, final maybe 0.05–0.12 and still rising at ep 20; smaller raw-vs-EMA gap |
| Expected if riceseg is irrelevant | same peak-then-decay shape as T5 → decay mechanism is generic (overfitting/objective drift) → investigate augmentation, per-epoch val of raw model, ATSS (T-queue) |
| Interaction risks | none — config-only. Do NOT compare absolute peak to T5's 0.1406 until both have 100-ep runs |
| Actual | 2026-07-09, 20 ep, config verified (scratch, lr 5e-4, clip 10, anchors 6). **Slow climb, NO early peak**: 0.0015 (ep2) → 0.0319 (ep6) → **peak 0.0450 (ep8)** → oscillates ~0.04 → mild late decline to 0.0246 (ep20). Raw vs EMA: 0.0173 vs 0.0450 (2.6× gap vs riceseg's 8–40×). AP large 0.000 (as expected — generic). Older notebook copy again: no Cell 4b, Cell 11 crash repeats |
| Verdict | ✅ **FORGETTING CONFIRMED.** Scratch (nothing to forget) shows the *opposite* dynamics of riceseg: gradual climb peaking at ep 8 vs riceseg's ep-4 spike-and-collapse. AND the research answer previews: riceseg's transient peak (0.1406) is **3.1× scratch's best (0.0450)** — the pretrained features are genuinely valuable; the job is to stop training from destroying them. Scratch's own mild late decline (0.044→0.025) shows a smaller generic component (overfitting/objective drift) — revisit after T7/T8. **Scratch baseline for the paper: 0.0450 @ 20 ep** (needs a 100-ep number later). → T7 (freeze backbone) is GO |

---

## Queue

*(Forgetting-defense plan + citations: see `RESEARCH_FORGETTING_DEFENSES.md`. The notebook in active/
already has a `FREEZE_BACKBONE` flag wired into Cells 3/6/7 — T7 is a two-line config flip.)*

- **T7 — DONE 2026-07-14** — LP-FT phase A: frozen riceseg backbone (57 BN → eval, 2.88M/26.38M
  trainable), 20 ep.
  **Actual:** 0.1005 (ep2) → dip 0.062 (ep6) → **steady monotonic climb to 0.1040 (ep18)**, holds
  0.1035 at ep20; AP@75 RISING at the end (0.0116 ep20 — best final-epoch AP@75 of any run).
  **Raw vs EMA: 0.0985 vs 0.1040 — the instability gap is GONE** (was 8–40×); raw AP@75 0.0115.
  Final-epoch AP@50 is 4× every previous run's final. Cell 4b GT percentiles (finally):
  width p50=20.8/p95=54.5, height p50=54.4/p95=109.6, sqrt(area) p50=34.1/p95=70.2.
  → **COCO-'large' GT is nearly absent (p95 sqrt-area 70 < 96) — AP-large 0.000 is a sample-size
  artifact, NOT a bug. Demotes the ATSS-large concern and T4b.** Anchors (11–341 px) comfortably
  cover p5–p95.
  **Verdict: ✅ SUCCESS — forgetting defense works exactly as LP-FT/TFA predict. Stable, converged,
  still slightly rising at ep 20. FREEZE_BACKBONE=True joins the settled config for riceseg runs.**
  Remaining gaps: FP flood persists (det/img pinned 300, saturated scores on canopy texture —
  precision problem), foreground/tail plants still missed. → T7b harvest first, then T8.

## T7b — Same config, 100 epochs  ⟵ CURRENT

| | |
|---|---|
| Status | PLANNED — next run (not a new variable; longer schedule of T7) |
| Change | `NUM_EPOCHS = 100`, `RUN_TAG = 'T7b'`. Nothing else |
| Why | T7's curve was still rising at ep 20 with lr already at 1e-5; a 100-ep cosine spends far longer at productive lr. AP@75 trend (0.0004 → 0.0116 over ep 12–20) suggests localization is still improving |
| Expected | AP@50 ≥ 0.14 (beat the T5 transient with a STABLE model); AP@75 ≥ 0.03 |
| Abort | plateau < 0.11 by ep 40 → head capacity is the ceiling; stop early, go to T8 (unfreeze @ 0.1× backbone lr from best T7b weights) |
| Actual | 2026-07-14, killed ~ep 44 (abort fired: 0.015 at ep 40). Climbed during warmup (0.053→0.086, lr 1e-4→4.8e-4), degraded exactly when lr reached & HELD ~4.5–5e-4 (ep 18+ → 0.015 plateau); train loss still falling. Backbone frozen → NOT forgetting: the 2.88M-param head itself overfits/drifts at sustained lr > ~2e-4 |
| Verdict | ❌ **FAIL — but it isolates the LR band cleanly.** Re-reading all runs: every productive phase (T7's entire climb to 0.104; T7b's warmup climb) happened at **lr ≤ ~2e-4**; every degradation at lr above it. T7 succeeded *because* its 20-ep cosine exited the destructive band quickly. Rule adopted: head trains productively only below ~2e-4 → T7c |

## T7c — 100 ep inside the productive LR band  ⟵ CURRENT

| | |
|---|---|
| Status | PLANNED |
| Change vs T7b | `BASE_LR = 0.00015`, `RUN_TAG = 'T7c'`. One variable. (Keep 100 ep, FREEZE_BACKBONE=True, everything else) |
| Expected | slow monotonic climb past T7's 0.104; target ≥ 0.14 stable; AP@75 ≥ 0.03 |
| Abort | < 0.09 by ep 40 → schedule exonerated; head capacity/objective is the ceiling → T8 (unfreeze @ 0.1× backbone lr) or precision work (FP flood) |
| Actual | 2026-07-14, killed ep 13. Peak 0.0758 (ep 4, during warmup), collapse to 0.005 by ep 12 at SUSTAINED lr 1.5e-4 |
| Verdict | ❌ FAIL — **falsifies the static LR-band rule from T7b.** Combined read of T7/T7b/T7c: the level doesn't matter, the SHAPE does. T7's entire climb happened while lr actively decayed 2e-4→1e-5 (ep 10–18); both long runs died in their flat phases at different lr levels. Conclusion: the 2.88M head reaches its optimum in ~8–15 effective epochs on 1,079 images — **a fast cosine anneal is early stopping in disguise; long flat schedules overshoot the optimum and memorize.** → T7d: modest anneal extension, not a stretch |

## T7d — T7's recipe, 32-epoch anneal  ⟵ CURRENT

| | |
|---|---|
| Status | PLANNED |
| Change vs T7 | `NUM_EPOCHS = 32` (was 20), `BASE_LR = 0.0005` (T7's value), `RUN_TAG = 'T7d'`. Same anneal shape, ~60% more time in the productive decay phase |
| Expected | stable finish ≥ 0.11, AP@75 > 0.012. If ≤ T7 (0.104) → schedule question CLOSED: 20-ep T7 is the optimum; move to precision work / T8 |
| Abort | none needed — 32 ep is cheap; let it finish |
| Actual | *(fill in)* |
| Verdict | *(fill in)* |
- **T8** — LP-FT phase B: unfreeze backbone at lr 0.1× via param groups, BN stays frozen; init from
  T7 best raw weights (deliberate exception to the fresh-run rule — log it).
- **T8b** — WiSE-FT sweep (eval-only, free): interpolate backbone weights toward riceseg init,
  α ∈ {0.25, 0.5, 0.75}, pick best on valid.
- **T9** — feature-space anchoring to frozen riceseg teacher (L2 on C3/C4/C5), reserve if T7/T8
  confirm forgetting but can't hold features.

- ~~T4 original entry below (now CURRENT above)~~ **Anchor-scale realignment** *(quantified 2026-07-09, structural audit pass 2)*.
  With `anchor_base_scale=3`, anchors span 5–76 px vs plants ~25–250 px. Measured best-possible
  centered IoU: stride-4 level ≤0.24 for small plants, ≤0.10 for medium — i.e. **76% of all anchors
  (196,608 stride-4 slivers) can never be positives**: pure negative mass, regression never trained
  → they ARE the 5–13 px sliver FPs in the run-#2 visuals. Large plants (100×200) max IoU **0.29 at
  ANY level** — foreground plants are unmatchable, which is why visuals show zero detections on them.
  Candidate fixes (pick ONE): per-level base scales (e.g. 6/5/4), raise base_scale to 5–6 (v4-era
  value), or drop the stride-4 level. Interactions: changes head anchor geometry only via
  AnchorGenerator args (no weight shapes — `num_anchors` stays 12), but any trained checkpoint's
  REGRESSION head is calibrated to old anchor sizes → fresh run only; ATSS candidate statistics
  shift wholesale; det/img and AP not comparable to earlier tests; eval decode unaffected (same
  anchors object).
- **T5** — ATSS per-location top-k (fix co-located-anchor distance ties; candidates = 9 nearest *cells*
  per level, not 9 anchors of the nearest cell). Also fixes the tie-break systematically excluding
  ar=1.0 anchors from candidacy (first-9-of-12 ordering). Risks: changes positive counts/statistics →
  loss scale and lr sensitivity; interacts with atss_topk=9 vs 12 anchors/location.
- **T6** — Scratch control, full protocol (the paper baseline). No code change; must run under
  whatever config T2–T5 settle on, same for all conditions.
- **T7** — riceseg vs scratch A/B with settled config (the actual research question).
- **T8** — imagenet ablation; **T9** — test-split eval ONCE; **T10** — YOLOv8s baseline.

## Cleared suspects (structural audit, pass 2 — 2026-07-09)

Verified CORRECT, stop re-investigating: backbone strides 4/8/16 match anchor strides (stem s2 +
layer1 s1 + layer2/3/4 s2 each); head-flatten order (H,W,A) matches anchor generation order in both
loss and decode; encode/decode delta math symmetric; letterbox/unpad round-trip (probe: median
best-IoU 0.69); CocoWeedDataset 5-column label alignment through augmentation; EMA update (raw≈EMA
empirically); riceseg transfer load (342/342 backbone tensors, keys identical by construction).

## Log hygiene

- After each test: append Actual + Verdict here, add the metrics row to
  `RESEARCH_PLAN_DETECTION_ACCURACY.md` Part C, and note config drift if any.
- Checkpoint names carry the test ID. Never reuse `weeddet_v7_riceseg_best.pth` unsuffixed.
