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
| Actual | *(fill in)* |
| Verdict | *(fill in)* |

---

## Queue (do NOT start until T1 has a verdict)

- **T2** — ATSS per-location top-k (fix co-located-anchor distance ties; candidates = 9 nearest *cells*
  per level, not 9 anchors of the nearest cell). Risks: changes positive counts/statistics → loss scale
  and lr sensitivity; interacts with atss_topk=9 vs 12 anchors/location.
- **T3** — Scratch control, full protocol (the paper baseline). No code change; must run under
  whatever loss config T1/T2 settle on, same for all conditions.
- **T4** — riceseg vs scratch A/B with settled config (the actual research question).
- **T5** — `missing`-keys assert on backbone load (safety, no behaviour change).
- **T6** — imagenet ablation; **T7** — test-split eval ONCE; **T8** — YOLOv8s baseline.

## Log hygiene

- After each test: append Actual + Verdict here, add the metrics row to
  `RESEARCH_PLAN_DETECTION_ACCURACY.md` Part C, and note config drift if any.
- Checkpoint names carry the test ID. Never reuse `weeddet_v7_riceseg_best.pth` unsuffixed.
