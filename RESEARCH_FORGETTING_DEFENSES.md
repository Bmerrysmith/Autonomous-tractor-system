# Research note — Defenses against backbone forgetting for AgriNav V7 (2026-07-09)

**Question:** best literature-backed method to stop the riceseg backbone from being destroyed during
detection fine-tuning (symptom: EMA peak at ep ~4, monotonic val-AP decay at every lr, raw model dead).

**Case profile:** in-domain segmentation-pretrained ResNet-50 backbone (fills 100% incl. stem);
random neck+head; 1,079 train images; batch size 2 (BN-hostile); one-stage anchor detector; EMA on.

## What the literature says, mapped to this case

1. **LP-FT — train the head first, then fine-tune** (Kumar et al., ICLR 2022, arXiv:2202.10054).
   Proves the mechanism we observe: with a random head, full fine-tuning distorts pretrained features
   while the head learns; linear-probe-then-fine-tune beats both extremes (≈+10% OOD). Our ep-2
   AP=0.10 spike is the "probe phase" happening by accident and then being destroyed.
2. **TFA — the detection-specific precedent** (Wang et al., ICML 2020, arXiv:2003.06957). In
   low-data detection, fine-tuning ONLY the last layers on a frozen feature extractor beat all
   meta-learning SOTA by 2–20 points. Direct precedent that frozen-backbone detection works and wins
   in small-data regimes.
3. **Mind the Backbone** (Saito et al., 2023, arXiv:2303.14744). Detection-specific: backbone
   feature distortion during fine-tuning tracks Relative Gradient Norm; recipes = regularization/
   architecture choices that minimize backbone gradient updates. Supports backbone lr ≈ 0.1× head.
4. **Frozen BN** (standard detection practice; Keras/TF transfer-learning guidance). At batch 2, BN
   running-stat drift corrupts features fastest; riceseg BN stats came from batch 12 and are
   in-domain → freeze backbone BN (eval mode) during fine-tuning. Safe here ONLY because riceseg
   fills the whole backbone (the old F1+F2 failure was ImageNet BN behind a random stem).
5. **WiSE-FT — weight-space interpolation** (Wortsman et al., CVPR 2022, arXiv:2109.01903).
   Post-hoc: interpolate fine-tuned backbone weights with the pretrained ones, sweep α on valid.
   Zero training cost; applies to the backbone only (no pretrained head exists). Free insurance on
   top of any run; EMA is the within-run cousin we already use.
6. **Anchor/distill penalties — L2-SP (arXiv:1802.01483-family), DELTA, EWC.** Principled but adds a
   tuned hyperparameter and (for feature distillation) a teacher forward pass per step. Hold in
   reserve; Benny's "distance-from-node" idea = this family, feature-space variant.

## Recommended recipe (ranked for this project)

**Phase A (T7): freeze the whole backbone** — params + BN eval — train neck+head ~20 ep.
Cheapest, strongest-precedent (TFA/LP-FT), fastest per-epoch (no backbone grads), zero new
hyperparameters. Expect: slower start than the ep-4 spike but a *stable* curve; success = AP holds
or climbs past 0.14 without decay.

**Phase B (T8): unfreeze at backbone lr = 0.1× head lr, BN stays frozen** (LP-FT phase 2 +
Mind-the-Backbone + FrozenBN). Start from Phase A weights. Only if Phase A plateaus below target.

**Free add-on:** WiSE-FT sweep on any finished run — interpolate backbone toward riceseg init,
α ∈ {0.25, 0.5, 0.75}, eval on valid (3 evals, no training).

**Reserve (T9):** feature-space anchoring to frozen riceseg backbone (L2 on C3/C4/C5 vs teacher),
λ tuned — only if A+B confirm forgetting but can't hold the features.

Run order stays: T6 scratch control first (discriminates the hypothesis + paper baseline).

## Sources

- Kumar et al. 2022, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution" — https://arxiv.org/abs/2202.10054
- Wang et al. 2020, "Frustratingly Simple Few-Shot Object Detection" (TFA) — https://arxiv.org/abs/2003.06957
- Saito et al. 2023, "Mind the Backbone: Minimizing Backbone Distortion for Robust Object Detection" — https://arxiv.org/abs/2303.14744
- Wortsman et al. 2022, "Robust fine-tuning of zero-shot models" (WiSE-FT) — https://arxiv.org/abs/2109.01903
- Keras transfer-learning guide (BN freezing) — https://keras.io/guides/transfer_learning/
