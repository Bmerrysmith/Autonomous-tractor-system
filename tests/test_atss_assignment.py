"""ATSS candidate-selection tests.

These pin the 2026-08-28 assigner fix. Before it, `_assign_atss` derived the
number of co-located anchor shapes by comparing anchor centres for float
equality, which returned 10 instead of 12 at every level, and then expanded each
selected cell to all of those shapes. The result was ~45 positives per object at
a mean assigned IoU of 0.50 -- i.e. most positives sat below the threshold AP50
itself uses -- while anchors averaging 0.73 went unselected.

The tests below are deliberately behavioural: they assert properties of the
assignment, not the shape of the implementation, so a future rewrite that keeps
the properties keeps passing.
"""

from __future__ import annotations

import pytest
import torch

from agrinav.models.weeddet_v6b import AnchorGenerator, WeedDetLoss, box_iou

IMG = 512
STRIDES = (4, 8, 16)


def _anchors():
    gen = AnchorGenerator(
        base_scale=3,
        aspect_ratios=(0.2, 0.33, 0.5, 1.0),
        scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
        strides=STRIDES,
    )
    feats = [torch.zeros(1, 1, IMG // s, IMG // s) for s in STRIDES]
    anchors = gen(feats, (IMG, IMG))
    return gen, anchors, list(gen.num_anchors_per_level)


def _synthetic_gt(n=24, seed=0):
    """Tall, small boxes matching the measured phase-2 distribution.

    Median GT in 512 letterbox space is 20.8 x 41.9 px (aspect w/h ~0.5).
    """
    g = torch.Generator().manual_seed(seed)
    cx = torch.rand(n, generator=g) * 400 + 56
    cy = torch.rand(n, generator=g) * 400 + 56
    w = torch.rand(n, generator=g) * 16 + 14  # 14-30 px
    h = w / (torch.rand(n, generator=g) * 0.3 + 0.4)  # aspect 0.4-0.7
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def _assign(mode, anchors, npl, gt, num_shapes=12):
    loss = WeedDetLoss(
        num_classes=2,
        use_atss=True,
        atss_topk=9,
        atss_candidate_mode=mode,
        num_shapes_per_location=num_shapes,
    )
    ious = box_iou(anchors, gt)
    pos, neg, best, quality = loss._assign_atss(anchors, gt, ious, npl)
    return pos, neg, best, quality, ious


# --------------------------------------------------------------------------
# The structural bug that caused the dilution
# --------------------------------------------------------------------------


def test_num_shapes_is_structural_not_float_derived():
    """12 shapes per location, from the config -- never from centre equality."""
    gen, anchors, npl = _anchors()
    assert gen.num_shapes == 12

    # Demonstrate why the old derivation was unsafe: `(x1 + x2) * 0.5` is not
    # exact in float32, so 2 of the 12 co-located shapes lose their centre.
    centers = (anchors[:, :2] + anchors[:, 2:]) * 0.5
    start = 0
    for n_level in npl:
        lvl = centers[start : start + n_level]
        float_derived = int((lvl == lvl[0]).all(dim=1).sum().item())
        assert float_derived == 10, (
            "the historical float-equality derivation is expected to return 10; "
            f"got {float_derived}. If this changed, the regression guard below "
            "no longer protects anything."
        )
        start += n_level


def test_anchor_index_layout_is_cell_major():
    """index == cell * num_shapes + shape, exactly. The selector relies on it."""
    gen, anchors, npl = _anchors()
    n_shapes = gen.num_shapes
    start = 0
    for level, n_level in enumerate(npl):
        assert n_level % n_shapes == 0
        widths = anchors[start : start + n_shapes, 2] - anchors[start : start + n_shapes, 0]
        # The next cell repeats the same 12 shapes.
        nxt = (
            anchors[start + n_shapes : start + 2 * n_shapes, 2]
            - anchors[start + n_shapes : start + 2 * n_shapes, 0]
        )
        torch.testing.assert_close(widths, nxt)
        start += n_level


# --------------------------------------------------------------------------
# Behavioural guarantees of the assignment
# --------------------------------------------------------------------------


def test_default_mode_is_not_diluted():
    """The property the fix exists to establish.

    Bounds are deliberately loose -- they are a regression guard, not a
    reproduction of an exact measurement. The shipped `legacy` mode produced
    45.7 pos/GT at mean IoU 0.50; anything in that region must fail here.
    """
    _, anchors, npl = _anchors()
    gt = _synthetic_gt()
    pos, _, _, quality, _ = _assign("cells_best_shape", anchors, npl, gt)

    per_gt = int(pos.sum()) / len(gt)
    assigned = quality[pos]

    assert 2.0 <= per_gt <= 15.0, f"positives per GT = {per_gt:.2f}"
    assert assigned.mean() >= 0.55, f"mean assigned IoU = {assigned.mean():.4f}"
    # The decisive one: positives must clear the threshold AP50 itself uses.
    frac_above_half = (assigned >= 0.5).float().mean()
    assert frac_above_half >= 0.85, f"only {frac_above_half:.3f} of positives at IoU >= 0.5"


def test_default_beats_legacy_on_every_axis():
    """Guards the direction of the fix, not its magnitude."""
    _, anchors, npl = _anchors()
    gt = _synthetic_gt()

    _, _, _, q_new, _ = _assign("cells_best_shape", anchors, npl, gt)
    pos_new, *_ = _assign("cells_best_shape", anchors, npl, gt)
    pos_old, _, _, q_old, _ = _assign("legacy", anchors, npl, gt)

    new_pos, old_pos = q_new[pos_new], q_old[pos_old]
    assert int(pos_new.sum()) < int(pos_old.sum())
    assert new_pos.mean() > old_pos.mean()
    assert (new_pos >= 0.5).float().mean() > (old_pos >= 0.5).float().mean()


def test_legacy_mode_still_reproduces_the_old_behaviour():
    """`legacy` exists so the ablation table can be regenerated. Keep it broken."""
    _, anchors, npl = _anchors()
    gt = _synthetic_gt()
    pos, _, _, quality, _ = _assign("legacy", anchors, npl, gt)
    per_gt = int(pos.sum()) / len(gt)
    assert per_gt > 20.0, (
        "legacy mode is supposed to reproduce the diluted assignment; "
        f"got {per_gt:.2f} positives per GT"
    )


@pytest.mark.parametrize("mode", WeedDetLoss.ATSS_CANDIDATE_MODES)
def test_every_gt_owns_at_least_one_positive(mode):
    _, anchors, npl = _anchors()
    gt = _synthetic_gt()
    pos, _, best, _, _ = _assign(mode, anchors, npl, gt)
    owned = torch.unique(best[pos])
    assert owned.numel() == len(
        gt
    ), f"{mode}: {len(gt) - owned.numel()} ground-truth boxes have no positive anchor"


@pytest.mark.parametrize("mode", WeedDetLoss.ATSS_CANDIDATE_MODES)
def test_positive_and_negative_masks_are_disjoint(mode):
    _, anchors, npl = _anchors()
    gt = _synthetic_gt()
    pos, neg, _, _, _ = _assign(mode, anchors, npl, gt)
    assert not bool((pos & neg).any())


def test_unknown_candidate_mode_is_rejected():
    with pytest.raises(ValueError, match="atss_candidate_mode"):
        WeedDetLoss(num_classes=2, atss_candidate_mode="topk")


def test_selection_is_deterministic():
    _, anchors, npl = _anchors()
    gt = _synthetic_gt()
    a = _assign("cells_best_shape", anchors, npl, gt)[0]
    b = _assign("cells_best_shape", anchors, npl, gt)[0]
    assert torch.equal(a, b)


def test_empty_gt_image_produces_finite_losses():
    """A frame with no objects must train as all-background, not crash or NaN."""
    gen, anchors, npl = _anchors()
    loss = WeedDetLoss(
        num_classes=2,
        atss_candidate_mode="cells_best_shape",
        num_shapes_per_location=gen.num_shapes,
    )

    total = anchors.shape[0]
    cls_logits = [torch.zeros(1, 2 * gen.num_shapes, IMG // s, IMG // s) for s in STRIDES]
    regs = [torch.zeros(1, 4 * gen.num_shapes, IMG // s, IMG // s) for s in STRIDES]
    targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}]

    out = loss(cls_logits, regs, anchors, targets, npl)

    assert int(out["num_pos_anchors"].item()) == 0
    for key in ("cls_loss", "reg_loss", "total_loss"):
        assert torch.isfinite(out[key]), f"{key} is not finite on an empty frame"
    assert float(out["reg_loss"]) == 0.0, "no boxes means nothing to regress"
    # Sanity: the anchor count the loss saw matches the generator's.
    assert sum(npl) == total
