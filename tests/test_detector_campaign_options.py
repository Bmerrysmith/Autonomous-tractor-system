"""Scientific invariants for the detector campaign's separate experimental arms."""

import math

import pytest
import torch

from agrinav.models import weeddet_v6b as wd
from agrinav.training.weeddet_train import _build_parser, load_checkpoint_model


def test_reference_varifocal_positive_weight_is_iou_without_focal_modulation():
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([0.25, 0.75, 0.0], dtype=torch.float64)
    loss, positive, negative = wd.ReferenceVarifocalLoss()(logits, targets, return_split=True)
    # At p=.5, BCE is log(2); positive weights sum to 1, negative weight=.75*.5**2.
    assert positive.item() == pytest.approx(math.log(2))
    assert negative.item() == pytest.approx(0.1875 * math.log(2))
    assert loss.item() == pytest.approx(positive.item() + negative.item())
    gradient = torch.autograd.grad(loss, logits)[0]
    assert gradient[:2].tolist() == pytest.approx([0.0625, -0.1875])
    assert torch.autograd.gradcheck(wd.ReferenceVarifocalLoss(), (logits, targets))


def test_legacy_loss_keeps_its_historical_zero_at_matching_soft_targets():
    target = torch.tensor([0.75])
    logits = torch.logit(target)
    assert wd.HardTargetFocalLikeLoss()(logits, target).item() == pytest.approx(0, abs=1e-12)
    assert wd.ReferenceVarifocalLoss()(logits, target).item() > 0
    assert wd.VariFocalLoss is wd.HardTargetFocalLikeLoss
    assert isinstance(wd.WeedDetLoss().varifocal, wd.HardTargetFocalLikeLoss)


def test_group_norm_changes_only_the_head_normalization():
    batch = wd.WeedDet(num_classes=2)
    group = wd.WeedDet(num_classes=2, head_norm="group")
    batch_bn = {name for name, m in batch.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}
    group_bn = {name for name, m in group.named_modules() if isinstance(m, torch.nn.BatchNorm2d)}
    assert batch_bn - group_bn == {"head.shared.2.seq.3"}
    assert not group_bn - batch_bn
    layer = group.head.shared[2].seq[3]
    assert isinstance(layer, torch.nn.GroupNorm) and layer.num_groups == 32
    data = torch.randn(2, 64, 8, 8)
    assert torch.equal(layer.train()(data), layer.eval()(data))


@pytest.mark.parametrize("kwargs", [{"head_norm": "typo"}, {"cls_loss_mode": "typo"}])
def test_invalid_campaign_options_fail_loudly(kwargs):
    with pytest.raises(ValueError):
        wd.WeedDet(num_classes=2, **kwargs)


def test_loader_restores_architecture_and_prediction_geometry(tmp_path):
    config = {
        "num_classes": 2,
        "anchor_base_scale": 4.0,
        "lsc_k": 5,
        "head_norm": "group",
        "cls_loss_mode": "varifocal",
        "cls_target_mode": "pred_iou",
    }
    model = wd.WeedDet(**config).eval()
    path = tmp_path / "variant.pth"
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
    restored = load_checkpoint_model(path)
    assert restored.anchor_gen.base_scale == 4
    assert isinstance(restored.criterion.varifocal, wd.ReferenceVarifocalLoss)
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        original = model._get_logits(images)
        loaded = restored._get_logits(images)
    assert torch.equal(original[2], loaded[2])
    for original_levels, loaded_levels in zip(original[:2], loaded[:2]):
        for expected, actual in zip(original_levels, loaded_levels):
            assert torch.equal(expected, actual)


def test_campaign_cli_options_preserve_legacy_defaults():
    parser = _build_parser()
    defaults = parser.parse_args([])
    assert defaults.cls_loss_mode is None and defaults.head_norm is None
    args = parser.parse_args(
        ["--cls-loss-mode", "varifocal", "--head-norm", "group", "--val-hard-nms"]
    )
    assert (args.cls_loss_mode, args.head_norm, args.val_use_soft_nms) == (
        "varifocal",
        "group",
        False,
    )
