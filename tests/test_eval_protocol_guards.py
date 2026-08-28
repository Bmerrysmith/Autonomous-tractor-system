"""Evaluation-protocol guards shared by both arms.

Two defects motivated these:

1. At a non-standard ``maxDets`` the COCO-named fields stop meaning what they
   say. pycocotools passes ``maxDets[2]`` for stats[1], stats[2] and stats[8],
   so ``ap50``/``ap75`` are computed at the non-standard budget and the field
   named ``ar_100`` is AR at that budget, while ``ap`` is the -1.0 sentinel.
   Publishing that dict verbatim reports "AR@100" for a number that is not.

2. ``weeddet_train`` refuses to validate against a file whose name starts with
   ``test``; ``baseline_det_control`` had no equivalent, so the two arms were
   not held to the same standard on checkpoint selection.
"""
from __future__ import annotations

import pytest

from agrinav.evaluation.metrics import CocoEvalResult


def _result(max_dets: int) -> CocoEvalResult:
    return CocoEvalResult(
        ap=0.18, ap50=0.56, ap75=0.06,
        ap_small=0.14, ap_medium=0.21, ap_large=0.30,
        ar_1=0.10, ar_10=0.25, ar_100=0.30,
        ar_small=0.24, ar_medium=0.33, ar_large=0.41,
        per_category_ap={1: 0.25, 2: 0.11},
        max_dets=max_dets,
        num_images=518,
        num_detections=51800,
        num_gt_annotations=15226,
        category_names={1: "rice_protect", 2: "weed_target"},
    )


# ---------------------------------------------------------------- maxDets

def test_standard_maxdets_emits_the_coco_field_names():
    data = _result(100).to_dict()
    assert data["is_standard_maxdets"] is True
    for field in ("ap", "ap50", "ap75", "ar_100"):
        assert field in data
    assert "ap_unavailable_reason" not in data


def test_non_standard_maxdets_renames_the_misleading_fields():
    data = _result(300).to_dict()
    assert data["is_standard_maxdets"] is False
    # The COCO-named fields must not be emitted under names that misdescribe them.
    for field in ("ap50", "ap75", "ar_100"):
        assert field not in data, f"{field} still emitted at maxDets=300"
    for field in ("ap50_at_maxdets_300", "ap75_at_maxdets_300", "ar_100_at_maxdets_300"):
        assert field in data


def test_non_standard_maxdets_drops_ap_and_explains_why():
    data = _result(300).to_dict()
    assert "ap" not in data, "the -1.0 sentinel must not be published as `ap`"
    assert "300" in data["ap_unavailable_reason"]
    assert "100" in data["ap_unavailable_reason"]


def test_renamed_values_are_preserved_not_recomputed():
    data = _result(300).to_dict()
    assert data["ap50_at_maxdets_300"] == pytest.approx(0.56)
    assert data["ar_100_at_maxdets_300"] == pytest.approx(0.30)


@pytest.mark.parametrize("max_dets", [1, 10, 50, 100, 300, 1000])
def test_to_dict_is_json_serialisable_at_any_budget(max_dets):
    import json
    json.dumps(_result(max_dets).to_dict())


# ------------------------------------------------------- sealed test split

@pytest.mark.parametrize("name", ["instances_test.coco.json", "test.json"])
def test_baseline_refuses_to_validate_on_the_sealed_test_split(tmp_path, name):
    from agrinav.training.baseline_det_control import BaselineError, train

    sealed = tmp_path / name
    sealed.write_text("{}", encoding="utf-8")

    with pytest.raises(BaselineError, match="sealed"):
        train(
            train_dataset=object(),
            config=_MinimalConfig(),
            out_dir=str(tmp_path / "out"),
            val_dataset=object(),
            val_ann_file=str(sealed),
        )


def test_the_real_split_filename_is_actually_caught():
    """The guard used to be `basename.startswith("test")`, which is False for
    every file this dataset ships -- the sealed split is `instances_test.coco.json`
    and the prefix is `instances`. Both arms' guards were decorative."""
    from agrinav.evaluation.metrics import names_a_test_split

    assert names_a_test_split("instances_test.coco.json")
    assert not "instances_test.coco.json".startswith("test"), (
        "if this ever becomes True the historical bug description is wrong")


@pytest.mark.parametrize("name,expected", [
    ("instances_test.coco.json", True),
    ("test.json", True),
    ("my-test-set.json", True),
    ("instances_valid.coco.json", False),
    ("instances_train.coco.json", False),
    ("latest_run.json", False),
    ("contest.json", False),
])
def test_token_boundaries_avoid_false_positives(name, expected):
    from agrinav.evaluation.metrics import names_a_test_split

    assert names_a_test_split(name) is expected


def test_baseline_guard_matches_the_weeddet_guard_wording():
    """Both arms must cite the same rule, so a reader cannot think they differ."""
    import inspect
    from agrinav.training import baseline_det_control, weeddet_train

    a = inspect.getsource(baseline_det_control)
    b = inspect.getsource(weeddet_train)
    for name, src in (("baseline_det_control", a), ("weeddet_train", b)):
        flat = " ".join(src.split())
        assert "the test split is sealed" in flat, f"{name} lacks the sealed-split refusal"
        assert "CLAUDE.md 13.3" in flat, f"{name} does not cite the rule"


class _MinimalConfig:
    """Enough surface for the guard to run before anything expensive."""
    seed = 42
    deterministic = False
    class_names = ("rice_protect", "weed_target")
    num_torchvision_classes = 3

    def validate(self):
        return None
