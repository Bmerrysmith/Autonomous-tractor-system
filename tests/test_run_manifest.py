"""Launch evidence must survive failures and distinguish recipe changes."""

import json

import pytest

from agrinav.training.run_manifest import (
    declarative_config,
    digest,
    file_sha256,
    read_manifest,
    start_run,
)


def _start(path, **config):
    return start_run(
        {"seed": 42, "num_epochs": 6, "val_ap_interval": 2, **config},
        path,
        trainer="fixture",
        train_dataset=[1],
        val_dataset=[1],
        protocol={},
        device="cpu",
    )


def test_runtime_addresses_do_not_change_config_hash():
    a = {"seed": 42, "train_dataset": object(), "backbone_init": lambda: None}
    b = {"seed": 42, "train_dataset": object(), "backbone_init": lambda: None}
    assert digest(declarative_config(a)) == digest(declarative_config(b))
    b["seed"] = 43
    assert digest(declarative_config(a)) != digest(declarative_config(b))


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf"), {1: "bad key"}])
def test_unknown_objects_and_nonfinite_config_fail_early(value):
    with pytest.raises(ValueError, match="non-declarative"):
        declarative_config({"unknown": value})


def test_start_marks_incomplete_and_preserves_source(tmp_path):
    provenance = _start(tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["completed"] is False
    assert status["epochs_completed"] == 0
    manifest = read_manifest(tmp_path, provenance)
    assert manifest["eval_epochs"] == [2, 4, 6]
    assert (
        file_sha256(tmp_path / manifest["source"]["archive"])
        == manifest["source"]["archive_sha256"]
    )
    original = (tmp_path / provenance["manifest_file"]).read_bytes()
    with pytest.raises(FileExistsError, match="fresh directory"):
        _start(tmp_path)
    assert (tmp_path / provenance["manifest_file"]).read_bytes() == original


def test_recipe_identity_excludes_seed_and_output_but_includes_training(tmp_path):
    a = _start(tmp_path / "a", checkpoint_dir="a", batch_size=8)
    b = _start(tmp_path / "b", checkpoint_dir="b", batch_size=8, seed=43)
    c = _start(tmp_path / "c", checkpoint_dir="c", batch_size=4, seed=43)
    assert a["recipe_sha256"] == b["recipe_sha256"]
    assert a["config_sha256"] != b["config_sha256"]
    assert a["recipe_sha256"] != c["recipe_sha256"]


def test_declared_annotations_cannot_override_actual_dataset(tmp_path):
    class Dataset(list):
        ann_file = tmp_path / "actual.json"

    dataset = Dataset([1])
    with pytest.raises(ValueError, match="actual annotation"):
        start_run(
            {"ann_file": str(tmp_path / "different.json")},
            tmp_path,
            trainer="fixture",
            train_dataset=dataset,
            val_dataset=None,
            protocol={},
            device="cpu",
        )


def test_subset_membership_changes_recipe_identity(tmp_path):
    class Dataset(list):
        pass

    a, b = Dataset([1]), Dataset([1])
    a.images = [{"id": 1, "file_name": "one.jpg"}]
    b.images = [{"id": 2, "file_name": "two.jpg"}]
    pa = start_run(
        {},
        tmp_path / "a",
        trainer="fixture",
        train_dataset=a,
        val_dataset=None,
        protocol={},
        device="cpu",
    )
    pb = start_run(
        {},
        tmp_path / "b",
        trainer="fixture",
        train_dataset=b,
        val_dataset=None,
        protocol={},
        device="cpu",
    )
    assert pa["recipe_sha256"] != pb["recipe_sha256"]
