"""Tests for the SAM re-seed input builder and the SAM2.1 backend contract.

The predicates under test are the ones whose failure is unrecoverable:
the sealed-split refusal, the sha256/geometry preflight, and the refusal of
text-style prompt kwargs. All CPU, no GPU, no network, no model download.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agrinav.data import build_sam_reseed_input as bsri
from agrinav.data.sam2_predictor import PredictorCounters, Sam2TransformersPredictor
from agrinav.data.sam_box_to_mask import SamPreflightError

# --------------------------------------------------------------------------
# fixtures: a miniature stand-in for the phase-2 rebuild tree
# --------------------------------------------------------------------------


def _png_bytes(width: int, height: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_split(root: Path, split: str, n_images: int, seed0: int) -> dict:
    images_dir = root / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 0
    for i in range(n_images):
        name = f"{split}_{i:03d}.png"
        raw = _png_bytes(32, 24, seed0 + i)
        (images_dir / name).write_bytes(raw)
        images.append(
            {
                "id": i + 1,
                "file_name": name,
                "width": 32,
                "height": 24,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "group_id": f"fam{i % 3}",
                "exif_reoriented": False,
                "source_split": split,
            }
        )
        for k in range(2):
            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": 1 + (k % 2),
                    "bbox": [2.0 + k, 3.0, 8.0, 6.0],
                    "area": 48.0,
                    "iscrowd": 0,
                }
            )
    doc = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "rice_protect"}, {"id": 2, "name": "weed_target"}],
    }
    path = root / "annotations" / f"instances_{split}.coco.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


@pytest.fixture()
def rebuild_root(tmp_path: Path) -> Path:
    root = tmp_path / "REBUILD"
    for split, n, seed0 in (("train", 6, 100), ("valid", 4, 200), ("test", 3, 300)):
        _write_split(root, split, n, seed0)
    pins = {}
    for split in ("train", "valid", "test"):
        key = f"annotations/instances_{split}.coco.json"
        pins[key] = hashlib.sha256((root / key).read_bytes()).hexdigest()
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "manifests" / "provenance.json").write_text(
        json.dumps({"json_sha256": pins}), encoding="utf-8"
    )
    return root


def _build(root: Path, out: Path, **kwargs):
    params = {
        "rebuild_root": root,
        "splits": ("train", "valid"),
        "sample_size": 0,
        "sample_seed": 0,
        "out_zip": out / "images.zip",
        "out_json": out / "proposals.json",
        "out_manifest": out / "manifest.json",
    }
    params.update(kwargs)
    return bsri.build(**params)


# --------------------------------------------------------------------------
# the sealed split
# --------------------------------------------------------------------------


def test_parse_splits_refuses_the_sealed_test_split():
    with pytest.raises(bsri.ReseedInputError, match="sealed"):
        bsri.parse_splits("train,test")


def test_parse_splits_refuses_a_typo_rather_than_selecting_nothing():
    with pytest.raises(bsri.ReseedInputError, match="unknown split"):
        bsri.parse_splits("train,vaild")


def test_assert_not_sealed_rejects_a_path_under_the_test_dir(rebuild_root: Path):
    with pytest.raises(bsri.ReseedInputError, match="sealed"):
        bsri.assert_not_sealed(rebuild_root / "images" / "test" / "test_000.png", rebuild_root)


def test_assert_not_sealed_rejects_a_path_outside_the_root(rebuild_root: Path, tmp_path: Path):
    with pytest.raises(bsri.ReseedInputError, match="outside"):
        bsri.assert_not_sealed(tmp_path / "elsewhere" / "x.png", rebuild_root)


def test_no_test_image_reaches_the_zip_or_the_proposal_doc(rebuild_root: Path, tmp_path: Path):
    out = tmp_path / "out"
    manifest = _build(rebuild_root, out)
    doc = json.loads((out / "proposals.json").read_text(encoding="utf-8"))

    test_shas = {
        im["sha256"]
        for im in json.loads(
            (rebuild_root / "annotations" / "instances_test.coco.json").read_text("utf-8")
        )["images"]
    }
    emitted_shas = {im["sha256"] for im in doc["images"]}
    assert emitted_shas.isdisjoint(test_shas)
    assert all(im["split"] in ("train", "valid") for im in doc["images"])

    with zipfile.ZipFile(out / "images.zip") as archive:
        members = archive.namelist()
    assert members and not any(m.startswith("test/") for m in members)
    assert manifest["counts"]["images"] == 10  # 6 train + 4 valid, no test


# --------------------------------------------------------------------------
# preflight: hashes and geometry
# --------------------------------------------------------------------------


def test_annotation_hash_drift_is_fatal(rebuild_root: Path, tmp_path: Path):
    path = rebuild_root / "annotations" / "instances_train.coco.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["images"][0]["group_id"] = "tampered"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(bsri.ReseedInputError, match="sha256 drift"):
        _build(rebuild_root, tmp_path / "out")


def test_image_byte_mismatch_is_fatal_and_never_substituted(rebuild_root: Path, tmp_path: Path):
    target = rebuild_root / "images" / "train" / "train_000.png"
    target.write_bytes(_png_bytes(32, 24, 9999))  # same size, different pixels
    with pytest.raises(bsri.ReseedInputError, match="sha256 mismatch"):
        _build(rebuild_root, tmp_path / "out")


def test_declared_size_mismatch_is_fatal(tmp_path: Path):
    raw = _png_bytes(32, 24, 7)
    path = tmp_path / "img.png"
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    with pytest.raises(bsri.ReseedInputError, match="decoded size"):
        bsri.verify_image_bytes(path, sha, 64, 24)


def test_missing_image_file_is_fatal(rebuild_root: Path, tmp_path: Path):
    (rebuild_root / "images" / "train" / "train_000.png").unlink()
    with pytest.raises(bsri.ReseedInputError, match="missing on disk"):
        _build(rebuild_root, tmp_path / "out")


def test_unmapped_category_is_fatal_not_a_silent_drop(rebuild_root: Path, tmp_path: Path):
    path = rebuild_root / "annotations" / "instances_train.coco.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["annotations"][0]["category_id"] = 7
    path.write_text(json.dumps(doc), encoding="utf-8")
    pins = json.loads((rebuild_root / "manifests" / "provenance.json").read_text("utf-8"))
    pins["json_sha256"]["annotations/instances_train.coco.json"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    (rebuild_root / "manifests" / "provenance.json").write_text(json.dumps(pins), encoding="utf-8")
    with pytest.raises(bsri.ReseedInputError, match="closed map"):
        _build(rebuild_root, tmp_path / "out")


# --------------------------------------------------------------------------
# sampling determinism
# --------------------------------------------------------------------------


def test_sample_is_deterministic_for_a_seed_and_independent_of_input_order():
    shas = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(50)]
    a = bsri.select_sample(shas, 10, 20260831)
    b = bsri.select_sample(list(reversed(shas)), 10, 20260831)
    assert a == b
    assert len(a) == 10


def test_different_seeds_select_different_samples():
    shas = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(50)]
    assert bsri.select_sample(shas, 10, 1) != bsri.select_sample(shas, 10, 2)


def test_sample_size_zero_selects_everything():
    shas = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(5)]
    assert bsri.select_sample(shas, 0, 1) == frozenset(shas)


def test_sampled_build_emits_exactly_the_requested_images(rebuild_root: Path, tmp_path: Path):
    manifest = _build(rebuild_root, tmp_path / "out", sample_size=4, sample_seed=20260831)
    assert manifest["counts"]["images"] == 4
    assert manifest["counts"]["boxes"] == 8  # 2 boxes per fixture image


# --------------------------------------------------------------------------
# the emitted document asserts no truth
# --------------------------------------------------------------------------


def test_emitted_proposals_carry_no_accepted_status_and_no_geometry(
    rebuild_root: Path, tmp_path: Path
):
    out = tmp_path / "out"
    _build(rebuild_root, out)
    blob = (out / "proposals.json").read_text(encoding="utf-8")
    for forbidden in ('"accepted"', '"adjudicated"', '"verified_empty"', '"treatment_eligible"'):
        assert forbidden not in blob
    doc = json.loads(blob)
    assert all(ann["segmentation"] == [] for ann in doc["annotations"])
    assert all(ann["review_status"] == "unreviewed_proposal" for ann in doc["annotations"])


def test_emitted_boxes_are_copied_verbatim_from_the_human_annotation(
    rebuild_root: Path, tmp_path: Path
):
    out = tmp_path / "out"
    _build(rebuild_root, out)
    doc = json.loads((out / "proposals.json").read_text(encoding="utf-8"))
    source = json.loads(
        (rebuild_root / "annotations" / "instances_train.coco.json").read_text("utf-8")
    )
    by_source_id = {a["id"]: a for a in source["annotations"]}
    checked = 0
    for ann in doc["annotations"]:
        if ann["source_split"] != "train":
            continue
        original = by_source_id[ann["source_annotation_id"]]
        assert ann["bbox"] == [float(v) for v in original["bbox"]]
        assert ann["category_id"] == original["category_id"]
        checked += 1
    assert checked > 0


def test_zip_members_match_proposal_file_names(rebuild_root: Path, tmp_path: Path):
    """sam_box_to_mask does archive.read(unit['file_name']); the two must agree."""
    out = tmp_path / "out"
    _build(rebuild_root, out)
    doc = json.loads((out / "proposals.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(out / "images.zip") as archive:
        members = set(archive.namelist())
        for image in doc["images"]:
            assert image["file_name"] in members
            raw = archive.read(image["file_name"])
            assert hashlib.sha256(raw).hexdigest() == image["sha256"]


# --------------------------------------------------------------------------
# the SAM2.1 backend refuses text prompts
# --------------------------------------------------------------------------


class _FakeProcessor:
    def __call__(self, **kwargs):
        raise AssertionError("not reached in these tests")

    def post_process_masks(self, *args, **kwargs):
        raise AssertionError("not reached in these tests")


def _predictor() -> Sam2TransformersPredictor:
    predictor = Sam2TransformersPredictor(
        model=object(),
        processor=_FakeProcessor(),
        device="cpu",
        dtype=None,
        counters=PredictorCounters(),
    )
    predictor._embeddings = object()
    predictor._original_sizes = object()
    return predictor


@pytest.mark.parametrize(
    "kwarg", ["text", "text_prompt", "label", "labels", "class_name", "phrase", "caption"]
)
def test_backend_refuses_every_text_style_prompt_kwarg(kwarg: str):
    with pytest.raises(SamPreflightError, match="forbidden"):
        _predictor().predict(
            box=np.array([0.0, 0.0, 1.0, 1.0]), multimask_output=True, **{kwarg: "weeds"}
        )


def test_backend_refuses_any_unexpected_kwarg():
    with pytest.raises(SamPreflightError, match="unexpected predict kwargs"):
        _predictor().predict(
            box=np.array([0.0, 0.0, 1.0, 1.0]), multimask_output=True, point_coords=[[1, 1]]
        )


def test_backend_refuses_predict_before_set_image():
    predictor = Sam2TransformersPredictor(
        model=object(),
        processor=_FakeProcessor(),
        device="cpu",
        dtype=None,
        counters=PredictorCounters(),
    )
    with pytest.raises(SamPreflightError, match="before set_image"):
        predictor.predict(box=np.array([0.0, 0.0, 1.0, 1.0]), multimask_output=True)
