"""Contract tests for the SAM2.1 box->mask stage.

This module also supplies ``make_stub_predictor``, which the stage's own
docstring advertises as the local CPU injection point::

    --predictor-factory tests.test_sam_box_to_mask:make_stub_predictor

That factory was referenced but did not exist, so the documented CPU path could
not be run and the stage's safety invariants had no test at all. It exists now.

Everything here is CPU-only: no GPU, no network, no model download. The two
things under test are the ones whose failure is unrecoverable — the candidate
set the stage emits, and whether the model pin is actually enforced rather than
merely recorded.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from agrinav.data import sam_box_to_mask as stage
from agrinav.data.sam_box_to_mask import (
    CANDIDATE_KIND_JITTER,
    CANDIDATE_KIND_MULTIMASK,
    CANDIDATE_KIND_SINGLE_MASK,
    CANDIDATE_KINDS,
    CANDIDATES_PER_BOX,
    DECODER_PASSES_PER_BOX,
    JITTER_CANDIDATES,
    MULTIMASK_CANDIDATES,
    SINGLE_MASK_CANDIDATES,
    SamPreflightError,
    candidates_for_box,
    clip_box,
    default_predictor_factory,
    jitter_box,
    run_thresholds,
)

IMAGE_WIDTH, IMAGE_HEIGHT = 64, 48
IMAGE_SHA = "a" * 64


class StubPredictor:
    """Deterministic CPU stand-in for ``SAM2ImagePredictor``.

    Draws a filled rectangle inset inside whatever box it is given, so the mask
    is a pure function of the prompt: a test can tell which box produced which
    candidate. Records every call so the *call sequence* is assertable, which is
    what distinguishes "ran the single-mask head on the unjittered box" from
    "relabelled an existing candidate".
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, ...], bool]] = []
        self.set_image_calls = 0

    def set_image(self, image: np.ndarray) -> None:
        self.set_image_calls += 1
        self._shape = np.asarray(image).shape[:2]

    def predict(
        self, *, box: np.ndarray, multimask_output: bool
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        xyxy = tuple(float(v) for v in np.asarray(box).reshape(-1)[:4])
        self.calls.append((xyxy, bool(multimask_output)))
        height, width = self._shape
        n = MULTIMASK_CANDIDATES if multimask_output else 1
        masks = np.zeros((n, height, width), dtype=bool)
        x0, y0, x1, y1 = (int(round(v)) for v in xyxy)
        for i in range(n):
            inset = i  # each multimask head returns a slightly different mask
            masks[
                i, max(0, y0 + inset) : max(0, y1 - inset), max(0, x0 + inset) : max(0, x1 - inset)
            ] = True
        scores = np.array([0.9 - 0.1 * i for i in range(n)], dtype=np.float32)
        return masks, scores, None


def make_stub_predictor() -> StubPredictor:
    """Zero-arg factory matching the ``--predictor-factory module:attr`` contract."""
    return StubPredictor()


@pytest.fixture()
def box() -> dict[str, Any]:
    return {
        "source_object_id": "coco-ann-1",
        "coco_annotation_id": 1,
        "category_id": 2,
        "label": "weed_target",
        "bbox": [10.0, 8.0, 20.0, 16.0],
    }


@pytest.fixture()
def emitted(box: dict[str, Any]) -> tuple[list[dict[str, Any]], StubPredictor]:
    predictor = make_stub_predictor()
    predictor.set_image(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8))
    candidates, reason = candidates_for_box(predictor, box, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_SHA)
    assert reason is None
    return candidates, predictor


# --------------------------------------------------------------------------
# the added single-mask candidate
# --------------------------------------------------------------------------


def test_the_single_mask_candidate_is_present_exactly_once(emitted):
    candidates, _ = emitted
    singles = [c for c in candidates if c["kind"] == CANDIDATE_KIND_SINGLE_MASK]
    assert len(singles) == SINGLE_MASK_CANDIDATES == 1


def test_single_mask_candidate_uses_the_unjittered_clipped_box(emitted, box):
    candidates, _ = emitted
    clipped = clip_box(box["bbox"], IMAGE_WIDTH, IMAGE_HEIGHT)
    single = next(c for c in candidates if c["kind"] == CANDIDATE_KIND_SINGLE_MASK)
    assert single["prompt_box"] == list(clipped)
    multimask = next(c for c in candidates if c["kind"] == CANDIDATE_KIND_MULTIMASK)
    assert single["prompt_box"] == multimask["prompt_box"], "same prompt as the multimask group"
    jitter_boxes = [c["prompt_box"] for c in candidates if c["kind"] == CANDIDATE_KIND_JITTER]
    assert single["prompt_box"] not in jitter_boxes, "must not be one of the jittered boxes"


def test_single_mask_candidate_was_produced_with_multimask_output_false(emitted, box):
    """The distinguishing property. Asserted on the call sequence, not the label."""
    candidates, predictor = emitted
    clipped = clip_box(box["bbox"], IMAGE_WIDTH, IMAGE_HEIGHT)
    exact_xyxy = tuple(stage.xywh_to_xyxy(clipped))

    single_mask_calls = [call for call in predictor.calls if call == (exact_xyxy, False)]
    assert len(single_mask_calls) == 1, "exactly one single-mask pass on the exact box"
    assert predictor.calls[-1] == (exact_xyxy, False), "and it is the last pass"
    assert predictor.calls[0] == (exact_xyxy, True), "the multimask pass still runs first"


def test_single_mask_candidate_carries_null_group_indices(emitted):
    candidates, _ = emitted
    single = next(c for c in candidates if c["kind"] == CANDIDATE_KIND_SINGLE_MASK)
    assert single["multimask_index"] is None
    assert single["jitter_index"] is None
    assert single["sam_pred_iou"] is not None
    assert single["area_px"] > 0
    assert set(single["rle"]) == {"size", "counts"}
    assert single["rle"]["size"] == [IMAGE_HEIGHT, IMAGE_WIDTH]


# --------------------------------------------------------------------------
# the pre-existing 8 candidates must be untouched
# --------------------------------------------------------------------------


def test_the_original_eight_candidates_are_unchanged(emitted, box):
    """Regression guard: adding a candidate must not renumber or alter the rest.

    Downstream jitter-agreement statistics select by ``kind`` and may store
    ``candidate_index``; either would break silently if the new candidate were
    inserted rather than appended.
    """
    candidates, _ = emitted
    clipped = clip_box(box["bbox"], IMAGE_WIDTH, IMAGE_HEIGHT)

    multimask = [c for c in candidates if c["kind"] == CANDIDATE_KIND_MULTIMASK]
    assert [c["candidate_index"] for c in multimask] == [0, 1, 2]
    assert [c["multimask_index"] for c in multimask] == [0, 1, 2]
    assert all(c["jitter_index"] is None for c in multimask)
    assert all(c["prompt_box"] == list(clipped) for c in multimask)

    jitters = [c for c in candidates if c["kind"] == CANDIDATE_KIND_JITTER]
    assert [c["candidate_index"] for c in jitters] == [3, 4, 5, 6, 7]
    assert [c["jitter_index"] for c in jitters] == [0, 1, 2, 3, 4]
    assert all(c["multimask_index"] is None for c in jitters)
    for k, candidate in enumerate(jitters):
        expected = jitter_box(
            clipped, IMAGE_SHA, box["source_object_id"], k, IMAGE_WIDTH, IMAGE_HEIGHT
        )
        assert candidate["prompt_box"] == list(expected)

    single = next(c for c in candidates if c["kind"] == CANDIDATE_KIND_SINGLE_MASK)
    assert single["candidate_index"] == 8, "appended last, so nothing is renumbered"


def test_candidate_count_and_kinds_match_the_constants(emitted):
    candidates, predictor = emitted
    assert len(candidates) == CANDIDATES_PER_BOX == 9
    assert len(predictor.calls) == DECODER_PASSES_PER_BOX == 7
    assert [c["candidate_index"] for c in candidates] == list(range(9))
    assert {c["kind"] for c in candidates} <= set(CANDIDATE_KINDS)


def test_run_thresholds_reports_the_count_actually_emitted(emitted):
    """Provenance claiming N while the loop emits M is the bug this prevents."""
    candidates, predictor = emitted
    thresholds = run_thresholds(shard_index=0, shard_count=1)
    assert thresholds["candidates_per_box"] == len(candidates)
    assert thresholds["decoder_passes_per_box"] == len(predictor.calls)
    assert thresholds["single_mask_candidates"] == SINGLE_MASK_CANDIDATES
    assert thresholds["multimask_candidates"] == MULTIMASK_CANDIDATES
    assert thresholds["jitter_candidates"] == JITTER_CANDIDATES
    assert thresholds["candidate_kinds"] == list(CANDIDATE_KINDS)
    emitted_kinds = {c["kind"] for c in candidates}
    assert emitted_kinds == set(thresholds["candidate_kinds"])


def test_degenerate_box_still_emits_no_candidates_and_a_reason(box):
    """The added pass must not resurrect a box that cannot be clipped."""
    predictor = make_stub_predictor()
    predictor.set_image(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8))
    offscreen = dict(box, bbox=[IMAGE_WIDTH + 5.0, 0.0, 10.0, 10.0])
    candidates, reason = candidates_for_box(
        predictor, offscreen, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_SHA
    )
    assert candidates == []
    assert reason == "prompt_box_degenerate_after_clip"
    assert predictor.calls == [], "no decoder pass at all for a degenerate box"


# --------------------------------------------------------------------------
# invariants the added candidate must not have weakened
# --------------------------------------------------------------------------


def test_no_candidate_carries_a_label_or_any_truth_marker(emitted):
    candidates, _ = emitted
    for candidate in candidates:
        assert "label" not in candidate
        assert "class_name" not in candidate
        assert "review_status" not in candidate


def test_provenance_template_asserts_nothing_accepted():
    provenance = stage.build_provenance(
        model_id="facebook/sam2.1-hiera-large",
        model_revision="0" * 40,
        thresholds=run_thresholds(0, 1),
        source_image_sha256="b" * 64,
        generated_at="2026-09-01T00:00:00Z",
    )
    assert provenance["review_status"] == "unreviewed"
    assert provenance["human_edit_state"] == "unreviewed"
    assert provenance["annotator_id"] is None
    assert provenance["reviewer_id"] is None
    assert "verified_empty" not in provenance
    assert "treatment_eligible" not in provenance


def test_label_map_is_closed_and_unmapped_ids_are_a_counted_drop():
    assert stage.map_category(1) == "rice_protect"
    assert stage.map_category(2) == "weed_target"
    assert stage.map_category(3) is None
    doc = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 4, "height": 4, "sha256": "c" * 64}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 1, 1]},
            {"id": 2, "image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1]},
        ],
    }
    units, drops = stage.index_proposals(doc)
    assert units[0]["boxes"] == []
    assert drops == {"category_id=3": 1, "orphan_annotation_missing_image": 1}


# --------------------------------------------------------------------------
# the pin must be enforced, not merely recorded
# --------------------------------------------------------------------------


def test_default_factory_refuses_a_placeholder_before_any_model_work():
    with pytest.raises(SamPreflightError, match="placeholder"):
        default_predictor_factory("facebook/sam2.1-hiera-large", "PIN_BEFORE_RUN")


def test_default_factory_refuses_a_branch_name():
    with pytest.raises(SamPreflightError, match="does not match"):
        default_predictor_factory("facebook/sam2.1-hiera-large", "refs/heads/main")


def test_default_factory_routes_to_the_pinning_backend_not_upstream_sam2(monkeypatch):
    """The upstream sam2 package drops ``revision``; it must not be reached."""
    captured: dict[str, Any] = {}
    sentinel = object()

    import agrinav.data.sam2_predictor as backend

    def fake_build_predictor(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(backend, "build_predictor", fake_build_predictor)
    monkeypatch.delitem(sys.modules, "sam2", raising=False)

    sha = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
    factory = default_predictor_factory("facebook/sam2.1-hiera-large", sha, device="cpu")
    assert factory() is sentinel
    assert captured["revision"] == sha
    assert captured["verify_revision"] is True
    assert "sam2" not in sys.modules, "the unpinned upstream package must not be imported"


def test_build_predictor_forwards_revision_to_both_from_pretrained_calls(monkeypatch):
    """The actual loader call must RECEIVE the sha, not absorb it into kwargs.

    This is the regression test for the upstream defect: ``sam2``'s
    ``from_pretrained`` accepted ``revision`` and silently dropped it. Asserting
    that the argument arrives is the only way to tell an enforced pin from a
    recorded one without hitting the network.

    Patches ``from_pretrained`` ON the real classes rather than replacing the
    classes on the module: ``transformers`` is a ``_LazyModule`` and a
    ``monkeypatch.setattr`` against the module object does not survive a
    subsequent ``from transformers import Sam2Model`` (verified, and the reason
    the naive version of this test passed vacuously against real weights).
    """
    from transformers import Sam2Model, Sam2Processor

    import agrinav.data.sam2_predictor as backend

    seen: dict[str, dict[str, Any]] = {}

    class FakeParameter:
        def numel(self) -> int:
            return 7

    class FakeModel:
        def to(self, device: str) -> "FakeModel":
            return self

        def eval(self) -> "FakeModel":
            return self

        def parameters(self) -> list[FakeParameter]:
            return [FakeParameter()]

    def fake_model_loader(model_id: str, **kwargs: Any) -> FakeModel:
        seen["model"] = {"model_id": model_id, **kwargs}
        return FakeModel()

    def fake_processor_loader(model_id: str, **kwargs: Any) -> object:
        seen["processor"] = {"model_id": model_id, **kwargs}
        return object()

    monkeypatch.setattr(Sam2Model, "from_pretrained", fake_model_loader)
    monkeypatch.setattr(Sam2Processor, "from_pretrained", fake_processor_loader)

    sha = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
    predictor = backend.build_predictor(
        model_id="facebook/sam2.1-hiera-large",
        revision=sha,
        device="cpu",
        verify_revision=False,
    )
    assert predictor is not None
    assert seen["model"]["revision"] == sha, "model weights must be pinned"
    assert seen["processor"]["revision"] == sha, "preprocessing must be pinned too"
    assert seen["model"]["model_id"] == "facebook/sam2.1-hiera-large"


def test_resolve_pinned_revision_queries_the_hub_with_the_requested_sha(monkeypatch):
    import huggingface_hub

    import agrinav.data.sam2_predictor as backend

    sha = "665f8e2ad61cf5f53d65644ff27c8ee525124610"
    asked: dict[str, Any] = {}

    class FakeInfo:
        def __init__(self, resolved: str) -> None:
            self.sha = resolved
            self.lastModified = "2025-08-15 21:19:57+00:00"

    class FakeApi:
        def model_info(self, model_id: str, revision: str | None = None, **kwargs: Any):
            asked["model_id"] = model_id
            asked["revision"] = revision
            return FakeInfo(sha)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    result = backend.resolve_pinned_revision("facebook/sam2.1-hiera-large", sha)
    assert asked["revision"] == sha
    assert result["resolved_sha"] == sha


def test_resolve_pinned_revision_refuses_when_the_hub_resolves_something_else(monkeypatch):
    import huggingface_hub

    import agrinav.data.sam2_predictor as backend

    class FakeInfo:
        sha = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        lastModified = None

    class FakeApi:
        def model_info(self, *args: Any, **kwargs: Any):
            return FakeInfo()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    with pytest.raises(SamPreflightError, match="not the pinned commit"):
        backend.resolve_pinned_revision(
            "facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610"
        )
