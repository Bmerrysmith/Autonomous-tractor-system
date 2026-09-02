#!/usr/bin/env python3
"""A real, revision-pinned SAM2.1 backend for ``sam_box_to_mask.BoxPredictor``.

WHY THIS FILE EXISTS INSTEAD OF ``default_predictor_factory``
------------------------------------------------------------
``sam_box_to_mask.default_predictor_factory`` builds the *official* ``sam2``
package predictor::

    SAM2ImagePredictor.from_pretrained(model_id, revision=revision)

That call does not pin anything. In the upstream package ``from_pretrained``
forwards ``**kwargs`` to ``build_sam2_hf`` -> ``build_sam2`` ->
``SAM2ImagePredictor.__init__``, none of which pass ``revision`` to
``hf_hub_download``; every one of them absorbs it into ``**kwargs``. The
download therefore resolves the repo's *current* ``main``, and the pinned sha
is recorded into provenance while having had no effect on the weights that
drew the masks. That is precisely the failure mode ``validate_model_revision``
exists to prevent, one layer lower down.

``transformers`` honours ``revision`` on ``from_pretrained`` for both the model
and the processor, so this backend can prove the pin: it resolves the commit
through ``huggingface_hub`` first and re-asserts that the resolved sha equals
the requested one. The pin is verified, not merely recorded.

WHAT THIS BACKEND MAY AND MAY NOT DO
------------------------------------
* It implements exactly two methods: ``set_image`` and ``predict(box=...,
  multimask_output=...)``. Geometry only.
* ``predict`` accepts **one box** and no text of any kind. There is no code
  path here that takes a string prompt, a class name or a label. The caller's
  label never reaches this object.
* It computes the image embedding once per ``set_image`` and reuses it for
  every box on that image, which is the amortisation the cost model in
  ``sam_box_to_mask`` depends on (median 33 boxes/image on the phase-2
  rebuild; a per-box re-encode is ~33x the work).
* It records timing counters for the pilot measurement, and nothing else. It
  writes no files and mutates no global state.

Usage (programmatic, preferred - the factory is a closure, not global state)::

    from agrinav.data.sam2_predictor import make_predictor_factory
    factory = make_predictor_factory(
        model_id="facebook/sam2.1-hiera-large",
        revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
    )

Pin check (no GPU, no weights download - HF metadata API only)::

    python -m agrinav.data.sam2_predictor --verify-revision \
        --model-id facebook/sam2.1-hiera-large \
        --revision 665f8e2ad61cf5f53d65644ff27c8ee525124610
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from agrinav.data.sam_box_to_mask import (
    FORBIDDEN_PREDICT_KWARGS,
    SamPreflightError,
    validate_model_revision,
)

#: Torch dtypes this backend will accept, by name. Closed set: an unrecognised
#: dtype string is an error, never a silent fallback to float32 (which would
#: change mask geometry relative to what the run manifest claims).
SUPPORTED_DTYPES: tuple[str, ...] = ("float32", "bfloat16", "float16")

#: Default compute dtype. bfloat16 matches upstream SAM2's own inference
#: recipe and is what any throughput number in a pilot report refers to.
DEFAULT_DTYPE = "bfloat16"


@dataclass
class PredictorCounters:
    """Measurement-only counters. Nothing here influences a mask."""

    set_image_calls: int = 0
    predict_calls: int = 0
    set_image_seconds: float = 0.0
    predict_seconds: float = 0.0
    model_load_seconds: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "set_image_calls": self.set_image_calls,
            "predict_calls": self.predict_calls,
            "set_image_seconds": round(self.set_image_seconds, 4),
            "predict_seconds": round(self.predict_seconds, 4),
            "model_load_seconds": round(self.model_load_seconds, 4),
            **self.extras,
        }


def resolve_pinned_revision(model_id: str, revision: str) -> dict[str, Any]:
    """Confirm ``revision`` is a real immutable commit of ``model_id``.

    Raises ``SamPreflightError`` if the sha cannot be confirmed. Returns the
    provenance fields a run manifest should carry (resolved sha, repo
    ``lastModified``, and the time we checked).
    """
    pinned = validate_model_revision(revision)
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(model_id, revision=pinned, files_metadata=False)
    except Exception as exc:  # network / 404 / auth - all fatal, all reported
        raise SamPreflightError(
            f"could not resolve {model_id}@{pinned} on the HuggingFace API: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    resolved = str(getattr(info, "sha", "") or "").lower()
    if not resolved.startswith(pinned.lower()) and not pinned.lower().startswith(resolved):
        raise SamPreflightError(
            f"{model_id}: requested revision {pinned} resolved to sha {resolved!r}; "
            "refusing to run against weights that are not the pinned commit."
        )
    last_modified = getattr(info, "lastModified", None)
    return {
        "model_id": model_id,
        "requested_revision": pinned,
        "resolved_sha": resolved,
        "repo_last_modified": str(last_modified) if last_modified is not None else None,
        "resolved_via": "huggingface_hub.HfApi().model_info",
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _torch_dtype(name: str) -> Any:
    if name not in SUPPORTED_DTYPES:
        raise SamPreflightError(
            f"unsupported dtype {name!r}; choose one of {list(SUPPORTED_DTYPES)}. "
            "Refusing to substitute a default: dtype changes mask geometry."
        )
    import torch

    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


class Sam2TransformersPredictor:
    """``BoxPredictor`` over ``transformers`` SAM2.1. Box prompts only."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str,
        dtype: Any,
        counters: PredictorCounters,
    ) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self.counters = counters
        self._embeddings: Any | None = None
        self._original_sizes: Any | None = None

    # -- BoxPredictor ----------------------------------------------------
    def set_image(self, image: np.ndarray) -> None:
        """Encode one RGB image. The embedding is reused for every box."""
        import torch

        started = time.perf_counter()
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise SamPreflightError(f"set_image expects HxWx3 RGB, got shape {arr.shape}")
        inputs = self._processor(images=arr, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            self._embeddings = self._model.get_image_embeddings(pixel_values)
        self._original_sizes = inputs["original_sizes"]
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        self.counters.set_image_calls += 1
        self.counters.set_image_seconds += time.perf_counter() - started

    def predict(
        self, *, box: np.ndarray, multimask_output: bool, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Decode masks for ONE xyxy box. ``**kwargs`` exists only to refuse."""
        import torch

        forbidden = FORBIDDEN_PREDICT_KWARGS & set(kwargs)
        if forbidden:
            raise SamPreflightError(
                f"text-style prompt kwargs are forbidden in this backend: {sorted(forbidden)}"
            )
        if kwargs:
            raise SamPreflightError(f"unexpected predict kwargs: {sorted(kwargs)}")
        if self._embeddings is None or self._original_sizes is None:
            raise SamPreflightError("predict() called before set_image()")

        started = time.perf_counter()
        xyxy = [float(v) for v in np.asarray(box).reshape(-1)[:4]]
        prompt = self._processor(
            input_boxes=[[xyxy]],
            original_sizes=self._original_sizes,
            return_tensors="pt",
        )
        input_boxes = prompt["input_boxes"].to(device=self._device, dtype=self._dtype)
        with torch.inference_mode():
            out = self._model(
                input_boxes=input_boxes,
                image_embeddings=self._embeddings,
                multimask_output=bool(multimask_output),
            )
        masks = self._processor.post_process_masks(
            out.pred_masks.float().cpu(), self._original_sizes, binarize=True
        )[0]
        masks_arr = np.asarray(masks)
        mask_np = masks_arr.reshape(-1, *masks_arr.shape[-2:]).astype(bool)
        scores_np = out.iou_scores.float().cpu().numpy().reshape(-1)
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        self.counters.predict_calls += 1
        self.counters.predict_seconds += time.perf_counter() - started
        return mask_np, scores_np, None


def build_predictor(
    *,
    model_id: str,
    revision: str,
    device: str = "cuda",
    dtype: str = DEFAULT_DTYPE,
    counters: PredictorCounters | None = None,
    verify_revision: bool = True,
) -> Sam2TransformersPredictor:
    """Load SAM2.1 at a verified pinned commit and wrap it as a BoxPredictor."""
    import torch
    from transformers import Sam2Model, Sam2Processor

    pinned = validate_model_revision(revision)
    if verify_revision:
        resolve_pinned_revision(model_id, pinned)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SamPreflightError(
            f"device={device!r} requested but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}). Refusing to silently fall back to CPU: "
            "a CPU run would produce a throughput number that means nothing for "
            "the GPU ETA this pilot exists to compute."
        )
    torch_dtype = _torch_dtype(dtype)
    counters = counters if counters is not None else PredictorCounters()

    started = time.perf_counter()
    # Typed as Any: the transformers stubs describe .to()/.eval() through a
    # decorator wrapper that does not round-trip the concrete model class.
    model: Any = Sam2Model.from_pretrained(model_id, revision=pinned, dtype=torch_dtype)
    model.to(device)
    model.eval()
    processor = Sam2Processor.from_pretrained(model_id, revision=pinned)
    counters.model_load_seconds = time.perf_counter() - started
    counters.extras["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
    counters.extras["device"] = device
    counters.extras["dtype"] = dtype
    return Sam2TransformersPredictor(model, processor, device, torch_dtype, counters)


def make_predictor_factory(
    *,
    model_id: str,
    revision: str,
    device: str = "cuda",
    dtype: str = DEFAULT_DTYPE,
    counters: PredictorCounters | None = None,
    verify_revision: bool = True,
) -> Callable[[], Sam2TransformersPredictor]:
    """Zero-arg factory closure for ``sam_box_to_mask.process(predictor_factory=...)``."""

    def _factory() -> Sam2TransformersPredictor:
        return build_predictor(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            counters=counters,
            verify_revision=verify_revision,
        )

    return _factory


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SAM2.1 transformers backend utilities")
    ap.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--revision", required=True)
    ap.add_argument(
        "--verify-revision",
        action="store_true",
        help="resolve the pin against the HuggingFace API and print provenance",
    )
    args = ap.parse_args(argv)
    try:
        if args.verify_revision:
            print(json.dumps(resolve_pinned_revision(args.model_id, args.revision), indent=2))
        else:
            print(validate_model_revision(args.revision))
    except SamPreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
