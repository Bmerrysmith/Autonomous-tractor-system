#!/usr/bin/env python3
"""Measured pilot driver for the SAM2.1 box->mask re-seed.

This module OWNS NO PIPELINE LOGIC. Every mask, every candidate, every drop and
every preflight is produced by ``sam_box_to_mask.process`` exactly as the
documented CLI would; this file only

  1. re-asserts the sealed-split guarantee at the run boundary,
  2. verifies the model pin against the HuggingFace API before any GPU work,
  3. wraps the call in wall-clock and CUDA peak-memory instrumentation, and
  4. writes a run manifest (environment, pin, counts, rates) so the throughput
     number in a report can be traced to the run that produced it.

It exists because peak CUDA memory can only be read inside the process that
allocated it, so an external ``time`` around the CLI cannot produce the number
the 12 GB headroom question needs.

Extrapolating from a pilot: the cost model is ``per-image encode`` +
``per-box decode``, and those scale with different totals (2,318 images vs
74,917 boxes on the phase-2 rebuild). The manifest therefore reports the two
separately, because an ETA built on images/sec alone is wrong by whatever the
sample's boxes-per-image happens to differ from the corpus.

CLI::

    python -m agrinav.data.sam_reseed_pilot \\
        --coco-zip artifacts/sam_reseed/pilot_images.zip \\
        --proposals-json artifacts/sam_reseed/pilot_proposals_unreviewed.coco.json \\
        --model-revision 665f8e2ad61cf5f53d65644ff27c8ee525124610 \\
        --out-shard artifacts/sam_reseed/sam_raw/pilot_shard_000.jsonl \\
        --out-manifest artifacts/sam_reseed/pilot_run_manifest.json

Output is a RAW CANDIDATE SIDECAR: unreviewed proposals, not annotation
records and not training truth.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agrinav.data import sam_box_to_mask
from agrinav.data.build_sam_reseed_input import SEALED_SPLITS
from agrinav.data.sam2_predictor import (
    DEFAULT_DTYPE,
    PredictorCounters,
    make_predictor_factory,
    resolve_pinned_revision,
)
from agrinav.data.sam_box_to_mask import SamPreflightError

NOT_TRUTH = (
    "RAW SAM CANDIDATE SIDECAR. Unreviewed proposals only: no annotation record, "
    "no accepted/adjudicated status, no training truth. Human review in CVAT is "
    "still the only step that can produce truth."
)


def assert_no_sealed_split(proposals: dict[str, Any]) -> dict[str, int]:
    """Refuse a proposal document that carries any sealed-split image.

    Defence in depth: the builder already excludes them, but this driver may be
    pointed at any JSON on disk and a leak here is unrecoverable.
    """
    counts: dict[str, int] = {}
    for image in proposals.get("images", []):
        split = image.get("split")
        counts[str(split)] = counts.get(str(split), 0) + 1
    leaked = sorted(set(counts) & {s for s in SEALED_SPLITS})
    if leaked:
        raise SamPreflightError(
            f"proposals document contains images from sealed split(s) {leaked} "
            f"(counts {({k: counts[k] for k in leaked})}). Refusing: the phase-2 "
            "test split is sealed (TEST_SPLIT_STATUS.md)."
        )
    if "None" in counts:
        raise SamPreflightError(
            f"{counts['None']} proposal images carry no 'split' field; a sealed "
            "split cannot be proven absent, so the run is refused."
        )
    return counts


def collect_environment(device: str) -> dict[str, Any]:
    """Record what actually ran. Never guesses; unknown stays null."""
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device_requested": device,
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_memory_bytes"] = int(props.total_memory)
            env["gpu_capability"] = f"{props.major}.{props.minor}"
    except ImportError as exc:  # torch is required for a GPU pilot; say so
        env["torch_import_error"] = str(exc)
    for module_name in ("torchvision", "transformers", "numpy"):
        try:
            env[f"{module_name}_version"] = __import__(module_name).__version__
        except ImportError:
            env[f"{module_name}_version"] = None
    try:
        env["nvidia_driver_version"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        env["nvidia_driver_version"] = None
    return env


def _git_state(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def derive_rates(
    stats: dict[str, Any], counters: dict[str, Any], wall_seconds: float
) -> dict[str, Any]:
    """Throughput and the per-image / per-box split the ETA needs."""
    images = int(stats.get("images_processed", 0))
    boxes = int(stats.get("boxes_emitted", 0))
    inference_seconds = float(counters.get("set_image_seconds", 0.0)) + float(
        counters.get("predict_seconds", 0.0)
    )
    overhead_seconds = max(
        0.0, wall_seconds - inference_seconds - float(counters.get("model_load_seconds", 0.0))
    )
    return {
        "wall_seconds": round(wall_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "non_inference_seconds": round(overhead_seconds, 3),
        "images_per_second": round(images / wall_seconds, 4) if wall_seconds else None,
        "boxes_per_second": round(boxes / wall_seconds, 4) if wall_seconds else None,
        "boxes_per_image": round(boxes / images, 3) if images else None,
        "encode_seconds_per_image": (
            round(float(counters.get("set_image_seconds", 0.0)) / images, 4) if images else None
        ),
        "decode_seconds_per_box": (
            round(float(counters.get("predict_seconds", 0.0)) / boxes, 5) if boxes else None
        ),
        "overhead_seconds_per_box": round(overhead_seconds / boxes, 5) if boxes else None,
        "decoder_calls_per_box": (
            round(float(counters.get("predict_calls", 0)) / boxes, 3) if boxes else None
        ),
    }


def run(
    *,
    coco_zip: Path,
    proposals_json: Path,
    out_shard: str,
    out_manifest: Path,
    model_id: str,
    model_revision: str,
    device: str,
    dtype: str,
    shard_index: int,
    shard_count: int,
    repo_root: Path,
) -> dict[str, Any]:
    """Run one measured shard and write the run manifest."""
    proposals = json.loads(proposals_json.read_text(encoding="utf-8"))
    split_counts = assert_no_sealed_split(proposals)
    pin = resolve_pinned_revision(model_id, model_revision)

    counters = PredictorCounters()
    factory = make_predictor_factory(
        model_id=model_id,
        revision=pin["requested_revision"],
        device=device,
        dtype=dtype,
        counters=counters,
        verify_revision=False,  # already verified above, once, before any GPU work
    )

    torch = None
    if device.startswith("cuda"):
        import torch as _torch

        torch = _torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before = int(torch.cuda.memory_allocated())
    else:
        memory_before = 0

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    clock = time.perf_counter()
    stats = sam_box_to_mask.process(
        zip_path=coco_zip,
        proposals=proposals,
        predictor_factory=factory,
        out_shard=out_shard,
        shard_index=shard_index,
        shard_count=shard_count,
        model_id=model_id,
        model_revision=pin["requested_revision"],
    )
    wall_seconds = time.perf_counter() - clock

    memory: dict[str, Any] = {"cuda_measured": bool(torch is not None)}
    if torch is not None:
        memory.update(
            {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "allocated_before_bytes": memory_before,
                "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            }
        )
        memory["peak_reserved_gib"] = round(memory["peak_reserved_bytes"] / 2**30, 3)
        memory["device_total_gib"] = round(memory["device_total_bytes"] / 2**30, 3)
        memory["headroom_gib"] = round(memory["device_total_gib"] - memory["peak_reserved_gib"], 3)

    manifest = {
        "schema_version": "agrinav.sam_reseed_pilot_run.v1",
        "warning": NOT_TRUTH,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_pin": pin,
        "predictor_backend": "agrinav.data.sam2_predictor.Sam2TransformersPredictor",
        "predictor_backend_note": (
            "transformers Sam2Model/Sam2Processor, which honour revision= on "
            "from_pretrained. The upstream sam2 package's SAM2ImagePredictor."
            "from_pretrained absorbs revision into **kwargs and would download "
            "the repo's current main instead."
        ),
        "dtype": dtype,
        "inputs": {
            "coco_zip": str(coco_zip),
            "proposals_json": str(proposals_json),
            "images_by_split": split_counts,
            "sealed_splits_absent": sorted(SEALED_SPLITS),
        },
        "sharding": {"shard_index": shard_index, "shard_count": shard_count},
        "stage_stats": stats,
        "predictor_counters": counters.as_dict(),
        "rates": derive_rates(stats, counters.as_dict(), wall_seconds),
        "gpu_memory": memory,
        "environment": collect_environment(device),
        "git": _git_state(repo_root),
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measured SAM2.1 box->mask re-seed pilot")
    ap.add_argument("--coco-zip", required=True, type=Path)
    ap.add_argument("--proposals-json", required=True, type=Path)
    ap.add_argument("--model-id", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--model-revision", required=True)
    ap.add_argument("--out-shard", required=True)
    ap.add_argument("--out-manifest", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=DEFAULT_DTYPE)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)

    try:
        manifest = run(
            coco_zip=args.coco_zip,
            proposals_json=args.proposals_json,
            out_shard=args.out_shard,
            out_manifest=args.out_manifest,
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            repo_root=args.repo_root,
        )
    except SamPreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(NOT_TRUTH)
    print(f"  manifest: {args.out_manifest}")
    for key, value in manifest["stage_stats"].items():
        print(f"  stat.{key}: {value}")
    for key, value in manifest["rates"].items():
        print(f"  rate.{key}: {value}")
    for key, value in manifest["gpu_memory"].items():
        print(f"  gpu.{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
