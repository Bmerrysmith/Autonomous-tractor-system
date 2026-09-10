"""Immutable launch evidence shared by both detector trainers.

Each attempt has its own manifest and source archive. ``status.json`` is an
atomic, mutable progress pointer; it is never the source of configuration truth.
Dataset hashes cover annotations and available split manifests, not a fresh
verification of image bytes. Run the dataset preflight before a research campaign.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "agrinav.experiment.v1"
RUNTIME_KEYS = {"train_dataset", "val_dataset", "backbone_init"}
# Exclude locations and logging from a within-recipe, across-seed comparison.
LOCATION_KEYS = {
    "checkpoint_dir",
    "out_dir",
    "ann_file",
    "images_root",
    "val_ann_file",
    "val_images_root",
    "data_root",
    "riceseg_backbone",
    "resume",
}
REPORTING_KEYS = {"save_every", "progress_interval", "no_progress", "dump_grad_norms"}


def declarative_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep values, never object reprs or process-specific memory addresses."""

    def convert(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, Mapping) and all(isinstance(k, str) for k in value):
            return {k: convert(v) for k, v in value.items()}
        raise ValueError(f"configuration contains a non-declarative {type(value).__name__}")

    return {k: convert(v) for k, v in config.items() if k not in RUNTIME_KEYS}


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _source_snapshot(out_dir: Path, run_id: str) -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    files = {
        str(p.relative_to(package)).replace("\\", "/"): p.read_bytes()
        for p in sorted(package.rglob("*.py"))
    }
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    filename = f"source-{run_id}.zip"
    with zipfile.ZipFile(out_dir / filename, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(f"agrinav/{name}", data)
    return {
        "sha256": digest(hashes),
        "files": hashes,
        "archive": filename,
        "archive_sha256": file_sha256(out_dir / filename),
    }


def _dataset_evidence(dataset: Any, ann_file: Any = None) -> dict[str, Any]:
    actual_ann = getattr(dataset, "ann_file", None)
    if ann_file and actual_ann and Path(ann_file).resolve() != Path(actual_ann).resolve():
        raise ValueError("configured annotations differ from the dataset's actual annotation file")
    ann_file = actual_ann or ann_file
    evidence: dict[str, Any] = {"images": len(dataset) if dataset is not None else None}
    images = getattr(dataset, "images", None)
    if isinstance(images, list):
        membership = [{"id": row["id"], "file_name": row["file_name"]} for row in images]
        evidence["selected_images_sha256"] = digest(membership)
        evidence["selected_images"] = membership
    if hasattr(dataset, "CLASS_NAMES"):
        evidence["class_names"] = list(dataset.CLASS_NAMES)
    for name in ("img_size", "augment"):
        if hasattr(dataset, name):
            evidence[name] = getattr(dataset, name)
    if ann_file:
        path = Path(ann_file).resolve()
        evidence.update(annotation_path=str(path), annotation_sha256=file_sha256(path))
        manifests = path.parent.parent / "manifests"
        evidence["manifests"] = {
            name: file_sha256(manifests / name)
            for name in ("split_membership.json", "grouped_split.json", "provenance.json")
            if (manifests / name).is_file()
        }
    return evidence


def start_run(
    config: Mapping[str, Any],
    out_dir: str | os.PathLike[str],
    *,
    trainer: str,
    train_dataset: Any,
    val_dataset: Any,
    protocol: Mapping[str, Any],
    device: str,
    val_ann_file: Any = None,
    resume_from: str | None = None,
    start_epoch: int = 1,
    external_model: bool = False,
) -> dict[str, Any]:
    """Capture evidence and mark incomplete before the first optimizer step.

    Fresh runs must use fresh output directories. Resumes retain prior manifests;
    they remain inspectable but are excluded from automatic research aggregation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not resume_from and any(
        (out_dir / name).exists() for name in ("metrics.jsonl", "run.json", "status.json")
    ):
        raise FileExistsError(f"run artifacts already exist in {out_dir}; use a fresh directory")
    safe = declarative_config(config)
    run_id = uuid.uuid4().hex
    data = {
        "train": _dataset_evidence(train_dataset, safe.get("ann_file")),
        "validation": _dataset_evidence(val_dataset, val_ann_file or safe.get("val_ann_file")),
    }
    source = _source_snapshot(out_dir, run_id)
    packages = {}
    for name in ("torch", "torchvision", "numpy", "pycocotools", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
    }
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        environment["cuda"] = torch.version.cuda
        environment["device_name"] = (
            torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None
        )
    initializer = getattr(config.get("backbone_init"), "ckpt_path", None)
    initializer = initializer or safe.get("riceseg_backbone")
    initialization = {
        "checkpoint_sha256": file_sha256(initializer) if initializer else None,
        "external_model": external_model,
        "untracked_callable": config.get("backbone_init") is not None and not initializer,
    }
    recipe_config = {
        k: v for k, v in safe.items() if k not in LOCATION_KEYS | REPORTING_KEYS | {"seed"}
    }
    data_identity = {
        split: {k: v for k, v in evidence.items() if k != "annotation_path"}
        for split, evidence in data.items()
    }
    recipe = {
        "trainer": trainer,
        "config": recipe_config,
        "data": data_identity,
        "source_sha256": source["sha256"],
        "environment": environment,
        "initialization": initialization,
        "protocol": dict(protocol),
    }
    interval = int(safe.get("val_ap_interval", 0) or 0)
    epochs = int(safe.get("num_epochs", 0))
    eval_epochs = [
        e for e in range(1, epochs + 1) if interval > 0 and (e % interval == 0 or e == epochs)
    ]
    porcelain = _git("status", "--porcelain")
    manifest = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "trainer": trainer,
        "config": safe,
        "config_sha256": digest(safe),
        "recipe_sha256": digest(recipe),
        "recipe": recipe,
        "seed": safe.get("seed"),
        "data": data,
        "source": source,
        "environment": environment,
        "protocol": dict(protocol),
        "eval_epochs": eval_epochs,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": None if porcelain is None else bool(porcelain),
        "git_status": porcelain,
        "command": sys.argv,
        "resume_from": str(resume_from) if resume_from else None,
        "resume_sha256": file_sha256(resume_from) if resume_from else None,
        "start_epoch": start_epoch,
    }
    filename = f"manifest-{run_id}.json"
    with (out_dir / filename).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    provenance = {
        k: manifest[k]
        for k in (
            "run_id",
            "seed",
            "git_commit",
            "git_branch",
            "git_dirty",
            "config_sha256",
            "recipe_sha256",
            "recorded_at",
        )
    }
    provenance.update(manifest_file=filename, manifest_sha256=digest(manifest))
    atomic_json(
        out_dir / "status.json",
        {
            "completed": False,
            "epochs_planned": epochs,
            "epochs_completed": start_epoch - 1,
            "provenance": provenance,
        },
    )
    return provenance


def read_manifest(run_dir: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a referenced immutable manifest before trusting its comparison keys."""
    filename = provenance.get("manifest_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("missing or invalid manifest reference")
    manifest = json.loads((run_dir / filename).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or digest(manifest) != provenance.get(
        "manifest_sha256"
    ):
        raise ValueError("manifest checksum or schema mismatch")
    if digest(manifest["config"]) != manifest["config_sha256"]:
        raise ValueError("configuration checksum mismatch")
    if digest(manifest["recipe"]) != manifest["recipe_sha256"]:
        raise ValueError("recipe checksum mismatch")
    source = manifest["source"]
    if (
        Path(source["archive"]).name != source["archive"]
        or file_sha256(run_dir / source["archive"]) != source["archive_sha256"]
    ):
        raise ValueError("source archive checksum mismatch")
    return manifest


def finish_run(
    out_dir: str | os.PathLike[str], status: dict[str, Any], checkpoints: list[str]
) -> None:
    """Bind completion to the metric log and checkpoint bytes, then publish status."""
    out_dir = Path(out_dir)
    artifacts = {
        name: file_sha256(out_dir / name) for name in checkpoints if (out_dir / name).is_file()
    }
    if not artifacts:
        raise ValueError("cannot complete a run without a checkpoint")
    artifacts["metrics.jsonl"] = file_sha256(out_dir / "metrics.jsonl")
    if (out_dir / "run.json").exists():
        artifacts["run.json"] = file_sha256(out_dir / "run.json")
    atomic_json(out_dir / "status.json", {**status, "artifact_sha256": artifacts})
