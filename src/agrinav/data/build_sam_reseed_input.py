#!/usr/bin/env python3
"""Build the (zip + proposals JSON) input pair that ``sam_box_to_mask`` requires.

WHY A BUILDER INSTEAD OF LOOSENING THE PREFLIGHT
------------------------------------------------
``sam_box_to_mask.process`` reads images out of a **zip** and refuses to touch
any image whose recorded ``sha256`` and ``width``/``height`` do not match the
bytes it actually decodes. The phase-2 rebuild
(``RICE_phase2_rebuild_2026-07-29/``) is a *directory* tree, so the two do not
connect. The wrong fix is to teach the SAM stage to read loose directories and
skip the hash check. The right fix is this: materialise a zip whose members are
exactly the images we intend to process, and a proposal document whose
``sha256``/``width``/``height`` are computed from those same bytes. Every
existing preflight then runs unchanged, on real data, and still fails closed.

THE SEALED SPLIT
----------------
``RICE_phase2_rebuild_2026-07-29`` carries a 261-image ``test`` split. A
previous phase-2 archive silently mis-exported 179 of those images into train
and voided two runs (``TEST_SPLIT_STATUS.md``). This builder therefore treats
``test`` as **sealed** through a closed allow-list rather than a deny-list:

* ``--splits`` is validated against ``SELECTABLE_SPLITS``; anything else,
  including a typo, is a hard error rather than a silently empty selection.
* after paths are resolved, every selected path is re-checked to confirm it
  does not live under a sealed split directory. Belt and braces, because the
  first check trusts ``--rebuild-root`` to be the tree we think it is.
* the emitted proposal document carries each image's ``split`` so the sealing
  survives into every downstream artifact.

Sampling for a pilot is content-addressed (``sha256(seed|image_sha)``), so the
same ``--sample-seed`` reproduces the same image set on any machine and in any
filesystem order, and re-running cannot quietly reshuffle it.

CLI::

    python -m agrinav.data.build_sam_reseed_input \\
        --rebuild-root ~/Downloads/RICE_phase2_rebuild_2026-07-29 \\
        --splits train,valid \\
        --sample-size 40 --sample-seed 20260831 \\
        --out-zip artifacts/sam_reseed/pilot_images.zip \\
        --out-json artifacts/sam_reseed/pilot_proposals_unreviewed.coco.json \\
        --out-manifest artifacts/sam_reseed/pilot_input_manifest.json

Everything it writes is an UNREVIEWED PROPOSAL INPUT. It mints no truth: it
copies human boxes verbatim and adds no geometry of its own.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agrinav.data.sam_box_to_mask import COCO_CATEGORY_TO_LABEL, SamPreflightError

#: Splits a caller may ever ask for. CLOSED allow-list, not a deny-list.
SELECTABLE_SPLITS: tuple[str, ...] = ("train", "valid")

#: Splits that must never be read, embedded, or emitted. See TEST_SPLIT_STATUS.md.
SEALED_SPLITS: frozenset[str] = frozenset({"test"})

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

#: Marker written into every artifact so no downstream reader can mistake it.
NOT_TRUTH = (
    "UNREVIEWED PROPOSAL INPUT. Human boxes copied verbatim; no mask, no label "
    "and no review decision is asserted here. Not training truth."
)


class ReseedInputError(SamPreflightError):
    """Raised when the input tree cannot be trusted. Always fatal."""


def parse_splits(raw: str) -> tuple[str, ...]:
    """Validate a comma-separated split list against the closed allow-list."""
    requested = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not requested:
        raise ReseedInputError("--splits is empty; nothing to do")
    sealed = [s for s in requested if s in SEALED_SPLITS]
    if sealed:
        raise ReseedInputError(
            f"refusing to read sealed split(s) {sealed}. The {sorted(SEALED_SPLITS)} "
            "split of the phase-2 rebuild is sealed (TEST_SPLIT_STATUS.md); a "
            "previous archive leaked 179 of its images into train and voided two "
            "runs. If a re-seed of it is ever genuinely wanted, that is a separate, "
            "explicitly authorised piece of work."
        )
    unknown = [s for s in requested if s not in SELECTABLE_SPLITS]
    if unknown:
        raise ReseedInputError(
            f"unknown split(s) {unknown}; selectable splits are "
            f"{list(SELECTABLE_SPLITS)} (a typo must not silently select nothing)"
        )
    return requested


def assert_not_sealed(path: Path, rebuild_root: Path) -> None:
    """Re-assert that a resolved image path is outside every sealed split dir."""
    try:
        relative = path.resolve().relative_to(rebuild_root.resolve())
    except ValueError as exc:
        raise ReseedInputError(f"{path} resolves outside --rebuild-root {rebuild_root}") from exc
    parts = {p.lower() for p in relative.parts}
    hit = parts & {s.lower() for s in SEALED_SPLITS}
    if hit:
        raise ReseedInputError(
            f"{relative} lies under sealed split component(s) {sorted(hit)}; refusing"
        )


def verify_annotation_hashes(rebuild_root: Path, splits: Sequence[str]) -> dict[str, str]:
    """Check each split's COCO file against ``manifests/provenance.json``.

    The rebuild pins its own outputs by sha256. If a COCO file has drifted, the
    split membership we are about to trust is not the one that was reviewed.
    """
    provenance_path = rebuild_root / "manifests" / "provenance.json"
    if not provenance_path.is_file():
        raise ReseedInputError(f"missing {provenance_path}; cannot verify annotation integrity")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    recorded = provenance.get("json_sha256")
    if not isinstance(recorded, dict):
        raise ReseedInputError(f"{provenance_path}: no json_sha256 block to verify against")

    actual: dict[str, str] = {}
    for split in splits:
        key = f"annotations/instances_{split}.coco.json"
        path = rebuild_root / "annotations" / f"instances_{split}.coco.json"
        if not path.is_file():
            raise ReseedInputError(f"missing {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = recorded.get(key)
        if expected is None:
            raise ReseedInputError(f"{key} is not pinned in {provenance_path}")
        if digest.lower() != str(expected).lower():
            raise ReseedInputError(
                f"{key} sha256 drift: file is {digest}, provenance.json pins {expected}. "
                "The split membership on disk is not the one that was reviewed."
            )
        actual[key] = digest
    return actual


def select_sample(image_shas: Sequence[str], sample_size: int, sample_seed: int) -> frozenset[str]:
    """Content-addressed deterministic subset. ``sample_size <= 0`` means all."""
    if sample_size <= 0 or sample_size >= len(image_shas):
        return frozenset(image_shas)
    ranked = sorted(
        image_shas,
        key=lambda sha: hashlib.sha256(f"{sample_seed}|{sha}".encode("utf-8")).hexdigest(),
    )
    return frozenset(ranked[:sample_size])


def _read_split(rebuild_root: Path, split: str) -> dict[str, Any]:
    path = rebuild_root / "annotations" / f"instances_{split}.coco.json"
    return json.loads(path.read_text(encoding="utf-8"))


def verify_image_bytes(path: Path, recorded_sha: Any, width: Any, height: Any) -> bytes:
    """Return the file bytes only if hash and decoded size match the record.

    Mirrors ``sam_box_to_mask.preflight_image`` exactly. Never substitutes a
    placeholder: an invalid width zeroes the dimension downstream and skips all
    geometry validation for that record.
    """
    if not isinstance(recorded_sha, str) or not SHA256_RE.match(recorded_sha):
        raise ReseedInputError(f"{path.name}: sha256 missing or malformed ({recorded_sha!r})")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ReseedInputError(f"{path.name}: width/height invalid ({width}x{height})")
    if not path.is_file():
        raise ReseedInputError(f"{path}: image file listed in COCO is missing on disk")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest().lower()
    if actual != recorded_sha.lower():
        raise ReseedInputError(
            f"{path.name}: sha256 mismatch (COCO records {recorded_sha.lower()}, "
            f"disk is {actual})"
        )
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:
        if im.size != (width, height):
            raise ReseedInputError(
                f"{path.name}: decoded size {im.size} != recorded ({width}, {height})"
            )
    return raw


def build(
    *,
    rebuild_root: Path,
    splits: Sequence[str],
    sample_size: int,
    sample_seed: int,
    out_zip: Path,
    out_json: Path,
    out_manifest: Path,
) -> dict[str, Any]:
    """Materialise the zip + proposals JSON pair and a provenance manifest."""
    if not rebuild_root.is_dir():
        raise ReseedInputError(f"--rebuild-root {rebuild_root} is not a directory")
    annotation_hashes = verify_annotation_hashes(rebuild_root, splits)

    # Pass 1: index every candidate image and its boxes, per split.
    per_split: dict[str, dict[str, Any]] = {}
    all_shas: list[str] = []
    for split in splits:
        doc = _read_split(rebuild_root, split)
        boxes_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in doc["annotations"]:
            if ann["category_id"] not in COCO_CATEGORY_TO_LABEL:
                raise ReseedInputError(
                    f"{split}: annotation {ann['id']} has category_id "
                    f"{ann['category_id']}, which is outside the closed map "
                    f"{sorted(COCO_CATEGORY_TO_LABEL)}. It would become a counted "
                    "drop downstream; resolve the class map before re-seeding."
                )
            boxes_by_image.setdefault(ann["image_id"], []).append(ann)
        per_split[split] = {"doc": doc, "boxes": boxes_by_image}
        for image in doc["images"]:
            sha = image.get("sha256")
            if not isinstance(sha, str) or not SHA256_RE.match(sha):
                raise ReseedInputError(
                    f"{split}/{image.get('file_name')}: sha256 missing or malformed"
                )
            all_shas.append(sha.lower())

    if len(set(all_shas)) != len(all_shas):
        duplicates = len(all_shas) - len(set(all_shas))
        raise ReseedInputError(
            f"{duplicates} duplicate image sha256 across {list(splits)}: the same "
            "pixels appear twice, which would produce two proposal rows for one "
            "object and is a cross-split leakage signal. Resolve before re-seeding."
        )

    selected = select_sample(all_shas, sample_size, sample_seed)

    # Pass 2: verify bytes, write the zip, and emit the proposal document.
    images_out: list[dict[str, Any]] = []
    annotations_out: list[dict[str, Any]] = []
    image_id = 0
    ann_id = 0
    label_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    reoriented = 0
    total_bytes = 0

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for split in splits:
            doc = per_split[split]["doc"]
            boxes_by_image = per_split[split]["boxes"]
            for image in sorted(doc["images"], key=lambda im: im["id"]):
                sha = str(image["sha256"]).lower()
                if sha not in selected:
                    continue
                source_path = rebuild_root / "images" / split / image["file_name"]
                assert_not_sealed(source_path, rebuild_root)
                raw = verify_image_bytes(
                    source_path, image["sha256"], image.get("width"), image.get("height")
                )
                member = f"{split}/{image['file_name']}"
                archive.writestr(member, raw)
                total_bytes += len(raw)

                image_id += 1
                reoriented += 1 if image.get("exif_reoriented") else 0
                split_counts[split] = split_counts.get(split, 0) + 1
                images_out.append(
                    {
                        "id": image_id,
                        "file_name": member,
                        "width": int(image["width"]),
                        "height": int(image["height"]),
                        "sha256": sha,
                        "split": split,
                        "group_id": image.get("group_id"),
                        "capture_family": image.get("group_id"),
                        "exif_reoriented": bool(image.get("exif_reoriented")),
                        "source_dataset": "rice_phase2_rebuild_2026-07-29",
                        "source_split": image.get("source_split"),
                        "review_status": "unreviewed_proposal",
                    }
                )
                for ann in sorted(boxes_by_image.get(image["id"], []), key=lambda a: a["id"]):
                    ann_id += 1
                    label = COCO_CATEGORY_TO_LABEL[ann["category_id"]]
                    label_counts[label] = label_counts.get(label, 0) + 1
                    annotations_out.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": int(ann["category_id"]),
                            "bbox": [float(v) for v in ann["bbox"]],
                            "area": float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])),
                            "iscrowd": int(ann.get("iscrowd", 0)),
                            "segmentation": [],
                            "source_annotation_id": int(ann["id"]),
                            "source_split": split,
                            "annotation_origin": "coco_human_box",
                            "review_status": "unreviewed_proposal",
                        }
                    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    coco = {
        "info": {
            "description": "RICE phase-2 rebuild human boxes, staged for SAM2.1 box->mask re-seed",
            "WARNING": NOT_TRUTH,
            "generated_at": generated_at,
            "generated_by": "agrinav.data.build_sam_reseed_input",
            "source_root": str(rebuild_root),
            "splits": list(splits),
            "sealed_splits_excluded": sorted(SEALED_SPLITS),
            "sample_size": sample_size,
            "sample_seed": sample_seed,
        },
        "licenses": [],
        "categories": [
            {"id": cid, "name": name} for cid, name in sorted(COCO_CATEGORY_TO_LABEL.items())
        ],
        "images": images_out,
        "annotations": annotations_out,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco), encoding="utf-8")

    manifest = {
        "schema_version": "agrinav.sam_reseed_input.v1",
        "generated_at": generated_at,
        "generated_by": "agrinav.data.build_sam_reseed_input",
        "warning": NOT_TRUTH,
        "source_root": str(rebuild_root),
        "splits_selected": list(splits),
        "splits_sealed_and_excluded": sorted(SEALED_SPLITS),
        "sample_size_requested": sample_size,
        "sample_seed": sample_seed,
        "sample_rule": "sort by sha256(f'{seed}|{image_sha256}'), take first N",
        "annotation_file_sha256": annotation_hashes,
        "counts": {
            "images": len(images_out),
            "images_by_split": split_counts,
            "boxes": len(annotations_out),
            "boxes_by_label": label_counts,
            "exif_reoriented_images": reoriented,
            "image_bytes": total_bytes,
        },
        "out_zip": str(out_zip),
        "out_zip_sha256": hashlib.sha256(out_zip.read_bytes()).hexdigest(),
        "out_json": str(out_json),
        "out_json_sha256": hashlib.sha256(out_json.read_bytes()).hexdigest(),
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage RICE rebuild boxes for a SAM2.1 re-seed")
    ap.add_argument("--rebuild-root", required=True, type=Path)
    ap.add_argument(
        "--splits",
        default=",".join(SELECTABLE_SPLITS),
        help=f"comma-separated, from {list(SELECTABLE_SPLITS)}; 'test' is refused",
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="pilot subset size; 0 or negative means every image in the splits",
    )
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--out-zip", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    args = ap.parse_args(argv)

    try:
        splits = parse_splits(args.splits)
        manifest = build(
            rebuild_root=args.rebuild_root,
            splits=splits,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            out_zip=args.out_zip,
            out_json=args.out_json,
            out_manifest=args.out_manifest,
        )
    except SamPreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote SAM re-seed input ({NOT_TRUTH})")
    print(f"  zip      : {manifest['out_zip']}")
    print(f"  proposals: {manifest['out_json']}")
    print(f"  manifest : {args.out_manifest}")
    for key, value in manifest["counts"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
