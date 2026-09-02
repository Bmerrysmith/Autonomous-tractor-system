#!/usr/bin/env python3
"""Convert a CVAT "COCO 1.0" export into ``agrinav.annotation_record.v1`` JSONL.

This closes the gap named in ``docs/cvat.md`` §8: CVAT is the review step, but
its export is COCO, and the gate that decides what may enter a split
(``agrinav data-validate``) reads the annotation-record wire format. Until this
converter existed, reviewed work sat in a holding format and could not become
training truth.

Translating the *shapes* is mechanical, because ``agrinav.data.cvat_labels``
generated the CVAT project from the same ontology: every CVAT attribute has a
schema field, and CVAT's empty default stands for JSON ``null``. What is not
mechanical is everything a COCO export does not carry. Those decisions are
listed here, because a converter that guesses them is worse than no converter.

record_id
    Derived, never random and never a counter, so a second export of the same
    reviewed image produces the same row id::

        record_id = f"{dataset_id}:{sha256(namespace, dataset_id,
                       dataset_version, image_id, source_image_sha256)[:32]}"

    Deliberately excluded from the digest: CVAT task/job/annotation ids (they are
    renumbered per export), timestamps, and the annotation content — a second
    review pass must update a row, not orphan it, and annotation versions live in
    ``review.annotation_version``. ``image_id`` is the split manifest's key for
    the image, i.e. its file name.

source.*
    From ``--manifest``, which is the authority (``docs/cvat.md`` §6: "if they
    disagree, the manifest wins"). ``source_image_sha256`` is the manifest's
    ``sha256`` — the bytes the annotator actually looked at — never its
    ``source_sha256``, which is the pre-EXIF-normalisation original and describes
    different pixels than the ones the shapes were drawn on. A ``width``/
    ``height`` disagreement between manifest and export is a hard error rather
    than a silent preference: it means the coordinates were drawn against
    different pixels, and rescaling them would be a guess
    (``--allow-dimension-mismatch`` records the manifest's dimensions anyway and
    leaves coordinates untouched, so the validator sees any out-of-bounds
    geometry). ``group_id`` and ``split`` come from the manifest verbatim; only
    the split *name* is translated (``valid`` -> ``validation``). An image in the
    export with no manifest entry is a loud error, never a made-up group.

review.*
    Not in a COCO export at all. ``--review-metadata`` accepts a sidecar
    (``agrinav.cvat_review_metadata.v1``) exported from the CVAT REST API; when
    it is absent, every identity and timestamp is an explicit ``null`` and every
    record is ``unreviewed``. No identity is ever invented, and no review state is
    ever upgraded: ``review_status`` is copied from what CVAT states, and if two
    CVAT-derived sources disagree the converter fails rather than pick the more
    flattering one. Note that CVAT's COCO 1.0 exporter writes ``coco_instances``
    only, so the ``image_review`` tag label — which is where ``review_status``,
    ``verified_empty`` and ``unusable`` live — is dropped on export. Without the
    sidecar a converted package is reviewed work in a holding format, not truth.
    That is the honest outcome, not a bug.

provenance.human_edit_state vs attributes.human_edit_action
    Different fields, different vocabularies. ``human_edit_action`` is per object
    and is copied from the CVAT attribute (empty -> ``null``).
    ``human_edit_state`` is the record-level *summary* the schema asks for, and is
    derived from those per-object decisions: ``manual`` records are
    ``human_only``; an unreviewed record is ``unreviewed``; all-``accepted`` is
    ``accepted_unchanged``; objects only deleted and re-added is ``replaced``;
    anything else is ``edited``.

verified_empty / treatment_eligible
    Three-valued CVAT selects (``unknown`` -> ``null``). "The export contained no
    annotations for this image" is the ontology's named forbidden inference and is
    never used: an image with no shapes and no human verified-empty statement gets
    ``verified_empty: null``. ``treatment_eligible`` is never inferred from class.

annotation_confidence
    Has no CVAT default on purpose (``cvat_labels`` rule 3), but the wire format
    has no null for it. An object whose annotator never set it is a hard error
    naming the image and the object — the converter will not invent a confidence.

Geometry is preserved, not normalised: polygons stay polygons, masks stay RLE
(``iscrowd=1`` -> ``semantic_mask``, else ``instance_mask``), and a rectangle
stays a ``bbox``. The one deliberate exception is a four-point segmentation that
is exactly its own bounding box: some CVAT/datumaro versions synthesise those
corners for rectangle shapes, and letting them through would silently promote box
work into canonical polygon truth. They are recorded as ``bbox`` (counted in the
summary; ``--keep-rectangle-polygons`` disables it). The raw COCO object,
including its derived ``bbox``, is preserved in ``provenance.original_proposal``.

CLI::

    agrinav data-cvat-to-records \\
        --coco export/annotations/instances_default.json \\
        --manifest .../manifests/split_membership.json \\
        --dataset-id rice_phase2 --dataset-version rebuild-2026-07-29 \\
        --review-metadata artifacts/cvat/review_metadata.json \\
        --out artifacts/cvat/rice_phase2_reviewed.jsonl

The output is then checked by the gate itself (this module runs it too, unless
``--skip-validation``)::

    agrinav data-validate artifacts/cvat/rice_phase2_reviewed.jsonl \\
        --ontology data/ontology.v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from agrinav.data.cvat_labels import NULLABLE_BOOLEAN_UNSET
from agrinav.data.validate_annotation_package import (
    ACCEPTED_STATUSES,
    ANNOTATION_CONFIDENCE_VALUES,
    HUMAN_EDIT_ACTIONS,
    OCCLUSION_VALUES,
    REVIEW_STATUSES,
    SCHEMA_VERSION,
    AnnotationValidationError,
    OntologyEntry,
    load_ontology,
    validate_packages,
)

#: Version tag of the ``--review-metadata`` sidecar this module reads.
REVIEW_METADATA_SCHEMA = "agrinav.cvat_review_metadata.v1"

#: Namespace mixed into every ``record_id`` digest. Changing it changes every
#: derived id, so it is versioned rather than implicit.
RECORD_ID_NAMESPACE = "agrinav.cvat_export_to_records.v1"

#: Hex characters of that digest kept in a ``record_id`` (128 bits).
RECORD_ID_DIGEST_LENGTH = 32

#: Manifest split name -> the ``source.split`` vocabulary of the wire format.
#: Roboflow-derived manifests say ``valid``; the schema says ``validation``.
MANIFEST_SPLIT_NAMES: dict[str, str] = {
    "train": "train",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "challenge": "challenge",
    "ood": "ood",
    "unassigned": "unassigned",
}

#: Optional capture-lineage fields copied from a manifest entry when it has them,
#: and emitted as explicit ``null`` when it does not.
OPTIONAL_SOURCE_FIELDS: tuple[str, ...] = (
    "country",
    "site_id",
    "field_id",
    "session_id",
    "capture_pass_id",
    "frame_id",
    "source_photo_id",
)

#: Manifest keys consumed into the ``source`` block. Everything else in an entry
#: is preserved under ``source.capture_metadata`` rather than dropped.
CONSUMED_MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "sha256",
        "source_image_sha256",
        "width",
        "height",
        "group_id",
        "split",
        *OPTIONAL_SOURCE_FIELDS,
    }
)

#: Fields a ``--review-metadata`` entry may state. Any other key is an error: a
#: mistyped ``reviewer_id`` must not silently become "no reviewer".
REVIEW_METADATA_FIELDS: tuple[str, ...] = (
    "annotator_id",
    "annotator_completed_at",
    "reviewer_id",
    "reviewed_at",
    "review_status",
    "annotation_version",
    "guide_version",
    "verified_empty",
    "unusable",
)

#: Image-level review state a COCO file could carry (image attributes, or an
#: ``image_review`` tag shape). Identities never come from the export.
EXPORT_IMAGE_STATE_FIELDS: tuple[str, ...] = ("review_status", "verified_empty", "unusable")

#: How many individual problems a failure message lists before truncating.
MAX_REPORTED_ERRORS = 20


class CvatConversionError(RuntimeError):
    """A CVAT export cannot be converted into annotation records as requested."""


@dataclass(frozen=True)
class ProposalProvenance:
    """What the shapes in this export were before human review.

    A COCO export cannot say whether they were drawn by hand, imported, or
    proposed by a model, so the operator states it. The per-method requirements
    checked here are the ones ``validate_annotation_package`` enforces; checking
    them up front makes the failure name the missing *argument* instead of
    surfacing as a schema error several steps later.
    """

    method: str = "imported"
    model_id: str | None = None
    model_revision: str | None = None
    prompt: str | None = None
    generated_at: str | None = None
    thresholds: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.method not in {"manual", "imported", "model_assisted", "model_silence"}:
            raise CvatConversionError(
                f"--proposal-method {self.method!r} is not one of manual, imported, "
                "model_assisted, model_silence"
            )
        if self.method in {"model_assisted", "model_silence"}:
            missing = [
                name
                for name, value in (
                    ("--proposal-model-id", self.model_id),
                    ("--proposal-model-revision", self.model_revision),
                    ("--proposal-prompt", self.prompt),
                    ("--proposal-generated-at", self.generated_at),
                    ("--proposal-thresholds", self.thresholds),
                )
                if not value
            ]
            if missing:
                raise CvatConversionError(
                    f"--proposal-method {self.method!r} is model-derived and the wire format "
                    f"requires its full lineage; missing: {', '.join(missing)}"
                )
        if self.method == "manual" and any(
            (self.model_id, self.model_revision, self.prompt, self.generated_at, self.thresholds)
        ):
            raise CvatConversionError(
                "--proposal-method manual means there was no proposal; drop the --proposal-* "
                "arguments or choose another method"
            )

    def as_record_fields(self, original_proposal: list[dict[str, Any]]) -> dict[str, Any]:
        """The ``provenance`` block minus ``human_edit_state`` (a derived summary)."""
        manual = self.method == "manual"
        return {
            "proposal_model_id": self.model_id,
            "proposal_model_revision": self.model_revision,
            "proposal_method": self.method,
            "prompt": self.prompt,
            "thresholds": self.thresholds,
            "generated_at": self.generated_at,
            "original_proposal": None if manual else original_proposal,
        }


@dataclass(frozen=True)
class ConversionOptions:
    """Everything the conversion needs that the COCO file does not contain."""

    dataset_id: str
    proposal: ProposalProvenance
    dataset_version: str | None = None
    review_defaults: Mapping[str, Any] = field(default_factory=dict)
    review_by_image: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    image_uri_prefix: str = ""
    image_label_name: str = "image_review"
    allow_dimension_mismatch: bool = False
    keep_rectangle_polygons: bool = False
    lineage: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageTask:
    """One COCO image, its ``(raw shape, label name)`` pairs, and its manifest entry."""

    image: Mapping[str, Any]
    objects: Sequence[tuple[Mapping[str, Any], str]]
    manifest_key: str
    manifest_entry: Mapping[str, Any]

    @property
    def file_name(self) -> str:
        return str(self.image.get("file_name") or self.manifest_key)


@dataclass(frozen=True)
class ObjectContext:
    """Per-object translation context (kept together so signatures stay short)."""

    record_id: str
    file_name: str
    ontology: Mapping[str, OntologyEntry]
    options: ConversionOptions


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_json(path: Path, *, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise CvatConversionError(f"{what} not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CvatConversionError(f"{what} {path} could not be read: {exc}") from exc


def sha256_file(path: Path) -> str:
    """Hash a file so every record can name the exact export/manifest it came from."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_coco_export(path: Path) -> dict[str, Any]:
    """Read a CVAT COCO 1.0 export and check it has the parts we need."""
    document = _read_json(path, what="CVAT COCO export")
    if not isinstance(document, dict):
        raise CvatConversionError(f"CVAT COCO export {path} is not a JSON object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(document.get(key), list):
            raise CvatConversionError(
                f"CVAT COCO export {path} has no {key!r} list; this does not look like a COCO "
                "1.0 export (Actions -> Export task dataset -> COCO 1.0)"
            )
    return document


def load_split_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Read the split manifest into ``file name -> entry``.

    Accepts the two shapes in use: a mapping of file name to entry (as in
    ``manifests/split_membership.json``), and a list of entries each carrying a
    ``file_name``/``image_id`` key, optionally wrapped in ``images``/``records``.
    """
    document = _read_json(path, what="split manifest")
    if isinstance(document, dict):
        for key in ("images", "records", "samples"):
            if key in document:
                document = document[key]
                break
    if isinstance(document, dict):
        mapped = {str(name): value for name, value in document.items() if isinstance(value, dict)}
        if mapped:
            return mapped
    if isinstance(document, list):
        listed: dict[str, dict[str, Any]] = {}
        for item in document:
            if not isinstance(item, dict):
                continue
            name = item.get("file_name") or item.get("image_id") or item.get("image")
            if isinstance(name, str) and name:
                listed[name] = item
        if listed:
            return listed
    raise CvatConversionError(
        f"split manifest {path} holds no usable entries; expected a mapping of file name to "
        "entry, or a list of entries carrying 'file_name'"
    )


def load_review_metadata(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read the review sidecar into ``(defaults, per-image state)``.

    The sidecar is how CVAT REST-API facts (who annotated, who reviewed, when, and
    the job's review state) reach a converter that only sees COCO::

        {"schema_version": "agrinav.cvat_review_metadata.v1",
         "defaults": {"guide_version": "v1"},
         "jobs": [{"job_id": 7, "images": ["a.jpg"], "annotator_id": "...",
                   "reviewer_id": "...", "review_status": "accepted", ...}],
         "images": {"a.jpg": {"verified_empty": true}}}

    Precedence is image entry over job entry over ``defaults``. Unknown keys are
    rejected rather than ignored.
    """
    document = _read_json(path, what="review metadata")
    if not isinstance(document, dict):
        raise CvatConversionError(f"review metadata {path} is not a JSON object")
    version = document.get("schema_version")
    if version not in (None, REVIEW_METADATA_SCHEMA):
        raise CvatConversionError(
            f"review metadata {path} declares schema_version {version!r}; expected "
            f"{REVIEW_METADATA_SCHEMA!r}"
        )
    defaults = _review_entry(document.get("defaults") or {}, where=f"{path.name}:defaults")
    by_image: dict[str, dict[str, Any]] = {}
    for index, job in enumerate(document.get("jobs") or []):
        if not isinstance(job, dict):
            raise CvatConversionError(f"{path.name}:jobs[{index}] is not an object")
        images = job.get("images")
        if not isinstance(images, list) or not images:
            raise CvatConversionError(
                f"{path.name}:jobs[{index}] has no 'images' list; a job's review state can only "
                "be applied to the images it names"
            )
        state = _review_entry(
            {key: value for key, value in job.items() if key not in ("job_id", "images")},
            where=f"{path.name}:jobs[{index}]",
        )
        for name in images:
            by_image[str(name)] = dict(state)
    for name, entry in (document.get("images") or {}).items():
        if not isinstance(entry, dict):
            raise CvatConversionError(f"{path.name}:images[{name!r}] is not an object")
        merged = dict(by_image.get(str(name), {}))
        merged.update(_review_entry(entry, where=f"{path.name}:images[{name!r}]"))
        by_image[str(name)] = merged
    return defaults, by_image


def _review_entry(entry: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Validate one review-metadata entry and return only the fields it states."""
    unexpected = sorted(set(entry) - set(REVIEW_METADATA_FIELDS))
    if unexpected:
        raise CvatConversionError(
            f"{where}: unknown review-metadata field(s) {unexpected}; expected any of "
            f"{list(REVIEW_METADATA_FIELDS)}"
        )
    state: dict[str, Any] = {}
    for key, value in entry.items():
        if value is None:
            continue
        if key in ("verified_empty", "unusable"):
            if not isinstance(value, bool):
                raise CvatConversionError(f"{where}.{key}: must be true, false, or null")
        elif not (isinstance(value, str) and value.strip()):
            raise CvatConversionError(f"{where}.{key}: must be a non-empty string or null")
        elif key == "review_status" and value not in REVIEW_STATUSES:
            raise CvatConversionError(
                f"{where}.review_status: {value!r} is not one of {sorted(REVIEW_STATUSES)}"
            )
        state[key] = value
    return state


# --------------------------------------------------------------------------- #
# Attribute translation
# --------------------------------------------------------------------------- #
def _text_attribute(attributes: Mapping[str, Any], name: str) -> str | None:
    """CVAT text/select attribute -> ``str | None``; its empty default is the null."""
    value = attributes.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tristate_attribute(attributes: Mapping[str, Any], name: str, *, where: str) -> bool | None:
    """CVAT three-valued select -> ``bool | None`` (``unknown``/empty is the null).

    Only for the two ``boolean | null`` ontology fields. ``occlusion``'s
    ``unknown`` is a real value and must not pass through here.
    """
    value = attributes.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", NULLABLE_BOOLEAN_UNSET):
        return None
    if text in ("true", "yes", "1"):
        return True
    if text in ("false", "no", "0"):
        return False
    raise CvatConversionError(
        f"{where}: attribute {name!r} is {value!r}; expected true, false, or "
        f"{NULLABLE_BOOLEAN_UNSET!r}"
    )


def _checkbox_attribute(attributes: Mapping[str, Any], name: str, *, where: str) -> bool:
    """CVAT checkbox -> ``bool``; absent means the checkbox default, ``false``."""
    value = attributes.get(name)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "false", "no", "0"):
        return False
    if text in ("true", "yes", "1"):
        return True
    raise CvatConversionError(f"{where}: attribute {name!r} is {value!r}; expected true or false")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _is_rectangle_ring(points: Sequence[float], bbox: Sequence[Any]) -> bool:
    """True when a four-point ring is exactly the corners of its own bounding box."""
    if len(points) != 8 or len(bbox) != 4:
        return False
    x, y, width, height = (float(value) for value in bbox)
    corners = {(x, y), (x + width, y), (x + width, y + height), (x, y + height)}
    ring = {(float(points[i]), float(points[i + 1])) for i in range(0, 8, 2)}
    return ring == corners


def build_geometry(
    annotation: Mapping[str, Any], *, where: str, keep_rectangle_polygons: bool = False
) -> tuple[dict[str, Any], bool]:
    """Translate one COCO shape into a wire-format geometry.

    Returns the geometry and whether an exporter-synthesised rectangle ring was
    recorded as a ``bbox`` instead of a polygon (see the module docstring).
    """
    segmentation = annotation.get("segmentation")
    bbox = annotation.get("bbox")
    if isinstance(segmentation, Mapping) and segmentation:
        return _mask_geometry(segmentation, annotation, where=where), False
    ring = _polygon_ring(segmentation, where=where)
    if ring is not None:
        if (
            not keep_rectangle_polygons
            and isinstance(bbox, list)
            and _is_rectangle_ring(ring, bbox)
        ):
            return {"type": "bbox", "bbox": [float(value) for value in bbox]}, True
        return {"type": "polygon", "polygon": ring}, False
    if isinstance(bbox, list) and len(bbox) == 4:
        return {"type": "bbox", "bbox": [float(value) for value in bbox]}, False
    raise CvatConversionError(f"{where}: the export carries no polygon, mask, or box geometry")


def _mask_geometry(
    segmentation: Mapping[str, Any], annotation: Mapping[str, Any], *, where: str
) -> dict[str, Any]:
    """COCO RLE -> ``instance_mask``/``semantic_mask`` (a crowd region is semantic)."""
    counts, size = segmentation.get("counts"), segmentation.get("size")
    if counts is None or not isinstance(size, list) or len(size) != 2:
        raise CvatConversionError(
            f"{where}: RLE segmentation needs 'counts' and a two-element 'size' [height, width]"
        )
    crowd = int(annotation.get("iscrowd") or 0) == 1
    return {
        "type": "semantic_mask" if crowd else "instance_mask",
        "rle": {"size": [int(size[0]), int(size[1])], "counts": counts},
    }


def _polygon_ring(segmentation: Any, *, where: str) -> list[float] | None:
    """The single coordinate ring of a COCO polygon segmentation, or ``None``."""
    if not isinstance(segmentation, list):
        return None
    parts = [part for part in segmentation if isinstance(part, list) and part]
    if len(parts) > 1:
        raise CvatConversionError(
            f"{where}: multi-part polygon ({len(parts)} rings). The wire format holds one ring "
            "per object; split it into separate shapes in CVAT rather than dropping rings"
        )
    if parts:
        return [float(value) for value in parts[0]]
    if segmentation and all(isinstance(value, (int, float)) for value in segmentation):
        return [float(value) for value in segmentation]
    return None


# --------------------------------------------------------------------------- #
# Derived identity and summaries
# --------------------------------------------------------------------------- #
def derive_record_id(
    *, dataset_id: str, dataset_version: str | None, image_id: str, source_image_sha256: str
) -> str:
    """Stable row id for one reviewed image (see the module docstring)."""
    payload = "\x1f".join(
        (
            RECORD_ID_NAMESPACE,
            dataset_id,
            dataset_version or "",
            image_id,
            source_image_sha256.lower(),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{dataset_id}:{digest[:RECORD_ID_DIGEST_LENGTH]}"


def derive_human_edit_state(
    actions: Sequence[str | None], *, method: str, review_status: str
) -> str:
    """Summarise per-object ``human_edit_action`` values as a record-level state."""
    if method == "manual":
        return "human_only"
    if review_status == "unreviewed":
        return "unreviewed"
    stated = {action for action in actions if action is not None}
    if not stated:
        # No objects at all *and* an accepted review: the human accepted the
        # import as it stood. Anything else (objects with no action set, a
        # rejected image) stays unreviewed — inventing a state would only hide
        # the gap, and the validator names it precisely.
        if actions or review_status not in ACCEPTED_STATUSES:
            return "unreviewed"
        return "accepted_unchanged"
    if stated == {"accepted"}:
        return "accepted_unchanged"
    if stated == {"added", "deleted"}:
        return "replaced"
    return "edited"


# --------------------------------------------------------------------------- #
# Record assembly
# --------------------------------------------------------------------------- #
def _image_state_from_attributes(attributes: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Image-level ontology attributes, if an export carried them at all."""
    state: dict[str, Any] = {}
    status = _text_attribute(attributes, "review_status")
    if status is not None:
        if status not in REVIEW_STATUSES:
            raise CvatConversionError(
                f"{where}: review_status {status!r} is not one of {sorted(REVIEW_STATUSES)}"
            )
        state["review_status"] = status
    verified_empty = _tristate_attribute(attributes, "verified_empty", where=where)
    if verified_empty is not None:
        state["verified_empty"] = verified_empty
    if "unusable" in attributes:
        state["unusable"] = _checkbox_attribute(attributes, "unusable", where=where)
    return state


def _merge_image_state(
    layers: Sequence[tuple[str, Mapping[str, Any]]], *, where: str
) -> dict[str, Any]:
    """Combine review-state layers, failing when two CVAT-derived sources disagree."""
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for origin, layer in layers:
        for key, value in layer.items():
            if key in merged and merged[key] != value:
                raise CvatConversionError(
                    f"{where}: {key} is {merged[key]!r} in {origins[key]} but {value!r} in "
                    f"{origin}. The converter does not choose between them — reconcile the "
                    "export and the review metadata"
                )
            merged[key] = value
            origins.setdefault(key, origin)
    return merged


def _build_annotation(
    raw: Mapping[str, Any], label: str, context: ObjectContext
) -> tuple[dict[str, Any], bool]:
    """Translate one COCO object into a wire-format annotation."""
    coco_id = raw.get("id")
    if not isinstance(coco_id, (int, str)) or isinstance(coco_id, bool) or coco_id == "":
        raise CvatConversionError(
            f"{context.file_name}: an object has no COCO 'id' ({coco_id!r}). Its id is the only "
            "identity this export gives the object, and the audit trail will not carry a made-up "
            "one"
        )
    where = f"{context.file_name}: object {coco_id}"
    raw_attributes = raw.get("attributes")
    attributes = raw_attributes if isinstance(raw_attributes, Mapping) else {}
    biological_class, decision_role, _ = context.ontology[label]

    confidence = _text_attribute(attributes, "annotation_confidence")
    if confidence not in ANNOTATION_CONFIDENCE_VALUES:
        raise CvatConversionError(
            f"{where} ({label}): annotation_confidence is {confidence!r}. CVAT leaves that "
            "attribute unset on purpose and the wire format has no null for it; set it in CVAT "
            f"to one of {sorted(ANNOTATION_CONFIDENCE_VALUES)}"
        )
    occlusion = _text_attribute(attributes, "occlusion") or "unknown"
    if occlusion not in OCCLUSION_VALUES:
        raise CvatConversionError(
            f"{where} ({label}): occlusion {occlusion!r} is not one of {sorted(OCCLUSION_VALUES)}"
        )
    action = _text_attribute(attributes, "human_edit_action")
    if action is not None and action not in HUMAN_EDIT_ACTIONS:
        raise CvatConversionError(
            f"{where} ({label}): human_edit_action {action!r} is not one of "
            f"{sorted(HUMAN_EDIT_ACTIONS)}"
        )
    if context.options.proposal.method == "manual" and action not in (None, "added"):
        raise CvatConversionError(
            f"{where} ({label}): human_edit_action {action!r} says this object came from a "
            "proposal, but --proposal-method manual says there was none"
        )
    geometry, demoted = build_geometry(
        raw, where=where, keep_rectangle_polygons=context.options.keep_rectangle_polygons
    )
    annotation = {
        "annotation_id": f"{context.record_id}#{coco_id}",
        "label": label,
        "biological_class": biological_class,
        "decision_role": decision_role,
        "geometry": geometry,
        "attributes": {
            # CVAT does not carry the upstream proposal's object id, so the
            # audit trail points at the object in *this* export instead of
            # claiming an upstream identity the export cannot know.
            "source_object_id": _text_attribute(attributes, "source_object_id")
            or f"cvat:{coco_id}",
            "species": _text_attribute(attributes, "species"),
            "growth_stage": _text_attribute(attributes, "growth_stage"),
            "occlusion": occlusion,
            "truncated": _checkbox_attribute(attributes, "truncated", where=where),
            "annotation_confidence": confidence,
            "treatment_eligible": _tristate_attribute(
                attributes, "treatment_eligible", where=where
            ),
            "human_edit_action": action,
        },
    }
    return annotation, demoted


def _dimensions(task: ImageTask, options: ConversionOptions) -> tuple[int, int]:
    """Manifest dimensions, refusing a disagreement the coordinates cannot survive."""
    entry, where = task.manifest_entry, task.file_name
    try:
        width, height = int(entry["width"]), int(entry["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CvatConversionError(
            f"{where}: manifest entry has no usable 'width'/'height' ({exc})"
        ) from exc
    coco_width, coco_height = task.image.get("width"), task.image.get("height")
    disagrees = (
        isinstance(coco_width, int)
        and isinstance(coco_height, int)
        and (coco_width, coco_height) != (width, height)
    )
    if disagrees and not options.allow_dimension_mismatch:
        raise CvatConversionError(
            f"{where}: the export says {coco_width}x{coco_height} but the manifest says "
            f"{width}x{height}. The shapes were drawn against one of them and the converter will "
            "not rescale coordinates; re-export from the manifest's images, or pass "
            "--allow-dimension-mismatch to record the manifest's dimensions unchanged"
        )
    return width, height


def _source_block(task: ImageTask, options: ConversionOptions) -> dict[str, Any]:
    """The ``source`` block, with the manifest as authority over the COCO file."""
    entry, where = task.manifest_entry, task.file_name
    digest = entry.get("sha256") or entry.get("source_image_sha256")
    if not isinstance(digest, str) or not digest.strip():
        raise CvatConversionError(
            f"{where}: manifest entry has no 'sha256'. The wire format identifies an image by its "
            "content hash, and a manifest 'source_sha256' is the pre-normalisation original, not "
            "the pixels these shapes were drawn on"
        )
    width, height = _dimensions(task, options)
    split = str(entry.get("split", "")).strip()
    if split not in MANIFEST_SPLIT_NAMES:
        raise CvatConversionError(
            f"{where}: manifest split {split!r} is not one of {sorted(MANIFEST_SPLIT_NAMES)}"
        )
    group_id = entry.get("group_id")
    if not isinstance(group_id, str) or not group_id.strip():
        raise CvatConversionError(
            f"{where}: manifest entry has no 'group_id'. Group-safe splits are the point of the "
            "manifest; the converter will not invent a group"
        )
    prefix = options.image_uri_prefix.rstrip("/")
    return {
        "dataset_id": options.dataset_id,
        "dataset_version": options.dataset_version,
        "image_uri": f"{prefix}/{where}" if prefix else where,
        "source_image_sha256": digest.lower(),
        "width": width,
        "height": height,
        **{name: _optional_text(entry.get(name)) for name in OPTIONAL_SOURCE_FIELDS},
        "group_id": group_id,
        "split": MANIFEST_SPLIT_NAMES[split],
        "capture_metadata": {
            "cvat_export": {
                "file_name": where,
                "coco_image_id": task.image.get("id"),
                **dict(options.lineage.get("export", {})),
            },
            "split_manifest": {
                "key": task.manifest_key,
                "entry": {k: v for k, v in entry.items() if k not in CONSUMED_MANIFEST_KEYS},
                **dict(options.lineage.get("manifest", {})),
            },
        },
    }


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _review_state(
    task: ImageTask, options: ConversionOptions
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[Mapping[str, Any], str]]]:
    """Resolve image-level review state, sidecar identities, and the object shapes."""
    where = task.file_name
    layers: list[tuple[str, Mapping[str, Any]]] = []
    image_attributes = task.image.get("attributes")
    if isinstance(image_attributes, Mapping):
        layers.append(
            (
                "the export's image attributes",
                _image_state_from_attributes(image_attributes, where=where),
            )
        )
    shapes: list[tuple[Mapping[str, Any], str]] = []
    for raw, label in task.objects:
        if label == options.image_label_name:
            raw_attributes = raw.get("attributes")
            layers.append(
                (
                    f"the export's {options.image_label_name!r} shape",
                    _image_state_from_attributes(
                        raw_attributes if isinstance(raw_attributes, Mapping) else {}, where=where
                    ),
                )
            )
        else:
            shapes.append((raw, label))

    sidecar = dict(options.review_defaults)
    sidecar.update(options.review_by_image.get(task.manifest_key, {}))
    sidecar.update(options.review_by_image.get(where, {}))
    layers.append(
        (
            "the review metadata",
            {key: sidecar[key] for key in EXPORT_IMAGE_STATE_FIELDS if key in sidecar},
        )
    )
    return _merge_image_state(layers, where=where), sidecar, shapes


def _convert_image(
    task: ImageTask, ontology: Mapping[str, OntologyEntry], options: ConversionOptions
) -> tuple[dict[str, Any], int]:
    """Build one annotation record from one COCO image and its shapes."""
    source = _source_block(task, options)
    record_id = derive_record_id(
        dataset_id=options.dataset_id,
        dataset_version=options.dataset_version,
        image_id=task.manifest_key,
        source_image_sha256=source["source_image_sha256"],
    )
    state, sidecar, shapes = _review_state(task, options)
    review_status = state.get("review_status", "unreviewed")

    annotations: list[dict[str, Any]] = []
    demoted = 0
    context = ObjectContext(record_id, task.file_name, ontology, options)
    for raw, label in shapes:
        if label not in ontology:
            raise CvatConversionError(
                f"{task.file_name}: object {raw.get('id')} has label {label!r}, which is not a "
                f"canonical ontology label ({sorted(ontology)}). An unmapped label is an error, "
                "not a shape to drop"
            )
        annotation, was_demoted = _build_annotation(raw, label, context)
        annotations.append(annotation)
        demoted += int(was_demoted)

    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "image_id": task.manifest_key,
        "source": source,
        "provenance": {
            **options.proposal.as_record_fields([dict(raw) for raw, _ in shapes]),
            "human_edit_state": derive_human_edit_state(
                [item["attributes"]["human_edit_action"] for item in annotations],
                method=options.proposal.method,
                review_status=review_status,
            ),
        },
        "review": {
            "annotator_id": sidecar.get("annotator_id"),
            "annotator_completed_at": sidecar.get("annotator_completed_at"),
            "reviewer_id": sidecar.get("reviewer_id"),
            "reviewed_at": sidecar.get("reviewed_at"),
            "review_status": review_status,
            "annotation_version": sidecar.get("annotation_version"),
            "guide_version": sidecar.get("guide_version"),
        },
        "verified_empty": state.get("verified_empty"),
        "unusable": bool(state.get("unusable", False)),
        "annotations": annotations,
    }
    return record, demoted


def _category_index(coco: Mapping[str, Any]) -> dict[int, str]:
    """COCO category id -> name, refusing entries that cannot identify a label."""
    index: dict[int, str] = {}
    for position, category in enumerate(coco.get("categories", [])):
        if not isinstance(category, Mapping) or "id" not in category or "name" not in category:
            raise CvatConversionError(f"categories[{position}] has no 'id'/'name'")
        try:
            index[int(category["id"])] = str(category["name"])
        except (TypeError, ValueError) as exc:
            raise CvatConversionError(f"categories[{position}] has a non-integer id") from exc
    return index


def _shapes_by_image(
    coco: Mapping[str, Any], categories: Mapping[int, str]
) -> dict[Any, list[tuple[Mapping[str, Any], str]]]:
    """Group ``(raw object, label name)`` pairs by COCO image id, in export order."""
    grouped: dict[Any, list[tuple[Mapping[str, Any], str]]] = {}
    for raw in coco.get("annotations", []):
        if not isinstance(raw, Mapping):
            raise CvatConversionError("annotations[] holds an entry that is not an object")
        try:
            label = categories.get(int(raw.get("category_id", -1)), "")
        except (TypeError, ValueError) as exc:
            raise CvatConversionError(
                f"annotation {raw.get('id')!r} has a non-integer category_id"
            ) from exc
        grouped.setdefault(raw.get("image_id"), []).append((raw, label))
    return grouped


def build_records(
    coco: Mapping[str, Any],
    manifest: Mapping[str, Mapping[str, Any]],
    ontology: Mapping[str, OntologyEntry],
    options: ConversionOptions,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert a whole export, reporting every problem rather than only the first."""
    categories = _category_index(coco)
    grouped = _shapes_by_image(coco, categories)
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    demoted = 0
    claimed: dict[str, str] = {}
    for image in coco.get("images", []):
        if not isinstance(image, Mapping):
            problems.append("images[] holds an entry that is not an object")
            continue
        file_name = str(image.get("file_name", "")).strip()
        manifest_key = _manifest_key(file_name, manifest)
        if manifest_key is None:
            problems.append(
                f"{file_name or image.get('id')}: no entry in the split manifest. Membership is "
                "decided by the manifest, so an image it does not know cannot be converted"
            )
            continue
        if manifest_key in claimed:
            problems.append(
                f"{file_name}: resolves to the same manifest entry {manifest_key!r} as "
                f"{claimed[manifest_key]}"
            )
            continue
        claimed[manifest_key] = file_name
        task = ImageTask(
            image, grouped.get(image.get("id"), []), manifest_key, manifest[manifest_key]
        )
        try:
            record, image_demoted = _convert_image(task, ontology, options)
        except CvatConversionError as exc:
            problems.append(str(exc))
            continue
        records.append(record)
        demoted += image_demoted

    if problems:
        raise CvatConversionError(_problem_report(problems))
    if not records:
        raise CvatConversionError("the export contains no images; nothing to convert")
    return records, {
        "images": len(records),
        "objects": sum(len(record["annotations"]) for record in records),
        "rectangle_polygons_demoted": demoted,
    }


def _manifest_key(file_name: str, manifest: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Match a COCO ``file_name`` to a manifest key, exactly or by base name."""
    if file_name in manifest:
        return file_name
    base = PurePosixPath(file_name.replace("\\", "/")).name
    return base if base in manifest else None


def _problem_report(problems: Sequence[str]) -> str:
    head = problems[:MAX_REPORTED_ERRORS]
    body = "\n".join(f"  - {problem}" for problem in head)
    if len(problems) > len(head):
        body += f"\n  ... and {len(problems) - len(head)} more"
    return f"{len(problems)} image(s) could not be converted:\n{body}"


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> int:
    """Write records as one JSON object per line, LF-terminated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _default_ontology() -> Path:
    """Repo-root ``data/ontology.v1.json`` (src layout: src/agrinav/data/<this>)."""
    return Path(__file__).resolve().parents[3] / "data" / "ontology.v1.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--coco", required=True, type=Path, help="CVAT COCO 1.0 export JSON")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="split manifest; the authority for hash, dimensions, group and split",
    )
    parser.add_argument("--out", required=True, type=Path, help="output JSONL package")
    parser.add_argument("--dataset-id", required=True, help="value for source.dataset_id")
    parser.add_argument("--dataset-version", default=None, help="default: explicit null")
    parser.add_argument(
        "--review-metadata",
        type=Path,
        default=None,
        help="CVAT REST-API review sidecar; without it every identity is null and every record "
        "stays unreviewed",
    )
    parser.add_argument(
        "--proposal-method",
        default="imported",
        choices=("manual", "imported", "model_assisted", "model_silence"),
        help="what the shapes were before review (default: imported)",
    )
    parser.add_argument("--proposal-model-id", default=None)
    parser.add_argument("--proposal-model-revision", default=None)
    parser.add_argument("--proposal-prompt", default=None)
    parser.add_argument("--proposal-generated-at", default=None)
    parser.add_argument(
        "--proposal-thresholds", default=None, help="JSON object of the proposal thresholds"
    )
    parser.add_argument(
        "--annotation-version", default=None, help="package-level review.annotation_version"
    )
    parser.add_argument("--guide-version", default=None, help="package-level review.guide_version")
    parser.add_argument("--image-uri-prefix", default="", help="prepended to source.image_uri")
    parser.add_argument(
        "--image-label-name",
        default="image_review",
        help="tag label carrying image-level state, if an export kept it",
    )
    parser.add_argument("--ontology", type=Path, default=_default_ontology())
    parser.add_argument(
        "--allow-dimension-mismatch",
        action="store_true",
        help="convert despite a manifest/export dimension disagreement (no rescaling)",
    )
    parser.add_argument(
        "--keep-rectangle-polygons",
        action="store_true",
        help="keep exporter-synthesised rectangle rings as polygons instead of boxes",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="do not run the data-validate gate on the output",
    )
    return parser


def _options_from_args(args: argparse.Namespace) -> ConversionOptions:
    """Turn parsed arguments into the options object, failing on bad combinations."""
    thresholds = None
    if args.proposal_thresholds:
        try:
            thresholds = json.loads(args.proposal_thresholds)
        except json.JSONDecodeError as exc:
            raise CvatConversionError(f"--proposal-thresholds is not valid JSON: {exc}") from exc
        if not isinstance(thresholds, dict):
            raise CvatConversionError("--proposal-thresholds must be a JSON object")
    defaults: dict[str, Any] = {}
    if args.annotation_version:
        defaults["annotation_version"] = args.annotation_version
    if args.guide_version:
        defaults["guide_version"] = args.guide_version
    by_image: dict[str, dict[str, Any]] = {}
    if args.review_metadata is not None:
        sidecar_defaults, by_image = load_review_metadata(args.review_metadata)
        defaults.update(sidecar_defaults)
    return ConversionOptions(
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
        proposal=ProposalProvenance(
            method=args.proposal_method,
            model_id=args.proposal_model_id,
            model_revision=args.proposal_model_revision,
            prompt=args.proposal_prompt,
            generated_at=args.proposal_generated_at,
            thresholds=thresholds,
        ),
        review_defaults=defaults,
        review_by_image=by_image,
        image_uri_prefix=args.image_uri_prefix,
        image_label_name=args.image_label_name,
        allow_dimension_mismatch=args.allow_dimension_mismatch,
        keep_rectangle_polygons=args.keep_rectangle_polygons,
        lineage={
            "export": {"path": args.coco.name, "sha256": sha256_file(args.coco)},
            "manifest": {"path": args.manifest.name, "sha256": sha256_file(args.manifest)},
        },
    )


def _tally(values: Iterable[Any]) -> str:
    totals: dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        totals[key] = totals.get(key, 0) + 1
    return " ".join(f"{name}={count}" for name, count in sorted(totals.items()))


def _summarise(records: Sequence[Mapping[str, Any]], counts: Mapping[str, int]) -> None:
    """Print what was produced, including the two facts that decide whether it is truth."""
    print(f"  images:          {counts['images']}")
    print(f"  objects:         {counts['objects']}")
    print(f"  review_status:   {_tally(r['review']['review_status'] for r in records)}")
    print(f"  split:           {_tally(r['source']['split'] for r in records)}")
    print(f"  verified_empty:  {_tally(r['verified_empty'] for r in records)}")
    geometry = (a["geometry"]["type"] for r in records for a in r["annotations"])
    print(f"  geometry:        {_tally(geometry)}")
    if counts["rectangle_polygons_demoted"]:
        print(
            f"  NOTE: {counts['rectangle_polygons_demoted']} exporter-synthesised rectangle "
            "ring(s) recorded as bbox, not polygon (--keep-rectangle-polygons to disable)."
        )
    if all(record["review"]["review_status"] == "unreviewed" for record in records):
        print("  NOTE: every record is unreviewed. This package is not training truth.")


def _report_validation(errors: Sequence[str], out: Path) -> int:
    for error in errors[:MAX_REPORTED_ERRORS]:
        print(f"error: {error}", file=sys.stderr)
    if len(errors) > MAX_REPORTED_ERRORS:
        print(f"error: ... and {len(errors) - MAX_REPORTED_ERRORS} more", file=sys.stderr)
    print(
        f"{out} was written but fails agrinav data-validate with {len(errors)} error(s); "
        "it must not enter a split.",
        file=sys.stderr,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        options = _options_from_args(args)
        ontology = load_ontology(args.ontology)
        coco = load_coco_export(args.coco)
        manifest = load_split_manifest(args.manifest)
        records, counts = build_records(coco, manifest, ontology, options)
    except (CvatConversionError, AnnotationValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_jsonl(records, args.out)
    print(f"Wrote {args.out}")
    _summarise(records, counts)
    if args.skip_validation:
        print(f"  Validate with: agrinav data-validate {args.out} --ontology {args.ontology}")
        return 0
    errors = validate_packages([args.out], ontology_path=args.ontology)
    if errors:
        return _report_validation(errors, args.out)
    print("  data-validate: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
