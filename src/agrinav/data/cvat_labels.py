#!/usr/bin/env python3
"""Generate a CVAT label specification from ``data/ontology.v1.json``.

Why this exists
---------------
CVAT is stage 4 of ``docs/automated_annotation_pipeline.md`` — the human review
step, and per that document "the only truth step". A CVAT project whose labels
were typed by hand into the web UI is a second, unversioned copy of the
ontology: it drifts, and nothing detects the drift. This module emits the label
spec *from* the ontology, so the annotation tool and the training pipeline agree
by construction.

The output is the JSON that CVAT accepts in the **Raw** tab of a project's label
editor, and that the REST API / ``cvat-cli`` accept as ``labels``. Shape of a
label (``cvat/apps/engine/models.py``: ``Label``, ``AttributeSpec``)::

    {"name": ..., "color": "#rrggbb", "type": ..., "attributes": [
        {"name": ..., "mutable": false, "input_type": ...,
         "default_value": ..., "values": [...]}
    ]}

``type`` is a CVAT ``ShapeType`` (rectangle, polygon, polyline, points, ellipse,
cuboid, mask, skeleton) plus ``any``/``tag``; ``input_type`` is a CVAT
``AttributeType`` (checkbox, radio, number, text, select).

Three ontology rules survive the translation, and they are the reason this is a
generator rather than a copy-paste:

1. **Geometry is not free.** Each canonical label declares which geometries are
   valid for it. ``--geometry rectangle`` is refused unless
   ``--allow-non-canonical-geometry`` is passed, because the ontology principle
   is "Boxes may be derived from masks, but boxes are not canonical treatment
   geometry" — a box task is a proposal/QA layer, not truth geometry, and the
   flag makes that an explicit choice rather than a silent one.

2. **Nullable booleans never become checkboxes.** ``verified_empty`` and
   ``treatment_eligible`` are ``boolean | null`` with default ``null``. A CVAT
   checkbox has no third state, so an unset attribute would export as ``false``
   — which is exactly the forbidden inference "model returned no boxes =>
   verified_empty". They become three-valued selects defaulting to ``unknown``.

3. **``annotation_confidence`` gets no default.** The ontology declares no
   default for it; picking one would put a value the annotator never chose into
   the record. It emits with ``default_value: ""`` so CVAT shows it unset.

Image-level ontology attributes (``verified_empty``, ``unusable``,
``review_status``) cannot hang off a shape, so they are emitted as a single
``tag`` label (``--image-label-name``, default ``image_review``).

CLI::

    python -m agrinav.data.cvat_labels --out artifacts/cvat/labels_polygon.json
    python -m agrinav.data.cvat_labels --geometry mask --out .../labels_mask.json
    python -m agrinav.data.cvat_labels --geometry rectangle \
        --allow-non-canonical-geometry --out artifacts/cvat/labels_boxqa.json

Setup and workflow: ``docs/cvat.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: CVAT ``ShapeType`` values, plus the two non-shape label types CVAT accepts.
#: Source: cvat/apps/engine/models.py (ShapeType, LabelType).
CVAT_LABEL_TYPES: frozenset[str] = frozenset(
    {
        "rectangle",
        "polygon",
        "polyline",
        "points",
        "ellipse",
        "cuboid",
        "mask",
        "skeleton",
        "any",
        "tag",
    }
)

#: CVAT ``AttributeType`` values. Source: cvat/apps/engine/models.py.
CVAT_ATTRIBUTE_TYPES: frozenset[str] = frozenset({"checkbox", "radio", "number", "text", "select"})

#: Ontology geometry name -> CVAT label type. ``semantic_mask`` has no instance
#: equivalent in CVAT; it is drawn with the same mask tool as ``instance_mask``.
ONTOLOGY_GEOMETRY_TO_CVAT: dict[str, str] = {
    "instance_mask": "mask",
    "semantic_mask": "mask",
    "polygon": "polygon",
}

#: Stable per-label colors, so overlays mean the same thing across every task.
#: Unlisted labels fall back to a deterministic hash of the name.
LABEL_COLORS: dict[str, str] = {
    "rice_protect": "#3ddb3d",
    "weed_target": "#ff355e",
    "unknown_vegetation": "#ffb300",
    "non_target_aquatic": "#00a8e8",
    "ground_exclusion": "#9e9e9e",
}

#: Color for the generated image-level tag label.
IMAGE_LABEL_COLOR = "#cccccc"

#: Attributes whose ``boolean | null`` type must not collapse to a checkbox.
#: Value is the string standing in for ``null``.
NULLABLE_BOOLEAN_UNSET = "unknown"

#: Enum members that *are* the "not yet stated" state. When an ontology enum
#: declares no default but contains one of these, it is the default: an image
#: nobody has opened should read ``unreviewed``, not blank. An enum with no such
#: member (``annotation_confidence``) is left unset instead — see rule 3.
UNSET_ENUM_MEMBERS: tuple[str, ...] = ("unreviewed", "unknown", "unspecified")

#: Per-object edit decision. Mirrors ``$defs/object_attributes.human_edit_action``
#: in ``data/schemas/annotation_record.v1.schema.json`` — the wire format
#: ``agrinav data-validate`` enforces — minus its ``null`` member, which CVAT
#: expresses as the empty default (the schema reads: "null only before human
#: review"). ``tests/test_cvat_labels.py`` fails if the two drift apart.
HUMAN_EDIT_ACTIONS: tuple[str, ...] = (
    "accepted",
    "edited",
    "deleted",
    "added",
    "reclassified",
    "split",
    "merged",
)

#: Ontology attributes that are per-object provenance CVAT already records
#: itself (it stores the annotator and the reviewer per job) or that belong to
#: the proposal layer's JSON rather than to a CVAT attribute.
SKIPPED_PROVENANCE_FIELDS: frozenset[str] = frozenset(
    {
        "proposal_model_id",
        "proposal_model_revision",
        "proposal_method",
        "prompt",
        "thresholds",
        "source_image_sha256",
        "generated_at",
        "original_proposal",
        "annotator_id",
        "reviewer_id",
    }
)


class OntologyTranslationError(RuntimeError):
    """The ontology cannot be expressed as a CVAT label spec as requested."""


def _default_ontology() -> Path:
    """Repo-root ``data/ontology.v1.json`` (src layout: src/agrinav/data/<this>)."""
    return Path(__file__).resolve().parents[3] / "data" / "ontology.v1.json"


def _color_for(name: str) -> str:
    """Stable color for a label: the fixed palette, else a hash of the name."""
    if name in LABEL_COLORS:
        return LABEL_COLORS[name]
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


def _select(name: str, values: list[str], default: str, *, mutable: bool = False) -> dict[str, Any]:
    if default and default not in values:
        raise OntologyTranslationError(
            f"attribute {name!r}: default {default!r} is not one of {values}"
        )
    return {
        "name": name,
        "mutable": mutable,
        "input_type": "select",
        "default_value": default,
        "values": list(values),
    }


def _enum_default(values: list[str], declared: Any) -> str:
    """Default for an ontology enum: the declared one, else its unset member, else blank.

    Inventing a substantive default would put a value in the record that no
    annotator chose (rule 3). Choosing the vocabulary's *own* unset member is
    not an invention — it is the state the record is already in.
    """
    if declared is not None:
        return str(declared)
    for candidate in UNSET_ENUM_MEMBERS:
        if candidate in values:
            return candidate
    return ""


def _attribute_from_spec(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Translate one ontology attribute spec into a CVAT attribute.

    The ontology uses three shapes: an ``enum`` list, a plain ``type``, or a
    ``[type, "null"]`` union. Each maps to exactly one CVAT ``input_type``.
    """
    if "enum" in spec:
        values = [str(v) for v in spec["enum"]]
        return _select(name, values, _enum_default(values, spec.get("default")))

    declared = spec.get("type")
    types = [declared] if isinstance(declared, str) else list(declared or [])
    nullable = "null" in types
    concrete = [t for t in types if t != "null"]
    if len(concrete) != 1:
        raise OntologyTranslationError(
            f"attribute {name!r}: expected exactly one concrete type, got {types!r}"
        )
    kind = concrete[0]

    if kind == "boolean":
        if nullable:
            # A checkbox has two states; a nullable boolean has three, and the
            # missing one is the safe one (see module docstring, rule 2).
            return _select(name, [NULLABLE_BOOLEAN_UNSET, "true", "false"], NULLABLE_BOOLEAN_UNSET)
        default = spec.get("default", False)
        return {
            "name": name,
            "mutable": False,
            "input_type": "checkbox",
            "default_value": "true" if default else "false",
            "values": ["false", "true"],
        }
    if kind == "string":
        default = spec.get("default")
        return {
            "name": name,
            "mutable": False,
            "input_type": "text",
            "default_value": "" if default is None else str(default),
            "values": [],
        }
    if kind in ("number", "integer"):
        default = spec.get("default")
        return {
            "name": name,
            "mutable": False,
            "input_type": "number",
            "default_value": "" if default is None else str(default),
            "values": [],
        }
    raise OntologyTranslationError(f"attribute {name!r}: unsupported ontology type {kind!r}")


def _review_status_values(ontology: dict[str, Any]) -> list[str]:
    """The one review-state vocabulary, taken from ``image_attributes``."""
    spec = ontology.get("image_attributes", {}).get("review_status", {})
    values = [str(v) for v in spec.get("enum", [])]
    if not values:
        raise OntologyTranslationError(
            "ontology image_attributes.review_status.enum is missing; "
            "there is no review vocabulary to give CVAT"
        )
    return values


def build_label_spec(
    ontology: dict[str, Any],
    *,
    geometry: str = "polygon",
    allow_non_canonical_geometry: bool = False,
    include_image_label: bool = True,
    image_label_name: str = "image_review",
) -> list[dict[str, Any]]:
    """Translate the ontology into a CVAT label specification.

    Args:
        ontology: parsed ``data/ontology.v1.json``.
        geometry: CVAT label type for the object labels (``polygon``, ``mask``,
            ``rectangle``, ``any``, ...).
        allow_non_canonical_geometry: permit a ``geometry`` the ontology does
            not list for a label. Required for box tasks; see module docstring.
        include_image_label: also emit the image-level ``tag`` label.
        image_label_name: name of that tag label.

    Returns:
        The label list, ready to paste into CVAT's Raw label editor.

    Raises:
        OntologyTranslationError: unknown geometry, a label whose ontology
            geometry list excludes ``geometry`` (without the override), or an
            attribute spec that has no CVAT equivalent.
    """
    if geometry not in CVAT_LABEL_TYPES:
        raise OntologyTranslationError(
            f"{geometry!r} is not a CVAT label type; expected one of " f"{sorted(CVAT_LABEL_TYPES)}"
        )

    canonical = ontology.get("canonical_labels")
    if not canonical:
        raise OntologyTranslationError("ontology has no canonical_labels")

    object_attributes = ontology.get("object_attributes", {})
    shared_attributes = [
        _attribute_from_spec(name, spec)
        for name, spec in object_attributes.items()
        if name not in SKIPPED_PROVENANCE_FIELDS
    ]
    # The ontology has to carry a review vocabulary at all, or CVAT cannot
    # record a review state anywhere. Fail on that here rather than emit a
    # project that silently cannot express "reviewed".
    _review_status_values(ontology)
    # Per-object accept / fix / reject. `review_status` is deliberately *not*
    # repeated per shape: in the wire format it is a record-level field, and
    # `$defs/annotation` sets additionalProperties: false, so a per-shape copy
    # would have nowhere to land on export.
    shared_attributes.append(_select("human_edit_action", list(HUMAN_EDIT_ACTIONS), ""))

    offenders: list[str] = []
    labels: list[dict[str, Any]] = []
    for entry in canonical:
        name = entry["name"]
        allowed = {
            ONTOLOGY_GEOMETRY_TO_CVAT[g]
            for g in entry.get("geometry", [])
            if g in ONTOLOGY_GEOMETRY_TO_CVAT
        }
        if geometry not in allowed and geometry != "any":
            offenders.append(f"{name} (ontology allows {sorted(allowed) or ['nothing mappable']})")
        labels.append(
            {
                "name": name,
                "color": _color_for(name),
                "type": geometry,
                "attributes": [dict(attribute) for attribute in shared_attributes],
            }
        )

    if offenders and not allow_non_canonical_geometry:
        raise OntologyTranslationError(
            f"geometry {geometry!r} is not canonical for: {'; '.join(offenders)}. "
            "The ontology principle is 'boxes are not canonical treatment geometry' — "
            "pass --allow-non-canonical-geometry to build a proposal/QA task anyway."
        )

    if include_image_label:
        labels.append(_build_image_label(ontology, image_label_name))
    return labels


def _build_image_label(ontology: dict[str, Any], name: str) -> dict[str, Any]:
    """Image-level ontology attributes as a CVAT ``tag`` label."""
    image_attributes = ontology.get("image_attributes", {})
    if not image_attributes:
        raise OntologyTranslationError("ontology has no image_attributes")
    return {
        "name": name,
        "color": IMAGE_LABEL_COLOR,
        "type": "tag",
        "attributes": [
            _attribute_from_spec(attribute, spec) for attribute, spec in image_attributes.items()
        ],
    }


def load_ontology(path: Path) -> dict[str, Any]:
    """Read and minimally sanity-check the ontology document."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OntologyTranslationError(
            f"ontology not found at {path}. Pass --ontology explicitly."
        ) from exc
    except json.JSONDecodeError as exc:
        raise OntologyTranslationError(f"ontology {path} is not valid JSON: {exc}") from exc
    if document.get("name") != "agrinav-perception-ontology":
        raise OntologyTranslationError(
            f"{path} does not look like the AgriNav ontology " f"(name={document.get('name')!r})"
        )
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ontology", type=Path, default=_default_ontology())
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument(
        "--geometry",
        default="polygon",
        help="CVAT label type for the object labels (default: polygon)",
    )
    parser.add_argument(
        "--allow-non-canonical-geometry",
        action="store_true",
        help="emit a geometry the ontology does not list (e.g. rectangle box QA)",
    )
    parser.add_argument(
        "--no-image-label",
        action="store_true",
        help="omit the image-level tag label",
    )
    parser.add_argument("--image-label-name", default="image_review")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ontology = load_ontology(args.ontology)
    labels = build_label_spec(
        ontology,
        geometry=args.geometry,
        allow_non_canonical_geometry=args.allow_non_canonical_geometry,
        include_image_label=not args.no_image_label,
        image_label_name=args.image_label_name,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    schema_version = ontology.get("schema_version", "unknown")
    print(f"Wrote {args.out}")
    print(f"  ontology:        {args.ontology} (schema {schema_version})")
    print(f"  geometry:        {args.geometry}")
    print(f"  labels:          {len(labels)}")
    for label in labels:
        print(f"    - {label['name']:<20} {label['type']:<10} {len(label['attributes'])} attrs")
    if args.allow_non_canonical_geometry:
        print("  NOTE: non-canonical geometry allowed — this is a proposal/QA task, not truth.")
    print("Paste into CVAT: project -> Raw labels tab. See docs/cvat.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
