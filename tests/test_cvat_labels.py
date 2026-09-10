"""The CVAT label spec must carry the ontology's rules, not just its names.

A label list that merely has the right five names is not enough: the three
things that make CVAT safe as the truth step are (a) nullable booleans that
cannot silently export as ``false``, (b) a geometry gate that makes a box task
an explicit decision, and (c) a review vocabulary that matches the one the
pipeline already validates against. Each is pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agrinav.data.cvat_labels import (
    CVAT_ATTRIBUTE_TYPES,
    CVAT_LABEL_TYPES,
    OntologyTranslationError,
    build_label_spec,
    load_ontology,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = REPO_ROOT / "data" / "ontology.v1.json"


@pytest.fixture(scope="module")
def ontology() -> dict[str, Any]:
    return load_ontology(ONTOLOGY_PATH)


def _by_name(labels: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [label for label in labels if label["name"] == name]
    assert matches, f"no label named {name!r} in {[label['name'] for label in labels]}"
    return matches[0]


def _attribute(label: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [a for a in label["attributes"] if a["name"] == name]
    assert matches, f"{label['name']} has no attribute {name!r}"
    return matches[0]


def test_every_canonical_label_is_emitted(ontology: dict[str, Any]) -> None:
    labels = build_label_spec(ontology, geometry="polygon")
    emitted = {label["name"] for label in labels}
    for entry in ontology["canonical_labels"]:
        assert entry["name"] in emitted


def test_emitted_types_are_valid_cvat_types(ontology: dict[str, Any]) -> None:
    for label in build_label_spec(ontology, geometry="polygon"):
        assert label["type"] in CVAT_LABEL_TYPES
        for attribute in label["attributes"]:
            assert attribute["input_type"] in CVAT_ATTRIBUTE_TYPES
            assert set(attribute) == {
                "name",
                "mutable",
                "input_type",
                "default_value",
                "values",
            }


def test_colors_are_hex_and_distinct_per_label(ontology: dict[str, Any]) -> None:
    labels = build_label_spec(ontology, geometry="polygon")
    colors = [label["color"] for label in labels]
    for color in colors:
        assert len(color) == 7 and color.startswith("#")
        int(color[1:], 16)
    assert len(set(colors)) == len(colors), "two labels share a color"


def test_rice_and_weed_do_not_share_a_color(ontology: dict[str, Any]) -> None:
    labels = build_label_spec(ontology, geometry="polygon")
    assert _by_name(labels, "rice_protect")["color"] != _by_name(labels, "weed_target")["color"]


# --- rule 2: nullable booleans must not become checkboxes --------------------


def test_nullable_boolean_becomes_three_valued_select(ontology: dict[str, Any]) -> None:
    """``treatment_eligible`` is boolean|null; a checkbox would export null as false."""
    labels = build_label_spec(ontology, geometry="polygon")
    attribute = _attribute(_by_name(labels, "weed_target"), "treatment_eligible")
    assert attribute["input_type"] == "select"
    assert attribute["default_value"] == "unknown"
    assert set(attribute["values"]) == {"unknown", "true", "false"}


def test_image_verified_empty_is_not_a_checkbox(ontology: dict[str, Any]) -> None:
    """'model returned no boxes => verified_empty' is a forbidden inference."""
    labels = build_label_spec(ontology, geometry="polygon")
    attribute = _attribute(_by_name(labels, "image_review"), "verified_empty")
    assert attribute["input_type"] == "select"
    assert attribute["default_value"] == "unknown"


def test_plain_boolean_stays_a_checkbox_defaulting_false(ontology: dict[str, Any]) -> None:
    """``unusable`` is a non-nullable boolean with default false — no third state needed."""
    labels = build_label_spec(ontology, geometry="polygon")
    attribute = _attribute(_by_name(labels, "image_review"), "unusable")
    assert attribute["input_type"] == "checkbox"
    assert attribute["default_value"] == "false"


# --- rule 3: no invented defaults -------------------------------------------


def test_annotation_confidence_has_no_default(ontology: dict[str, Any]) -> None:
    labels = build_label_spec(ontology, geometry="polygon")
    attribute = _attribute(_by_name(labels, "rice_protect"), "annotation_confidence")
    assert attribute["default_value"] == ""
    assert set(attribute["values"]) == {"certain", "probable", "uncertain"}


def test_enum_with_an_unset_member_defaults_to_it(ontology: dict[str, Any]) -> None:
    """``occlusion`` declares no default but its vocabulary owns 'unknown'."""
    labels = build_label_spec(ontology, geometry="polygon")
    attribute = _attribute(_by_name(labels, "rice_protect"), "occlusion")
    assert attribute["default_value"] == "unknown"


def test_image_review_status_is_not_left_blank(ontology: dict[str, Any]) -> None:
    """The enum owns an unset member, so the default is it — not an empty box."""
    labels = build_label_spec(ontology, geometry="polygon")
    image = _attribute(_by_name(labels, "image_review"), "review_status")
    assert image["default_value"] == "unreviewed"


def test_review_default_follows_the_vocabulary_not_its_order() -> None:
    """Reordering the enum must not change which state is the safe default."""
    reordered = {
        "name": "agrinav-perception-ontology",
        "canonical_labels": [{"name": "x", "geometry": ["polygon"]}],
        "object_attributes": {},
        "image_attributes": {"review_status": {"enum": ["accepted", "unreviewed"]}},
    }
    labels = build_label_spec(reordered, geometry="polygon")
    attribute = _attribute(_by_name(labels, "image_review"), "review_status")
    assert attribute["default_value"] == "unreviewed"


def test_enum_defaults_must_be_members() -> None:
    broken = {
        "name": "agrinav-perception-ontology",
        "canonical_labels": [{"name": "x", "geometry": ["polygon"]}],
        "object_attributes": {"occlusion": {"enum": ["none", "severe"], "default": "partial"}},
        "image_attributes": {"review_status": {"enum": ["unreviewed"]}},
    }
    with pytest.raises(OntologyTranslationError, match="not one of"):
        build_label_spec(broken, geometry="polygon")


# --- rule 1: the geometry gate ----------------------------------------------


def test_rectangle_is_refused_without_the_override(ontology: dict[str, Any]) -> None:
    """No ontology label lists a box geometry; a box task must be opted into."""
    with pytest.raises(OntologyTranslationError, match="not canonical"):
        build_label_spec(ontology, geometry="rectangle")


def test_rectangle_is_allowed_with_the_override(ontology: dict[str, Any]) -> None:
    labels = build_label_spec(ontology, geometry="rectangle", allow_non_canonical_geometry=True)
    assert _by_name(labels, "rice_protect")["type"] == "rectangle"


def test_polygon_and_mask_need_no_override(ontology: dict[str, Any]) -> None:
    for geometry in ("polygon", "mask"):
        labels = build_label_spec(ontology, geometry=geometry)
        assert _by_name(labels, "rice_protect")["type"] == geometry


def test_unknown_geometry_is_rejected(ontology: dict[str, Any]) -> None:
    with pytest.raises(OntologyTranslationError, match="not a CVAT label type"):
        build_label_spec(ontology, geometry="scribble")


def test_geometry_gate_names_the_offending_labels() -> None:
    """The error has to say which label blocked it, or it is not actionable."""
    partial = {
        "name": "agrinav-perception-ontology",
        "canonical_labels": [
            {"name": "mask_only", "geometry": ["instance_mask"]},
            {"name": "polygon_ok", "geometry": ["polygon"]},
        ],
        "object_attributes": {},
        "image_attributes": {"review_status": {"enum": ["unreviewed", "accepted"]}},
    }
    with pytest.raises(OntologyTranslationError) as excinfo:
        build_label_spec(partial, geometry="polygon")
    assert "mask_only" in str(excinfo.value)
    assert "polygon_ok" not in str(excinfo.value)


# --- review vocabulary -------------------------------------------------------


def test_review_status_matches_the_ontology_vocabulary(ontology: dict[str, Any]) -> None:
    labels = build_label_spec(ontology, geometry="polygon")
    expected = ontology["image_attributes"]["review_status"]["enum"]
    attribute = _attribute(_by_name(labels, "image_review"), "review_status")
    assert attribute["values"] == list(expected)


def test_review_status_defaults_to_unreviewed(ontology: dict[str, Any]) -> None:
    """An image nobody has looked at must not read as accepted."""
    labels = build_label_spec(ontology, geometry="polygon")
    assert _attribute(_by_name(labels, "image_review"), "review_status")["default_value"] == (
        "unreviewed"
    )


def test_shapes_do_not_carry_a_review_status(ontology: dict[str, Any]) -> None:
    """review_status is record-level in the wire format; a per-shape copy cannot export."""
    labels = build_label_spec(ontology, geometry="polygon")
    names = {a["name"] for a in _by_name(labels, "rice_protect")["attributes"]}
    assert "review_status" not in names
    assert "human_edit_action" in names


def test_missing_review_vocabulary_is_an_error() -> None:
    without = {
        "name": "agrinav-perception-ontology",
        "canonical_labels": [{"name": "x", "geometry": ["polygon"]}],
        "object_attributes": {},
        "image_attributes": {"unusable": {"type": "boolean", "default": False}},
    }
    with pytest.raises(OntologyTranslationError, match="review_status"):
        build_label_spec(without, geometry="polygon")


# --- agreement with the wire format ------------------------------------------
#
# CVAT is only useful here if what comes out of it can be expressed in
# `data/schemas/annotation_record.v1.schema.json`, which `agrinav data-validate`
# enforces and which sets additionalProperties: false. An attribute CVAT
# collects that the schema has no slot for is data that dies at export.


@pytest.fixture(scope="module")
def wire_schema() -> dict[str, Any]:
    path = REPO_ROOT / "data" / "schemas" / "annotation_record.v1.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _schema_object_attributes(schema: dict[str, Any]) -> dict[str, Any]:
    return schema["$defs"]["object_attributes"]["properties"]


def test_every_shape_attribute_has_a_slot_in_the_wire_format(
    ontology: dict[str, Any], wire_schema: dict[str, Any]
) -> None:
    labels = build_label_spec(ontology, geometry="polygon")
    allowed = set(_schema_object_attributes(wire_schema))
    emitted = {a["name"] for a in _by_name(labels, "rice_protect")["attributes"]}
    assert emitted <= allowed, f"no schema slot for: {sorted(emitted - allowed)}"


def test_every_image_attribute_has_a_slot_in_the_wire_format(
    ontology: dict[str, Any], wire_schema: dict[str, Any]
) -> None:
    labels = build_label_spec(ontology, geometry="polygon")
    record_level = set(wire_schema["properties"])
    review_level = set(wire_schema["properties"]["review"]["properties"])
    emitted = {a["name"] for a in _by_name(labels, "image_review")["attributes"]}
    unplaceable = emitted - record_level - review_level
    assert not unplaceable, f"no schema slot for: {sorted(unplaceable)}"


def test_shape_enums_match_the_wire_format(
    ontology: dict[str, Any], wire_schema: dict[str, Any]
) -> None:
    """Same vocabulary, or an export is rejected for a value CVAT offered."""
    labels = build_label_spec(ontology, geometry="polygon")
    schema_attributes = _schema_object_attributes(wire_schema)
    checked = 0
    for attribute in _by_name(labels, "weed_target")["attributes"]:
        spec = schema_attributes.get(attribute["name"], {})
        if "enum" not in spec:
            continue
        # CVAT has no null; the schema's null member is the empty default.
        expected = [str(v) for v in spec["enum"] if v is not None]
        assert sorted(attribute["values"]) == sorted(expected), attribute["name"]
        if None in spec["enum"]:
            assert attribute["default_value"] == "", attribute["name"]
        checked += 1
    assert checked >= 2, "expected several enum attributes to compare"


def test_human_edit_action_constant_tracks_the_schema(wire_schema: dict[str, Any]) -> None:
    from agrinav.data.cvat_labels import HUMAN_EDIT_ACTIONS

    schema_enum = _schema_object_attributes(wire_schema)["human_edit_action"]["enum"]
    assert list(HUMAN_EDIT_ACTIONS) == [v for v in schema_enum if v is not None]


def test_labels_match_the_wire_format_label_vocabulary(
    ontology: dict[str, Any], wire_schema: dict[str, Any]
) -> None:
    labels = build_label_spec(ontology, geometry="polygon", include_image_label=False)
    expected = set(wire_schema["$defs"]["annotation"]["properties"]["label"]["enum"])
    assert {label["name"] for label in labels} == expected


def test_attributes_are_not_shared_objects(ontology: dict[str, Any]) -> None:
    """Each label needs its own attribute dicts; aliasing corrupts every label at once."""
    labels = build_label_spec(ontology, geometry="polygon")
    first = _attribute(_by_name(labels, "rice_protect"), "occlusion")
    second = _attribute(_by_name(labels, "weed_target"), "occlusion")
    assert first is not second
    first["default_value"] = "MUTATED"
    assert second["default_value"] != "MUTATED"


def test_provenance_fields_do_not_become_attributes(ontology: dict[str, Any]) -> None:
    """CVAT records the annotator itself; model provenance belongs to the proposal JSON."""
    labels = build_label_spec(ontology, geometry="polygon")
    names = {a["name"] for a in _by_name(labels, "rice_protect")["attributes"]}
    assert not names & {"annotator_id", "reviewer_id", "proposal_model_id", "source_image_sha256"}


def test_image_label_can_be_omitted_and_renamed(ontology: dict[str, Any]) -> None:
    without = build_label_spec(ontology, geometry="polygon", include_image_label=False)
    assert all(label["type"] != "tag" for label in without)
    renamed = build_label_spec(ontology, geometry="polygon", image_label_name="frame_state")
    assert _by_name(renamed, "frame_state")["type"] == "tag"


# --- loading + CLI -----------------------------------------------------------


def test_load_ontology_rejects_a_foreign_document(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"name": "something-else"}), encoding="utf-8")
    with pytest.raises(OntologyTranslationError, match="does not look like"):
        load_ontology(path)


def test_load_ontology_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OntologyTranslationError, match="Pass --ontology"):
        load_ontology(tmp_path / "absent.json")


def test_cli_writes_json_cvat_can_parse(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "labels.json"
    assert main(["--ontology", str(ONTOLOGY_PATH), "--out", str(out)]) == 0
    labels = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(labels, list)
    assert {label["name"] for label in labels} >= {"rice_protect", "weed_target"}


def test_cli_fails_closed_on_a_box_task(tmp_path: Path) -> None:
    out = tmp_path / "labels.json"
    with pytest.raises(OntologyTranslationError):
        main(["--ontology", str(ONTOLOGY_PATH), "--out", str(out), "--geometry", "rectangle"])
    assert not out.exists(), "a refused spec must not be written"
