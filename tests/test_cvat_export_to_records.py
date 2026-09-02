"""The CVAT-COCO converter must not manufacture truth on the way through.

Every failure this module guards against is silent by construction: an image that
nobody reviewed arriving as accepted, an empty export becoming a verified empty,
a label quietly dropped, a split taken from the wrong authority, an invented
annotator. So the tests assert on the observable record — and on the real gate,
by calling ``validate_annotation_package`` rather than restating its rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agrinav.data.cvat_export_to_records import (
    ConversionOptions,
    CvatConversionError,
    ProposalProvenance,
    build_records,
    derive_record_id,
    load_review_metadata,
    load_split_manifest,
    main,
)
from agrinav.data.validate_annotation_package import load_ontology, validate_packages

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = REPO_ROOT / "data" / "ontology.v1.json"

RICE_POLYGON = [10.0, 10.0, 90.0, 10.0, 90.0, 70.0, 10.0, 70.0, 40.0, 40.0]
WEED_POLYGON = [100.0, 100.0, 160.0, 100.0, 160.0, 150.0, 100.0, 150.0, 120.0, 130.0]


# --------------------------------------------------------------------------- #
# Fixture construction
# --------------------------------------------------------------------------- #
def shape_attributes(**overrides: Any) -> dict[str, Any]:
    """Attributes as ``agrinav data-cvat-labels`` makes CVAT export them.

    Defaults mirror the generated spec: ``occlusion`` defaults to ``unknown``,
    the nullable booleans to ``unknown``, and ``human_edit_action`` to empty.
    """
    attributes: dict[str, Any] = {
        "occluded": False,
        "rotation": 0.0,
        "species": "",
        "growth_stage": "",
        "occlusion": "unknown",
        "truncated": False,
        "annotation_confidence": "certain",
        "treatment_eligible": "unknown",
        "human_edit_action": "",
    }
    attributes.update(overrides)
    return attributes


def coco_export(images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "info": {"contributor": "", "description": "", "version": ""},
        "categories": [
            {"id": 1, "name": "rice_protect", "supercategory": ""},
            {"id": 2, "name": "weed_target", "supercategory": ""},
        ],
        "images": images,
        "annotations": annotations,
    }


def coco_image(image_id: int, file_name: str, *, width: int = 200, height: int = 200) -> dict:
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0,
    }


def coco_polygon(
    annotation_id: int, image_id: int, category_id: int, polygon: list[float], **attributes: Any
) -> dict[str, Any]:
    xs, ys = polygon[0::2], polygon[1::2]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [list(polygon)],
        "area": bbox[2] * bbox[3],
        "bbox": bbox,
        "iscrowd": 0,
        "attributes": shape_attributes(**attributes),
    }


def coco_box(
    annotation_id: int, image_id: int, category_id: int, bbox: list[float], **attributes: Any
) -> dict[str, Any]:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [],
        "area": bbox[2] * bbox[3],
        "bbox": list(bbox),
        "iscrowd": 0,
        "attributes": shape_attributes(**attributes),
    }


def manifest_entry(digest_seed: str, *, split: str = "train", **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "bytes": 1234,
        "exif_reoriented": False,
        "group_id": "#7",
        "height": 200,
        "native_split": split,
        "sha256": digest_seed * 64,
        "source_sha256": digest_seed * 64,
        "split": split,
        "trained_on_legacy": False,
        "width": 200,
    }
    entry.update(overrides)
    return entry


@pytest.fixture(scope="module")
def ontology() -> dict[str, tuple[str, str, frozenset[str]]]:
    return load_ontology(ONTOLOGY_PATH)


def options(**overrides: Any) -> ConversionOptions:
    settings: dict[str, Any] = {
        "dataset_id": "rice_phase2",
        "dataset_version": "rebuild-2026-07-29",
        "proposal": ProposalProvenance(method="imported"),
    }
    settings.update(overrides)
    return ConversionOptions(**settings)


def accepted_review(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "annotator_id": "annotator-04",
        "annotator_completed_at": "2026-08-28T09:00:00Z",
        "reviewer_id": "reviewer-01",
        "reviewed_at": "2026-08-29T11:30:00Z",
        "review_status": "accepted",
        "annotation_version": "cvat-task-31-v2",
        "guide_version": "v1",
    }
    state.update(overrides)
    return state


def validate(records: list[dict[str, Any]], tmp_path: Path) -> list[str]:
    """Run the real gate over the produced package."""
    package = tmp_path / "package.jsonl"
    package.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8", newline="\n"
    )
    return validate_packages([package], ontology_path=ONTOLOGY_PATH)


# --------------------------------------------------------------------------- #
# The gate accepts the output
# --------------------------------------------------------------------------- #
def test_reviewed_export_round_trips_through_the_real_validator(ontology, tmp_path) -> None:
    """A reviewed polygon export must survive `agrinav data-validate` unmodified."""
    coco = coco_export(
        [coco_image(1, "train/a.jpg"), coco_image(2, "train/b.jpg")],
        [
            coco_polygon(1, 1, 1, RICE_POLYGON, human_edit_action="accepted"),
            coco_polygon(2, 1, 2, WEED_POLYGON, human_edit_action="edited"),
            coco_polygon(3, 2, 1, RICE_POLYGON, human_edit_action="accepted"),
        ],
    )
    manifest = {"a.jpg": manifest_entry("a"), "b.jpg": manifest_entry("b", split="valid")}
    records, counts = build_records(
        coco,
        manifest,
        ontology,
        options(
            review_by_image={
                "a.jpg": accepted_review(verified_empty=False),
                "b.jpg": accepted_review(verified_empty=True),
            }
        ),
    )

    assert validate(records, tmp_path) == []
    assert counts == {"images": 2, "objects": 3, "rectangle_polygons_demoted": 0}
    assert [record["source"]["split"] for record in records] == ["train", "validation"]


def test_unreviewed_export_also_validates_but_is_not_truth(ontology, tmp_path) -> None:
    coco = coco_export([coco_image(1, "train/a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON)])
    records, _ = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())

    assert validate(records, tmp_path) == []
    assert records[0]["review"]["review_status"] == "unreviewed"


# --------------------------------------------------------------------------- #
# Review state is copied, never upgraded or invented
# --------------------------------------------------------------------------- #
def test_unreviewed_export_stays_unreviewed(ontology) -> None:
    coco = coco_export(
        [coco_image(1, "train/a.jpg")],
        [coco_polygon(1, 1, 2, WEED_POLYGON, annotation_confidence="probable")],
    )
    (record,) = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())[0]

    assert record["review"]["review_status"] == "unreviewed"
    assert record["provenance"]["human_edit_state"] == "unreviewed"
    assert record["annotations"][0]["attributes"]["human_edit_action"] is None
    assert record["verified_empty"] is None


def test_missing_review_metadata_yields_nulls_not_identities(ontology) -> None:
    coco = coco_export([coco_image(1, "a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON)])
    (record,) = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())[0]

    assert record["review"] == {
        "annotator_id": None,
        "annotator_completed_at": None,
        "reviewer_id": None,
        "reviewed_at": None,
        "review_status": "unreviewed",
        "annotation_version": None,
        "guide_version": None,
    }


def test_review_metadata_supplies_identities_verbatim(ontology) -> None:
    coco = coco_export(
        [coco_image(1, "a.jpg")],
        [coco_polygon(1, 1, 1, RICE_POLYGON, human_edit_action="accepted")],
    )
    records, _ = build_records(
        coco,
        {"a.jpg": manifest_entry("a")},
        ontology,
        options(review_by_image={"a.jpg": accepted_review()}),
    )

    assert records[0]["review"]["annotator_id"] == "annotator-04"
    assert records[0]["review"]["reviewer_id"] == "reviewer-01"
    assert records[0]["review"]["review_status"] == "accepted"


def test_review_metadata_job_entries_apply_only_to_their_images(tmp_path) -> None:
    sidecar = tmp_path / "review.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "agrinav.cvat_review_metadata.v1",
                "defaults": {"guide_version": "v1"},
                "jobs": [
                    {
                        "job_id": 7,
                        "images": ["a.jpg"],
                        "annotator_id": "annotator-04",
                        "review_status": "accepted",
                    }
                ],
                "images": {"a.jpg": {"reviewer_id": "reviewer-01"}},
            }
        ),
        encoding="utf-8",
    )
    defaults, by_image = load_review_metadata(sidecar)

    assert defaults == {"guide_version": "v1"}
    assert by_image["a.jpg"]["annotator_id"] == "annotator-04"
    assert by_image["a.jpg"]["reviewer_id"] == "reviewer-01"
    assert "b.jpg" not in by_image


def test_review_metadata_rejects_unknown_fields(tmp_path) -> None:
    """A mistyped reviewer key must not silently become "no reviewer"."""
    sidecar = tmp_path / "review.json"
    sidecar.write_text(json.dumps({"images": {"a.jpg": {"reviewr_id": "x"}}}), encoding="utf-8")

    with pytest.raises(CvatConversionError, match="reviewr_id"):
        load_review_metadata(sidecar)


def test_conflicting_review_states_are_not_reconciled(ontology) -> None:
    """Export says in_review, sidecar says accepted: refuse rather than pick one."""
    image = coco_image(1, "a.jpg")
    image["attributes"] = {"review_status": "in_review"}
    coco = coco_export([image], [coco_polygon(1, 1, 1, RICE_POLYGON)])

    with pytest.raises(CvatConversionError, match="review_status"):
        build_records(
            coco,
            {"a.jpg": manifest_entry("a")},
            ontology,
            options(review_by_image={"a.jpg": {"review_status": "accepted"}}),
        )


# --------------------------------------------------------------------------- #
# Forbidden inferences
# --------------------------------------------------------------------------- #
def test_image_without_annotations_is_not_a_verified_empty(ontology, tmp_path) -> None:
    """An empty export is the ontology's named forbidden inference, not an empty."""
    coco = coco_export([coco_image(1, "a.jpg")], [])
    (record,) = build_records(
        coco,
        {"a.jpg": manifest_entry("a")},
        ontology,
        options(review_by_image={"a.jpg": accepted_review()}),
    )[0]

    assert record["annotations"] == []
    assert record["verified_empty"] is None
    assert validate([record], tmp_path) == []


def test_verified_empty_only_comes_from_a_human_statement(ontology) -> None:
    coco = coco_export([coco_image(1, "a.jpg")], [])
    (record,) = build_records(
        coco,
        {"a.jpg": manifest_entry("a")},
        ontology,
        options(review_by_image={"a.jpg": accepted_review(verified_empty=True)}),
    )[0]

    assert record["verified_empty"] is True
    assert record["provenance"]["human_edit_state"] == "accepted_unchanged"


def test_treatment_eligible_unknown_becomes_null(ontology) -> None:
    coco = coco_export([coco_image(1, "a.jpg")], [coco_polygon(1, 1, 2, WEED_POLYGON)])
    (record,) = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())[0]

    assert record["annotations"][0]["attributes"]["treatment_eligible"] is None


def test_unknown_label_names_the_image_and_the_label(ontology) -> None:
    coco = coco_export([coco_image(1, "field/a.jpg")], [coco_polygon(1, 1, 3, RICE_POLYGON)])
    coco["categories"].append({"id": 3, "name": "weeds", "supercategory": ""})

    with pytest.raises(CvatConversionError) as excinfo:
        build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())
    message = str(excinfo.value)
    assert "field/a.jpg" in message
    assert "'weeds'" in message


def test_unset_annotation_confidence_is_an_error_not_a_guess(ontology) -> None:
    coco = coco_export(
        [coco_image(1, "a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON, annotation_confidence="")]
    )

    with pytest.raises(CvatConversionError) as excinfo:
        build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())
    assert "annotation_confidence" in str(excinfo.value)
    assert "a.jpg" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_polygons_survive_as_polygons(ontology) -> None:
    coco = coco_export([coco_image(1, "a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON)])
    (record,) = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())[0]

    assert record["annotations"][0]["geometry"] == {"type": "polygon", "polygon": RICE_POLYGON}


def test_box_only_input_stays_box_only(ontology) -> None:
    coco = coco_export([coco_image(1, "a.jpg")], [coco_box(1, 1, 1, [10.0, 10.0, 40.0, 30.0])])
    (record,) = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())[0]

    assert record["annotations"][0]["geometry"] == {
        "type": "bbox",
        "bbox": [10.0, 10.0, 40.0, 30.0],
    }


def test_exporter_synthesised_rectangle_ring_is_not_promoted_to_polygon(ontology) -> None:
    """A ring that is exactly its own bbox is box work; polygon is canonical truth."""
    ring = [10.0, 10.0, 50.0, 10.0, 50.0, 40.0, 10.0, 40.0]
    coco = coco_export([coco_image(1, "a.jpg")], [coco_polygon(1, 1, 1, ring)])
    records, counts = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())

    assert records[0]["annotations"][0]["geometry"]["type"] == "bbox"
    assert counts["rectangle_polygons_demoted"] == 1

    kept, _ = build_records(
        coco, {"a.jpg": manifest_entry("a")}, ontology, options(keep_rectangle_polygons=True)
    )
    assert kept[0]["annotations"][0]["geometry"]["type"] == "polygon"


def test_mask_segmentation_stays_a_mask(ontology) -> None:
    annotation = coco_polygon(1, 1, 1, RICE_POLYGON)
    annotation["segmentation"] = {"size": [200, 200], "counts": "abc123"}
    coco = coco_export([coco_image(1, "a.jpg")], [annotation])
    (record,) = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())[0]

    assert record["annotations"][0]["geometry"]["type"] == "instance_mask"
    assert record["annotations"][0]["geometry"]["rle"]["size"] == [200, 200]


def test_multi_ring_polygon_is_refused_rather_than_truncated(ontology) -> None:
    annotation = coco_polygon(1, 1, 1, RICE_POLYGON)
    annotation["segmentation"] = [list(RICE_POLYGON), list(WEED_POLYGON)]
    coco = coco_export([coco_image(1, "a.jpg")], [annotation])

    with pytest.raises(CvatConversionError, match="multi-part polygon"):
        build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())


# --------------------------------------------------------------------------- #
# record_id
# --------------------------------------------------------------------------- #
def test_record_id_is_stable_across_runs_and_distinct_per_image(ontology) -> None:
    coco = coco_export(
        [coco_image(1, "train/a.jpg"), coco_image(2, "train/b.jpg")],
        [coco_polygon(1, 1, 1, RICE_POLYGON), coco_polygon(2, 2, 1, RICE_POLYGON)],
    )
    manifest = {"a.jpg": manifest_entry("a"), "b.jpg": manifest_entry("b")}
    first, _ = build_records(coco, manifest, ontology, options())
    second, _ = build_records(coco, manifest, ontology, options())

    assert [r["record_id"] for r in first] == [r["record_id"] for r in second]
    assert first[0]["record_id"] != first[1]["record_id"]


def test_record_id_ignores_cvat_ids_but_tracks_image_identity() -> None:
    """Re-exports renumber CVAT ids; the row id must not move with them."""
    base = {
        "dataset_id": "rice_phase2",
        "dataset_version": "rebuild-2026-07-29",
        "image_id": "a.jpg",
        "source_image_sha256": "a" * 64,
    }
    assert derive_record_id(**base) == derive_record_id(**base)
    assert derive_record_id(**base).startswith("rice_phase2:")
    assert derive_record_id(**{**base, "source_image_sha256": "b" * 64}) != derive_record_id(**base)
    assert derive_record_id(**{**base, "image_id": "b.jpg"}) != derive_record_id(**base)
    assert derive_record_id(**{**base, "dataset_version": None}) != derive_record_id(**base)


# --------------------------------------------------------------------------- #
# The manifest is the authority
# --------------------------------------------------------------------------- #
def test_manifest_supplies_hash_group_and_split_over_the_export(ontology) -> None:
    coco = coco_export([coco_image(1, "train/a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON)])
    manifest = {"a.jpg": manifest_entry("c", split="valid", group_id="#12")}
    (record,) = build_records(coco, manifest, ontology, options())[0]

    assert record["source"]["source_image_sha256"] == "c" * 64
    assert record["source"]["group_id"] == "#12"
    assert record["source"]["split"] == "validation"
    assert record["image_id"] == "a.jpg"


def test_dimension_disagreement_is_refused_then_resolved_for_the_manifest(ontology) -> None:
    coco = coco_export(
        [coco_image(1, "a.jpg", width=800, height=800)], [coco_polygon(1, 1, 1, RICE_POLYGON)]
    )
    manifest = {"a.jpg": manifest_entry("a")}

    with pytest.raises(CvatConversionError) as excinfo:
        build_records(coco, manifest, ontology, options())
    assert "800x800" in str(excinfo.value) and "200x200" in str(excinfo.value)

    (record,) = build_records(coco, manifest, ontology, options(allow_dimension_mismatch=True))[0]
    assert (record["source"]["width"], record["source"]["height"]) == (200, 200)


def test_image_missing_from_the_manifest_is_a_loud_error(ontology) -> None:
    coco = coco_export(
        [coco_image(1, "a.jpg"), coco_image(2, "ghost.jpg")],
        [coco_polygon(1, 1, 1, RICE_POLYGON)],
    )

    with pytest.raises(CvatConversionError) as excinfo:
        build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())
    assert "ghost.jpg" in str(excinfo.value)
    assert "manifest" in str(excinfo.value)


def test_two_exported_paths_with_one_manifest_entry_do_not_collapse(ontology) -> None:
    """train/a.jpg and valid/a.jpg cannot silently become one record."""
    coco = coco_export(
        [coco_image(1, "train/a.jpg"), coco_image(2, "valid/a.jpg")],
        [coco_polygon(1, 1, 1, RICE_POLYGON), coco_polygon(2, 2, 1, RICE_POLYGON)],
    )

    with pytest.raises(CvatConversionError, match="same manifest entry"):
        build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())


def test_manifest_without_a_group_id_is_refused(ontology) -> None:
    entry = manifest_entry("a")
    del entry["group_id"]
    coco = coco_export([coco_image(1, "a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON)])

    with pytest.raises(CvatConversionError, match="group_id"):
        build_records(coco, {"a.jpg": entry}, ontology, options())


def test_split_membership_manifest_shape_loads(tmp_path) -> None:
    path = tmp_path / "split_membership.json"
    path.write_text(json.dumps({"a.jpg": manifest_entry("a")}), encoding="utf-8")

    assert load_split_manifest(path)["a.jpg"]["split"] == "train"


# --------------------------------------------------------------------------- #
# Record-level edit state vs per-object edit action
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (["accepted", "accepted"], "accepted_unchanged"),
        (["accepted", "edited"], "edited"),
        (["deleted", "added"], "replaced"),
        (["reclassified"], "edited"),
    ],
)
def test_human_edit_state_summarises_the_object_actions(ontology, actions, expected) -> None:
    annotations = [
        coco_polygon(index, 1, 1, RICE_POLYGON, human_edit_action=action)
        for index, action in enumerate(actions, start=1)
    ]
    coco = coco_export([coco_image(1, "a.jpg")], annotations)
    (record,) = build_records(
        coco,
        {"a.jpg": manifest_entry("a")},
        ontology,
        options(review_by_image={"a.jpg": accepted_review()}),
    )[0]

    assert record["provenance"]["human_edit_state"] == expected
    assert [a["attributes"]["human_edit_action"] for a in record["annotations"]] == actions


def test_rejected_unusable_empty_record_is_not_summarised_as_accepted(ontology, tmp_path) -> None:
    """An image the reviewer threw out did not "accept the import unchanged"."""
    coco = coco_export([coco_image(1, "a.jpg")], [])
    (record,) = build_records(
        coco,
        {"a.jpg": manifest_entry("a")},
        ontology,
        options(
            review_by_image={
                "a.jpg": {
                    "annotator_id": "annotator-04",
                    "review_status": "rejected_unusable",
                    "unusable": True,
                }
            }
        ),
    )[0]

    assert record["provenance"]["human_edit_state"] == "unreviewed"
    assert record["unusable"] is True
    assert record["verified_empty"] is None
    assert validate([record], tmp_path) == []


def test_object_without_a_coco_id_is_refused_not_given_a_fake_one(ontology) -> None:
    annotation = coco_polygon(1, 1, 1, RICE_POLYGON)
    del annotation["id"]
    coco = coco_export([coco_image(1, "a.jpg")], [annotation])

    with pytest.raises(CvatConversionError, match="no COCO 'id'"):
        build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())


def test_manual_records_are_human_only_and_reject_proposal_actions(ontology, tmp_path) -> None:
    coco = coco_export(
        [coco_image(1, "a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON, human_edit_action="added")]
    )
    manual = options(
        proposal=ProposalProvenance(method="manual"),
        review_by_image={"a.jpg": accepted_review()},
    )
    (record,) = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, manual)[0]

    assert record["provenance"]["human_edit_state"] == "human_only"
    assert record["provenance"]["original_proposal"] is None
    assert validate([record], tmp_path) == []

    contradiction = coco_export(
        [coco_image(1, "a.jpg")],
        [coco_polygon(1, 1, 1, RICE_POLYGON, human_edit_action="accepted")],
    )
    with pytest.raises(CvatConversionError, match="manual"):
        build_records(contradiction, {"a.jpg": manifest_entry("a")}, ontology, manual)


def test_imported_records_keep_the_raw_objects(ontology) -> None:
    coco = coco_export([coco_image(1, "a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON)])
    (record,) = build_records(coco, {"a.jpg": manifest_entry("a")}, ontology, options())[0]

    raw = record["provenance"]["original_proposal"]
    assert raw[0]["bbox"] == coco["annotations"][0]["bbox"]
    assert raw[0]["attributes"]["occluded"] is False
    assert "__label__" not in raw[0]


def test_model_assisted_requires_its_full_lineage() -> None:
    with pytest.raises(CvatConversionError, match="--proposal-model-id"):
        ProposalProvenance(method="model_assisted")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_writes_a_package_that_passes_the_gate(tmp_path, capsys) -> None:
    coco_path = tmp_path / "instances_default.json"
    coco_path.write_text(
        json.dumps(
            coco_export(
                [coco_image(1, "train/a.jpg")],
                [coco_polygon(1, 1, 1, RICE_POLYGON, human_edit_action="accepted")],
            )
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "split_membership.json"
    manifest_path.write_text(json.dumps({"a.jpg": manifest_entry("a")}), encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({"images": {"a.jpg": accepted_review()}}), encoding="utf-8")
    out = tmp_path / "package.jsonl"

    code = main(
        [
            "--coco",
            str(coco_path),
            "--manifest",
            str(manifest_path),
            "--review-metadata",
            str(review_path),
            "--dataset-id",
            "rice_phase2",
            "--ontology",
            str(ONTOLOGY_PATH),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert "data-validate: passed" in capsys.readouterr().out
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert record["schema_version"] == "agrinav.annotation_record.v1"
    assert record["source"]["capture_metadata"]["split_manifest"]["path"] == manifest_path.name
    assert validate_packages([out], ontology_path=ONTOLOGY_PATH) == []


def test_cli_fails_when_an_accepted_package_does_not_pass_the_gate(tmp_path, capsys) -> None:
    """Accepted review, but no per-object decision: report the gate, exit non-zero."""
    coco_path = tmp_path / "instances_default.json"
    coco_path.write_text(
        json.dumps(coco_export([coco_image(1, "a.jpg")], [coco_polygon(1, 1, 1, RICE_POLYGON)])),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "split_membership.json"
    manifest_path.write_text(json.dumps({"a.jpg": manifest_entry("a")}), encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps({"images": {"a.jpg": accepted_review()}}), encoding="utf-8")
    out = tmp_path / "package.jsonl"

    code = main(
        [
            "--coco",
            str(coco_path),
            "--manifest",
            str(manifest_path),
            "--review-metadata",
            str(review_path),
            "--dataset-id",
            "rice_phase2",
            "--ontology",
            str(ONTOLOGY_PATH),
            "--out",
            str(out),
        ]
    )

    assert code == 1
    stderr = capsys.readouterr().err
    assert "human_edit_action" in stderr
    assert "must not enter a split" in stderr


def test_cli_reports_a_missing_manifest_entry_and_writes_nothing(tmp_path, capsys) -> None:
    coco_path = tmp_path / "instances_default.json"
    coco_path.write_text(
        json.dumps(coco_export([coco_image(1, "ghost.jpg")], [])), encoding="utf-8"
    )
    manifest_path = tmp_path / "split_membership.json"
    manifest_path.write_text(json.dumps({"a.jpg": manifest_entry("a")}), encoding="utf-8")
    out = tmp_path / "package.jsonl"

    code = main(
        [
            "--coco",
            str(coco_path),
            "--manifest",
            str(manifest_path),
            "--dataset-id",
            "rice_phase2",
            "--ontology",
            str(ONTOLOGY_PATH),
            "--out",
            str(out),
        ]
    )

    assert code == 2
    assert "ghost.jpg" in capsys.readouterr().err
    assert not out.exists()
