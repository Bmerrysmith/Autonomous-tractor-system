"""Tests for the blinded review sheet builder.

The predicates that matter here are the ones that would quietly invalidate 20
minutes of a reviewer's attention: stratum disjointness, sampling determinism,
the blinding, and the legacy-annotation lookup that has to survive ids colliding
across split files. All CPU, no GPU, no network, no image decoding.
"""

from __future__ import annotations

import collections
import json
from typing import Any

from agrinav.data import build_review_sheet as brs


def _obj(
    key: str,
    label: str = "rice_protect",
    new_iou: float = 0.80,
    old_iou: float = 0.70,
    area: float = 5000.0,
    components: int = 1,
    nonlargest: float = 0.0,
    legacy_split: str = "train",
    legacy_id: int = 1,
) -> dict[str, Any]:
    return {
        "image_sha256": key * 4,
        "source_object_id": f"coco-ann-{key}",
        "label": label,
        "new_box_iou": new_iou,
        "old_box_iou": old_iou,
        "box_area_px": area,
        "new_components": components,
        "new_nonlargest_area_fraction": nonlargest,
        "legacy_split": legacy_split,
        "legacy_annotation_id": legacy_id,
    }


# --------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------


def test_every_object_lands_in_at_most_one_stratum():
    """First-match-wins, so the counts printed on the sheet are unambiguous."""
    objects = [
        _obj("a", label="weed_target", nonlargest=0.4, components=3),
        _obj("b", label="weed_target", new_iou=0.9, old_iou=0.5),
        _obj("c", label="weed_target", new_iou=0.5, old_iou=0.9),
        _obj("d", label="weed_target", new_iou=0.70, old_iou=0.70),
        _obj("e", nonlargest=0.2, components=4),
        _obj("f", components=3, nonlargest=0.01),
        _obj("g", area=100.0, new_iou=0.9, old_iou=0.5),
        _obj("h", area=100.0, new_iou=0.5, old_iou=0.9),
        _obj("i", new_iou=0.9, old_iou=0.5),
        _obj("j", new_iou=0.5, old_iou=0.9),
        _obj("k", new_iou=0.70, old_iou=0.70),
    ]
    assigned = [brs.assign_stratum(o) for o in objects]
    assert all(a is not None for a in assigned)
    assert len(set(assigned)) == len(assigned), "each fixture should hit a distinct stratum"

    drawn, summary = brs.select_stratified(objects, seed=1)
    assert len(drawn) == len(objects)
    assert sum(row["drawn"] for row in summary) == len(objects)
    counts = collections.Counter(d["stratum"] for d in drawn)
    assert max(counts.values()) == 1


def test_fragment_strata_take_precedence_over_win_loss():
    """Fragmentation is the question, so it must not be diluted into win/loss."""
    fragmented_winner = _obj("z", label="weed_target", new_iou=0.95, old_iou=0.5, nonlargest=0.3)
    assert brs.assign_stratum(fragmented_winner) == "weed_fragment_substantial"


def test_speck_fragmentation_is_separated_from_substantial():
    speck = _obj("s", components=6, nonlargest=0.004)
    substantial = _obj("t", components=2, nonlargest=0.30)
    assert brs.assign_stratum(speck) == "rice_fragment_specks"
    assert brs.assign_stratum(substantial) == "rice_fragment_substantial"


def test_stratum_targets_cap_the_draw_but_never_exceed_the_pool():
    objects = [_obj(chr(97 + i), new_iou=0.9, old_iou=0.5) for i in range(50)]
    drawn, summary = brs.select_stratified(objects, seed=7)
    row = next(r for r in summary if r["stratum"] == "large_new_wins")
    assert row["pool"] == 50
    assert row["drawn"] == row["target"] < 50
    assert len(drawn) == row["target"]


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_same_seed_reproduces_the_same_draw_regardless_of_input_order():
    objects = [_obj(chr(97 + i), new_iou=0.9, old_iou=0.5) for i in range(30)]
    first, _ = brs.select_stratified(objects, seed=20260901)
    second, _ = brs.select_stratified(list(reversed(objects)), seed=20260901)
    assert [brs.object_key(o) for o in first] == [brs.object_key(o) for o in second]


def test_different_seeds_draw_different_objects():
    objects = [_obj(chr(97 + i), new_iou=0.9, old_iou=0.5) for i in range(30)]
    a, _ = brs.select_stratified(objects, seed=1)
    b, _ = brs.select_stratified(objects, seed=2)
    assert {brs.object_key(o) for o in a} != {brs.object_key(o) for o in b}


# --------------------------------------------------------------------------
# blinding
# --------------------------------------------------------------------------


def test_panels_are_complementary_and_deterministic():
    for key in (f"sha{i}|obj{i}" for i in range(50)):
        panels = brs.blind_panels(key, 20260901)
        assert set(panels) == {"legacy", "new_sm"}
        assert brs.blind_panels(key, 20260901) == panels


def test_blinding_is_not_all_one_way():
    keys = [f"sha{i}|obj{i}" for i in range(200)]
    first_panel = collections.Counter(brs.blind_panels(k, 20260901)[0] for k in keys)
    assert min(first_panel.values()) > 60, f"assignment too lopsided: {first_panel}"


def test_changing_the_blind_seed_reassigns_panels():
    keys = [f"sha{i}|obj{i}" for i in range(100)]
    a = [brs.blind_panels(k, 1)[0] for k in keys]
    b = [brs.blind_panels(k, 2)[0] for k in keys]
    assert a != b


def test_rendered_sheet_never_names_the_source_or_the_stratum_per_card():
    """A stratum such as weed_legacy_wins states which set wins.

    Printed next to each panel box IoU it hands over the answer, so it must not
    reach the card, the DOM, or the exported verdicts.
    """
    cards = [
        {
            "object_id": "001",
            "stratum": "weed_legacy_wins",
            "label": "weed_target",
            "box_w": 40,
            "box_h": 30,
            "iou_new_old": 0.8,
            "panels": {
                letter: {
                    "b64": "",
                    "width": 10,
                    "height": 10,
                    "box_iou": 0.7,
                    "components": 1,
                    "nonlargest_pct": "0.0%",
                }
                for letter in ("A", "B")
            },
        }
    ]
    summary = [
        {"stratum": "weed_legacy_wins", "description": "d", "pool": 44, "drawn": 1, "target": 8}
    ]
    context = {
        "title": "t",
        "generated_at": "now",
        "key_path": "k.json",
        "raw_shard": "s.jsonl",
        "sample_seed": 1,
        "blind_seed": 2,
        "pool_total": 1238,
        "drawn_total": 1,
        "images_compared": 37,
        "corpus_images": 2318,
        "pilot_fraction": "40 images",
        "drawn_weed": 1,
        "drawn_weed_pct": "100.0%",
        "pool_weed_pct": "15.8%",
        "drawn_small": 0,
        "drawn_fragmented": 0,
        "dropped": 0,
        "dropped_note": ".",
    }
    markup = brs.build_html(cards, summary, context)

    card_start = markup.index('data-object-id="001"')
    card_end = markup.index("</fieldset>", card_start)
    card = markup[card_start:card_end]
    assert "weed_legacy_wins" not in card
    assert "new_sm" not in card
    assert "legacy" not in card
    assert "data-stratum" not in markup
    # the stratum table above the cards is disclosure, and must still be there
    assert "weed_legacy_wins" in markup[:card_start]


# --------------------------------------------------------------------------
# the legacy id collision that silently rendered the wrong object
# --------------------------------------------------------------------------


def test_legacy_key_disambiguates_ids_that_collide_across_split_files():
    """Regression: legacy ids restart at 1 in every split file.

    instances_train numbers 1..56502 and instances_valid restarts at 1, so a
    bare-id lookup resolved valid#1 to train#1 -- a different object on a
    different image -- and rendered a polygon unrelated to the crop.
    """
    assert brs.legacy_key("train", 1) != brs.legacy_key("valid", 1)
    assert brs.legacy_key("valid", 1) != brs.legacy_key("test", 1)
    assert brs.legacy_key("train", 7) == brs.legacy_key("train", 7)


def test_load_legacy_polygons_returns_the_annotation_from_the_named_split(tmp_path):
    for split, marker in (("train", "TRAIN"), ("valid", "VALID")):
        (tmp_path / f"instances_{split}.coco.json").write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [{"id": 1, "bbox": [0, 0, 1, 1], "marker": marker}],
                }
            ),
            encoding="utf-8",
        )
    found = brs.load_legacy_polygons(tmp_path, ["train", "valid"], {brs.legacy_key("valid", 1)})
    assert list(found) == [brs.legacy_key("valid", 1)]
    assert found[brs.legacy_key("valid", 1)]["marker"] == "VALID"
