#!/usr/bin/env python3
"""Build a blinded, stratified side-by-side sheet for human polygon review.

THE ONE QUESTION THIS SHEET EXISTS TO ANSWER
--------------------------------------------
The 40-image re-pilot left a trade-off that measurement cannot settle: the new
SAM2.1 ``single_mask`` polygons localise better than the legacy polygons
(median box IoU 0.774 vs 0.759) but are fragmented far more often
(multi-component 13.6% vs 2.0%). Better localisation, messier shapes.

Which one costs a reviewer less to correct in CVAT is a judgement about human
effort, and the only instrument for it is a human. This module turns ~20 minutes
of that scarce attention into an answer.

WHY IT IS BLINDED
-----------------
A reviewer who can see which panel is the new one is scoring their expectation,
not the geometry. Left/right is therefore randomised per object from a recorded
seed, and the key is written to a SEPARATE file that the sheet does not embed
and does not link to by content.

Residual leak, stated on the sheet rather than hidden: per-panel component
counts are disclosed because a reviewer needs them to judge, and a panel showing
7 components is weak evidence of being the new set (legacy fragments 2% of the
time). The alternative -- withholding the number -- costs more than the leak.

WHY THE SAMPLING IS STRATIFIED AND DISCLOSED
--------------------------------------------
A uniform draw of 89 from 1238 would contain ~3 substantial-fragment cases,
which are the entire question. Strata are therefore deliberately unequal, first
match wins so they are disjoint, and both the pool size and the drawn count for
every stratum are printed on the sheet itself. Nobody should be able to mistake
89 hand-picked-by-rule objects for the corpus.

Selection inside a stratum is content-addressed
(``sha256(seed|image_sha|source_object_id)``), so the same seed reproduces the
same sheet on any machine and re-running cannot quietly reshuffle it.

WHAT THIS MODULE IS NOT
-----------------------
It renders. It selects nothing for training, writes no annotation record, and
asserts no truth: both polygon sets remain unreviewed proposals, and a reviewer
answering "A" has expressed a preference about correction effort, not accepted a
mask.

CLI::

    python -m agrinav.data.build_review_sheet \\
        --objects-jsonl artifacts/sam_reseed_2026-09-01/iou_objects_single_mask_exact_box.jsonl \\
        --raw-shard artifacts/sam_reseed_2026-09-01/sam_raw/pilot_shard_000.jsonl \\
        --coco-zip artifacts/sam_reseed_2026-09-01/pilot_images.zip \\
        --legacy-annotations-dir <intake>/detection/RICE/annotations \\
        --out-html reports/figures/sam_reseed_review_2026-09-01.html \\
        --out-key reports/figures/sam_reseed_review_2026-09-01.BLINDING_KEY.json \\
        --out-png-dir reports/figures/sam_reseed_review_2026-09-01_panels
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from agrinav.data.compare_sam_polygons import (
    BOX_MATCH_MIN_IOU,
    _decode_polygon,
    _decode_rle,
    box_iou,
    xywh_to_xyxy,
)

#: Panel geometry. A crop is padded around the human box so the reviewer can see
#: whether a stray component belongs to a neighbouring plant.
CROP_MARGIN_FRACTION = 0.35
CROP_MARGIN_MIN_PX = 10
PANEL_LONG_EDGE_PX = 280

#: Both panels are drawn with IDENTICAL styling. Any difference in colour,
#: width or alpha between them would defeat the blinding.
COLOUR_HUMAN_BOX = (0, 212, 255)  # BGR -> amber
COLOUR_MASK = (255, 229, 0)  # BGR -> cyan
MASK_FILL_ALPHA = 0.30

#: A mask whose non-largest components are below this share of its area is
#: "specks": one largest-component call from clean, so its fragmentation is a
#: post-processing question rather than a reviewer cost. At or above it, there
#: is real structure outside the main blob and a human has to look.
SUBSTANTIAL_FRAGMENT_FRACTION = 0.05
SMALL_OBJECT_AREA_PX = 32 * 32
#: Below this absolute box-IoU difference the two sets are treated as agreeing.
TIE_BAND = 0.02

Predicate = Callable[[dict[str, Any]], bool]


def _delta(obj: dict[str, Any]) -> float:
    return float(obj["new_box_iou"]) - float(obj["old_box_iou"])


def _is_weed(obj: dict[str, Any]) -> bool:
    return obj["label"] == "weed_target"


def _is_small(obj: dict[str, Any]) -> bool:
    return float(obj["box_area_px"]) < SMALL_OBJECT_AREA_PX


def _has_substantial_fragments(obj: dict[str, Any]) -> bool:
    return float(obj.get("new_nonlargest_area_fraction", 0.0)) >= SUBSTANTIAL_FRAGMENT_FRACTION


def _has_specks(obj: dict[str, Any]) -> bool:
    return int(obj.get("new_components", 1)) >= 2 and not _has_substantial_fragments(obj)


#: (name, human description, predicate, target draw). ORDER MATTERS: first match
#: wins, so the strata are disjoint and the counts on the sheet are unambiguous.
#: Fragment strata come first and take everything available, because they are
#: the question; the win/loss strata are sampled.
STRATA: tuple[tuple[str, str, Predicate, int], ...] = (
    (
        "weed_fragment_substantial",
        "weed_target where the new mask has real structure outside its largest "
        "component (>=5% of area). The decision-relevant class, hardest case.",
        lambda o: _is_weed(o) and _has_substantial_fragments(o),
        17,
    ),
    (
        "weed_new_wins",
        "weed_target, new mask localises better by more than 0.02 box IoU.",
        lambda o: _is_weed(o) and _delta(o) > TIE_BAND,
        8,
    ),
    (
        "weed_legacy_wins",
        "weed_target, legacy mask localises better by more than 0.02 box IoU.",
        lambda o: _is_weed(o) and _delta(o) < -TIE_BAND,
        8,
    ),
    (
        "weed_agree",
        "weed_target, the two agree within 0.02 box IoU.",
        lambda o: _is_weed(o) and abs(_delta(o)) <= TIE_BAND,
        5,
    ),
    (
        "rice_fragment_substantial",
        "rice_protect with substantial fragments (>=5% of area outside the " "largest component).",
        lambda o: _has_substantial_fragments(o),
        10,
    ),
    (
        "rice_fragment_specks",
        "rice_protect fragmented only into specks (<5% of area outside the "
        "largest component). Included so the reviewer can confirm that these "
        "really are cosmetic.",
        _has_specks,
        8,
    ),
    (
        "small_new_wins",
        "rice_protect under 32x32 px, new mask wins.",
        lambda o: _is_small(o) and _delta(o) > TIE_BAND,
        8,
    ),
    (
        "small_legacy_wins",
        "rice_protect under 32x32 px, legacy mask wins.",
        lambda o: _is_small(o) and _delta(o) < -TIE_BAND,
        8,
    ),
    (
        "large_new_wins",
        "rice_protect 32x32 px or larger, new mask wins.",
        lambda o: _delta(o) > TIE_BAND,
        6,
    ),
    (
        "large_legacy_wins",
        "rice_protect 32x32 px or larger, legacy mask wins.",
        lambda o: _delta(o) < -TIE_BAND,
        6,
    ),
    (
        "rice_agree",
        "rice_protect, the two agree within 0.02 box IoU.",
        lambda o: abs(_delta(o)) <= TIE_BAND,
        5,
    ),
)


def object_key(obj: dict[str, Any]) -> str:
    return f"{obj['image_sha256']}|{obj['source_object_id']}"


def assign_stratum(obj: dict[str, Any]) -> str | None:
    """First matching stratum wins, so every object lands in at most one."""
    for name, _description, predicate, _target in STRATA:
        if predicate(obj):
            return name
    return None


def _rank(seed: int, salt: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{salt}|{key}".encode("utf-8")).hexdigest()


def select_stratified(
    objects: Sequence[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (drawn objects in stratum order, per-stratum pool/draw counts)."""
    pools: dict[str, list[dict[str, Any]]] = {name: [] for name, _d, _p, _t in STRATA}
    for obj in objects:
        stratum = assign_stratum(obj)
        if stratum is not None:
            pools[stratum].append(obj)

    drawn: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for name, description, _predicate, target in STRATA:
        pool = sorted(pools[name], key=lambda o: _rank(seed, name, object_key(o)))
        take = pool if target <= 0 or target >= len(pool) else pool[:target]
        for obj in take:
            enriched = dict(obj)
            enriched["stratum"] = name
            drawn.append(enriched)
        summary.append(
            {
                "stratum": name,
                "description": description,
                "pool": len(pool),
                "drawn": len(take),
                "target": target,
            }
        )
    return drawn, summary


def blind_panels(obj_key: str, seed: int) -> tuple[str, str]:
    """Assign the two sources to panels A and B. Deterministic, per object."""
    digest = _rank(seed, "blind", obj_key)
    if int(digest[:8], 16) % 2 == 0:
        return "legacy", "new_sm"
    return "new_sm", "legacy"


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def crop_window(box_xyxy: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    margin_x = max(CROP_MARGIN_MIN_PX, (x1 - x0) * CROP_MARGIN_FRACTION)
    margin_y = max(CROP_MARGIN_MIN_PX, (y1 - y0) * CROP_MARGIN_FRACTION)
    cx0 = max(0, int(np.floor(x0 - margin_x)))
    cy0 = max(0, int(np.floor(y0 - margin_y)))
    cx1 = min(width, int(np.ceil(x1 + margin_x)))
    cy1 = min(height, int(np.ceil(y1 + margin_y)))
    return cx0, cy0, cx1, cy1


def render_panel(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    box_xyxy: Sequence[float],
    window: tuple[int, int, int, int],
) -> bytes:
    """Draw one panel: translucent mask + contour, then the human box. PNG bytes.

    Identical code path and styling for both sources; the caller decides which
    mask it is handed. Nothing here knows or records the provenance.
    """
    import cv2

    cx0, cy0, cx1, cy1 = window
    crop = np.ascontiguousarray(image_rgb[cy0:cy1, cx0:cx1])
    crop_mask = np.ascontiguousarray(mask[cy0:cy1, cx0:cx1]).astype(np.uint8)
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        raise ValueError(f"degenerate crop {crop.shape} for window {window}")

    scale = PANEL_LONG_EDGE_PX / max(crop.shape[0], crop.shape[1])
    target = (max(2, int(round(crop.shape[1] * scale))), max(2, int(round(crop.shape[0] * scale))))
    canvas = cv2.resize(crop, target, interpolation=cv2.INTER_CUBIC)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    big_mask = cv2.resize(crop_mask, target, interpolation=cv2.INTER_NEAREST)

    if np.any(big_mask):
        overlay = canvas.copy()
        overlay[big_mask.astype(bool)] = COLOUR_MASK
        canvas = cv2.addWeighted(overlay, MASK_FILL_ALPHA, canvas, 1.0 - MASK_FILL_ALPHA, 0.0)
        contours, _ = cv2.findContours(big_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, COLOUR_MASK, 2)

    bx0 = int(round((float(box_xyxy[0]) - cx0) * scale))
    by0 = int(round((float(box_xyxy[1]) - cy0) * scale))
    bx1 = int(round((float(box_xyxy[2]) - cx0) * scale))
    by1 = int(round((float(box_xyxy[3]) - cy0) * scale))
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), COLOUR_HUMAN_BOX, 2)

    ok, buffer = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buffer.tobytes()


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def load_new_masks(raw_shard: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """Index the chosen single_mask candidate RLE by object key."""
    out: dict[str, dict[str, Any]] = {}
    with raw_shard.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = f"{row['source_image_sha256']}|{row['source_object_id']}"
            if key not in wanted:
                continue
            for candidate in row["provenance"]["original_proposal"]["candidates"]:
                if candidate["kind"] == "single_mask":
                    out[key] = candidate
                    break
    return out


def legacy_key(split: str, annotation_id: int) -> str:
    """Legacy annotation ids are only unique WITHIN a split file.

    ``instances_train`` numbers 1..56502, ``instances_valid`` restarts at 1, and
    so does ``instances_test`` -- so every valid and test id collides with a
    train id. A bare-id lookup silently returns a different object on a
    different image, which renders a polygon that has nothing to do with the
    crop. The split is part of the key, always.
    """
    return f"{split}|{int(annotation_id)}"


def load_legacy_polygons(
    annotations_dir: Path, splits: Sequence[str], wanted: set[str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in splits:
        path = annotations_dir / f"instances_{split}.coco.json"
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        for ann in doc["annotations"]:
            key = legacy_key(split, ann["id"])
            if key in wanted:
                out[key] = ann
        del doc
    return out


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #14161a; --muted: #4a5057;
  --card: #f4f6f8; --line: #c3c9d0; --accent: #0b4fa8; --warn: #8a3b00;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14161a; --fg: #f2f4f7; --muted: #b9c0c8;
          --card: #1e2228; --line: #454d57; --accent: #8ab4ff; --warn: #ffb26b; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 1.5rem; background: var(--bg); color: var(--fg);
       font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .4rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 .6rem; border-bottom: 2px solid var(--line);
     padding-bottom: .3rem; }
h3 { font-size: 1rem; margin: 0; }
p, li { max-width: 78ch; }
.lede { font-size: 1.05rem; }
.callout { border-left: 4px solid var(--accent); background: var(--card);
           padding: .8rem 1rem; margin: 1rem 0; border-radius: 0 6px 6px 0; }
.callout.warn { border-left-color: var(--warn); }
table { border-collapse: collapse; margin: 1rem 0; font-size: .9rem; width: 100%; }
th, td { border: 1px solid var(--line); padding: .4rem .6rem; text-align: left; }
th { background: var(--card); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.card { border: 1px solid var(--line); border-radius: 8px; background: var(--card);
        padding: 1rem; margin: 1.2rem 0; }
.card > legend { font-weight: 700; padding: 0 .4rem; }
.meta { color: var(--muted); font-size: .85rem; margin: .2rem 0 .8rem; }
.panels { display: flex; flex-wrap: wrap; gap: 1rem; }
figure { margin: 0; }
figure img { display: block; border: 1px solid var(--line); border-radius: 4px;
             background: #000; }
figcaption { font-size: .82rem; color: var(--muted); margin-top: .3rem;
             font-variant-numeric: tabular-nums; }
.choices { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .8rem; }
.choices label { display: inline-flex; align-items: center; gap: .4rem;
                 min-height: 44px; min-width: 44px; padding: .3rem .8rem;
                 border: 1px solid var(--line); border-radius: 6px; cursor: pointer; }
.choices input { width: 20px; height: 20px; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
textarea { width: 100%; min-height: 9rem; font-family: ui-monospace, monospace;
           font-size: .8rem; background: var(--bg); color: var(--fg);
           border: 1px solid var(--line); border-radius: 6px; padding: .5rem; }
button { min-height: 44px; padding: .5rem 1.1rem; font-size: 1rem; cursor: pointer;
         border: 1px solid var(--line); border-radius: 6px;
         background: var(--card); color: var(--fg); }
.swatch { display: inline-block; width: .9rem; height: .9rem; vertical-align: -1px;
          border: 1px solid var(--fg); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important;
                                              animation: none !important; } }
"""

_SCRIPT = """
function collectVerdicts() {
  var rows = [];
  document.querySelectorAll('fieldset[data-object-id]').forEach(function (fs) {
    var picked = fs.querySelector('input[type=radio]:checked');
    rows.push({
      object_id: fs.getAttribute('data-object-id'),
      verdict: picked ? picked.value : 'unanswered'
    });
  });
  var answered = rows.filter(function (r) { return r.verdict !== 'unanswered'; });
  var out = { generated_at: new Date().toISOString(), answered: answered.length,
              total: rows.length, verdicts: rows };
  var box = document.getElementById('verdict-output');
  box.value = JSON.stringify(out, null, 2);
  document.getElementById('verdict-status').textContent =
    answered.length + ' of ' + rows.length + ' answered. Copy the text below and save it.';
  box.focus();
  box.select();
}
"""


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def build_html(
    cards: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    context: dict[str, Any],
) -> str:
    e = html.escape
    parts: list[str] = []
    parts.append(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(context['title'])}</title><style>{_STYLE}</style></head><body><main>"
    )

    parts.append(f"<h1>{e(context['title'])}</h1>")
    parts.append(
        '<p class="lede"><strong>You are deciding one thing: which panel would take '
        "less work to correct in CVAT.</strong> Not which is prettier, not which is "
        "more detailed &mdash; which one you would rather be handed as a starting "
        "point, counting the clicks to make it right.</p>"
    )

    parts.append('<div class="callout">')
    parts.append('<h2 style="margin-top:0;border:0">How to read a pair</h2>')
    parts.append(
        "<p>Both panels show the <em>same</em> image crop and the <em>same</em> "
        "human-drawn box. Only the polygon differs. Styling is identical on both "
        "sides on purpose.</p><ul>"
        '<li><span class="swatch" style="background:#00d4ff"></span> '
        "<strong>Amber rectangle</strong> &mdash; the human box. Reviewed geometry, "
        "identical in both panels.</li>"
        '<li><span class="swatch" style="background:#00e5ff"></span> '
        "<strong>Cyan outline and tint</strong> &mdash; the machine polygon under "
        "judgement.</li></ul>"
        "<p>Useful things to weigh: does the polygon miss part of the plant; does it "
        "spill onto a neighbour; are the extra blobs real leaf tips or noise you would "
        "have to delete; would you rather redraw from scratch than fix it.</p>"
        "</div>"
    )

    parts.append('<div class="callout warn">')
    parts.append('<h2 style="margin-top:0;border:0">This sheet is blinded</h2>')
    parts.append(
        "<p>Which side is the legacy polygon and which is the new SAM&nbsp;2.1 polygon "
        "is randomised per object from a recorded seed. <strong>Do not look up the key "
        "until you have finished.</strong> The key is at:</p>"
        f"<p><code>{e(context['key_path'])}</code></p>"
        "<p><strong>Known leak, so you can discount it:</strong> each panel's component "
        "count is printed, and the new set fragments far more often than the legacy "
        "set. A panel showing many components is weak evidence of being the new one. "
        "The number is shown anyway because you need it to judge the effort.</p>"
        "</div>"
    )

    parts.append("<h2>What was sampled, and what was not</h2>")
    parts.append(
        f"<p>These <strong>{context['drawn_total']} objects</strong> are drawn from "
        f"<strong>{context['pool_total']} paired objects</strong> across "
        f"{context['images_compared']} images of a 40-image pilot &mdash; which is "
        f"itself {context['pilot_fraction']} of the {context['corpus_images']}-image "
        "train+valid corpus. <strong>This is not a random sample and it is not the "
        "whole picture.</strong> Strata are deliberately unequal: fragmentation cases "
        "and <code>weed_target</code> are over-represented because they are the "
        "decision-relevant ones.</p>"
    )
    parts.append(
        "<p>Assignment is first-match-wins down the table, so the strata are disjoint. "
        f"Selection within a stratum is content-addressed with seed "
        f"<code>{context['sample_seed']}</code>: the same seed reproduces this exact "
        "sheet.</p>"
    )
    parts.append(
        '<table><caption class="meta">Per-stratum pool and draw</caption><thead><tr>'
        '<th scope="col">Stratum</th><th scope="col">Definition</th>'
        '<th scope="col" class="num">Pool</th><th scope="col" class="num">Drawn</th>'
        "</tr></thead><tbody>"
    )
    for row in summary:
        parts.append(
            f"<tr><th scope=\"row\"><code>{e(row['stratum'])}</code></th>"
            f"<td>{e(row['description'])}</td>"
            f"<td class=\"num\">{row['pool']}</td>"
            f"<td class=\"num\">{row['drawn']}</td></tr>"
        )
    parts.append(
        f'<tr><th scope="row">Total</th><td>&mdash;</td>'
        f"<td class=\"num\">{context['pool_total']}</td>"
        f"<td class=\"num\">{context['drawn_total']}</td></tr>"
    )
    parts.append("</tbody></table>")
    parts.append(
        f'<p class="meta">Composition of the drawn set: '
        f"{context['drawn_weed']} <code>weed_target</code> "
        f"({context['drawn_weed_pct']} of the sheet, against "
        f"{context['pool_weed_pct']} of the pool) &middot; "
        f"{context['drawn_small']} objects under 32&times;32&nbsp;px &middot; "
        f"{context['drawn_fragmented']} where the new mask is multi-component. "
        f"Objects dropped because a crop could not be rendered: "
        f"<strong>{context['dropped']}</strong>"
        f"{e(context['dropped_note'])}</p>"
    )

    parts.append("<h2>The pairs</h2>")
    parts.append(
        '<p class="meta">Per-panel numbers: box IoU against the human box, '
        "connected components, and the share of mask area outside its largest "
        "component.</p>"
    )

    for card in cards:
        oid = e(card["object_id"])
        # NOTE: the stratum is deliberately NOT rendered here, in the DOM or in
        # the exported verdicts. Names like "weed_legacy_wins" state which set
        # wins, which together with the per-panel box IoU below would hand the
        # reviewer the answer. object_id -> stratum lives in the key file, and
        # the join happens after unblinding.
        parts.append(f'<fieldset class="card" data-object-id="{oid}">')
        parts.append(f"<legend>Object {oid}</legend>")
        parts.append(
            f"<p class=\"meta\">class <strong>{e(card['label'])}</strong> &middot; "
            f"box {card['box_w']}&times;{card['box_h']}&nbsp;px &middot; "
            f"the two polygons overlap at IoU {_fmt(card['iou_new_old'])}</p>"
        )
        parts.append('<div class="panels">')
        for panel in ("A", "B"):
            data = card["panels"][panel]
            parts.append(
                "<figure>"
                f"<img src=\"data:image/png;base64,{data['b64']}\" "
                f"alt=\"Panel {panel} for object {oid}: {e(card['label'])} crop with the "
                f'human box and one candidate polygon outlined" '
                f"width=\"{data['width']}\" height=\"{data['height']}\">"
                f"<figcaption><strong>Panel {panel}</strong> &middot; "
                f"box IoU {_fmt(data['box_iou'])} &middot; "
                f"{data['components']} component{'' if data['components'] == 1 else 's'} "
                f"&middot; {data['nonlargest_pct']} outside largest</figcaption>"
                "</figure>"
            )
        parts.append("</div>")
        parts.append('<div class="choices" role="group">')
        for value, text in (
            ("A", "A is easier to correct"),
            ("B", "B is easier to correct"),
            ("same", "About the same"),
            ("both_bad", "Both need a redraw"),
        ):
            rid = f"o{oid}_{value}"
            parts.append(
                f'<label for="{rid}"><input type="radio" id="{rid}" '
                f'name="obj_{oid}" value="{value}"> {text}</label>'
            )
        parts.append("</div></fieldset>")

    parts.append("<h2>When you are done</h2>")
    parts.append(
        "<p>Press the button, then copy the JSON and save it next to this file. "
        "Nothing is uploaded anywhere; this page has no network access.</p>"
        "<p>Your answers carry only the object number. Which stratum each object "
        "came from is in the key file, so the analysis joins the two <em>after</em> "
        "you have finished &mdash; that is why the stratum is not shown on the "
        "cards above.</p>"
        '<p><button type="button" onclick="collectVerdicts()">'
        "Collect my answers</button></p>"
        '<p id="verdict-status" role="status" aria-live="polite"></p>'
        '<label for="verdict-output">Your answers as JSON</label>'
        '<textarea id="verdict-output" readonly '
        'placeholder="Press the button above."></textarea>'
    )
    parts.append(
        '<h2>Provenance</h2><p class="meta">'
        f"Generated {e(context['generated_at'])} by "
        "<code>agrinav.data.build_review_sheet</code>. "
        "Both polygon sets are <strong>unreviewed proposals</strong>; answering here "
        "expresses a preference about correction effort and accepts nothing. "
        f"Source shard: <code>{e(context['raw_shard'])}</code>. "
        f"Sample seed <code>{context['sample_seed']}</code>, blinding seed "
        f"<code>{context['blind_seed']}</code>.</p>"
    )
    parts.append(f"<script>{_SCRIPT}</script></main></body></html>")
    return "".join(parts)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build(
    *,
    objects_jsonl: Path,
    raw_shard: Path,
    coco_zip: Path,
    proposals_json: Path,
    legacy_dir: Path,
    legacy_splits: Sequence[str],
    out_html: Path,
    out_key: Path,
    out_png_dir: Path,
    sample_seed: int,
    blind_seed: int,
) -> dict[str, Any]:
    from PIL import Image

    objects = [json.loads(line) for line in objects_jsonl.read_text("utf-8").splitlines() if line]
    missing = [o for o in objects if "image_sha256" not in o]
    if missing:
        raise SystemExit(
            f"{len(missing)} object rows lack identity fields; regenerate them with "
            "a current agrinav.data.compare_sam_polygons (--out-objects-jsonl)."
        )

    drawn, summary = select_stratified(objects, sample_seed)
    keys = {object_key(o) for o in drawn}
    new_candidates = load_new_masks(raw_shard, keys)
    if any("legacy_split" not in o for o in drawn):
        raise SystemExit(
            "object rows lack 'legacy_split'; regenerate them with a current "
            "agrinav.data.compare_sam_polygons. A legacy annotation id alone is "
            "ambiguous across split files and resolves to the wrong object."
        )
    legacy_anns = load_legacy_polygons(
        legacy_dir,
        legacy_splits,
        {legacy_key(o["legacy_split"], o["legacy_annotation_id"]) for o in drawn},
    )
    proposals = json.loads(proposals_json.read_text(encoding="utf-8"))
    file_by_sha = {str(i["sha256"]).lower(): i["file_name"] for i in proposals["images"]}

    out_png_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    image_cache: dict[str, np.ndarray] = {}

    with zipfile.ZipFile(coco_zip, "r") as archive:
        for index, obj in enumerate(drawn, start=1):
            oid = f"{index:03d}"
            key = object_key(obj)
            reason = None
            try:
                sha = str(obj["image_sha256"]).lower()
                if sha not in image_cache:
                    with Image.open(io.BytesIO(archive.read(file_by_sha[sha]))) as handle:
                        image_cache[sha] = np.asarray(handle.convert("RGB"))
                image = image_cache[sha]
                candidate = new_candidates[key]
                legacy_ann = legacy_anns[
                    legacy_key(obj["legacy_split"], obj["legacy_annotation_id"])
                ]
                height, width = int(obj["height"]), int(obj["width"])
                new_mask = _decode_rle(candidate["rle"])
                old_mask = _decode_polygon(legacy_ann.get("segmentation"), height, width)
                if old_mask is None:
                    raise ValueError("legacy segmentation empty")
                box = obj["human_box_xyxy"]
                # Guard: the legacy annotation must be the SAME object the
                # comparison paired, or the panel shows an unrelated polygon over
                # this crop. A blank or wrong panel is worse than a hard failure,
                # so this is checked rather than trusted.
                agreement = box_iou(box, xywh_to_xyxy(legacy_ann["bbox"]))
                if agreement < BOX_MATCH_MIN_IOU:
                    raise ValueError(
                        f"legacy annotation {obj['legacy_split']}#"
                        f"{obj['legacy_annotation_id']} box IoU {agreement:.3f} "
                        f"against the human box; wrong object"
                    )
                window = crop_window(box, width, height)
                masks = {"legacy": old_mask, "new_sm": new_mask}
                panel_sources = blind_panels(key, blind_seed)
                panels: dict[str, Any] = {}
                for letter, source in zip(("A", "B"), panel_sources):
                    png = render_panel(image, masks[source], box, window)
                    path = out_png_dir / f"obj_{oid}_{letter}.png"
                    path.write_bytes(png)
                    side = "new" if source == "new_sm" else "old"
                    with Image.open(io.BytesIO(png)) as rendered:
                        pw, ph = rendered.size
                    panels[letter] = {
                        "b64": base64.b64encode(png).decode("ascii"),
                        "width": pw,
                        "height": ph,
                        "box_iou": obj[f"{side}_box_iou"],
                        "components": int(obj[f"{side}_components"]),
                        "nonlargest_pct": f"{100 * float(obj[f'{side}_nonlargest_area_fraction']):.1f}%",
                        "png": str(path),
                    }
            except (KeyError, ValueError, OSError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
            if reason is not None:
                dropped.append({"object_key": key, "stratum": obj["stratum"], "reason": reason})
                continue

            cards.append(
                {
                    "object_id": oid,
                    "stratum": obj["stratum"],
                    "label": obj["label"],
                    "box_w": int(round(box[2] - box[0])),
                    "box_h": int(round(box[3] - box[1])),
                    "iou_new_old": obj.get("iou_new_old"),
                    "panels": panels,
                }
            )
            key_rows.append(
                {
                    "object_id": oid,
                    "object_key": key,
                    "stratum": obj["stratum"],
                    "label": obj["label"],
                    "legacy_split": obj["legacy_split"],
                    "legacy_annotation_id": obj["legacy_annotation_id"],
                    "panel_A": panel_sources[0],
                    "panel_B": panel_sources[1],
                    "file_name": file_by_sha[str(obj["image_sha256"]).lower()],
                    "human_box_xyxy": obj["human_box_xyxy"],
                    "new_box_iou": obj["new_box_iou"],
                    "old_box_iou": obj["old_box_iou"],
                }
            )

    # Recount the strata over what actually rendered, not what was selected.
    rendered_by_stratum: dict[str, int] = {}
    for card in cards:
        rendered_by_stratum[card["stratum"]] = rendered_by_stratum.get(card["stratum"], 0) + 1
    for row in summary:
        row["drawn"] = rendered_by_stratum.get(row["stratum"], 0)

    weed = sum(1 for c in cards if c["label"] == "weed_target")
    small = sum(1 for c in cards if (c["box_w"] * c["box_h"]) < SMALL_OBJECT_AREA_PX)
    fragmented = sum(
        1
        for c in cards
        if c["panels"]["A"]["components"] >= 2 or c["panels"]["B"]["components"] >= 2
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pool_weed = sum(1 for o in objects if o["label"] == "weed_target")
    context = {
        "title": "SAM 2.1 re-seed: which polygon set is cheaper to correct?",
        "generated_at": generated_at,
        "key_path": str(out_key),
        "raw_shard": str(raw_shard),
        "sample_seed": sample_seed,
        "blind_seed": blind_seed,
        "pool_total": len(objects),
        "drawn_total": len(cards),
        "images_compared": len({o["image_sha256"] for o in objects}),
        "corpus_images": 2318,
        "pilot_fraction": "40 images",
        "drawn_weed": weed,
        "drawn_weed_pct": f"{100 * weed / len(cards):.1f}%" if cards else "--",
        "pool_weed_pct": f"{100 * pool_weed / len(objects):.1f}%",
        "drawn_small": small,
        "drawn_fragmented": fragmented,
        "dropped": len(dropped),
        "dropped_note": (
            "." if not dropped else " (listed in the blinding key file; none substituted)."
        ),
    }

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(build_html(cards, summary, context), encoding="utf-8")

    out_key.parent.mkdir(parents=True, exist_ok=True)
    out_key.write_text(
        json.dumps(
            {
                "WARNING": (
                    "BLINDING KEY. Do not open until the review is finished. Both "
                    "polygon sets are unreviewed proposals."
                ),
                "generated_at": generated_at,
                "sheet": str(out_html),
                "sample_seed": sample_seed,
                "blind_seed": blind_seed,
                "strata": summary,
                "dropped": dropped,
                "key": key_rows,
            },
            indent=2,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "html": str(out_html),
        "key": str(out_key),
        "png_dir": str(out_png_dir),
        "objects_rendered": len(cards),
        "objects_dropped": len(dropped),
        "html_bytes": out_html.stat().st_size,
        "strata": summary,
        "context": context,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Blinded side-by-side polygon review sheet")
    ap.add_argument("--objects-jsonl", required=True, type=Path)
    ap.add_argument("--raw-shard", required=True, type=Path)
    ap.add_argument("--coco-zip", required=True, type=Path)
    ap.add_argument("--proposals-json", required=True, type=Path)
    ap.add_argument("--legacy-annotations-dir", required=True, type=Path)
    ap.add_argument("--legacy-splits", default="train,valid,test")
    ap.add_argument("--out-html", required=True, type=Path)
    ap.add_argument("--out-key", required=True, type=Path)
    ap.add_argument("--out-png-dir", required=True, type=Path)
    ap.add_argument("--sample-seed", type=int, default=20260901)
    ap.add_argument("--blind-seed", type=int, default=20260901)
    args = ap.parse_args(argv)

    result = build(
        objects_jsonl=args.objects_jsonl,
        raw_shard=args.raw_shard,
        coco_zip=args.coco_zip,
        proposals_json=args.proposals_json,
        legacy_dir=args.legacy_annotations_dir,
        legacy_splits=[s.strip() for s in args.legacy_splits.split(",") if s.strip()],
        out_html=args.out_html,
        out_key=args.out_key,
        out_png_dir=args.out_png_dir,
        sample_seed=args.sample_seed,
        blind_seed=args.blind_seed,
    )
    print("Wrote blinded review sheet (UNREVIEWED PROPOSALS on both sides)")
    print(f"  html    : {result['html']} ({result['html_bytes'] / 1e6:.1f} MB)")
    print(f"  key     : {result['key']}")
    print(f"  panels  : {result['png_dir']}")
    print(f"  rendered: {result['objects_rendered']}  dropped: {result['objects_dropped']}")
    for row in result["strata"]:
        print(f"    {row['stratum']:<28} pool {row['pool']:>4}  drawn {row['drawn']:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
