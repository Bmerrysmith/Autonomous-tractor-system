"""Synthetic correctness tests for training/riceseg_pretrain.py.

These cover the transfer-learning defects called out in the July 20 deep audit
(docs/audits/2026-07-20/AGRINAV_FULL_DEEP_AUDIT_2026-07-20.md), specifically the
Section 17.1 acceptance gates:

  * RiceSEG country/site parsing returns China, India, Japan, Philippines, and
    Tanzania for the supplied archive layout (audit Section 9.4);
  * unknown mask values fail rather than clip (audit Section 9.4);
  * the RiceSEG overfit gate cannot write the production backbone path and fails
    when its threshold is missed (audit Section 9.4 / 17.1);
  * backbone loading requires the full expected key set (audit Section 9.4).

The tests are deterministic and CPU-only.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.riceseg_pretrain import (
    CLASS_NAMES,
    ConfMat,
    NUM_CLASSES,
    RiceSegDataset,
    RiceSegModel,
    _enforce_overfit_gate,
    _overfit_output_path,
    export_backbone,
    load_riceseg_backbone,
    parse_country_site,
    scan_riceseg,
    sha256_file,
)

import training.riceseg_pretrain as rp


# The supplied RiceSEG archive extracts to a single wrapper directory that then
# contains one directory per country; some countries add a site level.
WRAPPER = "global rice segmentation"
LAYOUT = {
    ("China", "GD"): "IMG_5014_subset_overlap_0_4",
    ("China", "HN"): "DSCF0083_subset_overlap_3_0",
    ("Japan", "TKO_1"): "2014_0712_080213_subset_overlap_0_0",
    ("India", None): "IMG_0560_subset_overlap_1_1",
    ("Philippines", None): "00864_subset_overlap_0_1",
    ("Tanzania", None): "103_subset_overlap_1_0",
}


def _write_pair(root: Path, country: str, site, stem: str, mask_values=(0, 1, 4)):
    parent = root / WRAPPER / country
    if site is not None:
        parent = parent / site
    rgb_dir = parent / "rgb"
    lab_dir = parent / "label"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 120, 30)).save(rgb_dir / f"{stem}.jpg")
    mask = np.zeros((8, 8), dtype=np.uint8)
    for i, v in enumerate(mask_values):
        mask[i % 8, :] = v
    Image.fromarray(mask, mode="L").save(lab_dir / f"{stem}.png")


def _build_archive(root: Path):
    for (country, site), stem in LAYOUT.items():
        _write_pair(root, country, site, stem)


class CountrySiteParsingTests(unittest.TestCase):
    def test_parse_country_site_handles_both_layout_shapes(self):
        root = Path("/data/RiceSEG")
        with_site = root / WRAPPER / "China" / "GD" / "rgb" / "IMG_5014.jpg"
        no_site = root / WRAPPER / "India" / "rgb" / "IMG_0560.jpg"
        self.assertEqual(parse_country_site(with_site, root), ("China", "GD"))
        self.assertEqual(parse_country_site(no_site, root), ("India", None))

    def test_scan_riceseg_returns_all_five_countries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _build_archive(root)
            pairs = scan_riceseg(root)
            countries = {p["country"] for p in pairs}
            self.assertEqual(
                countries,
                {"China", "India", "Japan", "Philippines", "Tanzania"},
            )
            # The wrapper directory must never leak into the country field.
            self.assertNotIn(WRAPPER, countries)

    def test_scan_riceseg_parses_site_or_none(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _build_archive(root)
            by_country = {}
            for p in pairs_by_country(scan_riceseg(root)):
                by_country.setdefault(p["country"], set()).add(p.get("site"))
            self.assertIn("GD", by_country["China"])
            self.assertEqual(by_country["India"], {None})

    def test_country_holdout_split_is_now_reachable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _build_archive(root)
            pairs = scan_riceseg(root)
            train, val = rp.split_pairs(pairs, holdout_country="Tanzania")
            self.assertTrue(val, "holdout country must select at least one tile")
            self.assertTrue(all(p["country"] == "Tanzania" for p in val))
            self.assertTrue(all(p["country"] != "Tanzania" for p in train))


def pairs_by_country(pairs):
    return pairs


class MaskValidationTests(unittest.TestCase):
    def _dataset_with_mask(self, td: Path, values):
        rgb = td / "img.jpg"
        lab = td / "mask.png"
        Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
        arr = np.zeros((8, 8), dtype=np.uint8)
        arr[0, : len(values)] = values
        Image.fromarray(arr, mode="L").save(lab)
        pair = {"rgb": str(rgb), "label": str(lab), "country": "China", "site": "GD", "source": "China/img"}
        return RiceSegDataset([pair], img_size=8)

    def test_valid_mask_values_pass(self):
        with tempfile.TemporaryDirectory() as td:
            ds = self._dataset_with_mask(Path(td), list(range(NUM_CLASSES)))
            x, y = ds[0]
            self.assertEqual(tuple(y.shape), (8, 8))
            self.assertTrue(int(y.max()) < NUM_CLASSES)

    def test_unknown_mask_value_raises(self):
        with tempfile.TemporaryDirectory() as td:
            ds = self._dataset_with_mask(Path(td), [0, 1, 200])
            with self.assertRaises(ValueError):
                _ = ds[0]

    def test_ignore_index_is_allowed_when_configured(self):
        with tempfile.TemporaryDirectory() as td:
            rgb = Path(td) / "img.jpg"
            lab = Path(td) / "mask.png"
            Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
            arr = np.zeros((8, 8), dtype=np.uint8)
            arr[0, 0] = 255
            Image.fromarray(arr, mode="L").save(lab)
            pair = {"rgb": str(rgb), "label": str(lab), "country": "China", "site": "GD", "source": "China/img"}
            ds = RiceSegDataset([pair], img_size=8, ignore_index=255)
            _, y = ds[0]
            self.assertEqual(int((y == 255).sum()), 1)


class BackboneLoadingTests(unittest.TestCase):
    def test_complete_backbone_roundtrips(self):
        seg = RiceSegModel()
        det = rp.wd.WeedDet(num_classes=1, anchor_base_scale=3, lsc_k=7, use_atss=True)
        with tempfile.TemporaryDirectory() as td:
            pth = Path(td) / "bk.pth"
            export_backbone(seg, pth)
            n = load_riceseg_backbone(det, pth, verbose=False)
            self.assertGreater(n, 150)

    def test_incomplete_backbone_is_rejected(self):
        seg = RiceSegModel()
        det = rp.wd.WeedDet(num_classes=1, anchor_base_scale=3, lsc_k=7, use_atss=True)
        with tempfile.TemporaryDirectory() as td:
            pth = Path(td) / "bk.pth"
            sd = export_backbone(seg, pth)
            # Drop a handful of backbone tensors to simulate a partial export.
            dropped = dict(sd)
            for k in list(dropped)[:5]:
                del dropped[k]
            torch.save(dropped, pth)
            with self.assertRaises(ValueError):
                load_riceseg_backbone(det, pth, verbose=False)


class OverfitGateTests(unittest.TestCase):
    def test_overfit_output_is_isolated_from_production(self):
        production = Path("riceseg_backbone.pth")
        isolated = _overfit_output_path(production)
        self.assertNotEqual(isolated.resolve(), production.resolve())
        self.assertNotEqual(isolated.name, production.name)

    def test_overfit_gate_fails_below_threshold(self):
        with self.assertRaises(SystemExit):
            _enforce_overfit_gate(0.10, 0.80)

    def test_overfit_gate_passes_at_or_above_threshold(self):
        # Must not raise.
        _enforce_overfit_gate(0.85, 0.80)


class MetricReportingTests(unittest.TestCase):
    def test_absent_classes_are_reported_not_silently_dropped(self):
        # Only classes 0 and 1 ever appear; the other four must be reported
        # absent rather than silently excluded from the headline mIoU.
        cm = ConfMat()
        pred = np.array([[0, 1], [0, 1]])
        gt = np.array([[0, 1], [0, 1]])
        cm.update(pred, gt)
        iou, miou, absent = cm.iou()
        self.assertEqual(set(absent), set(CLASS_NAMES[2:]))
        self.assertAlmostEqual(miou, 1.0, places=6)  # over present classes only
        self.assertEqual(len(iou), NUM_CLASSES)

    def test_all_classes_present_reports_no_absent(self):
        cm = ConfMat()
        diag = np.arange(NUM_CLASSES)
        cm.update(diag, diag)
        _, _, absent = cm.iou()
        self.assertEqual(absent, [])


class HashHelperTests(unittest.TestCase):
    def test_sha256_file_is_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a.bin"
            b = Path(td) / "b.bin"
            a.write_bytes(b"agrinav")
            b.write_bytes(b"agrinav")
            self.assertEqual(sha256_file(a), sha256_file(b))
            b.write_bytes(b"agrinav2")
            self.assertNotEqual(sha256_file(a), sha256_file(b))


if __name__ == "__main__":
    unittest.main()
