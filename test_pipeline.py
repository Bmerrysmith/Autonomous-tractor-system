"""
test_pipeline.py
================
Unit + integration tests for the AgriNav detection → discrimination pipeline.
No GPU or real checkpoint required — the model is replaced with a deterministic stub.

Run:
    python test_pipeline.py
    python test_pipeline.py -v        # verbose output
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── resolve inference/ package on path ───────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
INFERENCE  = REPO_ROOT / 'inference'
MODELS     = REPO_ROOT / 'models'
for p in (str(INFERENCE), str(MODELS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from PIL import Image as PILImage

# Modules under test
from detection_pipeline  import DetectionPipeline, DetectionResult, Detection, CLASS_NAMES, RICE_CLASS
from discrimination      import DiscriminationModule, SprayDecision, NozzleCommand, SprayZone
from pipeline_integration import AgriNavPipeline


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_test_image(w=800, h=600, path=None):
    """Create a solid-green JPEG and return its path."""
    img = PILImage.new('RGB', (w, h), color=(34, 139, 34))
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        path = tmp.name
        tmp.close()
    img.save(path)
    return path


def _model_output_factory(boxes_xyxy, scores):
    """
    Return a callable that mimics WeedDet forward pass (eval mode).
    boxes_xyxy : list of [x1,y1,x2,y2] in model space (1000×600)
    scores     : list of floats
    """
    def _forward(images, targets=None):
        device = images.device
        b = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device)
        s = torch.tensor(scores,     dtype=torch.float32, device=device)
        l = torch.zeros(len(scores), dtype=torch.int64,   device=device)
        return [{'boxes': b, 'scores': s, 'labels': l}]
    return _forward


def _make_stub_pipeline(boxes_xyxy, scores, threshold=0.50):
    """
    Build a DetectionPipeline with a stubbed model — no checkpoint needed.
    """
    pipe = object.__new__(DetectionPipeline)
    pipe.threshold = threshold
    pipe.device    = 'cpu'

    import torchvision.transforms as T
    pipe._transform = T.Compose([
        T.Resize((600, 1000)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    stub = MagicMock()
    stub.eval.return_value = stub
    stub.side_effect = _model_output_factory(boxes_xyxy, scores)
    pipe.model = stub
    return pipe


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Detection dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectionDataclass(unittest.TestCase):

    def test_is_rice(self):
        d = Detection(box=[0,0,100,100], score=0.9, label=RICE_CLASS, label_name='Rice')
        self.assertTrue(d.is_rice())

    def test_area(self):
        d = Detection(box=[0, 0, 200, 100], score=0.8, label=0, label_name='Rice')
        self.assertAlmostEqual(d.area, 20000.0)

    def test_center(self):
        d = Detection(box=[0, 0, 200, 100], score=0.8, label=0, label_name='Rice')
        cx, cy = d.center
        self.assertAlmostEqual(cx, 100.0)
        self.assertAlmostEqual(cy, 50.0)

    def test_zero_area_clamps(self):
        d = Detection(box=[50, 50, 30, 30], score=0.5, label=0, label_name='Rice')
        self.assertEqual(d.area, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: DetectionResult helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectionResult(unittest.TestCase):

    def _make_result(self, boxes_scores):
        dets = [
            Detection(box=b, score=s, label=0, label_name='Rice')
            for b, s in boxes_scores
        ]
        return DetectionResult(
            image_path='fake.jpg',
            orig_w=800, orig_h=600,
            detections=dets,
            threshold=0.5,
        )

    def test_rice_count(self):
        r = self._make_result([([0,0,100,100], 0.9), ([200,200,300,300], 0.7)])
        self.assertEqual(r.rice_count, 2)

    def test_image_area(self):
        r = self._make_result([])
        self.assertEqual(r.image_area, 800 * 600)

    def test_estimated_weed_pct_full_coverage(self):
        # Rice box covers entire image → weed ≈ 0 %
        r = self._make_result([([0, 0, 800, 600], 0.95)])
        self.assertAlmostEqual(r.estimated_weed_pct, 0.0, places=0)

    def test_estimated_weed_pct_no_rice(self):
        r = self._make_result([])
        self.assertAlmostEqual(r.estimated_weed_pct, 100.0)

    def test_summary_string(self):
        r = self._make_result([([0, 0, 100, 100], 0.8)])
        s = r.summary()
        self.assertIn('rice=1', s)
        self.assertIn('fake.jpg', s)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: DetectionPipeline (stubbed model)
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectionPipeline(unittest.TestCase):

    def setUp(self):
        self.img_path = _make_test_image(w=800, h=600)

    def tearDown(self):
        os.unlink(self.img_path)

    def test_run_returns_detection_result(self):
        # Two rice boxes in model space (1000×600)
        pipe = _make_stub_pipeline(
            boxes_xyxy=[[100, 50, 300, 200], [600, 100, 800, 400]],
            scores=[0.92, 0.75],
        )
        result = pipe.run(self.img_path)
        self.assertIsInstance(result, DetectionResult)

    def test_detections_sorted_by_score_desc(self):
        pipe = _make_stub_pipeline(
            boxes_xyxy=[[0,0,100,100], [200,200,300,300]],
            scores=[0.60, 0.90],
        )
        result = pipe.run(self.img_path)
        scores = [d.score for d in result.detections]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_threshold_filters_low_confidence(self):
        pipe = _make_stub_pipeline(
            boxes_xyxy=[[0,0,100,100], [200,200,300,300]],
            scores=[0.30, 0.90],  # first below 0.5 threshold
            threshold=0.50,
        )
        result = pipe.run(self.img_path)
        self.assertEqual(len(result.detections), 1)
        self.assertAlmostEqual(result.detections[0].score, 0.90, places=2)

    def test_box_rescaled_to_original_coords(self):
        # Model space 1000×600, original image 800×600
        # box x1=500 in model space → 500 * 800/1000 = 400 in original
        pipe = _make_stub_pipeline(
            boxes_xyxy=[[500, 300, 800, 500]],
            scores=[0.85],
        )
        result = pipe.run(self.img_path)
        x1 = result.detections[0].box[0]
        self.assertAlmostEqual(x1, 500 * 800 / 1000, places=1)

    def test_no_detections_above_threshold(self):
        pipe = _make_stub_pipeline(boxes_xyxy=[], scores=[])
        result = pipe.run(self.img_path)
        self.assertEqual(result.rice_count, 0)
        self.assertAlmostEqual(result.estimated_weed_pct, 100.0)

    def test_orig_dimensions_captured(self):
        pipe = _make_stub_pipeline(boxes_xyxy=[], scores=[])
        result = pipe.run(self.img_path)
        self.assertEqual(result.orig_w, 800)
        self.assertEqual(result.orig_h, 600)

    def test_inference_ms_positive(self):
        pipe = _make_stub_pipeline(boxes_xyxy=[], scores=[])
        result = pipe.run(self.img_path)
        self.assertGreaterEqual(result.inference_ms, 0.0)

    def test_label_name_mapped(self):
        pipe = _make_stub_pipeline(
            boxes_xyxy=[[0, 0, 100, 100]],
            scores=[0.80],
        )
        result = pipe.run(self.img_path)
        self.assertEqual(result.detections[0].label_name, 'Rice')


# ═══════════════════════════════════════════════════════════════════════════
# TEST: DiscriminationModule
# ═══════════════════════════════════════════════════════════════════════════

class TestDiscriminationModule(unittest.TestCase):

    def _result_with_rice(self, boxes_scores, orig_w=800, orig_h=600):
        dets = [
            Detection(box=b, score=s, label=0, label_name='Rice')
            for b, s in boxes_scores
        ]
        return DetectionResult(
            image_path='fake.jpg',
            orig_w=orig_w, orig_h=orig_h,
            detections=dets,
            threshold=0.5,
        )

    def test_no_rice_all_nozzles_spray(self):
        disc   = DiscriminationModule(num_nozzles=8)
        result = self._result_with_rice([])
        dec    = disc.process(result)
        self.assertTrue(all(c.spray for c in dec.nozzle_commands))

    def test_no_rice_reason_is_no_detection(self):
        disc   = DiscriminationModule(num_nozzles=4)
        result = self._result_with_rice([])
        dec    = disc.process(result)
        self.assertTrue(all(c.reason == 'no_detection' for c in dec.nozzle_commands))

    def test_rice_in_center_blocks_center_nozzles(self):
        # Image 800 wide, 4 nozzles → sectors 0–200, 200–400, 400–600, 600–800
        # Rice box 200–600 (covers sectors 1 and 2)
        disc   = DiscriminationModule(num_nozzles=4, rice_veto_threshold=0.5)
        result = self._result_with_rice([([200, 0, 600, 600], 0.90)])
        dec    = disc.process(result)

        spray_states = [c.spray for c in dec.nozzle_commands]
        # Sectors 0 and 3 → spray; sectors 1 and 2 → hold
        self.assertTrue(spray_states[0],  "Sector 0 should spray (no rice)")
        self.assertFalse(spray_states[1], "Sector 1 should hold (rice veto)")
        self.assertFalse(spray_states[2], "Sector 2 should hold (rice veto)")
        self.assertTrue(spray_states[3],  "Sector 3 should spray (no rice)")

    def test_rice_veto_threshold_respected(self):
        # Low-confidence rice (0.40) should NOT veto when threshold=0.50
        disc   = DiscriminationModule(num_nozzles=4, rice_veto_threshold=0.50)
        result = self._result_with_rice([([200, 0, 600, 600], 0.40)])
        dec    = disc.process(result)
        # No veto → all spray (no confident rice found)
        self.assertTrue(all(c.spray for c in dec.nozzle_commands))

    def test_high_confidence_rice_sets_reason(self):
        disc   = DiscriminationModule(num_nozzles=4, rice_veto_threshold=0.50)
        result = self._result_with_rice([([200, 0, 600, 600], 0.90)])
        dec    = disc.process(result)
        reasons = {c.nozzle_id: c.reason for c in dec.nozzle_commands}
        self.assertEqual(reasons[1], 'rice_protected')
        self.assertEqual(reasons[2], 'rice_protected')

    def test_nozzle_count_matches_config(self):
        for n in [4, 8, 16, 32]:
            disc   = DiscriminationModule(num_nozzles=n)
            result = self._result_with_rice([])
            dec    = disc.process(result)
            self.assertEqual(len(dec.nozzle_commands), n)

    def test_spray_zones_merge_contiguous(self):
        # All 4 nozzles spray → should produce 1 merged zone
        disc   = DiscriminationModule(num_nozzles=4)
        result = self._result_with_rice([])
        dec    = disc.process(result)
        self.assertEqual(len(dec.spray_zones), 1)
        self.assertEqual(dec.spray_zones[0].x_start, 0)
        self.assertEqual(dec.spray_zones[0].x_end,   800)

    def test_spray_zones_fragmented_by_rice(self):
        # 4 nozzles, rice in sector 1 only → 2 separate zones: [0] and [2,3]
        disc   = DiscriminationModule(num_nozzles=4, rice_veto_threshold=0.5)
        result = self._result_with_rice([([200, 0, 400, 600], 0.90)])
        dec    = disc.process(result)
        self.assertEqual(len(dec.spray_zones), 2)

    def test_weed_coverage_propagated(self):
        disc   = DiscriminationModule(num_nozzles=8)
        result = self._result_with_rice([])   # 100 % weed
        dec    = disc.process(result)
        self.assertAlmostEqual(dec.weed_coverage_pct, 100.0)

    def test_spray_decision_counts(self):
        disc   = DiscriminationModule(num_nozzles=4, rice_veto_threshold=0.5)
        result = self._result_with_rice([([200, 0, 600, 600], 0.90)])
        dec    = disc.process(result)
        self.assertEqual(dec.spray_count, 2)
        self.assertEqual(dec.hold_count,  2)

    def test_invalid_nozzles_raises(self):
        with self.assertRaises(ValueError):
            DiscriminationModule(num_nozzles=0)

    def test_invalid_veto_threshold_raises(self):
        with self.assertRaises(ValueError):
            DiscriminationModule(rice_veto_threshold=0.0)

    def test_nozzle_x_ranges_cover_full_width(self):
        disc   = DiscriminationModule(num_nozzles=8)
        result = self._result_with_rice([], orig_w=800)
        dec    = disc.process(result)
        self.assertEqual(dec.nozzle_commands[0].x_start, 0)
        self.assertEqual(dec.nozzle_commands[-1].x_end,  800)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Full AgriNavPipeline integration (stubbed model, real handoff)
# ═══════════════════════════════════════════════════════════════════════════

class TestAgriNavPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.img_path = _make_test_image(w=800, h=600)

    def tearDown(self):
        os.unlink(self.img_path)

    def _make_pipeline(self, boxes_xyxy, scores, nozzles=8):
        pipe = object.__new__(AgriNavPipeline)
        pipe.detector      = _make_stub_pipeline(boxes_xyxy, scores)
        pipe.discriminator = DiscriminationModule(
            num_nozzles=nozzles,
            rice_veto_threshold=0.50,
        )
        return pipe

    def test_run_frame_returns_spray_decision(self):
        pipe = self._make_pipeline([[100,50,400,300]], [0.88])
        dec  = pipe.run_frame(self.img_path, verbose=False)
        self.assertIsInstance(dec, SprayDecision)

    def test_detect_and_discriminate_split_calls_match_run_frame(self):
        boxes  = [[100, 50, 400, 300]]
        scores = [0.88]
        # run_frame
        pipe1 = self._make_pipeline(boxes, scores, nozzles=8)
        dec1  = pipe1.run_frame(self.img_path, verbose=False)

        # split calls
        pipe2 = self._make_pipeline(boxes, scores, nozzles=8)
        result = pipe2.detect(self.img_path)
        dec2   = pipe2.discriminate(result)

        self.assertEqual(dec1.spray_count, dec2.spray_count)
        self.assertEqual(dec1.hold_count,  dec2.hold_count)

    def test_zero_rice_all_nozzles_spray(self):
        pipe = self._make_pipeline([], [], nozzles=8)
        dec  = pipe.run_frame(self.img_path, verbose=False)
        self.assertEqual(dec.spray_count, 8)
        self.assertEqual(dec.hold_count,  0)

    def test_full_coverage_rice_all_nozzles_hold(self):
        # Rice box covers entire image width (0–1000 in model space)
        pipe = self._make_pipeline([[0, 0, 1000, 600]], [0.95], nozzles=8)
        dec  = pipe.run_frame(self.img_path, verbose=False)
        self.assertEqual(dec.hold_count,  8)
        self.assertEqual(dec.spray_count, 0)

    def test_overlay_saved_to_disk(self):
        overlay_dir  = tempfile.mkdtemp()
        overlay_path = os.path.join(overlay_dir, 'test_overlay.jpg')
        try:
            pipe = self._make_pipeline([[100, 50, 400, 300]], [0.85], nozzles=8)
            pipe.run_frame(self.img_path, verbose=False, save_overlay=overlay_path)
            self.assertTrue(os.path.exists(overlay_path))
            self.assertGreater(os.path.getsize(overlay_path), 0)
        finally:
            if os.path.exists(overlay_path):
                os.unlink(overlay_path)
            os.rmdir(overlay_dir)

    def test_protected_boxes_in_decision(self):
        pipe   = self._make_pipeline([[100, 50, 400, 300]], [0.88], nozzles=8)
        result = pipe.detect(self.img_path)
        dec    = pipe.discriminate(result)
        self.assertEqual(len(dec.protected_boxes), 1)
        self.assertAlmostEqual(dec.protected_boxes[0].score, 0.88, places=1)

    def test_multiple_rice_boxes_all_protected(self):
        pipe   = self._make_pipeline(
            [[0,0,200,200], [400,0,600,200], [800,0,1000,200]],
            [0.90, 0.85, 0.80],
            nozzles=8,
        )
        dec = pipe.run_frame(self.img_path, verbose=False)
        self.assertEqual(len(dec.protected_boxes), 3)


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [
        TestDetectionDataclass,
        TestDetectionResult,
        TestDetectionPipeline,
        TestDiscriminationModule,
        TestAgriNavPipelineIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2 if '-v' in sys.argv else 1)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
