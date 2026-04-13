"""
detection_pipeline.py
=====================
Object detection pipeline for AgriNav — wraps WeedDet with a clean, typed
interface that feeds the downstream DiscriminationModule.

Handoff contract (for Benny / discrimination author):
  DetectionPipeline.run(image_path) → DetectionResult

DetectionResult fields
──────────────────────
  image_path    str               source file path
  orig_w        int               original image width  (pixels)
  orig_h        int               original image height (pixels)
  detections    List[Detection]   boxes in ORIGINAL pixel space
  threshold     float             confidence gate used
  inference_ms  float             wall-clock time for model forward pass

Detection fields (one per detected object)
──────────────────────────────────────────
  box           [x1, y1, x2, y2]  xyxy, original pixel coords
  score         float             model confidence 0-1
  label         int               class index (0 = Rice)
  label_name    str               human-readable class name

Usage
─────
  from detection_pipeline import DetectionPipeline, DetectionResult

  pipe = DetectionPipeline(
      checkpoint_path='checkpoints/weeddet_best.pth',
      device='cuda',
      threshold=0.50,
  )
  result = pipe.run('field.jpg')
  # pass result directly to DiscriminationModule.process(result)
"""

import sys
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms as T

# ── Constants matching WeedDet training config ────────────────────────────
IMG_SIZE     = (600, 1000)   # (H, W) — must match training / inference_rice.py
CLASS_NAMES  = ['Rice']
RICE_CLASS   = 0


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES  —  the detection-to-discrimination handoff contract
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Single object detection in original-image pixel coordinates."""
    box:        List[float]   # [x1, y1, x2, y2]
    score:      float
    label:      int
    label_name: str

    @property
    def area(self) -> float:
        w = max(0.0, self.box[2] - self.box[0])
        h = max(0.0, self.box[3] - self.box[1])
        return w * h

    @property
    def center(self):
        return (
            (self.box[0] + self.box[2]) / 2.0,
            (self.box[1] + self.box[3]) / 2.0,
        )

    def is_rice(self) -> bool:
        return self.label == RICE_CLASS


@dataclass
class DetectionResult:
    """
    Complete output of one forward pass.  Pass the whole object to
    DiscriminationModule.process() — it carries everything needed for
    the spray-decision logic.
    """
    image_path:   str
    orig_w:       int
    orig_h:       int
    detections:   List[Detection]
    threshold:    float
    inference_ms: float = 0.0

    # ── convenience accessors ────────────────────────────────────────────
    @property
    def rice_detections(self) -> List[Detection]:
        return [d for d in self.detections if d.is_rice()]

    @property
    def rice_count(self) -> int:
        return len(self.rice_detections)

    @property
    def image_area(self) -> float:
        return float(self.orig_w * self.orig_h)

    @property
    def rice_area(self) -> float:
        return sum(d.area for d in self.rice_detections)

    @property
    def estimated_weed_pct(self) -> float:
        """Rough weed coverage (%) based on non-rice pixel area."""
        return max(0.0, (self.image_area - self.rice_area) / self.image_area * 100.0)

    def summary(self) -> str:
        return (
            f"DetectionResult | {Path(self.image_path).name} "
            f"({self.orig_w}×{self.orig_h}) | "
            f"rice={self.rice_count} | "
            f"weed≈{self.estimated_weed_pct:.0f}% | "
            f"{self.inference_ms:.1f} ms"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class DetectionPipeline:
    """
    Loads WeedDet and exposes a single run() method.

    Parameters
    ──────────
    checkpoint_path : str
        Path to weeddet_best.pth (or any WeedDet checkpoint).
    device : str
        'cuda' | 'cpu' — auto-selects CUDA if available when 'auto'.
    threshold : float
        Minimum confidence to keep a detection (default 0.50).
    model_module_path : str
        Folder containing weeddet_for_VSCode.py.  Defaults to
        ../models relative to this file.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        threshold: float = 0.50,
        model_module_path: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold

        # ── resolve device ──────────────────────────────────────────────
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # ── add models/ to sys.path so weeddet_for_VSCode is importable ─
        if model_module_path is None:
            model_module_path = str(
                Path(__file__).resolve().parent.parent / 'models'
            )
        if model_module_path not in sys.path:
            sys.path.insert(0, model_module_path)

        self.model = self._load_model()
        self._transform = T.Compose([
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ── model loading ─────────────────────────────────────────────────────

    def _load_model(self):
        import weeddet_for_VSCode as wd  # type: ignore

        ckpt  = torch.load(self.checkpoint_path, map_location=self.device)
        cfg   = ckpt.get('config', {})
        model = wd.WeedDet(num_classes=cfg.get('num_classes', len(CLASS_NAMES)))

        state_key = 'state_dict' if 'state_dict' in ckpt else 'model_state_dict'
        model.load_state_dict(ckpt[state_key])
        model.to(self.device).eval()

        epoch = ckpt.get('epoch', '?')
        loss  = ckpt.get('loss', float('nan'))
        print(
            f"[DetectionPipeline] Loaded checkpoint  "
            f"epoch={epoch}  loss={loss:.4f}  device={self.device}"
        )
        return model

    # ── preprocessing ────────────────────────────────────────────────────

    def _preprocess(self, image_path: str):
        from PIL import Image as PILImage
        img          = PILImage.open(image_path).convert('RGB')
        orig_w, orig_h = img.size
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        return tensor, orig_w, orig_h

    # ── postprocessing ───────────────────────────────────────────────────

    def _postprocess(
        self,
        raw_preds: dict,
        orig_w: int,
        orig_h: int,
        image_path: str,
        inference_ms: float,
    ) -> DetectionResult:
        tH, tW = IMG_SIZE
        detections: List[Detection] = []

        for i in range(len(raw_preds['boxes'])):
            sc  = raw_preds['scores'][i].item()
            if sc < self.threshold:
                continue
            lbl = int(raw_preds['labels'][i].item())
            box = raw_preds['boxes'][i].cpu().tolist()

            # Scale from model space (tW×tH) back to original pixel space
            scaled_box = [
                box[0] * orig_w / tW,
                box[1] * orig_h / tH,
                box[2] * orig_w / tW,
                box[3] * orig_h / tH,
            ]
            name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f'class_{lbl}'
            detections.append(Detection(
                box=scaled_box,
                score=sc,
                label=lbl,
                label_name=name,
            ))

        # Sort highest confidence first
        detections.sort(key=lambda d: d.score, reverse=True)

        return DetectionResult(
            image_path=image_path,
            orig_w=orig_w,
            orig_h=orig_h,
            detections=detections,
            threshold=self.threshold,
            inference_ms=inference_ms,
        )

    # ── public API ────────────────────────────────────────────────────────

    def run(self, image_path: str) -> DetectionResult:
        """
        Run detection on a single image.

        Returns a DetectionResult ready for DiscriminationModule.process().
        """
        tensor, orig_w, orig_h = self._preprocess(image_path)

        t0 = time.perf_counter()
        with torch.no_grad():
            raw = self.model(tensor)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return self._postprocess(raw[0], orig_w, orig_h, image_path, inference_ms)

    def run_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """Run detection on a list of images (sequentially)."""
        return [self.run(p) for p in image_paths]
