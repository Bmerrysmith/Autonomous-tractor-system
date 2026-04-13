"""
pipeline_integration.py
========================
Top-level AgriNav pipeline:  camera frame → detection → discrimination → spray.

Connects DetectionPipeline and DiscriminationModule end-to-end.
Use this module as the single entry-point for the tractor control loop.

Architecture
────────────

  ┌─────────────┐      DetectionResult       ┌────────────────────┐
  │ Detection   │ ────────────────────────► │ Discrimination     │
  │ Pipeline    │  boxes · scores · labels  │ Module             │
  │ (WeedDet)   │                           │                    │
  └─────────────┘                           └────────────────────┘
        │                                           │
   image frame                               SprayDecision
  (camera / file)                        nozzle_commands[]
                                         spray_zones[]
                                         weed_coverage_pct

Detection-to-Discrimination handoff note
─────────────────────────────────────────
  Benny (Bmerrysmith) owns the detection side.
  The discrimination author consumes DetectionResult via:

      result   = pipeline.detect(image_path)
      decision = pipeline.discriminate(result)

  Or in one call:

      decision = pipeline.run_frame(image_path)

  The DetectionResult dataclass is the *only* coupling point.
  Both sides can evolve independently as long as that contract holds.

Quickstart
──────────
  python pipeline_integration.py \\
      --checkpoint checkpoints/weeddet_best.pth \\
      --image     field.jpg \\
      --nozzles   16 \\
      --threshold 0.50 \\
      --save      result_overlay.jpg
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# ── resolve local imports ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from detection_pipeline import DetectionPipeline, DetectionResult
from discrimination      import DiscriminationModule, SprayDecision


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATED PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class AgriNavPipeline:
    """
    Unified detection + discrimination pipeline for AgriNav.

    Parameters
    ──────────
    checkpoint_path     : str   Path to weeddet_best.pth
    device              : str   'auto' | 'cuda' | 'cpu'
    detection_threshold : float Confidence gate for WeedDet (default 0.50)
    num_nozzles         : int   Spray-boom sectors (default 16)
    rice_veto_threshold : float Rice confidence to block a nozzle (default 0.50)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        detection_threshold: float = 0.50,
        num_nozzles: int = 16,
        rice_veto_threshold: float = 0.50,
    ):
        self.detector = DetectionPipeline(
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=detection_threshold,
        )
        self.discriminator = DiscriminationModule(
            num_nozzles=num_nozzles,
            rice_veto_threshold=rice_veto_threshold,
        )

    # ── handoff methods (usable independently by Benny or discrimination side)

    def detect(self, image_path: str) -> DetectionResult:
        """
        Run detection only.  Returns DetectionResult.
        Benny's side calls this; discrimination side calls discriminate().
        """
        return self.detector.run(image_path)

    def discriminate(self, result: DetectionResult) -> SprayDecision:
        """
        Run discrimination only given an existing DetectionResult.
        Discrimination author calls this with Benny's output.
        """
        return self.discriminator.process(result)

    # ── full frame pipeline ───────────────────────────────────────────────

    def run_frame(
        self,
        image_path: str,
        verbose: bool = True,
        save_overlay: Optional[str] = None,
    ) -> SprayDecision:
        """
        Full pipeline: detect → discriminate → optionally visualize.

        Returns a SprayDecision with nozzle commands ready for the
        tractor's valve controller.
        """
        # ── 1. Detection ─────────────────────────────────────────────────
        det_result = self.detect(image_path)
        if verbose:
            print(f"\n[Detection]  {det_result.summary()}")

        # ── 2. Discrimination ────────────────────────────────────────────
        decision = self.discriminate(det_result)
        if verbose:
            DiscriminationModule.print_decision(decision)

        # ── 3. Optional overlay visualization ────────────────────────────
        if save_overlay:
            _render_overlay(det_result, decision, save_overlay)

        return decision

    def run_folder(
        self,
        folder: str,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> list:
        """Run the full pipeline on every image in a folder."""
        from pathlib import Path as _P
        exts   = {'.jpg', '.jpeg', '.png'}
        paths  = sorted(
            p for p in _P(folder).iterdir()
            if p.suffix.lower() in exts
        )
        if not paths:
            print(f"No images found in {folder}")
            return []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        decisions = []
        for img_path in paths:
            overlay = None
            if save_dir:
                overlay = os.path.join(save_dir, img_path.stem + '_pipeline.jpg')
            decision = self.run_frame(
                str(img_path),
                verbose=verbose,
                save_overlay=overlay,
            )
            decisions.append(decision)

        print(f"\nProcessed {len(decisions)} images.")
        return decisions


# ═══════════════════════════════════════════════════════════════════════════
# OVERLAY RENDERER
# ═══════════════════════════════════════════════════════════════════════════

def _render_overlay(
    det_result: DetectionResult,
    decision: SprayDecision,
    save_path: str,
) -> None:
    """
    Draw rice boxes (green), spray zones (red tint), and nozzle sector lines
    onto the image and save.
    """
    from PIL import Image as PILImage, ImageDraw

    img  = PILImage.open(det_result.image_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')

    # Red tint over spray zones
    for zone in decision.spray_zones:
        draw.rectangle(
            [zone.x_start, zone.y_start, zone.x_end, zone.y_end],
            fill=(255, 0, 0, 40),
        )

    # Nozzle sector dividers
    for cmd in decision.nozzle_commands[1:]:
        x = cmd.x_start
        draw.line([(x, 0), (x, det_result.orig_h)], fill=(200, 200, 0, 180), width=1)

    # Green boxes for rice (protected)
    for det in decision.protected_boxes:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        draw.rectangle([x1, y1, x2, y2], fill=(0, 200, 0, 60))
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0, 255), width=3)
        label = f"Rice {det.score:.2f}"
        lw    = len(label) * 7 + 8
        draw.rectangle([x1, y1 - 22, x1 + lw, y1], fill=(0, 160, 0, 230))
        draw.text((x1 + 4, y1 - 20), label, fill=(255, 255, 255))

    # Legend
    draw.rectangle([6, 6, 310, 100], fill=(0, 0, 0, 170))
    draw.text((12, 10), "GREEN = Rice  (protected — do not spray)", fill=(0, 255, 0))
    draw.text((12, 30), "RED   = Weed zone  (spray target)",        fill=(255, 80, 80))
    draw.text((12, 50), f"Rice detections  : {det_result.rice_count}",
              fill=(220, 220, 220))
    draw.text((12, 70), f"Spray / Hold     : {decision.spray_count} / {decision.hold_count}",
              fill=(220, 220, 220))

    img.save(save_path)
    print(f"  Overlay saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='AgriNav: WeedDet detection + discrimination pipeline'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to weeddet_best.pth')
    parser.add_argument('--image',   default=None,
                        help='Single image to process')
    parser.add_argument('--folder',  default=None,
                        help='Folder of images to process')
    parser.add_argument('--save',    default=None,
                        help='Save overlay for single image')
    parser.add_argument('--save-dir', default='pipeline_output',
                        help='Output folder for batch overlays')
    parser.add_argument('--threshold', type=float, default=0.50,
                        help='Detection confidence threshold (default 0.50)')
    parser.add_argument('--nozzles',   type=int,   default=16,
                        help='Number of spray-boom nozzle sectors (default 16)')
    parser.add_argument('--veto-threshold', type=float, default=0.50,
                        help='Rice veto confidence threshold (default 0.50)')
    parser.add_argument('--device', default='auto',
                        help="'auto' | 'cuda' | 'cpu'")
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-frame output')
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide --image or --folder")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    pipe = AgriNavPipeline(
        checkpoint_path=args.checkpoint,
        device=args.device,
        detection_threshold=args.threshold,
        num_nozzles=args.nozzles,
        rice_veto_threshold=args.veto_threshold,
    )

    if args.image:
        pipe.run_frame(args.image, verbose=not args.quiet, save_overlay=args.save)
    else:
        pipe.run_folder(
            args.folder,
            save_dir=args.save_dir,
            verbose=not args.quiet,
        )


if __name__ == '__main__':
    main()
