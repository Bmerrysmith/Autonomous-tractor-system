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
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── resolve local imports ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from detection_pipeline import DetectionPipeline, DetectionResult, SafetyFlag
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

    # Default log location: project root (one level above inference/)
    _DEFAULT_LOG = Path(__file__).resolve().parent.parent / 'pipeline_log.jsonl'

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        detection_threshold: float = 0.50,
        num_nozzles: int = 16,
        rice_veto_threshold: float = 0.50,
        log_path: Optional[str] = None,
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
        self.log_path = Path(log_path) if log_path else self._DEFAULT_LOG

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

    # ── logging ───────────────────────────────────────────────────────────

    def _log_frame(self, result: DetectionResult) -> None:
        """
        Append one JSON line to pipeline_log.jsonl for this detection result.

        Fields
        ──────
        timestamp       ISO-8601 UTC timestamp
        image_filename  basename of the source image
        num_detections  total detections passing the confidence threshold
        avg_confidence  mean score across all detections (null if none)
        min_box_area    smallest detection area in pixels  (null if none)
        max_box_area    largest  detection area in pixels  (null if none)
        inference_ms    model forward-pass wall-clock time (ms)
        """
        scores = [d.score for d in result.detections]
        areas  = [d.area  for d in result.detections]

        record = {
            'timestamp':      datetime.now(timezone.utc).isoformat(),
            'image_filename': Path(result.image_path).name,
            'num_detections': len(result.detections),
            'avg_confidence': round(sum(scores) / len(scores), 4) if scores else None,
            'min_box_area':   round(min(areas), 2) if areas else None,
            'max_box_area':   round(max(areas), 2) if areas else None,
            'inference_ms':   round(result.inference_ms, 2),
        }

        with self.log_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')

    # ── full frame pipeline ───────────────────────────────────────────────

    def run_frame(
        self,
        image_path: str,
        verbose: bool = True,
        save_overlay: Optional[str] = None,
    ) -> SprayDecision:
        """
        Full pipeline: detect → log → discriminate → optionally visualize.

        Returns a SprayDecision with nozzle commands ready for the
        tractor's valve controller.
        """
        # ── 1. Detection ─────────────────────────────────────────────────
        det_result = self.detect(image_path)
        if verbose:
            print(f"\n[Detection]  {det_result.summary()}")

        # ── 2. Structured audit log (detection → discrimination handoff) ──
        self._log_frame(det_result)

        # ── 3. Discrimination ────────────────────────────────────────────
        decision = self.discriminate(det_result)
        if verbose:
            DiscriminationModule.print_decision(decision)

        # ── 4. Optional overlay visualization ────────────────────────────
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

    def run_video_stream(
        self,
        image_paths: list,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> list:
        """
        Process a sequence of image paths as a simulated video stream.

        Unlike run_folder(), this method maintains state across frames:
          - consecutive_empty_frames increments each frame where rice_count == 0.
          - It resets to 0 the moment rice is detected again.
          - Each frame's SafetyFlag is evaluated with that running counter.
          - A ROW_LOST flag (>= 3 consecutive empty frames) triggers a
            WARNING log and a simulated pause — the tractor control loop
            should halt forward motion at this point.

        Parameters
        ──────────
        image_paths : list[str]
            Ordered sequence of image file paths (e.g. camera frames 0..N).
        save_dir : str | None
            Directory to write overlay images, or None to skip.
        verbose : bool
            Print per-frame summaries and safety events.

        Returns
        ───────
        List of (SprayDecision, SafetyFlag) tuples, one per frame.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        consecutive_empty_frames = 0
        results = []

        for frame_idx, image_path in enumerate(image_paths):
            # ── Detection + log ──────────────────────────────────────────
            det_result = self.detect(image_path)
            self._log_frame(det_result)

            # ── Update consecutive empty frame counter ───────────────────
            if det_result.rice_count == 0:
                consecutive_empty_frames += 1
            else:
                consecutive_empty_frames = 0

            # ── Safety evaluation ────────────────────────────────────────
            flag = det_result.evaluate_safety(consecutive_empty_frames)

            if verbose:
                print(
                    f"\n[Stream frame {frame_idx:>4}]  {det_result.summary()}  "
                    f"| empty_streak={consecutive_empty_frames}  "
                    f"| safety={flag.value.upper()}"
                )

            if flag is SafetyFlag.ROW_LOST:
                print(
                    f"[WARNING] ROW_LOST — {consecutive_empty_frames} consecutive "
                    f"empty frames at '{Path(image_path).name}'.  "
                    f"Simulating tractor pause: halting spray output."
                )
                # In a live system this would signal the motion controller.
                # Discrimination still runs so the log is complete, but the
                # decision is flagged so callers can suppress valve commands.

            elif flag is SafetyFlag.OBSTACLE_WARNING and verbose:
                print(
                    f"[WARNING] OBSTACLE_WARNING detected at frame {frame_idx} "
                    f"('{Path(image_path).name}')."
                )

            # ── Discrimination ───────────────────────────────────────────
            decision = self.discriminate(det_result)
            if verbose:
                DiscriminationModule.print_decision(decision)

            # ── Optional overlay ─────────────────────────────────────────
            if save_dir:
                overlay = os.path.join(
                    save_dir,
                    f"{Path(image_path).stem}_frame{frame_idx:04d}.jpg"
                )
                _render_overlay(det_result, decision, overlay)

            results.append((decision, flag))

        # ── Stream summary ───────────────────────────────────────────────
        total   = len(results)
        lost    = sum(1 for _, f in results if f is SafetyFlag.ROW_LOST)
        obs     = sum(1 for _, f in results if f is SafetyFlag.OBSTACLE_WARNING)
        clear   = sum(1 for _, f in results if f is SafetyFlag.CLEAR)
        print(
            f"\n[Stream complete]  {total} frames  |  "
            f"CLEAR={clear}  OBSTACLE_WARNING={obs}  ROW_LOST={lost}"
        )
        return results


# ═══════════════════════════════════════════════════════════════════════════
# OVERLAY RENDERER
# ═══════════════════════════════════════════════════════════════════════════

def _load_font(size: int = 16):
    """
    Try common system TrueType fonts at *size* pt; fall back to PIL's
    built-in bitmap font if none are found.
    """
    from PIL import ImageFont

    candidates = [
        # Linux / Raspberry Pi
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        # macOS
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/Arial.ttf',
        # Windows
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _text_size(draw, text: str, font) -> tuple:
    """Return (width, height) of *text* rendered with *font*."""
    bbox = draw.textbbox((0, 0), text, font=font)   # Pillow ≥ 9.2
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _render_overlay(
    det_result: DetectionResult,
    decision: SprayDecision,
    save_path: str,
) -> None:
    """
    Draw rice boxes (green), spray zones (red tint), nozzle sector lines,
    and per-zone weed coverage labels onto the image and save.
    """
    from PIL import Image as PILImage, ImageDraw

    img  = PILImage.open(det_result.image_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')

    font       = _load_font(16)
    font_small = _load_font(13)

    # ── Red tint + weed-coverage label over spray zones ──────────────────
    image_area = det_result.image_area or 1.0
    for zone in decision.spray_zones:
        draw.rectangle(
            [zone.x_start, zone.y_start, zone.x_end, zone.y_end],
            fill=(255, 0, 0, 40),
        )
        # Weed coverage for this zone only (zone pixel area / image area)
        zone_area  = max(1, zone.width) * max(1, zone.height)
        zone_pct   = min(100.0, zone_area / image_area * 100.0)
        label      = f"Weed ~{zone_pct:.0f}%"
        lw, lh     = _text_size(draw, label, font_small)
        lx = zone.x_start + max(0, (zone.width - lw) // 2)
        ly = zone.y_start + 6
        # semi-transparent pill behind the text
        draw.rectangle([lx - 3, ly - 2, lx + lw + 3, ly + lh + 2],
                       fill=(180, 0, 0, 160))
        draw.text((lx, ly), label, fill=(255, 255, 255), font=font_small)

    # ── Nozzle sector dividers ────────────────────────────────────────────
    for cmd in decision.nozzle_commands[1:]:
        x = cmd.x_start
        draw.line([(x, 0), (x, det_result.orig_h)], fill=(200, 200, 0, 180), width=1)

    # ── Green boxes for rice (protected) ─────────────────────────────────
    for det in decision.protected_boxes:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        draw.rectangle([x1, y1, x2, y2], fill=(0, 200, 0, 60))
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0, 255), width=3)
        label      = f"Rice {det.score:.0%}"
        lw, lh     = _text_size(draw, label, font)
        draw.rectangle([x1, y1 - lh - 6, x1 + lw + 8, y1],
                       fill=(0, 160, 0, 230))
        draw.text((x1 + 4, y1 - lh - 3), label, fill=(255, 255, 255), font=font)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_lines = [
        ("GREEN = Rice  (protected — do not spray)", (0, 255, 0)),
        ("RED   = Weed zone  (spray target)",        (255, 80, 80)),
        (f"Rice detections  : {det_result.rice_count}", (220, 220, 220)),
        (f"Spray / Hold     : {decision.spray_count} / {decision.hold_count}",
         (220, 220, 220)),
    ]
    line_h   = _text_size(draw, "Ag", font)[1] + 6
    box_h    = line_h * len(legend_lines) + 10
    draw.rectangle([6, 6, 330, 6 + box_h], fill=(0, 0, 0, 170))
    for i, (text, colour) in enumerate(legend_lines):
        draw.text((12, 10 + i * line_h), text, fill=colour, font=font)

    img.save(save_path)
    print(f"  Overlay saved: {save_path}")


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
