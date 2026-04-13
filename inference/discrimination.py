"""
discrimination.py
=================
Discrimination module for AgriNav — consumes a DetectionResult from the
detection pipeline and decides where to spray.

Detection-to-Discrimination Handoff (for Benny / Bmerrysmith)
──────────────────────────────────────────────────────────────
  INPUT  : DetectionResult   (from detection_pipeline.DetectionPipeline)
  OUTPUT : SprayDecision     (spray commands + spray-zone map)

Inverted-detection logic recap
───────────────────────────────
  The model detects RICE (protected class).
  Anything NOT inside a high-confidence rice bounding box = weed → SPRAY.

  Rice veto rule (hard-coded safety):
    If any rice box overlaps a nozzle sector with score ≥ RICE_VETO_THRESHOLD,
    that sector is BLOCKED (do not spray) regardless of anything else.

Nozzle model
────────────
  The spray boom is divided into N equal sectors across the image width.
  For each sector we emit one NozzleCommand: spray=True or spray=False.
  Sector boundaries are expressed as pixel columns in original image space.

  A sector sprays if and only if:
    • NO rice detection with score ≥ rice_veto_threshold overlaps the sector.

SprayDecision fields
────────────────────
  protected_boxes    List[Detection]      rice detections (do not spray)
  spray_zones        List[SprayZone]      contiguous column ranges to spray
  nozzle_commands    List[NozzleCommand]  one per boom sector
  weed_coverage_pct  float                estimated % of image that is weed
  frame_summary      str                  one-line human-readable summary

NozzleCommand fields
────────────────────
  nozzle_id          int     0-indexed sector number
  x_start            int     left pixel column (original coords)
  x_end              int     right pixel column (original coords)
  spray              bool    True = open valve, False = close valve
  reason             str     'weed_zone' | 'weed_confirmed' | 'rice_protected'
                             | 'obstacle_blocked' | 'no_detection'

SprayZone fields
────────────────
  x_start, x_end    int     pixel column range (original coords)
  y_start, y_end    int     pixel row range (full image height by default)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from detection_pipeline import Detection, DetectionResult

# Default confidence threshold above which a rice detection vetoes spraying
RICE_VETO_THRESHOLD = 0.50


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NozzleCommand:
    """Spray command for one boom sector."""
    nozzle_id: int
    x_start:   int        # left  edge pixel (original image coords)
    x_end:     int        # right edge pixel (original image coords)
    spray:     bool
    reason:    str        # 'weed_zone' | 'rice_protected' | 'no_detection'

    def __str__(self):
        state = 'SPRAY' if self.spray else 'HOLD '
        return (
            f"Nozzle {self.nozzle_id:>3}  [{self.x_start:>5}–{self.x_end:>5}px]  "
            f"{state}  ({self.reason})"
        )


@dataclass
class SprayZone:
    """A contiguous horizontal region that should be sprayed."""
    x_start: int
    x_end:   int
    y_start: int
    y_end:   int

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        return self.y_end - self.y_start


@dataclass
class SprayDecision:
    """
    Complete spray decision for one camera frame.

    Feed nozzle_commands directly to the tractor's valve controller.
    """
    protected_boxes:   List['Detection']        # rice detections (HOLD)
    weed_boxes:        List['Detection']        # explicit weed detections
    obstacle_boxes:    List['Detection']        # obstacle detections (BLOCK)
    spray_zones:       List[SprayZone]          # regions to spray
    nozzle_commands:   List[NozzleCommand]      # per-sector valve states
    weed_coverage_pct: float                    # estimated % weed coverage
    frame_summary:     str = ''

    @property
    def spray_count(self) -> int:
        return sum(1 for c in self.nozzle_commands if c.spray)

    @property
    def hold_count(self) -> int:
        return sum(1 for c in self.nozzle_commands if not c.spray)


# ═══════════════════════════════════════════════════════════════════════════
# DISCRIMINATION MODULE
# ═══════════════════════════════════════════════════════════════════════════

class DiscriminationModule:
    """
    Translates a DetectionResult into a SprayDecision.

    Parameters
    ──────────
    num_nozzles : int
        Number of spray-boom sectors (nozzles).  Default 16.
    rice_veto_threshold : float
        Minimum rice confidence to block a nozzle sector.  Default 0.50.
        Set lower (e.g. 0.30) for a more conservative (rice-safe) policy.
    """

    def __init__(
        self,
        num_nozzles: int = 16,
        rice_veto_threshold: float = RICE_VETO_THRESHOLD,
    ):
        if num_nozzles < 1:
            raise ValueError("num_nozzles must be >= 1")
        if not (0.0 < rice_veto_threshold <= 1.0):
            raise ValueError("rice_veto_threshold must be in (0, 1]")

        self.num_nozzles          = num_nozzles
        self.rice_veto_threshold  = rice_veto_threshold

    # ── main entry point ──────────────────────────────────────────────────

    def process(self, result: 'DetectionResult') -> SprayDecision:
        """
        Convert a DetectionResult into a SprayDecision.

        Class priority (highest → lowest):
          1. Obstacle  — block the sector entirely (no spray, halt signal).
          2. Rice      — protect the sector (no spray).
          3. Weed      — spray confirmed.
          4. No detection — spray (inverted-detection default).
        """
        protected  = [d for d in result.detections
                      if d.is_rice() and d.score >= self.rice_veto_threshold]
        weeds      = result.weed_detections
        obstacles  = result.obstacle_detections

        nozzle_commands = self._build_nozzle_commands(
            protected, weeds, obstacles, result.orig_w, result.orig_h
        )
        spray_zones = self._build_spray_zones(
            nozzle_commands, result.orig_h
        )

        summary = (
            f"SprayDecision | "
            f"rice_protected={len(protected)} | "
            f"weeds={len(weeds)} | "
            f"obstacles={len(obstacles)} | "
            f"nozzles={self.num_nozzles} | "
            f"spray={sum(1 for c in nozzle_commands if c.spray)} | "
            f"hold={sum(1 for c in nozzle_commands if not c.spray)} | "
            f"weed≈{result.estimated_weed_pct:.0f}%"
        )

        return SprayDecision(
            protected_boxes=protected,
            weed_boxes=weeds,
            obstacle_boxes=obstacles,
            spray_zones=spray_zones,
            nozzle_commands=nozzle_commands,
            weed_coverage_pct=result.estimated_weed_pct,
            frame_summary=summary,
        )

    # ── nozzle command builder ────────────────────────────────────────────

    def _build_nozzle_commands(
        self,
        protected:  List['Detection'],
        weeds:      List['Detection'],
        obstacles:  List['Detection'],
        orig_w: int,
        orig_h: int,
    ) -> List[NozzleCommand]:
        """
        Divide the image width into num_nozzles equal sectors and decide
        each sector's spray state using the three-class priority:

          obstacle overlap  → HOLD  ('obstacle_blocked')
          rice overlap      → HOLD  ('rice_protected')
          weed confirmed    → SPRAY ('weed_confirmed')  [explicit weed box]
          no detection      → SPRAY ('weed_zone' / 'no_detection')
        """
        sector_w = orig_w / self.num_nozzles
        commands: List[NozzleCommand] = []

        for n in range(self.num_nozzles):
            x_start = int(round(n * sector_w))
            x_end   = int(round((n + 1) * sector_w))

            # Priority 1: obstacle blocks the sector
            if self._has_overlap(obstacles, x_start, x_end):
                commands.append(NozzleCommand(
                    nozzle_id=n, x_start=x_start, x_end=x_end,
                    spray=False, reason='obstacle_blocked'
                ))
                continue

            # Priority 2: rice protects the sector
            if self._overlapping_rice(protected, x_start, x_end):
                commands.append(NozzleCommand(
                    nozzle_id=n, x_start=x_start, x_end=x_end,
                    spray=False, reason='rice_protected'
                ))
                continue

            # Priority 3: explicit weed detection in sector
            if self._has_overlap(weeds, x_start, x_end):
                commands.append(NozzleCommand(
                    nozzle_id=n, x_start=x_start, x_end=x_end,
                    spray=True, reason='weed_confirmed'
                ))
                continue

            # Default: inverted-detection — no rice here → spray
            reason = 'weed_zone' if protected else 'no_detection'
            commands.append(NozzleCommand(
                nozzle_id=n, x_start=x_start, x_end=x_end,
                spray=True, reason=reason
            ))

        return commands

    def _overlapping_rice(
        self,
        protected: List['Detection'],
        x_start: int,
        x_end: int,
    ) -> bool:
        """
        Return True if any rice box has horizontal overlap with [x_start, x_end].
        Vertical overlap is not checked — the boom operates per column strip.
        """
        return self._has_overlap(protected, x_start, x_end)

    def _has_overlap(
        self,
        detections: List['Detection'],
        x_start: int,
        x_end: int,
    ) -> bool:
        """Return True if any detection box overlaps the column range [x_start, x_end]."""
        for det in detections:
            det_x1, _, det_x2, _ = det.box
            if not (det_x2 <= x_start or det_x1 >= x_end):
                return True
        return False

    # ── spray zone builder ────────────────────────────────────────────────

    def _build_spray_zones(
        self,
        commands: List[NozzleCommand],
        orig_h: int,
    ) -> List[SprayZone]:
        """
        Merge consecutive spray-ON nozzle sectors into contiguous SprayZones.
        """
        zones: List[SprayZone] = []
        zone_start: int | None = None
        zone_x_start: int | None = None

        for cmd in commands:
            if cmd.spray:
                if zone_start is None:
                    zone_start   = cmd.nozzle_id
                    zone_x_start = cmd.x_start
                zone_x_end = cmd.x_end
            else:
                if zone_start is not None:
                    zones.append(SprayZone(
                        x_start=zone_x_start,  # type: ignore[arg-type]
                        x_end=zone_x_end,      # type: ignore[name-defined]
                        y_start=0,
                        y_end=orig_h,
                    ))
                    zone_start   = None
                    zone_x_start = None

        # Flush trailing open zone
        if zone_start is not None:
            zones.append(SprayZone(
                x_start=zone_x_start,  # type: ignore[arg-type]
                x_end=commands[-1].x_end,
                y_start=0,
                y_end=orig_h,
            ))

        return zones

    # ── reporting ─────────────────────────────────────────────────────────

    @staticmethod
    def print_decision(decision: SprayDecision) -> None:
        """Pretty-print a SprayDecision to stdout."""
        print(f"\n{'═' * 60}")
        print(decision.frame_summary)
        print(f"{'─' * 60}")
        print(f"  Protected rice boxes : {len(decision.protected_boxes)}")
        for i, d in enumerate(decision.protected_boxes):
            b = d.box
            print(f"    [{i+1}] score={d.score:.3f}  "
                  f"box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
        print(f"  Weed detections      : {len(decision.weed_boxes)}")
        for i, d in enumerate(decision.weed_boxes):
            b = d.box
            print(f"    [{i+1}] score={d.score:.3f}  "
                  f"box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
        print(f"  Obstacle detections  : {len(decision.obstacle_boxes)}")
        for i, d in enumerate(decision.obstacle_boxes):
            b = d.box
            print(f"    [{i+1}] score={d.score:.3f}  "
                  f"box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
        print(f"  Spray zones          : {len(decision.spray_zones)}")
        for z in decision.spray_zones:
            print(f"    cols {z.x_start}–{z.x_end}  rows {z.y_start}–{z.y_end}")
        print(f"  Nozzle commands ({len(decision.nozzle_commands)}):")
        for cmd in decision.nozzle_commands:
            print(f"    {cmd}")
        print(f"{'═' * 60}\n")
