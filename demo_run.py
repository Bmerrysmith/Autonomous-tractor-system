"""
demo_run.py
===========
Live demo of the AgriNav detection → discrimination pipeline.
No GPU or real checkpoint required — uses a stubbed model with
hand-crafted test scenarios so you can see real pipeline output.

Run:
    python demo_run.py
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# ── path setup ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
for p in (str(REPO_ROOT / 'inference'), str(REPO_ROOT / 'models')):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torchvision.transforms as T
from PIL import Image as PILImage

from detection_pipeline   import DetectionPipeline, CLASS_NAMES
from discrimination       import DiscriminationModule
from pipeline_integration import AgriNavPipeline, _render_overlay

# ── ANSI colours ─────────────────────────────────────────────────────────
GRN  = '\033[92m'
RED  = '\033[91m'
YEL  = '\033[93m'
CYN  = '\033[96m'
BOLD = '\033[1m'
RST  = '\033[0m'

def _banner(title):
    w = 64
    print(f"\n{BOLD}{CYN}{'═'*w}{RST}")
    print(f"{BOLD}{CYN}  {title}{RST}")
    print(f"{BOLD}{CYN}{'═'*w}{RST}")

def _section(title):
    print(f"\n{BOLD}{YEL}── {title} {'─'*(56-len(title))}{RST}")


# ═══════════════════════════════════════════════════════════════════════════
# STUB HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_test_image(w=800, h=600):
    """Solid green field image, saved as a temp JPEG."""
    img  = PILImage.new('RGB', (w, h), color=(34, 139, 34))
    tmp  = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name


def _stub_pipeline(boxes_xyxy, scores, nozzles=16, threshold=0.50):
    """Build AgriNavPipeline with a fake model — no checkpoint needed."""
    def _fwd(images, targets=None):
        device = images.device
        b = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device) \
            if boxes_xyxy else torch.zeros((0, 4), device=device)
        s = torch.tensor(scores,     dtype=torch.float32, device=device) \
            if scores    else torch.zeros((0,),    device=device)
        l = torch.zeros(len(scores), dtype=torch.int64,   device=device)
        return [{'boxes': b, 'scores': s, 'labels': l}]

    pipe = object.__new__(AgriNavPipeline)

    det = object.__new__(DetectionPipeline)
    det.threshold  = threshold
    det.device     = 'cpu'
    det._transform = T.Compose([
        T.Resize((600, 1000)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    stub = MagicMock()
    stub.eval.return_value = stub
    stub.side_effect = _fwd
    det.model = stub

    pipe.detector      = det
    pipe.discriminator = DiscriminationModule(
        num_nozzles=nozzles,
        rice_veto_threshold=threshold,
    )
    return pipe


def _print_nozzle_bar(commands, width=64):
    """Print a visual nozzle bar: G=protected, R=spray."""
    n = len(commands)
    bar = ''
    for cmd in commands:
        bar += (f"{GRN}█{RST}" if not cmd.spray else f"{RED}█{RST}")
    # Scale to fixed width
    ratio = width / n
    scaled = ''
    for cmd in commands:
        ch = int(ratio)
        char = (f"{GRN}{'█'*ch}{RST}") if not cmd.spray else (f"{RED}{'█'*ch}{RST}")
        scaled += char
    print(f"  Boom: {scaled}")
    print(f"        {GRN}█ = rice-protected (HOLD){RST}   {RED}█ = weed zone (SPRAY){RST}")


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        'name'   : 'Scenario 1 — No rice detected (full weed field)',
        'desc'   : 'Model finds nothing above threshold → entire boom sprays.',
        'boxes'  : [],
        'scores' : [],
        'nozzles': 16,
    },
    {
        'name'   : 'Scenario 2 — Rice in centre of frame',
        'desc'   : 'Two rice plants in the middle → centre nozzles hold, edges spray.',
        'boxes'  : [[350, 100, 550, 450], [470, 200, 650, 500]],
        'scores' : [0.92, 0.87],
        'nozzles': 16,
    },
    {
        'name'   : 'Scenario 3 — Dense rice coverage (rice row)',
        'desc'   : 'Rice plants spanning most of the image width.',
        'boxes'  : [
            [50,  50,  300, 550],
            [280, 80,  520, 540],
            [500, 60,  730, 520],
            [710, 100, 950, 530],
        ],
        'scores' : [0.95, 0.91, 0.88, 0.84],
        'nozzles': 16,
    },
    {
        'name'   : 'Scenario 4 — Single low-confidence detection (below veto)',
        'desc'   : 'Rice at score 0.35 — below 0.50 veto threshold → treated as weed zone.',
        'boxes'  : [[300, 100, 700, 500]],
        'scores' : [0.35],
        'nozzles': 16,
    },
    {
        'name'   : 'Scenario 5 — Sparse, scattered rice (real-field mix)',
        'desc'   : 'Three isolated rice plants → three protected gaps in spray pattern.',
        'boxes'  : [
            [30,  200, 180, 400],
            [420, 150, 580, 420],
            [820, 220, 960, 450],
        ],
        'scores' : [0.94, 0.89, 0.76],
        'nozzles': 16,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(scenario, img_path, save_dir):
    _banner(scenario['name'])
    print(f"  {scenario['desc']}")

    pipe = _stub_pipeline(
        scenario['boxes'],
        scenario['scores'],
        nozzles=scenario['nozzles'],
    )

    # ── Detection ────────────────────────────────────────────────────────
    _section('Detection')
    result = pipe.detect(img_path)
    print(f"  Image        : {result.orig_w} × {result.orig_h} px")
    print(f"  Inference    : {result.inference_ms:.1f} ms")
    print(f"  Rice found   : {result.rice_count}")
    if result.detections:
        for i, d in enumerate(result.detections):
            b = d.box
            print(f"    [{i+1}] score={d.score:.2f}  "
                  f"box=[{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}]  "
                  f"label={d.label_name}")
    else:
        print(f"    {YEL}(none above threshold){RST}")
    print(f"  Weed est.    : ~{result.estimated_weed_pct:.0f}%")

    # ── Discrimination ───────────────────────────────────────────────────
    _section('Discrimination')
    decision = pipe.discriminate(result)
    print(f"  Nozzles      : {len(decision.nozzle_commands)} sectors")
    print(f"  SPRAY        : {decision.spray_count} nozzles")
    print(f"  HOLD         : {decision.hold_count}  nozzles  (rice protection)")
    print(f"  Spray zones  : {len(decision.spray_zones)} contiguous region(s)")

    _section('Boom visualisation')
    _print_nozzle_bar(decision.nozzle_commands)

    _section('Nozzle commands')
    for cmd in decision.nozzle_commands:
        state = f"{RED}SPRAY{RST}" if cmd.spray else f"{GRN}HOLD {RST}"
        print(f"  Nozzle {cmd.nozzle_id:>2}  "
              f"[{cmd.x_start:>4}–{cmd.x_end:>4}px]  {state}  ({cmd.reason})")

    # ── Overlay image ────────────────────────────────────────────────────
    _section('Overlay')
    slug = scenario['name'].split('—')[0].strip().replace(' ', '_').lower()
    out  = os.path.join(save_dir, f"{slug}.jpg")
    _render_overlay(result, decision, out)
    print(f"  Saved → {out}")


def main():
    img_path = _make_test_image(w=800, h=600)
    save_dir = os.path.join(REPO_ROOT, 'demo_output')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{BOLD}AgriNav Pipeline Demo{RST}")
    print(f"Using synthetic field image: {img_path}")
    print(f"Overlay output folder      : {save_dir}")

    try:
        for s in SCENARIOS:
            run_scenario(s, img_path, save_dir)
    finally:
        os.unlink(img_path)

    _banner('All scenarios complete')
    print(f"  Overlay images saved to: {BOLD}{save_dir}{RST}\n")


if __name__ == '__main__':
    main()
