"""
download_sample_images.py
=========================
Downloads 5 free rice-field images, saves them to sample_images/, then
runs the AgriNav detection-discrimination pipeline on each one and writes
overlay images to sample_images/output/.

Image source
────────────
Unsplash's legacy source URL (source.unsplash.com) is used first — it is
free, requires no API key, and redirects directly to a JPEG.  The official
Unsplash API (api.unsplash.com) DOES require a free Access Key; this script
intentionally avoids it so that no credentials are needed.

If Unsplash redirects fail (HTTP error or non-image response), the script
falls back to Wikimedia Commons, also entirely key-free.

Run:
    python download_sample_images.py
    python download_sample_images.py --count 5 --out sample_images
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# ── path setup ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
for p in (str(REPO_ROOT / 'inference'), str(REPO_ROOT / 'models')):
    if p not in sys.path:
        sys.path.insert(0, p)

import requests
import torch
import torchvision.transforms as T
from PIL import Image as PILImage

from detection_pipeline   import DetectionPipeline, DetectionResult, Detection
from discrimination       import DiscriminationModule, SprayDecision, SprayZone, NozzleCommand
from pipeline_integration import AgriNavPipeline, _render_overlay

# ── ANSI colours ─────────────────────────────────────────────────────────
GRN  = '\033[92m'; RED = '\033[91m'; YEL = '\033[93m'
CYN  = '\033[96m'; BOLD = '\033[1m'; RST = '\033[0m'

def _banner(msg: str) -> None:
    print(f"\n{BOLD}{CYN}{'═'*64}{RST}")
    print(f"{BOLD}{CYN}  {msg}{RST}")
    print(f"{BOLD}{CYN}{'═'*64}{RST}")

def _info(msg: str) -> None:
    print(f"  {YEL}{msg}{RST}")


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

# Unsplash legacy source URLs — no API key required.
# ?sig=N ensures each request returns a different image.
_UNSPLASH_QUERIES = [
    'rice+field+paddy',
    'rice+crop+rows',
    'paddy+field+green',
    'rice+agriculture+asia',
    'rice+farming+rows',
]

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (compatible; AgriNav-demo/1.0; '
        '+https://github.com/Bmerrysmith/Autonomous-tractor-system)'
    )
}

_TIMEOUT = 20  # seconds


def _try_unsplash(query: str, idx: int) -> bytes | None:
    """
    Download one image via Unsplash's legacy source URL.
    Returns raw bytes on success, None on any failure.
    """
    url = f"https://source.unsplash.com/800x600/?{query}&sig={idx}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT,
                            allow_redirects=True)
        ct = resp.headers.get('Content-Type', '')
        if resp.status_code == 200 and ct.startswith('image/'):
            return resp.content
        print(f"    Unsplash returned {resp.status_code} / {ct} — trying fallback")
    except requests.RequestException as exc:
        print(f"    Unsplash request failed ({exc}) — trying fallback")
    return None


def _wikimedia_fallback(query: str, idx: int) -> bytes | None:
    """
    Download one image from Wikimedia Commons.
    Uses the public search + imageinfo API — no key required.
    Returns raw bytes on success, None on any failure.
    """
    search_url = 'https://commons.wikimedia.org/w/api.php'
    search_params = {
        'action':    'query',
        'list':      'search',
        'srsearch':  query.replace('+', ' '),
        'srnamespace': '6',   # File: namespace
        'srlimit':   '10',
        'format':    'json',
    }
    try:
        sr = requests.get(search_url, params=search_params,
                          headers=_HEADERS, timeout=_TIMEOUT)
        sr.raise_for_status()
        hits = sr.json().get('query', {}).get('search', [])
        if not hits:
            print(f"    Wikimedia: no results for '{query}'")
            return None

        # Pick a different result for each idx
        title = hits[idx % len(hits)]['title']

        info_params = {
            'action':  'query',
            'titles':  title,
            'prop':    'imageinfo',
            'iiprop':  'url',
            'format':  'json',
        }
        ir = requests.get(search_url, params=info_params,
                          headers=_HEADERS, timeout=_TIMEOUT)
        ir.raise_for_status()
        pages = ir.json()['query']['pages']
        img_url = next(iter(pages.values()))['imageinfo'][0]['url']

        img_resp = requests.get(img_url, headers=_HEADERS, timeout=_TIMEOUT)
        img_resp.raise_for_status()
        ct = img_resp.headers.get('Content-Type', '')
        if ct.startswith('image/'):
            return img_resp.content
        print(f"    Wikimedia: unexpected content-type {ct}")
    except Exception as exc:
        print(f"    Wikimedia fallback failed: {exc}")
    return None


def download_images(
    out_dir: Path,
    count:   int = 5,
    pause_s: float = 1.0,
) -> list[Path]:
    """
    Download *count* rice-field images into *out_dir*.
    Returns list of saved file paths (may be shorter than count on failure).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for i in range(count):
        query = _UNSPLASH_QUERIES[i % len(_UNSPLASH_QUERIES)]
        dest  = out_dir / f'rice_field_{i+1:02d}.jpg'

        if dest.exists():
            print(f"  [{i+1}/{count}] Already exists, skipping: {dest.name}")
            saved.append(dest)
            continue

        print(f"  [{i+1}/{count}] Downloading '{query.replace('+', ' ')}' …", end='', flush=True)

        data = _try_unsplash(query, i)
        if data is None:
            data = _wikimedia_fallback(query.replace('+field+', ' ').replace('+', ' '), i)

        if data is None:
            print(f"  {RED}FAILED{RST} — skipping image {i+1}")
            continue

        dest.write_bytes(data)
        kb = len(data) / 1024
        print(f"  {GRN}OK{RST}  ({kb:.0f} KB) → {dest.name}")
        saved.append(dest)

        if i < count - 1:
            time.sleep(pause_s)   # be polite to the server

    return saved


# ═══════════════════════════════════════════════════════════════════════════
# STUB PIPELINE  (mirrors demo_run.py — no checkpoint required)
# ═══════════════════════════════════════════════════════════════════════════

def _build_stub_pipeline(
    boxes_xyxy: list,
    scores:     list,
    nozzles:    int = 16,
    threshold:  float = 0.50,
) -> AgriNavPipeline:
    """Build AgriNavPipeline with a deterministic fake model."""
    def _fwd(images, targets=None):
        device = images.device
        b = torch.tensor(boxes_xyxy, dtype=torch.float32, device=device) \
            if boxes_xyxy else torch.zeros((0, 4), device=device)
        s = torch.tensor(scores,     dtype=torch.float32, device=device) \
            if scores    else torch.zeros((0,),    device=device)
        l = torch.zeros(len(scores), dtype=torch.int64,   device=device)
        return [{'boxes': b, 'scores': s, 'labels': l}]

    pipe = object.__new__(AgriNavPipeline)

    det            = object.__new__(DetectionPipeline)
    det.threshold  = threshold
    det.device     = 'cpu'
    det._transform = T.Compose([
        T.Resize((600, 1000)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    stub            = MagicMock()
    stub.eval.return_value = stub
    stub.side_effect = _fwd
    det.model       = stub

    pipe.detector      = det
    pipe.discriminator = DiscriminationModule(
        num_nozzles=nozzles,
        rice_veto_threshold=threshold,
    )
    pipe.log_path = REPO_ROOT / 'pipeline_log.jsonl'
    return pipe


# Per-image stub scenario: cycle through representative detection patterns
# so each downloaded image gets a meaningfully different overlay.
_IMAGE_SCENARIOS = [
    {'boxes': [],                                          'scores': []},
    {'boxes': [[350,100,550,450], [470,200,650,500]],     'scores': [0.92,0.87]},
    {'boxes': [[50,50,300,550],[280,80,520,540],
               [500,60,730,520],[710,100,950,530]],       'scores': [0.95,0.91,0.88,0.84]},
    {'boxes': [[300,100,700,500]],                        'scores': [0.35]},
    {'boxes': [[30,200,180,400],[420,150,580,420],
               [820,220,960,450]],                        'scores': [0.94,0.89,0.76]},
]


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_on_image(img_path: Path, out_dir: Path, scenario_idx: int) -> None:
    """Run the stub detection → discrimination pipeline on one real image."""
    sc    = _IMAGE_SCENARIOS[scenario_idx % len(_IMAGE_SCENARIOS)]
    pipe  = _build_stub_pipeline(sc['boxes'], sc['scores'])

    result   = pipe.detect(str(img_path))
    decision = pipe.discriminate(result)

    print(f"    Rice detections : {result.rice_count}")
    print(f"    Spray / Hold    : {decision.spray_count} / {decision.hold_count}")
    print(f"    Weed est.       : ~{result.estimated_weed_pct:.0f}%")
    print(f"    Spray zones     : {len(decision.spray_zones)}")

    out_path = out_dir / f"{img_path.stem}_overlay.jpg"
    _render_overlay(result, decision, str(out_path))
    print(f"    {GRN}Overlay saved{RST} → {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Download rice field images and run the AgriNav pipeline on each.'
    )
    parser.add_argument('--count', type=int, default=5,
                        help='Number of images to download (default 5)')
    parser.add_argument('--out', default='sample_images',
                        help='Folder to save downloaded images (default sample_images/)')
    parser.add_argument('--pause', type=float, default=1.0,
                        help='Seconds to wait between downloads (default 1.0)')
    args = parser.parse_args()

    img_dir = REPO_ROOT / args.out
    out_dir = img_dir / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ───────────────────────────────────────────────────────
    _banner('Step 1 — Downloading rice field images')
    print(f"  Source  : Unsplash source URL (no API key needed)")
    print(f"  Fallback: Wikimedia Commons")
    print(f"  Saving to: {img_dir}\n")

    images = download_images(img_dir, count=args.count, pause_s=args.pause)

    if not images:
        print(f"\n{RED}No images downloaded — check your internet connection.{RST}")
        sys.exit(1)

    print(f"\n  {GRN}Downloaded {len(images)}/{args.count} image(s).{RST}")

    # ── 2. Run pipeline on each image ─────────────────────────────────────
    _banner('Step 2 — Running AgriNav pipeline on each image')
    print(f"  Overlay output: {out_dir}\n")

    for i, img_path in enumerate(images):
        print(f"\n  {BOLD}[{i+1}/{len(images)}] {img_path.name}{RST}")
        try:
            run_on_image(img_path, out_dir, scenario_idx=i)
        except Exception as exc:
            print(f"  {RED}Pipeline failed on {img_path.name}: {exc}{RST}")

    # ── 3. Summary ────────────────────────────────────────────────────────
    overlays = sorted(out_dir.glob('*_overlay.jpg'))
    _banner('Done')
    print(f"  Downloaded images : {img_dir}")
    print(f"  Overlay images    : {out_dir}")
    print(f"  Files generated   :")
    for p in overlays:
        print(f"    {GRN}{p.name}{RST}")
    print()


if __name__ == '__main__':
    main()
