"""
inference_rice.py
=================
Run WeedDet inference on a single image or a folder.

Visualization logic (inverted detection):
  GREEN box  = Rice detected  → PROTECTED zone, do not spray
  RED overlay = everything else → WEED target zone, spray here

Usage:
    # Single image, display result
    python inference_rice.py --model checkpoints/weeddet_best.pth \
                             --image field.jpg

    # Single image, save result
    python inference_rice.py --model checkpoints/weeddet_best.pth \
                             --image field.jpg --save result.jpg

    # Batch folder
    python inference_rice.py --model checkpoints/weeddet_best.pth \
                             --folder ./field_images/ --save-dir ./results/

    # Lower threshold for more detections
    python inference_rice.py --model checkpoints/weeddet_best.pth \
                             --image field.jpg --threshold 0.3
"""

import os
import sys
import argparse
import torch
import torchvision.transforms as T
from pathlib import Path

IMG_SIZE     = (600, 1000)   # (H, W) — must match training config
RICE_THRESH  = 0.5


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device='cuda'):
    if '/content' not in sys.path:
        sys.path.insert(0, '/content')
    import weeddet_for_VSCode as wd

    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt.get('config', {})
    model = wd.WeedDet(num_classes=cfg.get('num_classes', 1))

    key = 'state_dict' if 'state_dict' in ckpt else 'model_state_dict'
    model.load_state_dict(ckpt[key])
    model.to(device).eval()
    print(f"Model loaded: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(model, image_path, device='cuda', threshold=RICE_THRESH):
    """
    Returns list of rice detection dicts: {box, score}
    Boxes are in ORIGINAL image pixel coordinates.
    """
    from PIL import Image as PILImage

    tf = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img          = PILImage.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    x            = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        results = model(x)

    preds     = results[0]
    tH, tW   = IMG_SIZE
    detections = []

    for i in range(len(preds['boxes'])):
        sc = preds['scores'][i].item()
        if sc < threshold:
            continue
        box = preds['boxes'][i].cpu().tolist()
        detections.append({
            'box'  : [
                box[0] * orig_w / tW,
                box[1] * orig_h / tH,
                box[2] * orig_w / tW,
                box[3] * orig_h / tH,
            ],
            'score': sc,
        })

    return detections, orig_w, orig_h


# ── Visualization ────────────────────────────────────────────────────────────

def visualize(image_path, detections, orig_w, orig_h, save_path=None):
    """
    Draw rice detections on image.
      Green box + fill = rice (protected)
      Red overlay      = weed zone (spray target)
    """
    from PIL import Image as PILImage, ImageDraw

    img  = PILImage.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')

    # Full red overlay — weed zone default
    draw.rectangle([0, 0, orig_w, orig_h], fill=(255, 0, 0, 45))

    # Green boxes for each rice detection
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        sc = det['score']

        # Green fill inside box (neutralize red overlay)
        draw.rectangle([x1, y1, x2, y2], fill=(0, 200, 0, 60))
        # Green outline
        draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0, 255), width=3)
        # Label background + text
        label = f"Rice {sc:.2f}"
        lw    = len(label) * 7 + 8
        draw.rectangle([x1, y1 - 22, x1 + lw, y1], fill=(0, 160, 0, 230))
        draw.text((x1 + 4, y1 - 20), label, fill=(255, 255, 255))

    # Legend
    draw.rectangle([6, 6, 255, 82], fill=(0, 0, 0, 170))
    draw.text((12, 12), "GREEN = Rice  (protected — do not spray)",
              fill=(0, 255, 0))
    draw.text((12, 34), "RED   = Weed zone  (spray target)",
              fill=(255, 80, 80))
    draw.text((12, 56), f"Rice detections: {len(detections)}",
              fill=(220, 220, 220))

    if save_path:
        img.save(save_path)
        print(f"  Saved → {save_path}")
    else:
        img.show()

    return img


# ── Per-image report ─────────────────────────────────────────────────────────

def report(image_path, detections, orig_w, orig_h):
    total_area = orig_w * orig_h
    rice_area  = sum(
        max(0, d['box'][2] - d['box'][0]) *
        max(0, d['box'][3] - d['box'][1])
        for d in detections
    )
    weed_pct = max(0, (total_area - rice_area) / total_area * 100)

    print(f"\n{'─'*50}")
    print(f"Image : {image_path}")
    print(f"Size  : {orig_w} × {orig_h}")
    print(f"Rice  : {len(detections)} detection(s)")
    for i, d in enumerate(detections):
        b = d['box']
        print(f"  [{i+1}] score={d['score']:.3f}  "
              f"box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")
    if not detections:
        print("  ⚠ No rice detected — entire image is weed zone")
    else:
        print(f"  Estimated weed zone: ~{weed_pct:.0f}% of image")
    print('─'*50)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WeedDet rice inference')
    parser.add_argument('--model',     required=True,
                        help='Path to weeddet_best.pth checkpoint')
    parser.add_argument('--image',     default=None,
                        help='Single image file to process')
    parser.add_argument('--folder',    default=None,
                        help='Folder of images to process')
    parser.add_argument('--save',      default=None,
                        help='Save path for single image output')
    parser.add_argument('--save-dir',  default='inference_output',
                        help='Output folder for batch results')
    parser.add_argument('--threshold', type=float, default=RICE_THRESH,
                        help=f'Detection threshold (default {RICE_THRESH})')
    parser.add_argument('--cpu',       action='store_true',
                        help='Force CPU inference')
    args = parser.parse_args()

    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    print(f"Device    : {device}")
    print(f"Threshold : {args.threshold}")

    if not os.path.exists(args.model):
        print(f"ERROR: checkpoint not found: {args.model}")
        sys.exit(1)

    model = load_model(args.model, device)

    if args.image:
        # Single image
        dets, w, h = run_inference(model, args.image, device, args.threshold)
        report(args.image, dets, w, h)
        visualize(args.image, dets, w, h, args.save)

    elif args.folder:
        # Batch folder
        os.makedirs(args.save_dir, exist_ok=True)
        paths = sorted([
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"Processing {len(paths)} images ...")
        for img_path in paths:
            try:
                dets, w, h = run_inference(model, img_path, device, args.threshold)
                out_name   = Path(img_path).stem + '_detected.jpg'
                out_path   = os.path.join(args.save_dir, out_name)
                visualize(img_path, dets, w, h, out_path)
                print(f"  {Path(img_path).name}: {len(dets)} rice detections")
            except Exception as e:
                print(f"  SKIP {img_path}: {e}")
    else:
        print("ERROR: provide --image or --folder")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
