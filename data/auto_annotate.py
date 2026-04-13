"""
auto_annotate.py
================
Uses Grounding DINO (HuggingFace Transformers) to automatically generate
PASCAL VOC XML bounding box annotations for unannotated rice field images.

Replaces manual annotation via LabelImg for large datasets.

Usage:
    # Annotate all images in a folder
    python auto_annotate.py --images /path/to/images --output /path/to/annotations

    # Preview first image before batch run (recommended first time)
    python auto_annotate.py --images ./images --output ./annotations --preview

    # Lower threshold = more boxes (more false positives)
    python auto_annotate.py --images ./images --output ./annotations --threshold 0.15

Requirements:
    pip install transformers torch Pillow

Model: IDEA-Research/grounding-dino-tiny (HuggingFace)
Text prompt: "plant . crop . rice plant . green plant . paddy plant"

Threshold tuning guide:
    0.35 → strict, fewer boxes, fewer false positives, may miss plants
    0.25 → balanced starting point (default)
    0.15 → permissive, more boxes, more coverage, more false positives
"""

import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

from PIL import Image


# ── Grounding DINO setup ────────────────────────────────────────────────────

def load_grounding_dino(device='cuda'):
    """
    Load Grounding DINO Tiny from HuggingFace.
    Falls back to CPU automatically if no GPU.
    """
    from transformers import pipeline

    print("Loading Grounding DINO... (first run downloads ~700MB model)")
    pipe = pipeline(
        "zero-shot-object-detection",
        model="IDEA-Research/grounding-dino-tiny",
        device=0 if device == 'cuda' else -1,
    )
    print("Model loaded.")
    return pipe


# ── VOC XML writer ──────────────────────────────────────────────────────────

def save_pascal_voc_xml(image_path, boxes, labels, img_w, img_h, output_dir):
    """
    Save detected boxes as a PASCAL VOC XML annotation file.

    Args:
        image_path:  path to source image (for filename/path fields)
        boxes:       list of [x1, y1, x2, y2] in pixel coordinates
        labels:      list of class name strings ('Rice' for all)
        img_w, img_h: image dimensions in pixels
        output_dir:  directory to write XML file into
    """
    stem     = Path(image_path).stem
    xml_path = os.path.join(output_dir, stem + '.xml')

    ann = ET.Element('annotation')
    ET.SubElement(ann, 'folder').text   = 'images'
    ET.SubElement(ann, 'filename').text = Path(image_path).name
    ET.SubElement(ann, 'path').text     = str(image_path)

    src = ET.SubElement(ann, 'source')
    ET.SubElement(src, 'database').text = 'GroundingDINO-AutoAnnotated'

    sz = ET.SubElement(ann, 'size')
    ET.SubElement(sz, 'width').text  = str(img_w)
    ET.SubElement(sz, 'height').text = str(img_h)
    ET.SubElement(sz, 'depth').text  = '3'

    ET.SubElement(ann, 'segmented').text = '0'

    for box, label in zip(boxes, labels):
        x1 = max(0, min(int(box[0]), img_w - 1))
        y1 = max(0, min(int(box[1]), img_h - 1))
        x2 = max(0, min(int(box[2]), img_w))
        y2 = max(0, min(int(box[3]), img_h))

        if x2 <= x1 or y2 <= y1:   # skip degenerate boxes
            continue

        obj = ET.SubElement(ann, 'object')
        ET.SubElement(obj, 'name').text      = label
        ET.SubElement(obj, 'pose').text      = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bb = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bb, 'xmin').text = str(x1)
        ET.SubElement(bb, 'ymin').text = str(y1)
        ET.SubElement(bb, 'xmax').text = str(x2)
        ET.SubElement(bb, 'ymax').text = str(y2)

    raw    = ET.tostring(ann, encoding='unicode')
    pretty = minidom.parseString(raw).toprettyxml(indent='  ')

    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty)

    return xml_path


# ── Preview ─────────────────────────────────────────────────────────────────

def preview_detections(image_path, boxes, scores):
    """Show a preview image with detected boxes (requires display)."""
    try:
        from PIL import ImageDraw
        img  = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
            draw.text((x1 + 4, y1 + 4), f"Rice {score:.2f}",
                      fill=(0, 220, 0))
        draw.text((10, 10),
                  f"{len(boxes)} rice detections",
                  fill=(255, 255, 0))
        img.show()
        print(f"\nPreview: {len(boxes)} boxes found.")
        input("Press ENTER to continue annotating all images...")
    except Exception as e:
        print(f"Preview failed: {e}")


# ── Main annotation loop ────────────────────────────────────────────────────

# Text prompt for Grounding DINO (period-separated phrases)
# The pipeline accepts a single text string, NOT a list of labels.
DETECTION_PROMPT = "plant . crop . rice plant . green plant . paddy plant"

# Normalize all detected labels to 'Rice' in the XML
LABEL = 'Rice'


def annotate_folder(images_dir, output_dir, threshold=0.25,
                    device='cuda', preview=False, report_every=10):
    """
    Run Grounding DINO on all images in images_dir and write PASCAL VOC XMLs.

    Args:
        images_dir:   folder containing .jpg / .png images
        output_dir:   folder to write .xml annotation files
        threshold:    Grounding DINO confidence threshold
        device:       'cuda' or 'cpu'
        preview:      show first image preview before batch run
        report_every: print progress every N images
    """
    os.makedirs(output_dir, exist_ok=True)

    img_paths = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not img_paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(img_paths)} images in {images_dir}")
    print(f"Threshold: {threshold}  |  Prompt: {DETECTION_PROMPT}")
    print(f"Output:    {output_dir}\n")

    pipe = load_grounding_dino(device)

    total_boxes = 0
    skipped     = 0
    first       = True

    for idx, img_path in enumerate(img_paths, start=1):
        try:
            pil_img      = Image.open(img_path).convert('RGB')
            img_w, img_h = pil_img.size
        except Exception as e:
            print(f"  SKIP {img_path}: {e}")
            skipped += 1
            continue

        # Run detection
        results = pipe(
            pil_img,
            text=DETECTION_PROMPT,
            threshold=threshold,
        )

        # Extract boxes and scores
        boxes  = []
        scores = []
        for r in results:
            box = r['box']  # {'xmin':..., 'ymin':..., 'xmax':..., 'ymax':...}
            x1, y1 = box['xmin'], box['ymin']
            x2, y2 = box['xmax'], box['ymax']
            if (x2 - x1) > 2 and (y2 - y1) > 2:   # skip degenerate
                boxes.append([x1, y1, x2, y2])
                scores.append(r['score'])

        # Preview first image
        if preview and first and boxes:
            preview_detections(img_path, boxes, scores)
            first = False

        # Write XML
        labels = [LABEL] * len(boxes)
        save_pascal_voc_xml(img_path, boxes, labels,
                            img_w, img_h, output_dir)
        total_boxes += len(boxes)

        if idx % report_every == 0 or idx == len(img_paths):
            print(f"  [{idx:4d}/{len(img_paths)}]  "
                  f"{os.path.basename(img_path):<40s}  "
                  f"{len(boxes):3d} boxes")

    print(f"\nDone.")
    print(f"  Images processed : {len(img_paths) - skipped}")
    print(f"  Images skipped   : {skipped}")
    print(f"  Total boxes saved: {total_boxes}")
    print(f"  Avg boxes/image  : {total_boxes/(len(img_paths)-skipped+1e-6):.1f}")
    print(f"  XMLs written to  : {output_dir}")


# ── Colab-specific convenience wrapper ─────────────────────────────────────

def annotate_for_colab(images_dir, output_dir, threshold=0.15):
    """
    Simplified wrapper for Google Colab.
    Uses threshold=0.15 (more permissive) to maximize coverage.

    Example:
        from auto_annotate import annotate_for_colab
        annotate_for_colab(
            images_dir='/content/dataset/images',
            output_dir='/content/dataset/annotations',
        )
    """
    annotate_folder(
        images_dir=images_dir,
        output_dir=output_dir,
        threshold=threshold,
        device='cuda',
        preview=False,
        report_every=25,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Grounding DINO auto-annotation for rice detection')
    parser.add_argument('--images',    required=True,
                        help='Directory containing input images')
    parser.add_argument('--output',    required=True,
                        help='Directory to write XML annotation files')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='Detection threshold (0.15–0.35, default 0.25)')
    parser.add_argument('--device',    default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--preview',   action='store_true',
                        help='Preview first image before batch run')
    parser.add_argument('--report',    type=int, default=10,
                        help='Print progress every N images')
    args = parser.parse_args()

    annotate_folder(
        images_dir=args.images,
        output_dir=args.output,
        threshold=args.threshold,
        device=args.device,
        preview=args.preview,
        report_every=args.report,
    )
