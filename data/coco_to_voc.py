"""
coco_to_voc.py
==============
Converts a COCO JSON annotation file to PASCAL VOC XML format.
Used to prepare the merged_dataset.zip (COCO) for WeedDet training (VOC).

Key filter: SKIPS any annotation whose bounding box covers ≥ 95% of the
image area. These are "full-image placeholder boxes" produced by the
disease-severity datasets (Mild/Healthy/Moderate/Severe) that have
classification labels but no real localization boxes. Only images with
genuine per-plant boxes pass through.

Usage:
    python coco_to_voc.py \
        --json merged_annotations.json \
        --images /path/to/images \
        --output /path/to/voc_dataset

    Output layout:
        /path/to/voc_dataset/
            images/        <- copied images
            annotations/   <- one .xml per image (only real-box images)
"""

import os
import json
import shutil
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path


# Fraction of image area a box must stay BELOW to be considered real
PLACEHOLDER_AREA_THRESHOLD = 0.95


def is_placeholder_box(bbox, img_w, img_h,
                        threshold=PLACEHOLDER_AREA_THRESHOLD):
    """
    Return True if the box covers >= threshold of the image area.
    These are full-image severity labels, not per-plant detections.

    bbox: [x, y, w, h] COCO format
    """
    x, y, w, h = bbox
    box_area = w * h
    img_area = img_w * img_h
    if img_area == 0:
        return True
    return (box_area / img_area) >= threshold


def write_voc_xml(image_path, boxes_xyxy, img_w, img_h, output_dir):
    """
    Write a PASCAL VOC XML annotation file.

    boxes_xyxy: list of [x1, y1, x2, y2] in pixel coordinates
    All boxes are written as class 'Rice' (single-class setup).
    """
    stem     = Path(image_path).stem
    xml_path = os.path.join(output_dir, 'annotations', stem + '.xml')

    ann = ET.Element('annotation')
    ET.SubElement(ann, 'folder').text   = 'images'
    ET.SubElement(ann, 'filename').text = Path(image_path).name
    ET.SubElement(ann, 'path').text     = os.path.join(
        output_dir, 'images', Path(image_path).name)

    sz = ET.SubElement(ann, 'size')
    ET.SubElement(sz, 'width').text  = str(img_w)
    ET.SubElement(sz, 'height').text = str(img_h)
    ET.SubElement(sz, 'depth').text  = '3'

    ET.SubElement(ann, 'segmented').text = '0'

    for box in boxes_xyxy:
        x1 = max(0, min(int(box[0]), img_w - 1))
        y1 = max(0, min(int(box[1]), img_h - 1))
        x2 = max(0, min(int(box[2]), img_w))
        y2 = max(0, min(int(box[3]), img_h))

        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue

        obj = ET.SubElement(ann, 'object')
        ET.SubElement(obj, 'name').text      = 'Rice'
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


def convert(json_path, images_dir, output_dir,
            copy_images=True, report_every=100):
    """
    Convert a COCO JSON annotation file to PASCAL VOC XML format.

    Args:
        json_path:    path to COCO JSON file
        images_dir:   directory containing image files referenced in the JSON
        output_dir:   root output directory (creates images/ and annotations/)
        copy_images:  if True, copy images to output_dir/images/
        report_every: print progress every N images
    """
    out_img_dir = os.path.join(output_dir, 'images')
    out_ann_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    print(f"Loading {json_path} ...")
    with open(json_path) as f:
        coco = json.load(f)

    # Build image id → image metadata lookup
    img_lookup = {img['id']: img for img in coco['images']}

    # Build image id → list of annotations
    ann_by_img = {}
    for ann in coco['annotations']:
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    # Build basename → full path lookup for images on disk
    disk_lookup = {}
    for root_dir, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                disk_lookup[f] = os.path.join(root_dir, f)

    total       = len(img_lookup)
    written     = 0
    skipped_notfound = 0
    skipped_noboxes  = 0

    print(f"Processing {total} images ...")

    for img_id, img_meta in img_lookup.items():
        # Find image on disk
        basename = Path(img_meta['file_name']).name
        img_path = disk_lookup.get(basename)

        if img_path is None:
            # Try absolute path stored in file_name
            if os.path.exists(img_meta['file_name']):
                img_path = img_meta['file_name']
            else:
                skipped_notfound += 1
                continue

        img_w = img_meta.get('width',  0)
        img_h = img_meta.get('height', 0)

        # If dimensions missing, read from file
        if img_w == 0 or img_h == 0:
            try:
                from PIL import Image as PILImage
                with PILImage.open(img_path) as pil:
                    img_w, img_h = pil.size
            except Exception:
                skipped_notfound += 1
                continue

        # Collect non-placeholder boxes for this image
        anns = ann_by_img.get(img_id, [])
        boxes_xyxy = []
        for ann in anns:
            bbox = ann['bbox']   # [x, y, w, h]
            if is_placeholder_box(bbox, img_w, img_h):
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w > 1 and h > 1:
                boxes_xyxy.append([x, y, x + w, y + h])

        if not boxes_xyxy:
            skipped_noboxes += 1
            continue

        # Copy image
        if copy_images:
            dst = os.path.join(out_img_dir, basename)
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)

        # Write XML
        write_voc_xml(img_path, boxes_xyxy, img_w, img_h, output_dir)
        written += 1

        if written % report_every == 0:
            print(f"  Written {written}/{total - skipped_notfound - skipped_noboxes} ...")

    print(f"\nConversion complete.")
    print(f"  Total images       : {total}")
    print(f"  Written (with boxes): {written}")
    print(f"  Skipped (not found): {skipped_notfound}")
    print(f"  Skipped (no real boxes / placeholder only): {skipped_noboxes}")
    print(f"  Output dir         : {output_dir}")
    return written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert COCO JSON to PASCAL VOC XML for WeedDet training')
    parser.add_argument('--json',    required=True,
                        help='Path to COCO JSON annotation file')
    parser.add_argument('--images',  required=True,
                        help='Directory containing image files')
    parser.add_argument('--output',  required=True,
                        help='Output directory (creates images/ and annotations/)')
    parser.add_argument('--no-copy', action='store_true',
                        help='Do not copy images (only write XMLs)')
    parser.add_argument('--report',  type=int, default=100,
                        help='Progress report interval')
    args = parser.parse_args()

    convert(
        json_path=args.json,
        images_dir=args.images,
        output_dir=args.output,
        copy_images=not args.no_copy,
        report_every=args.report,
    )
