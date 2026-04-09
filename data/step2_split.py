"""
step2_split.py
==============
Validate PASCAL VOC XML annotations and generate train/val split files.
Filters out images with no valid (non-degenerate) bounding boxes.

Usage:
    python step2_split.py --root /path/to/voc_dataset
    python step2_split.py --root /path/to/voc_dataset --split 0.85
"""

import os
import glob
import random
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def validate_xml(xml_path, min_box_size=2):
    """
    Return (valid, box_count, classes) for one XML file.
    valid = True if at least one non-degenerate box exists.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return False, 0, []

    root    = tree.getroot()
    boxes   = 0
    classes = []

    for obj in root.findall('object'):
        name = obj.find('name')
        bb   = obj.find('bndbox')
        if name is None or bb is None:
            continue
        try:
            x1 = float(bb.find('xmin').text)
            y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text)
            y2 = float(bb.find('ymax').text)
            if (x2 - x1) >= min_box_size and (y2 - y1) >= min_box_size:
                boxes += 1
                classes.append(name.text.strip())
        except (TypeError, ValueError):
            continue

    return boxes > 0, boxes, classes


def generate_splits(root_dir, train_ratio=0.8, seed=42,
                    min_box_size=2, verbose=True):
    """
    Scan annotations dir, validate XMLs, and write train.txt / val.txt.

    Args:
        root_dir:    VOC dataset root (contains images/ and annotations/)
        train_ratio: fraction of images for training (default 0.8)
        seed:        random seed for reproducibility
        min_box_size: minimum width/height for a valid box (pixels)
    """
    ann_dir = os.path.join(root_dir, 'annotations')
    img_dir = os.path.join(root_dir, 'images')

    if not os.path.exists(ann_dir):
        # Try flat structure
        ann_dir = root_dir
        img_dir = root_dir

    xml_paths = sorted(glob.glob(ann_dir + '/**/*.xml', recursive=True))

    if not xml_paths:
        print(f"ERROR: No XML files found in {ann_dir}")
        return

    print(f"Scanning {len(xml_paths)} XML files in {ann_dir} ...")

    valid_stems    = []
    skipped_nobox  = []
    skipped_parse  = []
    total_boxes    = 0
    class_counter  = Counter()

    for xml_path in xml_paths:
        stem  = Path(xml_path).stem
        valid, n_boxes, classes = validate_xml(xml_path, min_box_size)

        if valid:
            # Check image exists
            img_jpg = os.path.join(img_dir, stem + '.jpg')
            img_png = os.path.join(img_dir, stem + '.png')
            if os.path.exists(img_jpg) or os.path.exists(img_png):
                valid_stems.append(stem)
                total_boxes += n_boxes
                for cls in classes:
                    class_counter[cls] += 1
            else:
                skipped_nobox.append(stem)  # image missing
        else:
            skipped_nobox.append(stem)

    print(f"\nResults:")
    print(f"  Valid (image + boxes) : {len(valid_stems)}")
    print(f"  Skipped (no real boxes or missing image): {len(skipped_nobox)}")
    print(f"  Total boxes           : {total_boxes}")
    print(f"  Avg boxes/image       : {total_boxes/max(len(valid_stems),1):.1f}")

    print(f"\nClass distribution:")
    for cls, count in sorted(class_counter.items(), key=lambda x: -x[1]):
        print(f"  {cls:<35s}  {count:>6d} boxes")

    # Split
    random.seed(seed)
    random.shuffle(valid_stems)
    split      = int(len(valid_stems) * train_ratio)
    train_ids  = valid_stems[:split]
    val_ids    = valid_stems[split:]

    train_txt = os.path.join(root_dir, 'train.txt')
    val_txt   = os.path.join(root_dir, 'val.txt')

    with open(train_txt, 'w') as f:
        f.write('\n'.join(train_ids))
    with open(val_txt, 'w') as f:
        f.write('\n'.join(val_ids))

    print(f"\nSplit (seed={seed}, ratio={train_ratio}):")
    print(f"  Train : {len(train_ids):4d} images → {train_txt}")
    print(f"  Val   : {len(val_ids):4d}  images → {val_txt}")
    print("\nDone. Run train_rice.py or colab_weeddet_train.py to start training.")

    return train_ids, val_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',      required=True,
                        help='VOC dataset root directory')
    parser.add_argument('--split',     type=float, default=0.8,
                        help='Train ratio (default 0.8)')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--min-box',   type=int,   default=2,
                        help='Min box size in pixels to be considered valid')
    args = parser.parse_args()

    generate_splits(
        root_dir=args.root,
        train_ratio=args.split,
        seed=args.seed,
        min_box_size=args.min_box,
    )
