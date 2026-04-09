"""
colab_weeddet_train.py
======================
Complete Google Colab training cell for WeedDet.

Paste this entire file into a single Colab code cell and run it.

What this does, in order:
  1. Mounts Google Drive
  2. Unzips merged_dataset.zip from Drive
  3. Converts COCO JSON → PASCAL VOC XML (filtering placeholder boxes)
  4. Installs FixedWeedDataset to fix the bounding box coordinate-space bug
  5. Launches training with tqdm progress bars

────────────────────────────────────────────────────────────────────────────
CRITICAL BUG EXPLANATION (already fixed below, do not remove):

  WeedDet resizes input images to (600, 1000).
  VOC XML boxes are stored in the ORIGINAL image pixel space.
  If boxes are not rescaled, IoU between ground truth and anchors → 0.
  Zero positive anchors → zero gradient → loss collapses to 0.0000.
  PyTorch throws no error. Training appears to run but learns nothing.

  Fix: FixedWeedDataset applies sx = 1000/orig_w, sy = 600/orig_h
  to all box coordinates in __getitem__.
────────────────────────────────────────────────────────────────────────────

Drive layout expected:
  MyDrive/
    merged_dataset.zip          <- 1.6GB merged COCO dataset
    weeddet_checkpoints/        <- created automatically
"""

import os
import sys
import zipfile
import json
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

import torch
import torch.nn as nn


# ── 0. Config ───────────────────────────────────────────────────────────────

ZIP_ON_DRIVE    = '/content/drive/MyDrive/merged_dataset.zip'
EXTRACT_DIR     = '/content/dataset'
CHECKPOINT_DIR  = '/content/drive/MyDrive/weeddet_checkpoints'

TRAINING_CONFIG = {
    'data_root'       : EXTRACT_DIR,
    'num_classes'     : 1,           # rice only (inverted logic)
    'batch_size'      : 2,
    'num_epochs'      : 12,
    'base_lr'         : 0.01,
    'momentum'        : 0.9,
    'weight_decay'    : 0.0001,
    'warmup_iters'    : 500,
    'lr_decay_epochs' : [8, 11],
    'img_size'        : (600, 1000),
    'device'          : 'cuda',
    'checkpoint_dir'  : CHECKPOINT_DIR,
    'freeze_bn'       : True,
    'num_workers'     : 2,
}


# ── 1. Mount Drive ──────────────────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("Drive mounted.")


# ── 2. Unzip dataset ────────────────────────────────────────────────────────

if not os.path.exists(EXTRACT_DIR):
    print(f"Extracting {ZIP_ON_DRIVE} → {EXTRACT_DIR} ...")
    with zipfile.ZipFile(ZIP_ON_DRIVE, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("Extraction complete.")
else:
    print(f"Dataset already extracted at {EXTRACT_DIR}")

# Quick sanity check
import glob
imgs = glob.glob(EXTRACT_DIR + '/**/*.jpg', recursive=True) + \
       glob.glob(EXTRACT_DIR + '/**/*.png', recursive=True)
jsons = glob.glob(EXTRACT_DIR + '/**/*.json', recursive=True)
print(f"Found {len(imgs)} images, {len(jsons)} JSON files in dataset")


# ── 3. Convert COCO JSON → VOC XML (if not already done) ───────────────────

VOC_DIR     = os.path.join(EXTRACT_DIR, 'voc')
VOC_IMG_DIR = os.path.join(VOC_DIR, 'images')
VOC_ANN_DIR = os.path.join(VOC_DIR, 'annotations')

def _is_placeholder(bbox, img_w, img_h, thresh=0.95):
    x, y, w, h = bbox
    return (w * h) / (img_w * img_h + 1e-6) >= thresh

def _write_xml(basename, boxes_xyxy, img_w, img_h, out_ann_dir, img_dir_name):
    stem     = Path(basename).stem
    xml_path = os.path.join(out_ann_dir, stem + '.xml')
    ann = ET.Element('annotation')
    ET.SubElement(ann, 'folder').text   = img_dir_name
    ET.SubElement(ann, 'filename').text = basename
    sz = ET.SubElement(ann, 'size')
    ET.SubElement(sz, 'width').text  = str(img_w)
    ET.SubElement(sz, 'height').text = str(img_h)
    ET.SubElement(sz, 'depth').text  = '3'
    ET.SubElement(ann, 'segmented').text = '0'
    for box in boxes_xyxy:
        x1 = max(0, min(int(box[0]), img_w-1))
        y1 = max(0, min(int(box[1]), img_h-1))
        x2 = max(0, min(int(box[2]), img_w))
        y2 = max(0, min(int(box[3]), img_h))
        if x2-x1 <= 1 or y2-y1 <= 1:
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
    raw = ET.tostring(ann, encoding='unicode')
    pretty = minidom.parseString(raw).toprettyxml(indent='  ')
    with open(xml_path, 'w') as f:
        f.write(pretty)
    return xml_path

if not os.path.exists(VOC_ANN_DIR) or \
        len(glob.glob(VOC_ANN_DIR + '/*.xml')) == 0:
    print("Converting COCO JSON → PASCAL VOC XML ...")
    os.makedirs(VOC_IMG_DIR, exist_ok=True)
    os.makedirs(VOC_ANN_DIR, exist_ok=True)

    # Build image-path lookup
    disk_lookup = {}
    for p in imgs:
        disk_lookup[os.path.basename(p)] = p

    written = 0
    for json_path in jsons:
        with open(json_path) as f:
            coco = json.load(f)

        img_meta = {img['id']: img for img in coco['images']}
        ann_by   = {}
        for ann in coco['annotations']:
            ann_by.setdefault(ann['image_id'], []).append(ann)

        for img_id, meta in img_meta.items():
            basename = os.path.basename(meta['file_name'])
            src_path = disk_lookup.get(basename, meta['file_name'])
            if not os.path.exists(src_path):
                continue

            iw = meta.get('width', 0)
            ih = meta.get('height', 0)
            if iw == 0 or ih == 0:
                try:
                    from PIL import Image as _PIL
                    with _PIL.open(src_path) as _p:
                        iw, ih = _p.size
                except Exception:
                    continue

            boxes_xyxy = []
            for ann in ann_by.get(img_id, []):
                if _is_placeholder(ann['bbox'], iw, ih):
                    continue
                x, y, w, h = [float(v) for v in ann['bbox']]
                if w > 1 and h > 1:
                    boxes_xyxy.append([x, y, x+w, y+h])

            if not boxes_xyxy:
                continue

            # Copy image
            dst = os.path.join(VOC_IMG_DIR, basename)
            if not os.path.exists(dst):
                shutil.copy2(src_path, dst)

            _write_xml(basename, boxes_xyxy, iw, ih, VOC_ANN_DIR, 'images')
            written += 1

    print(f"Converted {written} images with real boxes → {VOC_ANN_DIR}")
else:
    written = len(glob.glob(VOC_ANN_DIR + '/*.xml'))
    print(f"VOC annotations already exist ({written} XMLs) — skipping conversion")

# Update training config to point at VOC dataset
TRAINING_CONFIG['data_root'] = VOC_DIR


# ── 4. Generate train/val split ─────────────────────────────────────────────

import random

train_txt = os.path.join(VOC_DIR, 'train.txt')
val_txt   = os.path.join(VOC_DIR, 'val.txt')

if not os.path.exists(train_txt):
    xml_files = glob.glob(VOC_ANN_DIR + '/*.xml')
    stems     = [Path(f).stem for f in xml_files]
    random.seed(42)
    random.shuffle(stems)
    split = int(len(stems) * 0.8)
    with open(train_txt, 'w') as f:
        f.write('\n'.join(stems[:split]))
    with open(val_txt, 'w') as f:
        f.write('\n'.join(stems[split:]))
    print(f"Split: {split} train / {len(stems)-split} val")
else:
    with open(train_txt) as f:
        n_train = len(f.read().splitlines())
    with open(val_txt) as f:
        n_val = len(f.read().splitlines())
    print(f"Split already exists: {n_train} train / {n_val} val")


# ── 5. Install module ───────────────────────────────────────────────────────

# weeddet_for_VSCode.py must be in /content/ or on sys.path
if '/content' not in sys.path:
    sys.path.insert(0, '/content')

import weeddet_for_VSCode as wd
print(f"WeedDet module loaded: {wd.__file__}")


# ── 6. FixedWeedDataset — BOUNDING BOX COORDINATE SCALING FIX ──────────────
#
# WHY THIS EXISTS:
#   The WeedDataset in weeddet_for_VSCode.py correctly resizes images to
#   (600, 1000) in __getitem__. However an older version did NOT rescale
#   the box coordinates to match the resized dimensions, leaving boxes in
#   the original image pixel space.
#
#   Result: IoU(ground_truth_boxes, anchors) = 0 for all anchors.
#   Result: no positive anchor assignments, no gradient, loss → 0.0000.
#   PyTorch does not raise an error. Training appears to run normally.
#
#   The current weeddet_for_VSCode.py already includes the fix in
#   WeedDataset.__getitem__. This class is kept as an explicit guard in
#   case the module version you are using does not have it.
# ───────────────────────────────────────────────────────────────────────────

class FixedWeedDataset(wd.WeedDataset):
    """
    Explicit bounding box coordinate scaling guard.
    Applies sx = TARGET_W / orig_w, sy = TARGET_H / orig_h
    to all box coordinates in __getitem__.
    """
    TARGET_H = 600
    TARGET_W = 1000

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        if result is None:
            return None
        img, target = result
        if target is None or 'boxes' not in target:
            return img, target
        boxes = target['boxes']
        if boxes.numel() == 0:
            return img, target

        orig_h = target.get('orig_h', self.TARGET_H)
        orig_w = target.get('orig_w', self.TARGET_W)
        sx = self.TARGET_W / max(orig_w, 1)
        sy = self.TARGET_H / max(orig_h, 1)

        boxes = boxes.clone().float()
        boxes[:, 0] *= sx   # xmin
        boxes[:, 2] *= sx   # xmax
        boxes[:, 1] *= sy   # ymin
        boxes[:, 3] *= sy   # ymax
        target['boxes'] = boxes
        return img, target


# Monkey-patch — train_with_progress picks this up automatically
wd.WeedDataset = FixedWeedDataset
print("FixedWeedDataset installed. Coordinate scaling is active.")


# ── 7. GPU check ────────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected!\n"
        "Go to: Runtime → Change runtime type → Hardware accelerator → GPU"
    )
print(f"GPU: {torch.cuda.get_device_name(0)}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── 8. Launch training ──────────────────────────────────────────────────────

print("\nTraining config:")
for k, v in TRAINING_CONFIG.items():
    print(f"  {k:<20s}: {v}")
print()

wd.train_with_progress(TRAINING_CONFIG)
