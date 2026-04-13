"""
voc_dataset.py
==============
PyTorch Dataset for PASCAL VOC XML annotated image directories.

Designed to load the NCHU RiceSeedlingDataset detection split and any
other VOC-format dataset (e.g. output of data/auto_annotate.py or
data/coco_to_voc.py) for WeedDet retraining.

-----------------------------------------------------------------------------
RiceSeedlingDataset layout (after extracting RiceSeedlingDetection.tgz):
    <root>/
        JPEGImages/          ← RGB .jpg images (1527 × 1527 px)
        Annotations/         ← one .xml per image (VOC format)

Our coco_to_voc.py / auto_annotate.py output layout:
    <root>/
        images/              ← RGB images
        annotations/         ← one .xml per image

Both layouts are auto-detected.
-----------------------------------------------------------------------------

Class name mapping
──────────────────
RiceSeedlingDataset uses 'rice seedling' (lowercase, space).  Our codebase
uses 'Rice', 'Weed', 'Obstacle'.  The default CLASS_MAP converts both.
Override CLASS_MAP to add more classes as your dataset evolves.

Default CLASS_MAP:
    'rice seedling' → 0   (RICE_CLASS)
    'rice'          → 0
    'weed'          → 1   (WEED_CLASS)
    'obstacle'      → 2   (OBSTACLE_CLASS)

-----------------------------------------------------------------------------

CRITICAL — bounding box coordinate scaling
──────────────────────────────────────────
VOC XML stores boxes in original image pixel space.
WeedDet is trained at IMG_SIZE = (H=600, W=1000).
All box coordinates are rescaled by sx = 1000/orig_w, sy = 600/orig_h
in __getitem__ so they live in model space.  This mirrors FixedWeedDataset
in training/colab_weeddet_train.py.

-----------------------------------------------------------------------------

Usage
─────
    from data.voc_dataset import VocRiceDataset
    from torch.utils.data import DataLoader

    ds = VocRiceDataset(
        root='/content/RiceSeedlingDetection',
        use_exgr=True,      # apply ExGR vegetation enhancement
        augment=True,       # random horizontal flip
    )
    loader = DataLoader(ds, batch_size=2, shuffle=True,
                        collate_fn=VocRiceDataset.collate_fn)

    # Combine with existing training dataset:
    from torch.utils.data import ConcatDataset
    merged = ConcatDataset([existing_weeddet_dataset, ds])

-----------------------------------------------------------------------------

ExGR vegetation enhancement
────────────────────────────
When use_exgr=True the same NCHU preprocessing technique is applied:
    ExGR = 3G - 2.4R - B   (on normalised [0,1] pixel values)
Positive values mark vegetation; a boost of (1 + alpha * ExGR) is applied
to all three RGB channels before the standard ImageNet normalisation.
This is the same _apply_exgr() method added to DetectionPipeline.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


# ── Constants matching WeedDet training config ────────────────────────────
TARGET_H = 600
TARGET_W = 1000

# Default class-name → integer-label mapping
# Keys are lower-cased and stripped before lookup.
DEFAULT_CLASS_MAP: Dict[str, int] = {
    'rice seedling': 0,
    'rice':          0,
    'weed':          1,
    'obstacle':      2,
}

# ImageNet statistics used by WeedDet
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# ExGR helper  (mirrors DetectionPipeline._apply_exgr)
# ═══════════════════════════════════════════════════════════════════════════

def apply_exgr(img: Image.Image, alpha: float = 0.30) -> Image.Image:
    """
    Apply Excess Green minus Excess Red vegetation enhancement.

    ExGR = 3G - 2.4R - B on [0,1] normalised pixel values.
    Boosts vegetation pixels by (1 + alpha * ExGR_normalised) multiplicative
    factor across all three channels, leaving non-vegetation unchanged.

    Args:
        img   : PIL Image in RGB mode
        alpha : boost strength in [0, 1] (default 0.30)

    Returns:
        Vegetation-enhanced PIL Image in RGB mode
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0     # (H, W, 3)  [0,1]
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]

    exgr = 3.0 * G - 2.4 * R - B                        # ExGR index
    exgr = np.clip(exgr, 0.0, None)                     # keep vegetation only
    max_val = exgr.max()
    if max_val > 0:
        exgr = exgr / max_val                            # normalise to [0,1]

    boost = 1.0 + alpha * exgr[..., np.newaxis]         # (H, W, 1) broadcast
    enhanced = np.clip(arr * boost, 0.0, 1.0)
    return Image.fromarray((enhanced * 255).astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════════════
# VOC XML parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_voc_xml(xml_path: str, class_map: Dict[str, int]):
    """
    Parse a PASCAL VOC XML annotation file.

    Returns:
        boxes  : list of [xmin, ymin, xmax, ymax] in original pixel space
        labels : list of integer class labels
        img_w  : image width from XML (0 if missing)
        img_h  : image height from XML (0 if missing)
        filename : image filename from XML
    """
    tree  = ET.parse(xml_path)
    root  = tree.getroot()

    filename = (root.findtext('filename') or
                Path(xml_path).stem + '.jpg')

    size_el = root.find('size')
    img_w = int(size_el.findtext('width',  default='0')) if size_el else 0
    img_h = int(size_el.findtext('height', default='0')) if size_el else 0

    boxes: List[List[float]] = []
    labels: List[int]        = []

    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip().lower()
        label = class_map.get(name)
        if label is None:
            # Unknown class — skip silently
            continue

        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext('xmin', default='0'))
            ymin = float(bndbox.findtext('ymin', default='0'))
            xmax = float(bndbox.findtext('xmax', default='0'))
            ymax = float(bndbox.findtext('ymax', default='0'))
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels, img_w, img_h, filename


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class VocRiceDataset(torch.utils.data.Dataset):
    """
    PASCAL VOC dataset loader compatible with WeedDet's training loop.

    Handles both NCHU layout (JPEGImages/ + Annotations/) and the layout
    produced by our data scripts (images/ + annotations/).

    All bounding boxes are rescaled to WeedDet model space (600 × 1000)
    in __getitem__, matching the FixedWeedDataset fix in colab_weeddet_train.py.

    Args:
        root         : Root directory containing images and annotations.
        split_file   : Optional path to a plain-text file listing image
                       stems (one per line) to use as this split.  When
                       None all images that have a matching XML are used.
        class_map    : Dict mapping lower-cased VOC object names to integer
                       labels.  Defaults to DEFAULT_CLASS_MAP.
        use_exgr     : Apply ExGR vegetation enhancement before normalisation.
        exgr_alpha   : ExGR boost strength (default 0.30).
        augment      : Apply random horizontal flip during training.
        transforms   : Custom torchvision transforms applied AFTER ExGR and
                       BEFORE normalisation.  When None, only Resize + ToTensor
                       + Normalize are applied.
    """

    def __init__(
        self,
        root: str,
        split_file: Optional[str] = None,
        class_map: Optional[Dict[str, int]] = None,
        use_exgr: bool = False,
        exgr_alpha: float = 0.30,
        augment: bool = False,
        transforms: Optional[T.Compose] = None,
    ):
        self.root       = Path(root)
        self.class_map  = class_map if class_map is not None else DEFAULT_CLASS_MAP
        self.use_exgr   = use_exgr
        self.exgr_alpha = exgr_alpha
        self.augment    = augment

        # ── locate images and annotations subdirectories ─────────────────
        self.img_dir = self._find_subdir(['JPEGImages', 'images', 'imgs', '.'])
        self.ann_dir = self._find_subdir(['Annotations', 'annotations', 'labels', '.'])

        # ── build sample list ────────────────────────────────────────────
        if split_file is not None:
            stems = [l.strip() for l in Path(split_file).read_text().splitlines()
                     if l.strip()]
        else:
            stems = [p.stem for p in self.ann_dir.glob('*.xml')]

        self.samples: List[Tuple[Path, Path]] = []
        for stem in stems:
            xml_path = self.ann_dir / (stem + '.xml')
            if not xml_path.exists():
                continue
            # Find matching image (try common extensions)
            img_path = self._find_image(stem)
            if img_path is None:
                continue
            self.samples.append((img_path, xml_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid (image, XML) pairs found under '{root}'.\n"
                f"  Images dir  : {self.img_dir}\n"
                f"  Annotations : {self.ann_dir}\n"
                "Check that JPEGImages/ (or images/) and Annotations/ exist."
            )

        # ── base transform (resize + tensor + normalise) ─────────────────
        if transforms is not None:
            self._base_transform = transforms
        else:
            self._base_transform = T.Compose([
                T.Resize((TARGET_H, TARGET_W)),
                T.ToTensor(),
                T.Normalize(_MEAN, _STD),
            ])

    # ── private helpers ───────────────────────────────────────────────────

    def _find_subdir(self, candidates: List[str]) -> Path:
        for name in candidates:
            p = self.root / name
            if p.is_dir():
                return p
        return self.root

    def _find_image(self, stem: str) -> Optional[Path]:
        for ext in ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'):
            p = self.img_dir / (stem + ext)
            if p.exists():
                return p
        return None

    # ── dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, xml_path = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Optional ExGR enhancement
        if self.use_exgr:
            img = apply_exgr(img, alpha=self.exgr_alpha)

        # Optional random horizontal flip (augmentation)
        flipped = False
        if self.augment and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = True

        # Parse annotation
        boxes, labels, xml_w, xml_h, _ = parse_voc_xml(str(xml_path), self.class_map)

        # Use XML dimensions if image could not provide them
        if orig_w == 0:
            orig_w = xml_w
        if orig_h == 0:
            orig_h = xml_h

        # Scale boxes from original pixel space → model space (TARGET_W × TARGET_H)
        sx = TARGET_W / max(orig_w, 1)
        sy = TARGET_H / max(orig_h, 1)

        scaled_boxes: List[List[float]] = []
        for box in boxes:
            xmin = box[0] * sx
            ymin = box[1] * sy
            xmax = box[2] * sx
            ymax = box[3] * sy
            if flipped:
                # mirror x coordinates after horizontal flip
                xmin, xmax = TARGET_W - xmax, TARGET_W - xmin
            # Clamp to model space
            xmin = max(0.0, min(xmin, TARGET_W))
            ymin = max(0.0, min(ymin, TARGET_H))
            xmax = max(0.0, min(xmax, TARGET_W))
            ymax = max(0.0, min(ymax, TARGET_H))
            if xmax > xmin and ymax > ymin:
                scaled_boxes.append([xmin, ymin, xmax, ymax])

        # Drop any labels whose boxes became degenerate after clamping
        valid_labels = labels[:len(scaled_boxes)]

        # Convert to tensors
        if scaled_boxes:
            boxes_t  = torch.as_tensor(scaled_boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(valid_labels, dtype=torch.long)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.long)

        target = {
            'boxes':   boxes_t,
            'labels':  labels_t,
            'orig_w':  orig_w,
            'orig_h':  orig_h,
            'image_id': torch.tensor(idx),
        }

        # Apply base transform (resize is already encoded in sx/sy for boxes;
        # the Resize here only acts on the PIL image tensor).
        tensor = self._base_transform(img)

        return tensor, target

    # ── DataLoader collate helper ─────────────────────────────────────────

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for variable-size target dicts.

        Usage:
            loader = DataLoader(ds, batch_size=2, collate_fn=VocRiceDataset.collate_fn)
        """
        images  = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return torch.stack(images, dim=0), targets

    # ── convenience ──────────────────────────────────────────────────────

    def class_distribution(self) -> Dict[int, int]:
        """Return a count of annotations per class label across all samples."""
        from collections import Counter
        counts: Counter = Counter()
        for _, xml_path in self.samples:
            _, labels, *_ = parse_voc_xml(str(xml_path), self.class_map)
            counts.update(labels)
        return dict(counts)

    def __repr__(self) -> str:
        return (
            f"VocRiceDataset(root='{self.root}', "
            f"samples={len(self.samples)}, "
            f"use_exgr={self.use_exgr}, "
            f"augment={self.augment})"
        )
