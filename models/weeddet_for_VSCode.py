"""
weeddet_for_VSCode.py
=====================
WeedDet — Full PyTorch Implementation
Based on: Peng et al. (2022) "Weed Detection in Paddy Field Using an
Improved RetinaNet Network", Computers and Electronics in Agriculture,
vol. 199, p. 107179.

Architecture:
  Input RGB image (1000×600)
      │
  Det-ResNet-50 backbone   ← replaces standard ResNet-50 first block
      │  C3 (512ch), C4 (1024ch), C5 (2048ch)
  eFPN (efficient FPN)     ← only P3, P4, P5  (no P6/P7 — saves 7.71M params)
      │  256ch per level
  ERetina-Head             ← 1 conv 64ch + Large Separable Conv (k=7)
      │
  Predictions              ← boxes + class scores + IACS confidence

Loss:
  Regression    : SmoothL1 + GIoU
  Classification: VariFocal Loss

Training:
  SGD  lr=0.01  momentum=0.9  weight_decay=1e-4
  Batch 2  ·  12 epochs  ·  2× dataset repeat per epoch
  Linear warmup 500 iters  →  MultiStepLR decay at epochs [8, 11]
  Freeze BatchNorm (batch_size=2 too small for stable BN stats)

Single-class rice config (AgriNav):
  num_classes = 1  (Rice only)
  Detection logic INVERTED: rice = protected, everything else = spray target
"""

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms, box_iou

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ===========================================================================
# SECTION 1 — BACKBONE: Det-ResNet-50
# ===========================================================================

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)


class DetResidualBlock(nn.Module):
    """
    Initial residual block replacing MaxPool in standard ResNet stem.
    Channels: 16 → 32, stride=2 for controlled spatial downsampling.
    Preserves fine detail critical for small-object detection.
    """
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(16, 32, 1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))


class Bottleneck(nn.Module):
    """Standard ResNet-50 bottleneck block."""
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample:
            res = self.downsample(x)
        return self.relu(out + res)


def _make_layer(in_ch, mid_ch, blocks, stride=1):
    out_ch = mid_ch * Bottleneck.expansion
    downsample = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch))
    layers = [Bottleneck(in_ch, mid_ch, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(Bottleneck(out_ch, mid_ch))
    return nn.Sequential(*layers)


class DetResNet50(nn.Module):
    """
    Det-ResNet-50: modified ResNet-50 backbone.
    Replaces 7×7 stem + MaxPool with two 3×3 convs + DetResidualBlock.
    Returns C3 (512ch), C4 (1024ch), C5 (2048ch).
    """
    def __init__(self):
        super().__init__()
        # Modified stem — preserves spatial detail
        self.stem = nn.Sequential(
            ConvBnRelu(3,  16, k=3, s=1, p=1),
            ConvBnRelu(16, 16, k=3, s=1, p=1),
            DetResidualBlock(),              # 32ch, stride=2
        )
        # Standard ResNet-50 bottleneck stages
        self.layer1 = _make_layer(32,  64, 3, stride=1)   # → 256ch  (C2)
        self.layer2 = _make_layer(256, 128, 4, stride=2)  # → 512ch  (C3)
        self.layer3 = _make_layer(512, 256, 6, stride=2)  # → 1024ch (C4)
        self.layer4 = _make_layer(1024,512, 3, stride=2)  # → 2048ch (C5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


# ===========================================================================
# SECTION 2 — NECK: eFPN (Efficient Feature Pyramid Network)
# ===========================================================================

class eFPN(nn.Module):
    """
    Efficient FPN: outputs only P3, P4, P5.
    P6 and P7 are NOT built — saves ~7.71M parameters vs. standard FPN.
    All pyramid levels normalized to 256 channels.
    """
    def __init__(self, c3_ch=512, c4_ch=1024, c5_ch=2048, feat=256):
        super().__init__()
        # Lateral 1×1 projections
        self.lat3 = nn.Conv2d(c3_ch, feat, 1)
        self.lat4 = nn.Conv2d(c4_ch, feat, 1)
        self.lat5 = nn.Conv2d(c5_ch, feat, 1)
        # Output 3×3 smoothing
        self.out3 = nn.Conv2d(feat, feat, 3, padding=1)
        self.out4 = nn.Conv2d(feat, feat, 3, padding=1)
        self.out5 = nn.Conv2d(feat, feat, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:],
                                            mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:],
                                            mode='nearest')
        return self.out3(p3), self.out4(p4), self.out5(p5)


# ===========================================================================
# SECTION 3 — HEAD: ERetina-Head
# ===========================================================================

class LargeSeparableConv(nn.Module):
    """
    Decomposes k×k conv into k×1 then 1×k.
    Expands receptive field without quadratic parameter growth.
    Default k=7 (from paper ablation).
    """
    def __init__(self, ch, k=7):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(ch, ch, (k, 1), padding=(k//2, 0),
                      groups=ch, bias=False),
            nn.Conv2d(ch, ch, (1, k), padding=(0, k//2),
                      groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)


class ERetinaHead(nn.Module):
    """
    Efficient Retina Head: 1 conv (64ch) + LSC instead of 4 convs (256ch).
    Shared weights across FPN levels.
    """
    def __init__(self, in_ch=256, num_classes=1, num_anchors=9, lsc_k=7):
        super().__init__()
        mid = 64
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            LargeSeparableConv(mid, lsc_k),
        )
        self.cls_head = nn.Conv2d(mid, num_classes * num_anchors, 3, padding=1)
        self.reg_head = nn.Conv2d(mid, 4 * num_anchors, 3, padding=1)

        # Bias init for focal/varifocal loss stability
        prior_prob = 0.01
        bias_val   = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_val)
        nn.init.normal_(self.reg_head.weight,  std=0.01)
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, features):
        cls_outs, reg_outs = [], []
        for feat in features:
            x   = self.shared(feat)
            cls_outs.append(self.cls_head(x))
            reg_outs.append(self.reg_head(x))
        return cls_outs, reg_outs


# ===========================================================================
# SECTION 4 — ANCHOR GENERATOR
# ===========================================================================

class AnchorGenerator(nn.Module):
    """
    Generates anchors for P3/P4/P5 pyramid levels.
    base_scale=6 from paper ablation study.
    3 aspect ratios × 3 scales = 9 anchors per location.
    """
    def __init__(self, base_scale=6,
                 aspect_ratios=(0.5, 1.0, 2.0),
                 scales=(1.0, 2**(1/3), 2**(2/3)),
                 strides=(8, 16, 32)):
        super().__init__()
        self.base_scale    = base_scale
        self.aspect_ratios = aspect_ratios
        self.scales        = scales
        self.strides       = strides

    @torch.no_grad()
    def forward(self, features, img_shape):
        all_anchors = []
        for feat, stride in zip(features, self.strides):
            H, W = feat.shape[-2:]
            anchors = self._make_anchors(H, W, stride, feat.device)
            all_anchors.append(anchors)
        return torch.cat(all_anchors, dim=0)  # [total_anchors, 4] xyxy

    def _make_anchors(self, H, W, stride, device):
        base = self.base_scale * stride
        anchors = []
        for ar in self.aspect_ratios:
            for sc in self.scales:
                w = base * sc * math.sqrt(ar)
                h = base * sc / math.sqrt(ar)
                anchors.append([-w/2, -h/2, w/2, h/2])
        anchors = torch.tensor(anchors, dtype=torch.float32, device=device)

        cy = (torch.arange(H, dtype=torch.float32, device=device) + 0.5) * stride
        cx = (torch.arange(W, dtype=torch.float32, device=device) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(cy, cx, indexing='ij')
        shifts = torch.stack([grid_x.flatten(), grid_y.flatten(),
                               grid_x.flatten(), grid_y.flatten()], dim=1)

        return (shifts[:, None, :] + anchors[None, :, :]).reshape(-1, 4)


# ===========================================================================
# SECTION 5 — LOSS FUNCTIONS
# ===========================================================================

class GIoULoss(nn.Module):
    def forward(self, pred, target):
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        inter_w  = (inter_x2 - inter_x1).clamp(min=0)
        inter_h  = (inter_y2 - inter_y1).clamp(min=0)
        inter    = inter_w * inter_h

        pw = (pred[:, 2]   - pred[:, 0]).clamp(min=0)
        ph = (pred[:, 3]   - pred[:, 1]).clamp(min=0)
        tw = (target[:, 2] - target[:, 0]).clamp(min=0)
        th = (target[:, 3] - target[:, 1]).clamp(min=0)
        union = pw * ph + tw * th - inter + 1e-6

        iou    = inter / union
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])
        enc    = ((enc_x2 - enc_x1) * (enc_y2 - enc_y1)).clamp(min=1e-6)
        giou   = iou - (enc - union) / enc
        return (1 - giou).mean()


class VariFocalLoss(nn.Module):
    """
    VariFocal Loss: replaces standard Focal Loss.
    Weights positive anchors by their IACS score (IoU-Aware Classification Score).
    alpha=0.75, gamma=2.0 from paper.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # pred:   [N, C] raw logits
        # target: [N, C] soft labels (IACS scores for positives, 0 for negatives)
        p    = pred.sigmoid()
        pos  = target > 0
        neg  = ~pos

        weight = torch.where(
            pos,
            target * (target - p).abs().pow(self.gamma),
            self.alpha * p.pow(self.gamma),
        )
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (loss * weight).sum()


class WeedDetLoss(nn.Module):
    """
    Combined loss: SmoothL1 + GIoU (regression) + VariFocal (classification).
    Uses IACS-style soft labels for VariFocal.
    """
    def __init__(self, num_classes=1, iou_threshold=0.4,
                 neg_iou_threshold=0.3):
        super().__init__()
        self.num_classes       = num_classes
        self.iou_threshold     = iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.giou_loss         = GIoULoss()
        self.varifocal         = VariFocalLoss()

    def encode(self, anchors, gt_boxes):
        """Encode ground truth boxes as offsets from anchors."""
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
        ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
        gw = gt_boxes[:, 2] - gt_boxes[:, 0]
        gh = gt_boxes[:, 3] - gt_boxes[:, 1]
        gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
        gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
        dx = (gx - ax) / (aw + 1e-6)
        dy = (gy - ay) / (ah + 1e-6)
        dw = torch.log(gw / (aw + 1e-6) + 1e-6)
        dh = torch.log(gh / (ah + 1e-6) + 1e-6)
        return torch.stack([dx, dy, dw, dh], dim=1)

    def decode(self, anchors, deltas):
        """Decode predicted deltas back to absolute boxes."""
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
        ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
        px = deltas[:, 0] * aw + ax
        py = deltas[:, 1] * ah + ay
        pw = torch.exp(deltas[:, 2].clamp(max=4)) * aw
        ph = torch.exp(deltas[:, 3].clamp(max=4)) * ah
        return torch.stack([px - pw/2, py - ph/2,
                            px + pw/2, py + ph/2], dim=1)

    def forward(self, cls_logits, regs, anchors, targets):
        B          = len(targets)
        cls_flat   = [torch.cat([
            c.permute(0,2,3,1).reshape(B, -1, self.num_classes)
            for c in cls_logits], dim=1)]  # [B, A, C]
        reg_flat   = torch.cat([
            r.permute(0,2,3,1).reshape(B, -1, 4)
            for r in regs], dim=1)          # [B, A, 4]
        cls_flat   = cls_flat[0]

        total_cls  = torch.tensor(0.0, device=anchors.device)
        total_reg  = torch.tensor(0.0, device=anchors.device)
        num_pos    = 0

        for b in range(B):
            gt_boxes  = targets[b]['boxes'].to(anchors.device)
            gt_labels = targets[b]['labels'].to(anchors.device)

            iacs = torch.zeros(len(anchors), self.num_classes,
                               device=anchors.device)

            if len(gt_boxes) == 0:
                valid     = torch.ones(len(anchors), dtype=torch.bool,
                                       device=anchors.device)
                total_cls = total_cls + self.varifocal(cls_flat[b], iacs)
                continue

            ious          = box_iou(anchors, gt_boxes)   # [A, G]
            max_iou, best = ious.max(dim=1)

            pos = max_iou >= self.iou_threshold
            neg = max_iou <  self.neg_iou_threshold
            ign = ~pos & ~neg

            pos_idx  = pos.nonzero(as_tuple=True)[0]
            num_pos += len(pos_idx)

            if len(pos_idx) > 0:
                matched_gt  = gt_boxes[best[pos_idx]]
                tgt_deltas  = self.encode(anchors[pos_idx], matched_gt)
                pred_deltas = reg_flat[b][pos_idx]
                pred_boxes  = self.decode(anchors[pos_idx], pred_deltas)

                smooth_l1   = F.smooth_l1_loss(pred_deltas, tgt_deltas,
                                               reduction='sum')
                total_reg   = total_reg + smooth_l1 + \
                              self.giou_loss(pred_boxes, matched_gt) * len(pos_idx)

                matched_cls = gt_labels[best[pos_idx]].clamp(0, self.num_classes-1)
                iacs[pos_idx, matched_cls] = max_iou[pos_idx]

            valid     = ~ign
            total_cls = total_cls + self.varifocal(cls_flat[b][valid],
                                                   iacs[valid])

        norm = max(num_pos, 1)
        return {
            'cls_loss'  : total_cls / norm,
            'reg_loss'  : total_reg / norm,
            'total_loss': (total_cls + total_reg) / norm,
        }


# ===========================================================================
# SECTION 6 — FULL MODEL
# ===========================================================================

class WeedDet(nn.Module):
    """
    WeedDet: complete one-stage paddy weed detector.
    Single-class rice config for AgriNav (inverted spray logic).
    """
    CLASS_NAMES = ['Rice']

    def __init__(self, num_classes=1, anchor_base_scale=4, lsc_k=7):
        super().__init__()
        self.num_classes = num_classes
        self.backbone    = DetResNet50()
        self.fpn         = eFPN(512, 1024, 2048, 256)
        self.head        = ERetinaHead(256, num_classes, 9, lsc_k)
        self.anchor_gen  = AnchorGenerator(
            base_scale=anchor_base_scale,
            aspect_ratios=(0.5, 1.0, 2.0),
            scales=(1.0, 2**(1/3), 2**(2/3)),
            strides=(8, 16, 32),
        )
        self.criterion = WeedDetLoss(num_classes=num_classes)

    def forward(self, images, targets=None):
        c3, c4, c5       = self.backbone(images)
        p3, p4, p5       = self.fpn(c3, c4, c5)
        features         = [p3, p4, p5]
        cls_logits, regs = self.head(features)
        anchors          = self.anchor_gen(features, images.shape[-2:])

        if self.training:
            assert targets is not None, "targets required in train mode"
            return self.criterion(cls_logits, regs, anchors, targets)

        return self._decode(cls_logits, regs, anchors, images.shape[-2:])

    def _decode(self, cls_logits, regs, anchors, img_shape,
                score_thr=0.05, nms_thr=0.3, max_dets=300):
        B     = cls_logits[0].shape[0]
        C     = self.num_classes
        cls_f = torch.cat([
            c.permute(0,2,3,1).reshape(B,-1,C) for c in cls_logits], dim=1)
        reg_f = torch.cat([
            r.permute(0,2,3,1).reshape(B,-1,4) for r in regs], dim=1)

        H, W = img_shape
        results = []
        for b in range(B):
            scores, labels = cls_f[b].sigmoid().max(dim=1)
            keep   = scores > score_thr
            scores = scores[keep]; labels = labels[keep]
            deltas = reg_f[b][keep]
            anch   = anchors[keep]

            aw = anch[:,2]-anch[:,0]; ah = anch[:,3]-anch[:,1]
            ax = (anch[:,0]+anch[:,2])*0.5; ay = (anch[:,1]+anch[:,3])*0.5
            px = deltas[:,0]*aw+ax; py = deltas[:,1]*ah+ay
            pw = torch.exp(deltas[:,2].clamp(max=4))*aw
            ph = torch.exp(deltas[:,3].clamp(max=4))*ah
            boxes = torch.stack([
                (px-pw/2).clamp(0,W), (py-ph/2).clamp(0,H),
                (px+pw/2).clamp(0,W), (py+ph/2).clamp(0,H)], dim=1)

            keep2  = nms(boxes, scores, nms_thr)[:max_dets]
            results.append({
                'boxes' : boxes[keep2],
                'scores': scores[keep2],
                'labels': labels[keep2],
            })
        return results


# ===========================================================================
# SECTION 7 — DATASET: WeedDataset (PASCAL VOC XML)
# ===========================================================================

class WeedDataset(Dataset):
    """
    PASCAL VOC XML dataset loader for WeedDet.

    Expected structure:
      root/
        images/        ← .jpg or .png
        annotations/   ← .xml (PASCAL VOC format)
        train.txt      ← image stems, one per line
        val.txt

    Bounding box coordinates are rescaled in __getitem__ to match the
    resized image dimensions. Do NOT apply additional scaling externally.
    """

    CLASS_NAMES = ['Rice']

    def __init__(self, root, split='train',
                 img_size=(600, 1000), augment=False):
        from PIL import Image as PILImage
        self.PILImage = PILImage
        self.root     = root
        self.img_size = img_size   # (H, W)
        self.augment  = augment
        self.class_to_idx = {n: i for i, n in enumerate(self.CLASS_NAMES)}

        split_file = os.path.join(root, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.ids = [l.strip() for l in f if l.strip()]
        else:
            ann_dir = os.path.join(root, 'annotations')
            self.ids = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(ann_dir) if f.endswith('.xml')
            )

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def _parse_xml(self, xml_path):
        """Parse PASCAL VOC XML, return boxes [N,4] xyxy and labels [N]."""
        try:
            tree = ET.parse(xml_path)
        except Exception:
            return torch.zeros((0,4), dtype=torch.float32), \
                   torch.zeros((0,), dtype=torch.int64), 0, 0

        root = tree.getroot()
        size = root.find('size')
        orig_w = int(size.find('width').text)  if size is not None else 0
        orig_h = int(size.find('height').text) if size is not None else 0

        boxes, labels = [], []
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is None:
                continue
            cls_name = name.text.strip()
            # Map any rice-related name to index 0
            if cls_name.lower() in {'rice', 'rice plant', 'oryza sativa'}:
                cls_idx = 0
            elif cls_name in self.class_to_idx:
                cls_idx = self.class_to_idx[cls_name]
            else:
                continue  # skip unknown classes

            bb = obj.find('bndbox')
            if bb is None:
                continue
            x1 = float(bb.find('xmin').text)
            y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text)
            y2 = float(bb.find('ymax').text)
            if x2 - x1 > 1 and y2 - y1 > 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_idx)

        if boxes:
            return (torch.tensor(boxes,  dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.int64),
                    orig_w, orig_h)
        return (torch.zeros((0,4), dtype=torch.float32),
                torch.zeros((0,),  dtype=torch.int64),
                orig_w, orig_h)

    def _augment(self, img, boxes):
        """Apply augmentations: flip, translate, brightness, blur, equalize."""
        import random
        from PIL import ImageFilter, ImageEnhance, ImageOps, ImageChops
        w, h = img.size

        if random.random() > 0.5:
            img = img.transpose(self.PILImage.FLIP_LEFT_RIGHT)
            if len(boxes):
                boxes = boxes.clone()
                boxes[:, [0,2]] = w - boxes[:, [2,0]]

        if random.random() > 0.5 and len(boxes):
            dx = int(random.uniform(-0.1, 0.1) * w)
            dy = int(random.uniform(-0.1, 0.1) * h)
            img = ImageChops.offset(img, dx, dy)
            boxes = boxes.clone()
            boxes[:, [0,2]] = (boxes[:, [0,2]] + dx).clamp(0, w)
            boxes[:, [1,3]] = (boxes[:, [1,3]] + dy).clamp(0, h)

        if random.random() > 0.5:
            img = ImageEnhance.Brightness(img).enhance(
                random.uniform(0.7, 1.3))

        if random.random() > 0.3:
            img = img.filter(
                ImageFilter.GaussianBlur(random.uniform(0, 1.0)))

        if random.random() > 0.5:
            img = ImageOps.equalize(img)

        return img, boxes

    def __getitem__(self, idx):
        stem    = self.ids[idx]
        img_dir = os.path.join(self.root, 'images')
        ann_dir = os.path.join(self.root, 'annotations')

        img_path = os.path.join(img_dir, stem + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, stem + '.png')
        if not os.path.exists(img_path):
            return None

        img = self.PILImage.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        boxes, labels, xml_w, xml_h = self._parse_xml(
            os.path.join(ann_dir, stem + '.xml'))

        if self.augment and len(boxes):
            img, boxes = self._augment(img, boxes)

        tH, tW = self.img_size
        img = img.resize((tW, tH), self.PILImage.BILINEAR)

        # Scale boxes to match resized image — CRITICAL
        if len(boxes):
            boxes = boxes.clone()
            boxes[:, [0,2]] *= tW / orig_w
            boxes[:, [1,3]] *= tH / orig_h

        return self.transform(img), {
            'boxes'  : boxes,
            'labels' : labels,
            'orig_h' : orig_h,
            'orig_w' : orig_w,
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    imgs = torch.stack([b[0] for b in batch])
    tgts = [b[1] for b in batch]
    return imgs, tgts


# ===========================================================================
# SECTION 8 — SCHEDULER
# ===========================================================================

class WarmupMultiStepLR:
    """
    Linear LR warmup for first `warmup_iters` steps.
    Used together with MultiStepLR for epoch-level decay.
    """
    def __init__(self, optimizer, warmup_iters=500, warmup_factor=0.001):
        self.opt          = optimizer
        self.warmup_iters = warmup_iters
        self.factor       = warmup_factor
        self._iter        = 0
        self._base_lrs    = [g['lr'] for g in optimizer.param_groups]

    def step_iter(self):
        self._iter += 1
        if self._iter <= self.warmup_iters:
            alpha = self._iter / self.warmup_iters
            lr_f  = self.factor + (1 - self.factor) * alpha
            for g, base in zip(self.opt.param_groups, self._base_lrs):
                g['lr'] = base * lr_f

    @property
    def warmup_done(self):
        return self._iter >= self.warmup_iters


# ===========================================================================
# SECTION 9 — TRAINING LOOP
# ===========================================================================

def train(config):
    """
    Original training loop (no tqdm). Use train_with_progress for Colab.
    """
    device = torch.device(
        config.get('device', 'cuda')
        if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = WeedDet(num_classes=config.get('num_classes', 1)).to(device)

    if config.get('freeze_bn', True):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        print("BatchNorm frozen.")

    img_size = config.get('img_size', (600, 1000))
    train_ds = WeedDataset(config['data_root'], 'train', img_size, augment=True)
    val_ds   = WeedDataset(config['data_root'], 'val',   img_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=config.get('batch_size', 2),
        shuffle=True, collate_fn=collate_fn,
        num_workers=config.get('num_workers', 2), pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('base_lr', 0.01),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 0.0001))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.get('lr_decay_epochs', [8, 11]),
        gamma=0.1)

    warmup   = WarmupMultiStepLR(optimizer,
                                 warmup_iters=config.get('warmup_iters', 500))
    ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss  = float('inf')
    num_epochs = config.get('num_epochs', 12)

    for epoch in range(1, num_epochs + 1):
        model.train()
        if config.get('freeze_bn', True):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        epoch_loss = 0.0
        n_batches  = 0

        for repeat in range(2):   # 2× dataset repeat per epoch (paper)
            for i, (imgs, tgts) in enumerate(train_loader):
                if not isinstance(imgs, torch.Tensor):
                    continue
                imgs = imgs.to(device)
                tgts = [{k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in t.items()} for t in tgts]

                optimizer.zero_grad()
                losses = model(imgs, tgts)
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if not warmup.warmup_done:
                    warmup.step_iter()

                epoch_loss += losses['total_loss'].item()
                n_batches  += 1

                if i % 50 == 0:
                    print(f"[E{epoch:02d}/{num_epochs}][R{repeat+1}/2]"
                          f"[B{i:04d}/{len(train_loader)}] "
                          f"loss={losses['total_loss'].item():.4f} "
                          f"cls={losses['cls_loss'].item():.4f} "
                          f"reg={losses['reg_loss'].item():.4f} "
                          f"lr={optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch:02d} avg loss: {avg_loss:.4f}\n")
        lr_scheduler.step()   # AFTER the batch loop

        if avg_loss < best_loss:
            best_loss = avg_loss
            path = os.path.join(ckpt_dir, 'weeddet_best.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': best_loss, 'config': config}, path)
            print(f"  Best model saved -> {path}")

        if epoch % 4 == 0:
            path = os.path.join(ckpt_dir, f'weeddet_epoch{epoch}.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'loss': avg_loss, 'config': config}, path)

    print("Training complete.")
    return model


def train_with_progress(config):
    """
    Training loop with tqdm progress bars — preferred for Colab/Kaggle.
    Identical to train() but with per-batch and per-epoch progress bars.
    """
    device = torch.device(
        config.get('device', 'cuda')
        if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    model = WeedDet(num_classes=config.get('num_classes', 1)).to(device)

    if config.get('freeze_bn', True):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    img_size = config.get('img_size', (600, 1000))
    train_ds = WeedDataset(config['data_root'], 'train', img_size, augment=True)
    train_loader = DataLoader(
        train_ds, batch_size=config.get('batch_size', 2),
        shuffle=True, collate_fn=collate_fn,
        num_workers=config.get('num_workers', 2), pin_memory=True)

    print(f"Train: {len(train_ds)} images | {len(train_loader)} batches/epoch")

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('base_lr', 0.01),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 0.0001))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.get('lr_decay_epochs', [8, 11]),
        gamma=0.1)

    warmup   = WarmupMultiStepLR(optimizer,
                                 warmup_iters=config.get('warmup_iters', 500))
    ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss  = float('inf')
    num_epochs = config.get('num_epochs', 12)

    epoch_bar = tqdm(range(1, num_epochs + 1), desc='Epochs') \
                if TQDM_AVAILABLE else range(1, num_epochs + 1)

    for epoch in epoch_bar:
        model.train()
        if config.get('freeze_bn', True):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        epoch_loss = 0.0
        n_batches  = 0

        for repeat in range(2):
            batch_iter = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f'E{epoch:02d} R{repeat+1}/2',
                leave=False
            ) if TQDM_AVAILABLE else enumerate(train_loader)

            for i, (imgs, tgts) in batch_iter:
                if not isinstance(imgs, torch.Tensor):
                    continue
                imgs = imgs.to(device)
                tgts = [{k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in t.items()} for t in tgts]

                optimizer.zero_grad()
                losses = model(imgs, tgts)
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if not warmup.warmup_done:
                    warmup.step_iter()

                l = losses['total_loss'].item()
                epoch_loss += l
                n_batches  += 1

                if TQDM_AVAILABLE:
                    batch_iter.set_postfix(
                        loss=f'{l:.4f}',
                        cls=f'{losses["cls_loss"].item():.4f}',
                        reg=f'{losses["reg_loss"].item():.4f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.5f}')

        avg_loss = epoch_loss / max(n_batches, 1)
        if TQDM_AVAILABLE:
            epoch_bar.set_postfix(avg_loss=f'{avg_loss:.4f}',
                                  best=f'{best_loss:.4f}')
        print(f"\nEpoch {epoch:02d}/{num_epochs}  avg_loss={avg_loss:.4f}  "
              f"best={best_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

        lr_scheduler.step()   # once per epoch — NOT inside batch loop

        if avg_loss < best_loss:
            best_loss = avg_loss
            path = os.path.join(ckpt_dir, 'weeddet_best.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': best_loss, 'config': config}, path)
            print(f"  ★ Best checkpoint -> {path}")

        if epoch % 4 == 0:
            path = os.path.join(ckpt_dir, f'weeddet_epoch{epoch}.pth')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'loss': avg_loss, 'config': config}, path)
            print(f"  Checkpoint saved  -> {path}")

    print("\nTraining complete.")
    return model


# ===========================================================================
# SECTION 10 — INFERENCE UTILITIES
# ===========================================================================

def load_model(path, device='cuda'):
    """Load a trained WeedDet checkpoint."""
    ckpt  = torch.load(path, map_location=device)
    cfg   = ckpt.get('config', {})
    model = WeedDet(num_classes=cfg.get('num_classes', 1))
    key   = 'state_dict' if 'state_dict' in ckpt else 'model_state_dict'
    model.load_state_dict(ckpt[key])
    model.to(device).eval()
    print(f"Loaded epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    return model


def predict(model, image_path, img_size=(600, 1000),
            score_thr=0.5, nms_thr=0.5, device='cuda'):
    """Run WeedDet on a single image. Returns list of detection dicts."""
    from PIL import Image as PILImage
    tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img      = PILImage.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    x        = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        results = model(x)

    preds    = results[0]
    tH, tW  = img_size
    dets     = []
    for i in range(len(preds['boxes'])):
        sc = preds['scores'][i].item()
        if sc < score_thr:
            continue
        box = preds['boxes'][i].cpu().tolist()
        dets.append({
            'box'  : [box[0]*orig_w/tW, box[1]*orig_h/tH,
                      box[2]*orig_w/tW, box[3]*orig_h/tH],
            'score': sc,
            'label': WeedDet.CLASS_NAMES[preds['labels'][i].item()]
                     if preds['labels'][i].item() < len(WeedDet.CLASS_NAMES)
                     else 'unknown',
        })
    return dets


# ===========================================================================
# QUICK BUILD CHECK
# ===========================================================================

if __name__ == '__main__':
    import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = WeedDet(num_classes=1).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.2f}M  (paper: ~24.33M)")

    # Training forward pass
    model.train()
    imgs = torch.randn(2, 3, 600, 1000).to(device)
    tgts = [
        {'boxes' : torch.tensor([[100,100,300,400]], dtype=torch.float32).to(device),
         'labels': torch.tensor([0], dtype=torch.int64).to(device),
         'orig_h': 600, 'orig_w': 1000},
        {'boxes' : torch.zeros((0,4), dtype=torch.float32).to(device),
         'labels': torch.zeros((0,),  dtype=torch.int64).to(device),
         'orig_h': 600, 'orig_w': 1000},
    ]
    losses = model(imgs, tgts)
    print(f"Train loss: total={losses['total_loss'].item():.4f} "
          f"cls={losses['cls_loss'].item():.4f} "
          f"reg={losses['reg_loss'].item():.4f}")

    # Inference forward pass + FPS
    model.eval()
    x = torch.randn(1, 3, 600, 1000).to(device)
    with torch.no_grad():
        model(x)  # warm-up
        t0 = time.time()
        for _ in range(30):
            model(x)
        fps = 30 / (time.time() - t0)
    print(f"Inference FPS: {fps:.1f}  (paper target: 24.3 FPS)")
    print("Build check PASSED ✓")
