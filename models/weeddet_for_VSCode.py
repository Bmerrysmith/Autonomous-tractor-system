"""
WeedDet — Full Implementation from Scratch
Based on: Peng et al. (2022) "Weed Detection in Paddy Field Using an
Improved RetinaNet Network", Computers and Electronics in Agriculture,
vol. 199, p. 107179.

Architecture overview:
  Input RGB image (1000x600)
      |
  Det-ResNet-50 backbone  <-- replaces standard ResNet-50 first block
      | C3, C4, C5 feature maps
  eFPN (efficient FPN)    <-- only P3, P4, P5 (no P6/P7)
      | three scale feature maps
  ERetina-Head            <-- 1x conv, 64ch, large separable convolution
      | classification + regression subnetworks
  Predictions             <-- bounding boxes + 9-class labels + IACS scores

Loss functions:
  Regression : SmoothL1 + GIoU (combined)
  Classification : VariFocal Loss (replaces Focal Loss)

Training:
  Optimizer  : SGD, momentum=0.9, weight_decay=0.0001
  Batch size : 2
  Epochs     : 12 (1x scheduler)
  LR warmup  : linear over first 500 iterations
  LR decay   : x0.1 at epoch 8 and epoch 11
  Hardware   : single NVIDIA GPU (paper used GTX TITAN X 12GB)
  Labels     : PASCAL VOC format XML

1 Classes:
  0 = Rice
"""


import math
import os
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms, box_iou


# ---------------------------------------------------------------------------
# SECTION 1 — BACKBONE: Det-ResNet-50
# ---------------------------------------------------------------------------
# The paper's key modification to ResNet-50:
#   Standard ResNet-50 starts with:
#     Conv2d(3, 64, 7x7, stride=2) -> BN -> ReLU -> MaxPool2d(3, stride=2)
#     This immediately reduces spatial resolution to 1/4 of input.
#   Det-ResNet-50 replaces this with:
#     Conv2d(3, 16, 3x3, stride=1) -> BN -> ReLU
#     Conv2d(16, 16, 3x3, stride=1) -> BN -> ReLU
#     DetResidualBlock(16 -> 32)
#   This preserves early spatial detail critical for small object detection.
# ---------------------------------------------------------------------------

class DetResidualBlock(nn.Module):
    """
    Initial residual block in Det-ResNet.
    Replaces the max-pooling downsampling of the standard ResNet stem.
    Channels: 16 -> 32, stride=2 for controlled spatial reduction.
    """
    def __init__(self, in_channels=16, out_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """Standard ResNet bottleneck block. Expansion=4."""
    expansion = 4

    def __init__(self, in_channels, mid_channels,
                 stride=1, downsample=None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class DetResNet50(nn.Module):
    """
    Det-ResNet-50 backbone.

    Key change from standard ResNet-50:
      Standard stem: Conv(7x7, stride=2) + MaxPool -> resolution /4
      Det stem:      Conv(3x3) + Conv(3x3) + DetResidualBlock -> resolution /2
                     This keeps more spatial detail for detection.

    Output feature maps:
      C3: stride 8  (layer2 output, 512 channels)
      C4: stride 16 (layer3 output, 1024 channels)
      C5: stride 32 (layer4 output, 2048 channels)
    """

    def __init__(self):
        super().__init__()

        # Modified stem: two 3x3 convs + residual block
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # Det residual block: stride=2, 16 -> 32 channels
        self.det_block = DetResidualBlock(16, 32)

        # Standard ResNet-50 bottleneck layers
        # layer1: 32  -> 256  channels, 3 blocks, stride=1
        # layer2: 256 -> 512  channels, 4 blocks, stride=2 -> C3
        # layer3: 512 -> 1024 channels, 6 blocks, stride=2 -> C4
        # layer4: 1024-> 2048 channels, 3 blocks, stride=2 -> C5
        self.layer1 = self._make_layer(32,   64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(256,  128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512,  256, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        self._init_weights()

    def _make_layer(self, in_channels, mid_channels, blocks, stride):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [Bottleneck(in_channels, mid_channels,
                             stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, mid_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.det_block(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


# ---------------------------------------------------------------------------
# SECTION 2 — NECK: eFPN (Efficient Feature Pyramid Network)
# ---------------------------------------------------------------------------
# Standard RetinaNet FPN: P3-P7 (5 levels) for MS COCO extreme scales.
# Paddy weed targets have consistent scale -> P6, P7 are unnecessary.
# eFPN removes P6 and P7, keeping only P3, P4, P5.
# This saves ~7.71M parameters and +2.8 FPS with no accuracy loss.
# Anchor base scale = 6 (paper ablation Figure 6).
# ---------------------------------------------------------------------------

class eFPN(nn.Module):
    """
    Efficient Feature Pyramid Network.
    Produces P3, P4, P5 only. No P6, P7.
    All output levels have feature_size=256 channels.
    """

    def __init__(self, c3_channels=512, c4_channels=1024,
                 c5_channels=2048, feature_size=256):
        super().__init__()

        # 1x1 lateral convolutions to unify channel depth
        self.lateral_c5 = nn.Conv2d(c5_channels, feature_size,
                                    kernel_size=1, bias=False)
        self.lateral_c4 = nn.Conv2d(c4_channels, feature_size,
                                    kernel_size=1, bias=False)
        self.lateral_c3 = nn.Conv2d(c3_channels, feature_size,
                                    kernel_size=1, bias=False)

        # 3x3 output convolutions after top-down merging
        self.output_p5 = nn.Conv2d(feature_size, feature_size,
                                   kernel_size=3, padding=1, bias=False)
        self.output_p4 = nn.Conv2d(feature_size, feature_size,
                                   kernel_size=3, padding=1, bias=False)
        self.output_p3 = nn.Conv2d(feature_size, feature_size,
                                   kernel_size=3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, c3, c4, c5):
        # Lateral projections
        lat5 = self.lateral_c5(c5)
        lat4 = self.lateral_c4(c4)
        lat3 = self.lateral_c3(c3)

        # Top-down pathway
        p5 = self.output_p5(lat5)

        up5 = F.interpolate(lat5, size=lat4.shape[-2:], mode='nearest')
        p4 = self.output_p4(lat4 + up5)

        up4 = F.interpolate(lat4 + up5, size=lat3.shape[-2:],
                            mode='nearest')
        p3 = self.output_p3(lat3 + up4)

        return p3, p4, p5


# ---------------------------------------------------------------------------
# SECTION 3 — DETECTION HEAD: ERetina-Head
# ---------------------------------------------------------------------------
# Standard RetinaNet head: 4x(Conv 3x3, 256ch) per subnet — 80 COCO classes
# ERetina-Head: 1x(Conv 64ch) + Large Separable Convolution
#
# Large Separable Convolution (LSC):
#   kxk -> (kx1) then (1xk)
#   Expands receptive field without quadratic parameter cost.
#   Avoids grid effect of dilated convolution.
#
# Two parallel subnetworks (shared weights across FPN levels):
#   Classification -> (B, num_anchors * num_classes, H, W)
#   Regression     -> (B, num_anchors * 4, H, W)
# ---------------------------------------------------------------------------

class LargeSeparableConv(nn.Module):
    """
    Large Separable Convolution: decomposes kxk into (kx1) + (1xk).
    Expands receptive field at low parameter cost.
    """
    def __init__(self, channels, k=7):
        super().__init__()
        self.conv_h = nn.Conv2d(
            channels, channels,
            kernel_size=(k, 1), padding=(k // 2, 0), bias=False
        )
        self.conv_v = nn.Conv2d(
            channels, channels,
            kernel_size=(1, k), padding=(0, k // 2), bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv_v(self.conv_h(x))))


class ERetinaHead(nn.Module):
    """
    Efficient Retina Head.

    Per-subnet architecture:
      Conv2d(256 -> 64, 3x3) -> BN -> ReLU
      LargeSeparableConv(64, k=7)
      Conv2d(64 -> output_channels, 1x1)

    Weights are shared across all FPN levels (P3, P4, P5).
    """

    def __init__(self, in_channels=256, num_classes=1,
                 num_anchors=9, lsc_k=7):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification subnet
        self.cls_conv = nn.Conv2d(in_channels, 64,
                                  kernel_size=3, padding=1, bias=False)
        self.cls_bn   = nn.BatchNorm2d(64)
        self.cls_lsc  = LargeSeparableConv(64, k=lsc_k)
        self.cls_pred = nn.Conv2d(64, num_anchors * num_classes,
                                  kernel_size=1)

        # Regression subnet
        self.reg_conv = nn.Conv2d(in_channels, 64,
                                  kernel_size=3, padding=1, bias=False)
        self.reg_bn   = nn.BatchNorm2d(64)
        self.reg_lsc  = LargeSeparableConv(64, k=lsc_k)
        self.reg_pred = nn.Conv2d(64, num_anchors * 4,
                                  kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        # Prior probability init for classification (from RetinaNet)
        # Prevents exploding loss at training start
        prior_prob = 0.01
        bias_val = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_val)
        nn.init.normal_(self.cls_pred.weight,  std=0.01)
        nn.init.normal_(self.reg_pred.weight,  std=0.01)
        nn.init.constant_(self.reg_pred.bias,  0.0)
        for m in [self.cls_conv, self.reg_conv]:
            nn.init.normal_(m.weight, std=0.01)

    def forward(self, features):
        """
        Args:
            features: [p3, p4, p5] list of feature maps
        Returns:
            cls_logits: list of (B, num_anchors*num_classes, H, W)
            reg_deltas: list of (B, num_anchors*4, H, W)
        """
        cls_logits, reg_deltas = [], []
        for feat in features:
            # Classification branch
            c = self.relu(self.cls_bn(self.cls_conv(feat)))
            c = self.cls_lsc(c)
            cls_logits.append(self.cls_pred(c))
            # Regression branch
            r = self.relu(self.reg_bn(self.reg_conv(feat)))
            r = self.reg_lsc(r)
            reg_deltas.append(self.reg_pred(r))
        return cls_logits, reg_deltas


# ---------------------------------------------------------------------------
# SECTION 4 — ANCHOR GENERATOR
# ---------------------------------------------------------------------------
# eFPN uses base_scale=6 (paper ablation Figure 6, best among 4-8).
# 3 aspect_ratios x 3 scales = 9 anchors per location per level.
# Strides: P3=8, P4=16, P5=32
# ---------------------------------------------------------------------------

class AnchorGenerator(nn.Module):
    """
    Generates anchor boxes for P3, P4, P5.
    base_scale=6 from paper ablation.
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
        self.num_anchors   = len(aspect_ratios) * len(scales)

    def _anchors_for_level(self, stride, h, w, device):
        base_size = stride * self.base_scale
        templates = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                bw = base_size * scale / math.sqrt(ratio)
                bh = base_size * scale * math.sqrt(ratio)
                templates.append([-bw/2, -bh/2, bw/2, bh/2])

        templates = torch.tensor(templates, dtype=torch.float32,
                                 device=device)

        # Grid of anchor centers
        cx = (torch.arange(w, device=device).float() + 0.5) * stride
        cy = (torch.arange(h, device=device).float() + 0.5) * stride
        gy, gx = torch.meshgrid(cy, cx, indexing='ij')
        shifts = torch.stack([gx.reshape(-1), gy.reshape(-1),
                               gx.reshape(-1), gy.reshape(-1)], dim=1)

        # (H*W, 1, 4) + (1, num_anchors, 4) -> (H*W*num_anchors, 4)
        anchors = (shifts.unsqueeze(1) +
                   templates.unsqueeze(0)).reshape(-1, 4)
        return anchors

    def forward(self, feature_maps, image_size):
        all_anchors = []
        for feat, stride in zip(feature_maps, self.strides):
            h, w = feat.shape[-2:]
            all_anchors.append(
                self._anchors_for_level(stride, h, w, feat.device)
            )
        return torch.cat(all_anchors, dim=0)


# ---------------------------------------------------------------------------
# SECTION 5 — LOSS FUNCTIONS
# ---------------------------------------------------------------------------

class SmoothL1Loss(nn.Module):
    """
    Smooth L1 / Huber Loss for bounding box regression offsets.
    L2 behaviour for small errors (stable gradients),
    L1 behaviour for large errors (outlier robustness).
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.beta,
            0.5 * diff.pow(2) / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.sum()


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss.
    Treats all 4 box coordinates as a unit — models their joint geometry.
    Provides non-zero gradient even when predicted and GT boxes
    do not overlap (where standard IoU loss gives zero gradient).

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C = smallest box enclosing both A and B.
    Loss = 1 - GIoU
    """

    def forward(self, pred, target):
        """
        Args:
            pred, target: (N, 4) in [x1, y1, x2, y2] format
        """
        # Intersection
        ix1 = torch.max(pred[:, 0], target[:, 0])
        iy1 = torch.max(pred[:, 1], target[:, 1])
        ix2 = torch.min(pred[:, 2], target[:, 2])
        iy2 = torch.min(pred[:, 3], target[:, 3])
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

        # Areas
        pred_area = (
            (pred[:, 2] - pred[:, 0]).clamp(0) *
            (pred[:, 3] - pred[:, 1]).clamp(0)
        )
        tgt_area = (
            (target[:, 2] - target[:, 0]).clamp(0) *
            (target[:, 3] - target[:, 1]).clamp(0)
        )
        union = pred_area + tgt_area - inter + 1e-7
        iou   = inter / union

        # Enclosing box
        ex1 = torch.min(pred[:, 0], target[:, 0])
        ey1 = torch.min(pred[:, 1], target[:, 1])
        ex2 = torch.max(pred[:, 2], target[:, 2])
        ey2 = torch.max(pred[:, 3], target[:, 3])
        encl = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0) + 1e-7

        giou = iou - (encl - union) / encl
        return (1 - giou).sum()


class VariFocalLoss(nn.Module):
    """
    VariFocal Loss — classification loss for WeedDet.

    Improvement over Focal Loss:
      Focal Loss trains classification and localization quality branches
      independently but inference multiplies them — a train/test mismatch.
      VariFocal Loss resolves this with IoU-Aware Classification Score (IACS):
        target = IoU(pred_box, gt_box) for positives (not just 1/0)
        target = 0 for negatives

    Positive samples (q > 0):
      VFL = -q * (q*log(p) + (1-q)*log(1-p))
    Negative samples (q = 0):
      VFL = -alpha * p^gamma * log(1-p)

    From paper Table 9:
      Focal Loss alone    -> 88.6% mAP
      VariFocal Loss alone -> 89.1% mAP (+0.5%)
      VFL + GIoU combined -> 94.1% mAP
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, targets_iacs):
        """
        Args:
            pred_logits:  (N, C) raw logits
            targets_iacs: (N, C) soft targets (IoU for positives, 0 for negatives)
        """
        p = pred_logits.sigmoid()

        pos_mask = (targets_iacs > 0).float()

        # Positive: -q * (q*log(p) + (1-q)*log(1-p))
        pos_loss = -targets_iacs * (
            targets_iacs.clamp(min=1e-8).log() +
            F.binary_cross_entropy_with_logits(
                pred_logits, targets_iacs, reduction='none'
            )
        )

        # Negative: -alpha * p^gamma * log(1-p)
        neg_loss = (
            -self.alpha *
            p.pow(self.gamma) *
            (1 - p).clamp(min=1e-8).log()
        )

        loss = pos_mask * pos_loss + (1 - pos_mask) * neg_loss
        return loss.sum()


class WeedDetLoss(nn.Module):
    """
    Combined multi-task loss.

    Regression    : SmoothL1 + GIoU  (combined in Equation 1 of paper)
    Classification: VariFocal Loss    (replaces Focal Loss)

    Anchor assignment:
      Positive : max_iou >= iou_threshold (0.5)
      Negative : max_iou <  neg_iou_threshold (0.4)
      Ignored  : 0.4 <= max_iou < 0.5
    """

    def __init__(self, num_classes=1, alpha_vfl=0.75, gamma_vfl=2.0,
                 iou_threshold=0.5, neg_iou_threshold=0.4):
        super().__init__()
        self.num_classes       = num_classes
        self.iou_threshold     = iou_threshold
        self.neg_iou_threshold = neg_iou_threshold

        self.smooth_l1  = SmoothL1Loss()
        self.giou_loss  = GIoULoss()
        self.varifocal  = VariFocalLoss(alpha=alpha_vfl, gamma=gamma_vfl)

    @staticmethod
    def encode(anchors, gt_boxes):
        """Delta encode GT boxes relative to anchors."""
        ax = (anchors[:, 0] + anchors[:, 2]) / 2
        ay = (anchors[:, 1] + anchors[:, 3]) / 2
        aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-8)
        ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-8)
        gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-8)
        gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-8)
        return torch.stack([
            (gx - ax) / aw,
            (gy - ay) / ah,
            torch.log(gw / aw),
            torch.log(gh / ah)
        ], dim=1)

    @staticmethod
    def decode(anchors, deltas):
        """Decode predicted deltas to absolute boxes."""
        ax = (anchors[:, 0] + anchors[:, 2]) / 2
        ay = (anchors[:, 1] + anchors[:, 3]) / 2
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        px = deltas[:, 0] * aw + ax
        py = deltas[:, 1] * ah + ay
        pw = torch.exp(deltas[:, 2].clamp(max=4)) * aw
        ph = torch.exp(deltas[:, 3].clamp(max=4)) * ah
        return torch.stack([px - pw/2, py - ph/2,
                             px + pw/2, py + ph/2], dim=1)

    def forward(self, cls_logits_list, reg_deltas_list,
                anchors, targets):
        """
        Args:
            cls_logits_list : list of (B, A*C, H, W) per FPN level
            reg_deltas_list : list of (B, A*4, H, W) per FPN level
            anchors         : (total_A, 4) all anchors
            targets         : list of {boxes:(N,4), labels:(N,)} per image
        """
        B = len(targets)

        # Flatten all FPN levels into (B, total_A, ...)
        cls_flat, reg_flat = [], []
        for cl, rl in zip(cls_logits_list, reg_deltas_list):
            b, _, H, W = cl.shape
            cls_flat.append(
                cl.permute(0,2,3,1).reshape(b, -1, self.num_classes)
            )
            reg_flat.append(
                rl.permute(0,2,3,1).reshape(b, -1, 4)
            )
        cls_flat = torch.cat(cls_flat, dim=1)  # (B, total_A, C)
        reg_flat = torch.cat(reg_flat, dim=1)  # (B, total_A, 4)

        total_cls = cls_flat.new_zeros(1)
        total_reg = cls_flat.new_zeros(1)
        num_pos   = 0

        for b in range(B):
            gt_boxes  = targets[b]['boxes'].to(anchors.device)
            gt_labels = targets[b]['labels'].to(anchors.device)

            # IACS targets: all zeros initially
            iacs = cls_flat.new_zeros(len(anchors), self.num_classes)

            if len(gt_boxes) == 0:
                total_cls = total_cls + self.varifocal(
                    cls_flat[b], iacs
                )
                continue

            iou_mat   = box_iou(anchors, gt_boxes)  # (total_A, num_gt)
            max_iou, best_gt = iou_mat.max(dim=1)

            pos  = max_iou >= self.iou_threshold
            neg  = max_iou <  self.neg_iou_threshold
            ign  = ~pos & ~neg

            pos_idx = pos.nonzero(as_tuple=True)[0]
            num_pos += len(pos_idx)

            # Regression loss on positive anchors
            if len(pos_idx) > 0:
                matched_gt  = gt_boxes[best_gt[pos_idx]]
                tgt_deltas  = self.encode(anchors[pos_idx], matched_gt)
                pred_deltas = reg_flat[b][pos_idx]
                pred_boxes  = self.decode(anchors[pos_idx], pred_deltas)

                total_reg = total_reg + (
                    self.smooth_l1(pred_deltas, tgt_deltas) +
                    self.giou_loss(pred_boxes, matched_gt)
                )

                # Set IACS targets for positive anchors
                matched_labels = gt_labels[best_gt[pos_idx]]
                matched_ious   = max_iou[pos_idx]
                iacs[pos_idx, matched_labels] = matched_ious

            # Classification loss (exclude ignored anchors)
            valid = ~ign
            total_cls = total_cls + self.varifocal(
                cls_flat[b][valid], iacs[valid]
            )

        norm = max(num_pos, 1)
        return {
            'cls_loss'  : total_cls / norm,
            'reg_loss'  : total_reg / norm,
            'total_loss': (total_cls + total_reg) / norm
        }


# ---------------------------------------------------------------------------
# SECTION 6 — FULL WEEDDET MODEL
# ---------------------------------------------------------------------------

class WeedDet(nn.Module):
    """
    WeedDet: Complete one-stage detector for paddy weed detection.

    Pipeline:
      RGB Image -> Det-ResNet-50 -> eFPN -> ERetina-Head -> Predictions

    Paper results on custom paddy dataset:
      mAP@0.5  : 94.1%
      FPS      : 24.3 (NVIDIA GTX TITAN X)
      Params   : 24.33M
      GFLOPs   : 54.72
    """

    CLASS_NAMES = [
        'Rice'
    ]

    def __init__(self, num_classes=1, anchor_base_scale=4, lsc_k=7):
        super().__init__()
        self.num_classes = num_classes

        self.backbone   = DetResNet50()
        self.fpn        = eFPN(512, 1024, 2048, 256)
        self.head       = ERetinaHead(256, num_classes, 9, lsc_k)
        self.anchor_gen = AnchorGenerator(
            base_scale=anchor_base_scale,
            aspect_ratios=(0.5, 1.0, 2.0),
            scales=(1.0, 2**(1/3), 2**(2/3)),
            strides=(8, 16, 32)
        )
        self.criterion = WeedDetLoss(num_classes=num_classes)

    def forward(self, images, targets=None):
        c3, c4, c5       = self.backbone(images)
        p3, p4, p5       = self.fpn(c3, c4, c5)
        features         = [p3, p4, p5]
        cls_logits, regs = self.head(features)
        anchors          = self.anchor_gen(features, images.shape[-2:])

        if self.training:
            assert targets is not None
            return self.criterion(cls_logits, regs, anchors, targets)

        return self._decode_predictions(cls_logits, regs, anchors,
                                        images.shape)

    def _decode_predictions(self, cls_list, reg_list, anchors,
                             img_shape, score_thr=0.05, nms_thr=0.5,
                             max_dets=100):
        B = img_shape[0]
        H, W = img_shape[-2:]

        cls_flat, reg_flat = [], []
        for cl, rl in zip(cls_list, reg_list):
            b, _, Hf, Wf = cl.shape
            cls_flat.append(
                cl.permute(0,2,3,1).reshape(b, -1, self.num_classes)
            )
            reg_flat.append(
                rl.permute(0,2,3,1).reshape(b, -1, 4)
            )
        cls_flat = torch.cat(cls_flat, dim=1)
        reg_flat = torch.cat(reg_flat, dim=1)

        ax = (anchors[:,0] + anchors[:,2]) / 2
        ay = (anchors[:,1] + anchors[:,3]) / 2
        aw = anchors[:,2] - anchors[:,0]
        ah = anchors[:,3] - anchors[:,1]

        results = []
        for b in range(B):
            d = reg_flat[b]
            px = d[:,0]*aw + ax
            py = d[:,1]*ah + ay
            pw = torch.exp(d[:,2].clamp(max=4)) * aw
            ph = torch.exp(d[:,3].clamp(max=4)) * ah

            boxes = torch.stack([
                (px - pw/2).clamp(0, W),
                (py - ph/2).clamp(0, H),
                (px + pw/2).clamp(0, W),
                (py + ph/2).clamp(0, H)
            ], dim=1)

            scores, labels = cls_flat[b].sigmoid().max(dim=1)
            keep = scores > score_thr
            boxes, scores, labels = (boxes[keep], scores[keep],
                                      labels[keep])

            if len(boxes):
                keep_nms = nms(boxes, scores, nms_thr)[:max_dets]
                boxes  = boxes[keep_nms]
                scores = scores[keep_nms]
                labels = labels[keep_nms]

            results.append({
                'boxes'      : boxes,
                'scores'     : scores,
                'labels'     : labels,
                'class_names': [self.CLASS_NAMES[l]
                                for l in labels.tolist()]
            })
        return results


# ---------------------------------------------------------------------------
# SECTION 7 — DATASET: PASCAL VOC FORMAT
# ---------------------------------------------------------------------------

class WeedDataset(Dataset):
    """
    PASCAL VOC XML dataset for WeedDet.

    Directory layout:
      root/
        images/       <-- .jpg or .png files
        annotations/  <-- PASCAL VOC .xml files
        train.txt     <-- image IDs (one per line, no extension)
        val.txt
    """

    CLASS_NAMES = WeedDet.CLASS_NAMES

    def __init__(self, root, split='train',
                 img_size=(600, 1000), augment=False):
        from PIL import Image
        self.PILImage  = Image
        self.root      = root
        self.img_size  = img_size  # (H, W)
        self.augment   = augment
        self.cls2idx   = {n: i for i, n in enumerate(self.CLASS_NAMES)}

        ann_dir = os.path.join(root, 'annotations')
        split_file = os.path.join(root, f'{split}.txt')

        if os.path.exists(split_file):
            with open(split_file) as f:
                self.ids = [l.strip() for l in f if l.strip()]
        else:
            self.ids = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(ann_dir)
                if f.endswith('.xml')
            )

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

        # Pillow 10+ moved constants into sub-enums; support both
        try:
            self._bilinear = self.PILImage.Resampling.BILINEAR
            self._flip_lr  = self.PILImage.Transpose.FLIP_LEFT_RIGHT
        except AttributeError:
            self._bilinear = self.PILImage.BILINEAR
            self._flip_lr  = self.PILImage.FLIP_LEFT_RIGHT

    def __len__(self):
        return len(self.ids)

    def _parse_xml(self, path):
        root = ET.parse(path).getroot()
        boxes, labels = [], []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            if name not in self.cls2idx:
                continue
            bb = obj.find('bndbox')
            x1 = float(bb.find('xmin').text)
            y1 = float(bb.find('ymin').text)
            x2 = float(bb.find('xmax').text)
            y2 = float(bb.find('ymax').text)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cls2idx[name])
        return (torch.tensor(boxes,  dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long))

    def _augment(self, img, boxes):
        """
        Augmentation from paper Section 2.1.3:
        Horizontal Flip, Random Shift, Random Brightness,
        Gaussian Blur, CLAHE approximation.
        """
        import random
        from PIL import ImageFilter, ImageEnhance, ImageOps, ImageChops

        w, h = img.size

        if random.random() > 0.5:
            img = img.transpose(self._flip_lr)
            if len(boxes):
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        if random.random() > 0.5 and len(boxes):
            dx = int(random.uniform(-0.1, 0.1) * w)
            dy = int(random.uniform(-0.1, 0.1) * h)
            img = ImageChops.offset(img, dx, dy)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] + dx).clamp(0, w)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] + dy).clamp(0, h)

        if random.random() > 0.5:
            img = ImageEnhance.Brightness(img).enhance(
                random.uniform(0.7, 1.3)
            )

        if random.random() > 0.3:
            img = img.filter(
                ImageFilter.GaussianBlur(random.uniform(0, 1.0))
            )

        if random.random() > 0.5:
            img = ImageOps.equalize(img)

        return img, boxes

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_dir  = os.path.join(self.root, 'images')
        ann_dir  = os.path.join(self.root, 'annotations')

        img_path = os.path.join(img_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id + '.png')

        img = self.PILImage.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        boxes, labels = self._parse_xml(
            os.path.join(ann_dir, img_id + '.xml')
        )

        if self.augment and len(boxes):
            img, boxes = self._augment(img, boxes)

        tH, tW = self.img_size
        img = img.resize((tW, tH), self._bilinear)

        if len(boxes):
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= tW / orig_w
            boxes[:, [1, 3]] *= tH / orig_h

        return self.transform(img), {
            'boxes': boxes, 'labels': labels, 'image_id': img_id
        }


def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


# ---------------------------------------------------------------------------
# SECTION 8 — TRAINING
# ---------------------------------------------------------------------------

class WarmupMultiStepLR:
    """
    Linear warmup for first `warmup_iters` steps,
    then MultiStepLR decay at specified epochs.
    """
    def __init__(self, optimizer, warmup_iters=500,
                 warmup_factor=0.001):
        self.opt          = optimizer
        self.warmup_iters = warmup_iters
        self.factor       = warmup_factor
        self._iter        = 0
        self._base_lrs    = [g['lr'] for g in optimizer.param_groups]

    def step_iter(self):
        self._iter += 1
        if self._iter <= self.warmup_iters:
            alpha = self._iter / self.warmup_iters
            lr_factor = self.factor + (1 - self.factor) * alpha
            for g, base in zip(self.opt.param_groups, self._base_lrs):
                g['lr'] = base * lr_factor

    @property
    def warmup_done(self):
        return self._iter >= self.warmup_iters


def train(config):
    """
    Train WeedDet.

    Required config keys:
      data_root  : str   - path to dataset root
      device     : str   - 'cuda' or 'cpu'

    Optional config keys (paper defaults shown):
      num_classes       : 9
      batch_size        : 2
      num_epochs        : 12
      base_lr           : 0.01
      momentum          : 0.9
      weight_decay      : 0.0001
      warmup_iters      : 500
      lr_decay_epochs   : [8, 11]
      img_size          : (600, 1000)
      freeze_bn         : True
      num_workers       : 2
      checkpoint_dir    : 'checkpoints'
    """
    device = torch.device(
        config.get('device', 'cuda')
        if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    model = WeedDet(
        num_classes=config.get('num_classes', 1)
    ).to(device)

    # Freeze BN (batch_size=2 is too small for stable statistics)
    if config.get('freeze_bn', True):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        print("BatchNorm frozen.")

    img_size = config.get('img_size', (600, 1000))

    train_ds = WeedDataset(config['data_root'], 'train',
                           img_size, augment=True)
    val_ds   = WeedDataset(config['data_root'], 'val',
                           img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.get('batch_size', 2),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 2),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('base_lr', 0.01),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 0.0001)
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.get('lr_decay_epochs', [8, 11]),
        gamma=0.1
    )

    warmup = WarmupMultiStepLR(
        optimizer,
        warmup_iters=config.get('warmup_iters', 500)
    )

    ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float('inf')
    num_epochs = config.get('num_epochs', 12)

    for epoch in range(1, num_epochs + 1):
        model.train()
        if config.get('freeze_bn', True):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        epoch_loss = 0.0
        n_batches  = 0

        # Paper uses 2x dataset repeat per epoch
        for repeat in range(2):
            for i, (imgs, tgts) in enumerate(train_loader):
                imgs = imgs.to(device)
                tgts = [{k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in t.items()} for t in tgts]

                optimizer.zero_grad()
                losses = model(imgs, tgts)
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

                if not warmup.warmup_done:
                    warmup.step_iter()

                epoch_loss += losses['total_loss'].item()
                n_batches  += 1

                if i % 50 == 0:
                    print(
                        f"[E{epoch}/{num_epochs}][R{repeat+1}/2]"
                        f"[B{i}/{len(train_loader)}] "
                        f"loss={losses['total_loss'].item():.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6f}"
                    )

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}\n")

        lr_scheduler.step()

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, os.path.join(ckpt_dir, 'weeddet_best.pth'))
            print(f"  Saved best checkpoint (loss={best_loss:.4f})")

        if epoch % 4 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(ckpt_dir, f'weeddet_e{epoch}.pth'))

    return model


# ---------------------------------------------------------------------------
# SECTION 9 — INFERENCE
# ---------------------------------------------------------------------------

def load_model(path, device='cuda'):
    """Load WeedDet from checkpoint."""
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt.get('config', {})
    model = WeedDet(num_classes=cfg.get('num_classes', 1))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print(f"Loaded epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    return model


def predict(model, image_path, img_size=(600, 1000),
            score_thr=0.5, nms_thr=0.5, device='cuda'):
    """
    Run WeedDet on a single image.
    Returns list of {box, score, label, class_name} dicts.
    """
    from PIL import Image as PI
    tf = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    img = PI.open(image_path).convert('RGB')
    x   = tf(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        results = model(x)

    preds = results[0]
    keep  = preds['scores'] > score_thr
    return [
        {
            'box'       : preds['boxes'][i].cpu().numpy(),
            'score'     : preds['scores'][i].item(),
            'label'     : preds['labels'][i].item(),
            'class_name': preds['class_names'][i]
        }
        for i in range(len(preds['boxes'])) if keep[i]
    ]


# ---------------------------------------------------------------------------
# SECTION 10 — MAIN: BUILD CHECK + ARCHITECTURE SUMMARY
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("WeedDet — build and forward pass check")
    print("=" * 60)

    model = WeedDet(num_classes=1, anchor_base_scale=4, lsc_k=7)
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"Parameters (total)    : {total/1e6:.2f}M")
    print(f"Parameters (trainable): {trainable/1e6:.2f}M")
    print(f"Paper reports         : 24.33M")
    print()

    # Dummy forward pass — training mode
    imgs = torch.randn(2, 3, 600, 1000)
    tgts = [
        {
            'boxes' : torch.tensor([[100., 50., 300., 200.],
                                    [400., 100., 600., 350.]]),
            'labels': torch.tensor([0, 0])
        },
        {
            'boxes' : torch.tensor([[200., 150., 450., 400.]]),
            'labels': torch.tensor([0])
        }
    ]

    model.train()
    loss = model(imgs, tgts)
    print("Training pass OK")
    print(f"  cls_loss   = {loss['cls_loss'].item():.4f}")
    print(f"  reg_loss   = {loss['reg_loss'].item():.4f}")
    print(f"  total_loss = {loss['total_loss'].item():.4f}")
    print()

    # Inference mode
    model.eval()
    with torch.no_grad():
        preds = model(imgs)
    print("Inference pass OK")
    print(f"  Image 0 detections: {len(preds[0]['boxes'])}")
    print(f"  Image 1 detections: {len(preds[1]['boxes'])}")
    print()

    # FPS benchmark (CPU)
    single = torch.randn(1, 3, 600, 1000)
    with torch.no_grad():
        for _ in range(3):
            model(single)
        t0 = time.time()
        for _ in range(10):
            model(single)
        fps = 10 / (time.time() - t0)
    print(f"CPU FPS: {fps:.1f}  (paper: 24.3 on GTX TITAN X)")
    print()

    print("=" * 60)
    print("Architecture summary")
    print("=" * 60)
    rows = [
        ("Backbone",  "Det-ResNet-50"),
        ("",          "  stem: 3x3 + 3x3 + DetResidualBlock"),
        ("",          "  layers 1-4: standard bottleneck"),
        ("Neck",      "eFPN  (P3, P4, P5 only)"),
        ("",          "  anchor base scale: 6"),
        ("Head",      "ERetina-Head"),
        ("",          "  1x conv (64ch) + LSC (k=7)"),
        ("",          "  cls subnet + reg subnet"),
        ("Reg loss",  "SmoothL1 + GIoU"),
        ("Cls loss",  "VariFocal Loss (IACS targets)"),
        ("Classes",   "9  (rice + 8 weed species)"),
        ("Input",     "1000 x 600 RGB"),
    ]
    for k, v in rows:
        print(f"  {k:<12} {v}")
    print()

    # To train on your own data:
    # config = {
    #     'data_root'       : '/path/to/dataset',
    #     'num_classes'     : 9,
    #     'batch_size'      : 2,
    #     'num_epochs'      : 12,
    #     'base_lr'         : 0.01,
    #     'momentum'        : 0.9,
    #     'weight_decay'    : 0.0001,
    #     'warmup_iters'    : 500,
    #     'lr_decay_epochs' : [8, 11],
    #     'img_size'        : (600, 1000),
    #     'device'          : 'cuda',
    #     'checkpoint_dir'  : 'checkpoints',
    #     'freeze_bn'       : True,
    #     'num_workers'     : 4,
    # }
    # train(config)
