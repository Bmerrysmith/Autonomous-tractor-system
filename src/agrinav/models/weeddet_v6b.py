"""
weeddet_v6b.py
==============
WeedDet v6b — Phase 1 fixes, with F3 REVERTED after empirical failure.

v6 (first run) set the VariFocal target to IoU(pred box, GT). On 1,077 images
this starved the classifier at cold start: pred IoU ~0 early -> positive target
~0 -> no positive gradient -> scores collapse -> AP@50 = 0.008 (vs v5GPT 0.166
on the SAME clean data). See RESEARCH_PLAN Part C, 2026-07-04.

v6b keeps the good fixes and reverts the bad one:
  F1: ImageNet-pretrained backbone loading  -> load_imagenet_backbone()   [keep]
  F2: BN policy — freeze ONLY pretrained BN; train random-init BN          [keep]
  F3: VariFocal/IACS target = anchor-to-GT assignment IoU (v5GPT behaviour,
      which bootstraps the classifier). Optional pred-IoU target behind the
      `vfl_use_pred_iou` flag for future ablation.                         [REVERTED]
  F6: augmentation probabilities corrected (blur p=0.30, equalize p=0.15)  [keep]
Based on: Peng et al. (2022) "Weed Detection in Paddy Field Using an
Improved RetinaNet Network", Computers and Electronics in Agriculture,
vol. 199, p. 107179.

Architecture:
    Input RGB image (1000×600)
    │
    Det-ResNet-50 backbone ← replaces standard ResNet-50 first block
    │  C3 (512ch), C4 (1024ch), C5 (2048ch)
    eFPN (efficient FPN) ← three levels aligned to P2/P3/P4 strides
    │  256ch per level
    ERetina-Head ← 1 conv 64ch + Large Separable Conv (k=7)
    │
    Predictions ← boxes + class scores + IACS confidence

Loss:
    Regression    : SmoothL1 + CIoU
    Classification: VariFocal Loss

Training:
    SGD lr=0.01 momentum=0.9 weight_decay=1e-4
    Batch 2  ·  epochs configured via config  ·  2× dataset repeat per epoch
    Linear warmup (~5% of steps) → cosine decay
    Freeze BatchNorm (batch_size=2 too small for stable BN stats)

Single-class rice config (AgriNav):
    num_classes = 1 (Rice only)
    Detection logic INVERTED: rice = protected, everything else = spray target
"""

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision.transforms as T
except Exception:
    # Lightweight torchvision.transforms fallback. Keeps this script importable in
    # environments where torchvision C++ ops are unavailable.
    class _Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, x):
            for transform in self.transforms:
                x = transform(x)
            return x

    class _ToTensor:
        def __call__(self, img):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            data = torch.as_tensor(bytearray(img.tobytes()), dtype=torch.uint8)
            data = data.view(img.size[1], img.size[0], 3)
            return data.permute(2, 0, 1).contiguous().float().div(255.0)

    class _Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

        def __call__(self, tensor):
            return (tensor - self.mean) / self.std

    class _T:
        Compose = _Compose
        ToTensor = _ToTensor
        Normalize = _Normalize

    T = _T()

from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ImageNet normalisation constants (exported for inference notebooks)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ===========================================================================
# LOCAL BOX / POSTPROCESSING OPS
# ===========================================================================

def box_iou(boxes1, boxes2, eps=1e-7):
    """Torch-only IoU for xyxy boxes. Avoids depending on compiled torchvision ops."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=eps)


def elementwise_box_iou(boxes1, boxes2, eps=1e-7):
    """IoU of aligned pairs: boxes1[i] vs boxes2[i] (xyxy). Returns [N]."""
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    return inter / (area1 + area2 - inter + eps)


# NMS lives in the canonical postprocessor so training-time validation, offline
# evaluation, and any future runtime cannot drift into three different
# protocols. Re-exported here because notebooks and older code import them from
# this module. `nms` keeps its historical name; `hard_nms` is the canonical one.
from agrinav.inference.postprocess import (  # noqa: E402
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_NMS_IOU,
    DEFAULT_PRE_NMS_TOPK,
    DEFAULT_SCORE_THRESHOLD,
    decode_batch,
    invert_letterbox,
)
from agrinav.inference.postprocess import hard_nms as nms  # noqa: E402
from agrinav.inference.postprocess import soft_nms  # noqa: E402


# ===========================================================================
# LETTERBOX UTILITIES
# ===========================================================================

def translate_image_and_boxes(img, boxes, labels, dx, dy,
                              fill=(114, 114, 114), min_size=2):
    """Translate visible image content AND its boxes by the SAME (+dx, +dy).

    AUDIT FIX (P0-1). Pillow's affine tuple maps OUTPUT coordinates back into
    INPUT coordinates, so `(1, 0, dx, 0, 1, dy)` samples from x+dx and makes
    content move LEFT for a positive dx — while the old code added +dx to the
    boxes, moving labels RIGHT. That systematically corrupted ~half of all
    augmented training samples. Negating the offsets in the affine tuple makes
    content move by (+dx, +dy), matching the box update.

    AUDIT FIX (P1-10). `labels` are filtered with the SAME keep mask as boxes,
    so a clipped-away object never leaves its class behind to be mis-assigned
    to a different box by a prefix fallback.

    Returns (img, boxes, labels).
    """
    from PIL import Image as PILImage
    w, h = img.size
    img = img.transform(
        img.size, PILImage.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=fill)
    if boxes is not None and len(boxes):
        boxes = boxes.clone().float()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] + dx).clamp(0, w)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] + dy).clamp(0, h)
        keep = (((boxes[:, 2] - boxes[:, 0]) >= min_size) &
                ((boxes[:, 3] - boxes[:, 1]) >= min_size))
        boxes = boxes[keep]
        if labels is not None and len(labels) == len(keep):
            labels = labels[keep]
    return img, boxes, labels


def letterbox_pil(img, size=512, fill=114):
    """
    Resize PIL image to square `size` with grey letterbox padding.
    Returns (letterboxed_img, scale_x, scale_y, pad_left, pad_top).

    AUDIT FIX (P1-8): the integer resize means the realised x and y scales
    differ slightly from the nominal `size / max(w, h)`. Returning the exact
    per-axis scales keeps the forward and inverse mappings consistent (the old
    single nominal scale introduced up to ~1 px of avoidable error, which
    matters for tiny objects and AP75).
    """
    from PIL import Image as PILImage
    w, h = img.size
    nominal = size / max(w, h)
    nw, nh = int(w * nominal), int(h * nominal)
    nw, nh = max(nw, 1), max(nh, 1)
    scale_x, scale_y = nw / w, nh / h
    resized = img.resize((nw, nh), PILImage.BILINEAR)
    canvas = PILImage.new('RGB', (size, size), (fill, fill, fill))
    pad_left = (size - nw) // 2
    pad_top  = (size - nh) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale_x, scale_y, pad_left, pad_top


def unpad_boxes(boxes, scale_x, scale_y, pad_left, pad_top,
                orig_w=None, orig_h=None):
    """Invert letterbox transform on predicted boxes → original image coords.

    Uses the same exact per-axis scales returned by `letterbox_pil`, and
    delegates to the canonical inverse in `agrinav.inference.postprocess` so
    training, evaluation, and inference share one geometry path.

    Pass `orig_w`/`orig_h` to clip to the original image. Without them the
    mapping is unclipped, which is how detections that reached into the grey
    letterbox padding used to come back with negative coordinates — real area,
    as far as pycocotools is concerned. Callers that will score or draw these
    boxes should always pass the original size; `predict_image` does.
    """
    mapped, _keep = invert_letterbox(
        boxes, scale_x, scale_y, pad_left, pad_top, orig_w=orig_w, orig_h=orig_h)
    return mapped

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
        self.stem = nn.Sequential(
            ConvBnRelu(3, 16, k=3, s=1, p=1),
            ConvBnRelu(16, 16, k=3, s=1, p=1),
            DetResidualBlock(),
        )
        self.layer1 = _make_layer(32,  64,  3, stride=1)
        self.layer2 = _make_layer(256, 128, 4, stride=2)
        self.layer3 = _make_layer(512, 256, 6, stride=2)
        self.layer4 = _make_layer(1024, 512, 3, stride=2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

# ---------------------------------------------------------------------------
# v6 FIX (F1): ImageNet-pretrained weights for the compatible part of the
# backbone. Det-ResNet-50 differs from torchvision ResNet-50 only in the stem
# and the first bottleneck of layer1 (in_ch 32 vs 64). Everything else
# (layer1.1-2, layer2, layer3, layer4 = ~92% of backbone params) is
# channel-identical and loads directly.
# ---------------------------------------------------------------------------

_PRETRAINED_PREFIXES = (
    'backbone.layer1.1', 'backbone.layer1.2',
    'backbone.layer2', 'backbone.layer3', 'backbone.layer4',
)


def load_imagenet_backbone(model, verbose=True):
    """
    Map torchvision resnet50 ImageNet weights into WeedDet's Det-ResNet-50.

    Key translation (torchvision -> this implementation):
        layerX.Y.convN.weight   -> backbone.layerX.Y.convN.0.weight
        layerX.Y.bnN.*          -> backbone.layerX.Y.convN.1.*
        layerX.Y.downsample.*   -> backbone.layerX.Y.downsample.*  (identical)
    Skipped: torchvision stem (custom Det stem here), layer1.0 (in_ch 32 vs
    64), fc.

    Returns (n_loaded, n_missed) and PRINTS the counts — a silent no-op load
    is the exact failure mode this project has been bitten by before, so the
    assert refuses to continue if the load did not actually happen.
    """
    try:
        import torchvision.models as tvm
        try:
            tv_sd = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).state_dict()
        except AttributeError:  # torchvision < 0.13
            tv_sd = tvm.resnet50(pretrained=True).state_dict()
    except Exception as exc:
        # AUDIT FIX (P1-9): fail closed. Returning (0, 0) here silently turned a
        # labelled "ImageNet" experiment into an unlabelled scratch run, because
        # callers did not check the match count.
        raise RuntimeError(
            f'ImageNet backbone load FAILED to fetch torchvision resnet50: {exc}. '
            'Refusing to continue — an unchecked failure would mislabel a scratch '
            'run as ImageNet-initialised. Fix connectivity/weights, or explicitly '
            'request a scratch run.'
        ) from exc

    mapped = {}
    for k, v in tv_sd.items():
        if not k.startswith('layer'):
            continue                      # skip stem conv1/bn1 and fc
        if k.startswith('layer1.0.'):
            continue                      # incompatible in_ch (32 vs 64)
        parts = k.split('.')
        block = parts[2]
        if block.startswith('conv'):
            nk = f'{parts[0]}.{parts[1]}.{block}.0.{parts[3]}'
        elif block.startswith('bn'):
            nk = f"{parts[0]}.{parts[1]}.conv{block[-1]}.1." + '.'.join(parts[3:])
        elif block == 'downsample':
            nk = k
        else:
            continue
        mapped[f'backbone.{nk}'] = v

    model_sd = model.state_dict()
    loadable = {k: v for k, v in mapped.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    missed = sorted(set(mapped) - set(loadable))
    model.load_state_dict(loadable, strict=False)

    if verbose:
        n_backbone = sum(1 for k in model_sd if k.startswith('backbone.'))
        print(f'[pretrained] loaded {len(loadable)}/{n_backbone} backbone tensors '
              f'from ImageNet; {len(missed)} mapped keys failed to match.')
        if missed:
            print('[pretrained] first misses:', missed[:5])
    assert len(loadable) > 150, (
        'Pretrained load looks broken — expected >150 matched tensors. '
        'Do NOT train from this state; fix the key mapping first.')
    return len(loadable), len(missed)


def apply_bn_policy(model, pretrained_loaded=True, verbose=False):
    """
    v6 FIX (F2): freeze BatchNorm ONLY where pretrained running stats exist.
    Randomly-initialised BN (stem, layer1.0, FPN/head) stays trainable —
    freezing BN at random init turns it into a permanent identity op, which
    is what crippled v5 training.

    IMPORTANT: model.train() re-enables train mode on every BN, so re-apply
    this immediately after EVERY model.train() call (each epoch, and before
    any train-mode val-loss computation).
    """
    frozen = trainable = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.BatchNorm2d):
            continue
        if pretrained_loaded and name.startswith(_PRETRAINED_PREFIXES):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            frozen += 1
        else:
            if model.training:
                m.train()
            for p in m.parameters():
                p.requires_grad = True
            trainable += 1
    if verbose:
        print(f'[bn-policy] frozen(pretrained)={frozen}  trainable(random-init)={trainable}')
    return frozen, trainable


def all_bn_eval(model):
    """Put every BN in eval mode (use running stats; no stat updates).
    For train-mode val-loss computation: model.train(); all_bn_eval(model)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ===========================================================================
# SECTION 2 — NECK: eFPN
# ===========================================================================

class eFPN(nn.Module):
    """
    Efficient FPN with three output levels. Because Det-ResNet removes the
    standard max-pool, these feature maps align to strides 4/8/16 for 512px
    inputs, i.e. P2/P3/P4 for small-object coverage.
    """
    def __init__(self, c3_ch=512, c4_ch=1024, c5_ch=2048, feat=256):
        super().__init__()
        self.lat3 = nn.Conv2d(c3_ch, feat, 1)
        self.lat4 = nn.Conv2d(c4_ch, feat, 1)
        self.lat5 = nn.Conv2d(c5_ch, feat, 1)
        self.out3 = nn.Conv2d(feat, feat, 3, padding=1)
        self.out4 = nn.Conv2d(feat, feat, 3, padding=1)
        self.out5 = nn.Conv2d(feat, feat, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        return self.out3(p3), self.out4(p4), self.out5(p5)

# ===========================================================================
# SECTION 3 — HEAD: ERetina-Head
# ===========================================================================

class LargeSeparableConv(nn.Module):
    """Decomposes k×k conv into k×1 then 1×k. Default k=7 (from paper ablation)."""
    def __init__(self, ch, k=7):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(ch, ch, (k, 1), padding=(k//2, 0), groups=ch, bias=False),
            nn.Conv2d(ch, ch, (1, k), padding=(0, k//2), groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class ERetinaHead(nn.Module):
    """Efficient Retina Head: 1 conv (64ch) + LSC instead of 4 convs (256ch)."""
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

        prior_prob = 0.01
        bias_val = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_val)
        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, features):
        cls_outs, reg_outs = [], []
        for feat in features:
            x = self.shared(feat)
            cls_outs.append(self.cls_head(x))
            reg_outs.append(self.reg_head(x))
        return cls_outs, reg_outs

# ===========================================================================
# SECTION 4 — ANCHOR GENERATOR
# ===========================================================================

class AnchorGenerator(nn.Module):
    """
    Generates anchors for the three eFPN levels.
    Approved analysis fix: base_scale=3 and tall-object ratios
    (0.2, 0.33, 0.5, 1.0). Strides are aligned to the actual feature maps
    produced by Det-ResNet/eFPN (4/8/16), giving the first level P2 coverage.
    4 aspect ratios x 3 scales = 12 anchors per location.
    """
    def __init__(self, base_scale=3,
                 aspect_ratios=(0.2, 0.33, 0.5, 1.0),
                 scales=(1.0, 2**(1/3), 2**(2/3)),
                 strides=(4, 8, 16)):
        super().__init__()
        self.base_scale    = base_scale
        self.aspect_ratios = aspect_ratios
        self.scales        = scales
        self.strides       = strides
        self.num_anchors_per_level = []

    @torch.no_grad()
    def forward(self, features, img_shape):
        all_anchors = []
        self.num_anchors_per_level = []
        for level, feat in enumerate(features):
            H, W = feat.shape[-2:]
            stride = self.strides[level] if self.strides is not None else max(
                img_shape[0] / H, img_shape[1] / W
            )
            anchors = self._make_anchors(H, W, stride, feat.device)
            self.num_anchors_per_level.append(anchors.shape[0])
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

class CIoULoss(nn.Module):
    """Complete-IoU loss for tighter localization than GIoU on tall weed boxes."""
    def forward(self, pred, target):
        eps = 1e-7

        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        inter_w  = (inter_x2 - inter_x1).clamp(min=0)
        inter_h  = (inter_y2 - inter_y1).clamp(min=0)
        inter    = inter_w * inter_h

        pw = (pred[:, 2]   - pred[:, 0]).clamp(min=eps)
        ph = (pred[:, 3]   - pred[:, 1]).clamp(min=eps)
        tw = (target[:, 2] - target[:, 0]).clamp(min=eps)
        th = (target[:, 3] - target[:, 1]).clamp(min=eps)
        union = pw * ph + tw * th - inter + eps
        iou   = inter / union

        pcx = (pred[:, 0] + pred[:, 2]) * 0.5
        pcy = (pred[:, 1] + pred[:, 3]) * 0.5
        tcx = (target[:, 0] + target[:, 2]) * 0.5
        tcy = (target[:, 1] + target[:, 3]) * 0.5
        center_dist = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)

        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])
        enc_diag = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

        v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)).pow(2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)

        ciou = iou - (center_dist / enc_diag) - alpha * v
        return (1 - ciou).mean()


class HardTargetFocalLikeLoss(nn.Module):
    """Custom focal-style binary classification loss with HARD positive targets.

    AUDIT FIX (P1-1) — HONEST NAMING. This is deliberately *not* called
    "VariFocal Loss", because it does not implement the published VarifocalNet
    objective:

      * published VFL weights positives by the target IoU score itself and
        trains an IoU-Aware Classification Score (IACS);
      * this implementation weights positives by ``target * |target - p|^gamma``
        and, with ``cls_hard_target=True``, sets positive targets to 1.0.

    So classification confidence here is NOT trained to rank boxes by
    localisation quality — a plausible contributor to the observed
    high-confidence/poor-localisation duplicates and near-zero AP75. If a real
    VFL/IACS comparison is wanted it must be implemented separately and
    ablated against this loss under one locked assigner and evaluator.

    ``VariFocalLoss`` remains as a backwards-compatible alias.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        p   = pred.sigmoid()
        pos = target > 0
        weight = torch.where(
            pos,
            target * (target - p).abs().pow(self.gamma),
            self.alpha * p.pow(self.gamma),
        )
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (loss * weight).sum()


# Backwards-compatible alias. The name is historically inaccurate (see the
# class docstring); prefer HardTargetFocalLikeLoss in new code.
VariFocalLoss = HardTargetFocalLikeLoss


class WeedDetLoss(nn.Module):
    """Combined loss: SmoothL1 + CIoU (regression) + VariFocal (classification)."""
    def __init__(self, num_classes=1, iou_threshold=0.5, neg_iou_threshold=0.4,
                 use_atss=True, atss_topk=9, vfl_use_pred_iou=False):
        super().__init__()
        self.vfl_use_pred_iou  = vfl_use_pred_iou   # False => anchor-GT IoU target (bootstraps)
        self.cls_hard_target   = True   # v7 FIX: positives get target 1.0 (see forward)
        self.atss_all_neg      = True   # T1 FIX (2026-07-09): no ignore band in ATSS mode.
        # Rationale: ~pos anchors with max_iou>=0.4 previously got ZERO cls gradient
        # ("ignore band"). Standard ATSS has no such band. Under hard 1.0 targets those
        # unsupervised anchors saturated to score 1.00 -> FP floods (V7 run #2 collapse).
        # Set False to reproduce the old behaviour for ablation. See AUDIT_2026-07-09_V7.md
        # and TEST_LOG.md entry T1.
        self.num_classes       = num_classes
        self.iou_threshold     = iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.use_atss          = use_atss
        self.atss_topk         = atss_topk
        self.ciou_loss         = CIoULoss()
        self.varifocal         = VariFocalLoss()

    def encode(self, anchors, gt_boxes):
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
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
        ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
        px = deltas[:, 0] * aw + ax
        py = deltas[:, 1] * ah + ay
        pw = torch.exp(deltas[:, 2].clamp(max=4)) * aw
        ph = torch.exp(deltas[:, 3].clamp(max=4)) * ah
        return torch.stack([px - pw/2, py - ph/2, px + pw/2, py + ph/2], dim=1)

    def _force_one_positive_per_gt(self, ious, pos, best, quality):
        """Guarantee every GT owns at least one positive anchor, collision-safe.

        AUDIT FIX (P1-3). The previous vectorised write
        ``pos[gt_best_anchor] = True`` used advanced indexing, so if two GTs
        picked the SAME best anchor the later write overwrote the earlier and a
        ground-truth object silently ended up with no positive. Here the
        strongest GT claims a contested anchor and the loser falls back to its
        next-best *unclaimed* anchor.
        """
        device = ious.device
        gt_best_iou, gt_best_anchor = ious.max(dim=0)
        claimed = torch.zeros(ious.shape[0], dtype=torch.bool, device=device)
        for g in torch.argsort(gt_best_iou, descending=True).tolist():
            col = ious[:, g].masked_fill(claimed, -1.0)
            a = int(col.argmax().item())
            if float(col[a]) < 0.0:          # degenerate: fewer anchors than GTs
                a = int(gt_best_anchor[g].item())
            claimed[a] = True
            pos[a] = True
            best[a] = g
            quality[a] = ious[a, g]
        return pos, best, quality

    def _assign_atss(self, anchors, gt_boxes, ious, num_anchors_per_level=None):
        """
        ATSS-style adaptive positive assignment.
        Falls back to fixed IoU thresholds if per-level counts are unavailable.
        """
        max_iou, best = ious.max(dim=1)

        if (not self.use_atss) or not num_anchors_per_level:
            pos = max_iou >= self.iou_threshold
            neg = max_iou < self.neg_iou_threshold
            # The fallback path must also guarantee per-GT coverage (audit P1-3).
            pos, best, quality = self._force_one_positive_per_gt(
                ious, pos.clone(), best.clone(), max_iou.clone())
            return pos, neg & ~pos, best, quality

        num_gt = gt_boxes.shape[0]
        device = anchors.device
        centers = (anchors[:, :2] + anchors[:, 2:]) * 0.5
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5
        distances = ((centers[:, None, :] - gt_centers[None, :, :]) ** 2).sum(dim=2)

        # AUDIT FIX (P1-2): select top-k distinct CELLS per level, then take every
        # anchor shape at those cells. All co-located shapes share one centre, so
        # the old per-anchor top-k could return k shapes at a single cell instead
        # of k nearby spatial locations, destroying the neighbourhood ATSS needs
        # to compute its mean+std IoU threshold.
        candidate_idxs = []
        start = 0
        for n_level in num_anchors_per_level:
            end = start + n_level
            if end <= start:
                continue
            lvl_centers = centers[start:end]
            # Anchors sharing the first centre = shapes per cell (grid centres are unique).
            per_cell = int((lvl_centers == lvl_centers[0]).all(dim=1).sum().item()) or 1
            n_cells = max(n_level // per_cell, 1)
            cell_dist = distances[start:end:per_cell][:n_cells]      # [n_cells, num_gt]
            k = min(self.atss_topk, n_cells)
            _, topk_cells = cell_dist.topk(k, dim=0, largest=False)  # [k, num_gt]
            cell_base = start + topk_cells * per_cell                # [k, num_gt]
            offsets = torch.arange(per_cell, device=device).view(per_cell, 1, 1)
            idxs = (cell_base.unsqueeze(0) + offsets).reshape(k * per_cell, num_gt)
            candidate_idxs.append(idxs)
            start = end

        if not candidate_idxs:
            pos = max_iou >= self.iou_threshold
            neg = max_iou < self.neg_iou_threshold
            return pos, neg, best, max_iou

        candidate_idxs = torch.cat(candidate_idxs, dim=0)  # [K, num_gt]
        gt_arange = torch.arange(num_gt, device=device)
        candidate_ious = ious[candidate_idxs, gt_arange]

        iou_thresh = candidate_ious.mean(dim=0) + candidate_ious.std(dim=0)
        candidate_centers = centers[candidate_idxs]
        gt = gt_boxes.unsqueeze(0)
        inside_gt = (
            (candidate_centers[:, :, 0] >= gt[:, :, 0]) &
            (candidate_centers[:, :, 1] >= gt[:, :, 1]) &
            (candidate_centers[:, :, 0] <= gt[:, :, 2]) &
            (candidate_centers[:, :, 1] <= gt[:, :, 3])
        )
        is_pos = (candidate_ious >= iou_thresh.unsqueeze(0)) & inside_gt

        pos_ious = ious.new_full(ious.shape, -1.0)
        pos_ious[candidate_idxs, gt_arange] = torch.where(
            is_pos, candidate_ious, candidate_ious.new_full(candidate_ious.shape, -1.0)
        )

        assigned_iou, assigned_gt = pos_ious.max(dim=1)
        pos = assigned_iou >= 0
        best = assigned_gt.clamp(min=0)

        # Guarantee one positive per GT, collision-safe (audit P1-3).
        pos, best, assigned_iou = self._force_one_positive_per_gt(
            ious, pos, best, assigned_iou)

        # Post-condition: every ground-truth object owns at least one positive.
        assert torch.unique(best[pos]).numel() >= min(num_gt, int(pos.sum().item())), (
            'ATSS assignment orphaned a ground-truth object')

        # T1 FIX: all non-positives are negatives (ATSS-paper semantics).
        # Old ignore band (max_iou in [neg_iou_threshold, pos)) kept behind the flag.
        if self.atss_all_neg:
            neg = ~pos
        else:
            neg = ~pos & (max_iou < self.neg_iou_threshold)
        quality = torch.where(pos, assigned_iou.clamp(min=0), max_iou)
        return pos, neg, best, quality

    def forward(self, cls_logits, regs, anchors, targets, num_anchors_per_level=None):
        B = len(targets)
        cls_flat = [torch.cat([
            c.permute(0,2,3,1).reshape(B, -1, self.num_classes)
            for c in cls_logits], dim=1)]
        reg_flat = torch.cat([
            r.permute(0,2,3,1).reshape(B, -1, 4)
            for r in regs], dim=1)
        cls_flat = cls_flat[0]

        total_cls = torch.tensor(0.0, device=anchors.device)
        total_reg = torch.tensor(0.0, device=anchors.device)
        num_pos   = 0

        for b in range(B):
            gt_boxes  = targets[b]['boxes'].to(anchors.device)
            gt_labels = targets[b]['labels'].to(anchors.device)
            iacs      = torch.zeros(len(anchors), self.num_classes, device=anchors.device)

            if len(gt_boxes) == 0:
                total_cls = total_cls + self.varifocal(cls_flat[b], iacs)
                continue

            ious = box_iou(anchors, gt_boxes)
            pos, neg, best, quality_iou = self._assign_atss(
                anchors, gt_boxes, ious, num_anchors_per_level
            )
            ign = ~pos & ~neg

            pos_idx  = pos.nonzero(as_tuple=True)[0]
            num_pos += len(pos_idx)

            if len(pos_idx) > 0:
                matched_gt  = gt_boxes[best[pos_idx]]
                tgt_deltas  = self.encode(anchors[pos_idx], matched_gt)
                pred_deltas = reg_flat[b][pos_idx]
                pred_boxes  = self.decode(anchors[pos_idx], pred_deltas)
                smooth_l1   = F.smooth_l1_loss(pred_deltas, tgt_deltas, reduction='sum')
                total_reg   = total_reg + smooth_l1 + \
                              self.ciou_loss(pred_boxes, matched_gt) * len(pos_idx)
                matched_cls = gt_labels[best[pos_idx]].clamp(0, self.num_classes-1)
                # v6b: classification target for positives.
                # Default = anchor-to-GT assignment IoU (quality_iou) — strong,
                # stable positive signal from step 1, which bootstraps the
                # classifier (v5GPT behaviour, AP@50 0.166 on this data).
                # Optional pred-box IoU target (true VFL) behind the flag; it
                # cold-starved training here (AP@50 0.008) so it is OFF.
                # v7 FIX: hard target 1.0 for positives. The anchor-GT IoU
                # quality target capped confidence (~0.06) and decoupled score
                # from final box quality -> overfit-16 stalled at AP@50 0.037 with
                # the boxes themselves at median IoU 0.69. Hard 1.0 restored
                # confidence (overfit-16 AP@50 0.6+). Old quality target kept
                # behind cls_hard_target=False for ablation.
                if self.vfl_use_pred_iou:
                    with torch.no_grad():
                        target_q = elementwise_box_iou(pred_boxes, matched_gt).clamp_(0.0, 1.0)
                elif self.cls_hard_target:
                    target_q = torch.ones(len(pos_idx), device=anchors.device)
                else:
                    target_q = quality_iou[pos_idx].clamp(0.0, 1.0)
                iacs[pos_idx, matched_cls] = target_q

            valid     = ~ign
            total_cls = total_cls + self.varifocal(cls_flat[b][valid], iacs[valid])

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
    """WeedDet: complete one-stage paddy weed detector."""
    CLASS_NAMES = ['Rice']

    def __init__(self, num_classes=1, anchor_base_scale=3, lsc_k=7, use_atss=True,
                 vfl_use_pred_iou=False):
        super().__init__()
        self.num_classes = num_classes
        self.backbone    = DetResNet50()
        self.fpn         = eFPN(512, 1024, 2048, 256)
        self.head        = ERetinaHead(256, num_classes, 12, lsc_k)
        self.anchor_gen  = AnchorGenerator(
            base_scale=anchor_base_scale,
            aspect_ratios=(0.2, 0.33, 0.5, 1.0),
            scales=(1.0, 2**(1/3), 2**(2/3)),
            strides=(4, 8, 16),
        )
        self.criterion   = WeedDetLoss(num_classes=num_classes, use_atss=use_atss,
                                       vfl_use_pred_iou=vfl_use_pred_iou)

    def forward(self, images, targets=None):
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        c3, c4, c5       = self.backbone(images)
        p3, p4, p5       = self.fpn(c3, c4, c5)
        features         = [p3, p4, p5]
        cls_logits, regs = self.head(features)
        anchors          = self.anchor_gen(features, images.shape[-2:])

        if self.training:
            assert targets is not None
            return self.criterion(cls_logits, regs, anchors, targets,
                                  getattr(self.anchor_gen, 'num_anchors_per_level', None))
        return self._decode(cls_logits, regs, anchors, images.shape[-2:])

    def _get_logits(self, images):
        """Separate logit extraction for external decode (eval notebooks)."""
        if isinstance(images, (list, tuple)):
            images = torch.stack(images)
        c3, c4, c5       = self.backbone(images)
        p3, p4, p5       = self.fpn(c3, c4, c5)
        features         = [p3, p4, p5]
        cls_logits, regs = self.head(features)
        anchors          = self.anchor_gen(features, images.shape[-2:])
        return cls_logits, regs, anchors, images.shape[-2:]

    def _decode(self, cls_logits, regs, anchors, img_shape,
                score_thr=DEFAULT_SCORE_THRESHOLD, nms_thr=DEFAULT_NMS_IOU,
                max_dets=DEFAULT_MAX_DETECTIONS,
                output_thr=None, use_soft_nms=True, soft_nms_sigma=0.5,
                pre_nms_topk=DEFAULT_PRE_NMS_TOPK):
        """Decode head outputs to per-image detections in letterboxed space.

        Delegates to `agrinav.inference.postprocess.decode_batch`, the single
        canonical postprocessor. This method used to keep only the top-scoring
        class per anchor and then run one class-agnostic NMS pass over every
        surviving box, so a rice detection could suppress an overlapping weed
        detection and the losing class hypothesis never reached the metric.

        score_thr     : pre-suppression candidate gate (0.05 = COCO convention)
        nms_thr       : IoU threshold for suppression, applied within a class
        max_dets      : per-image cap after suppression (100 = COCO maxDets)
        output_thr    : optional gate after suppression; defaults to score_thr
        use_soft_nms  : decay overlapping scores instead of deleting boxes
        pre_nms_topk  : per-class candidate cap (per class, not global: the
                        dataset is ~6.8:1 rice-to-weed, and a shared cap is one
                        the majority class wins)
        """
        return decode_batch(
            cls_logits, regs, anchors, img_shape,
            num_classes=self.num_classes,
            score_threshold=score_thr,
            nms_iou=nms_thr,
            max_detections=max_dets,
            output_threshold=output_thr,
            use_soft_nms=use_soft_nms,
            soft_nms_sigma=soft_nms_sigma,
            pre_nms_topk=pre_nms_topk,
        )

# ===========================================================================
# SECTION 7 — DATASET: WeedDataset (PASCAL VOC XML)
# ===========================================================================

class WeedDataset(Dataset):
    """
    PASCAL VOC XML dataset loader for WeedDet.
    Expected structure:
        root/
          images/           ← .jpg or .png
          annotations/      ← .xml (PASCAL VOC format)
          train.txt         ← image stems, one per line
          val.txt
    """
    CLASS_NAMES = ['Rice']

    def __init__(self, root, split='train',
                 img_size=512, augment=False):
        from PIL import Image as PILImage
        self.PILImage = PILImage
        self.root      = root
        self.img_size  = img_size
        self.augment   = augment
        self.class_to_idx = {n: i for i, n in enumerate(self.CLASS_NAMES)}

        split_file = os.path.join(root, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.ids = [l.strip() for l in f if l.strip()]
        else:
            ann_dir  = os.path.join(root, 'annotations')
            self.ids = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(ann_dir) if f.endswith('.xml')
            )

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.ids)

    def _parse_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
        except Exception:
            return (torch.zeros((0,4), dtype=torch.float32),
                    torch.zeros((0,),  dtype=torch.int64), 0, 0)

        root     = tree.getroot()
        size     = root.find('size')
        orig_w   = int(size.find('width').text)  if size is not None else 0
        orig_h   = int(size.find('height').text) if size is not None else 0
        boxes, labels = [], []

        for obj in root.findall('object'):
            name = obj.find('name')
            if name is None: continue
            cls_name = name.text.strip()
            if cls_name.lower() in {'rice', 'rice plant', 'oryza sativa'}:
                cls_idx = 0
            elif cls_name in self.class_to_idx:
                cls_idx = self.class_to_idx[cls_name]
            else:
                continue
            bb = obj.find('bndbox')
            if bb is None: continue
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
                torch.zeros((0,),  dtype=torch.int64), orig_w, orig_h)

    def _augment(self, img, boxes, labels=None):
        """Geometric + photometric augmentation.

        AUDIT FIX (P0-1/P1-10): translation now moves image content and boxes in
        the SAME direction, and `labels` are carried through every keep mask so
        boxes and classes can never desynchronise. Returns (img, boxes, labels).
        """
        import random
        from PIL import ImageFilter, ImageEnhance, ImageOps

        w, h = img.size

        # Horizontal flip
        if random.random() > 0.5:
            img = img.transpose(self.PILImage.FLIP_LEFT_RIGHT)
            if len(boxes):
                boxes = boxes.clone()
                boxes[:, [0,2]] = w - boxes[:, [2,0]]

        # Translation +/-10% (joint image+box+label transform)
        if random.random() > 0.5 and len(boxes):
            dx = int(random.uniform(-0.1, 0.1) * w)
            dy = int(random.uniform(-0.1, 0.1) * h)
            img, boxes, labels = translate_image_and_boxes(
                img, boxes, labels, dx, dy)

        # Brightness
        if random.random() > 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

        # Colour saturation
        if random.random() > 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))

        # Gaussian blur — v6 FIX (F6): p=0.30, radius <=0.5
        # (previous condition `> 0.3` fired at p=0.70 with radius up to 1.0)
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 0.5)))

        # Equalise — v6 FIX (F6): p=0.15 (previous condition fired at p=0.50)
        if random.random() < 0.15:
            img = ImageOps.equalize(img)

        return img, boxes, labels

    def __getitem__(self, idx):
        stem     = self.ids[idx]
        img_dir  = os.path.join(self.root, 'images')
        ann_dir  = os.path.join(self.root, 'annotations')
        img_path = os.path.join(img_dir, stem + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, stem + '.png')
        if not os.path.exists(img_path):
            return None

        img    = self.PILImage.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        boxes, labels, xml_w, xml_h = self._parse_xml(
            os.path.join(ann_dir, stem + '.xml'))

        if self.augment and len(boxes):
            img, boxes, labels = self._augment(img, boxes, labels)

        # Letterbox to square (exact per-axis scales — audit P1-8)
        img_lb, sx, sy, pl, pt = letterbox_pil(img, self.img_size)

        # Scale boxes to letterbox space
        if len(boxes):
            boxes = boxes.clone().float()
            boxes[:, [0,2]] = (boxes[:, [0,2]] * sx + pl).clamp(0, self.img_size)
            boxes[:, [1,3]] = (boxes[:, [1,3]] * sy + pt).clamp(0, self.img_size)
            keep  = ((boxes[:,2]-boxes[:,0]) >= 1) & ((boxes[:,3]-boxes[:,1]) >= 1)
            boxes  = boxes[keep]
            # AUDIT FIX (P1-10): never fall back to a prefix slice — that
            # silently re-labels surviving boxes with another object's class.
            labels = labels[keep]

        return self.transform(img_lb), {
            'boxes' : boxes,
            'labels': labels,
            'orig_h': orig_h,
            'orig_w': orig_w,
        }


class CocoWeedDataset(WeedDataset):
    """
    COCO-format loader for the pre-split output of data/split_coco.py:
        root/
          train/  ← .jpg images + _annotations.coco.json
          valid/
          test/
    Replaces the VOC WeedDataset AND the in-notebook COCO-GT rebuild:
    for pycocotools eval use `COCO(ds.ann_file)` directly and iterate
    `ds.items()` — the image_ids match the json, so no GT conversion.

    class_names picks which categories to train on (by name), mapped to
    contiguous label indices in the given order. Default rice-only
    (num_classes=1, matches v5GPT/v6b). Use ('rice','weed') for 2-class.
    """

    def __init__(self, root, split='train', img_size=512, augment=False,
                 class_names=('rice',)):
        import json as _json
        from PIL import Image as PILImage
        self.PILImage = PILImage
        self.root, self.split = root, split
        self.img_size, self.augment = img_size, augment
        self.CLASS_NAMES  = list(class_names)
        self.class_to_idx = {n: i for i, n in enumerate(self.CLASS_NAMES)}
        self.ann_file = os.path.join(root, split, '_annotations.coco.json')
        with open(self.ann_file) as f:
            coco = _json.load(f)
        name_by_id = {c['id']: c['name'] for c in coco['categories']}
        self.catid_to_idx = {cid: self.class_to_idx[n]
                             for cid, n in name_by_id.items()
                             if n in self.class_to_idx}
        assert self.catid_to_idx, (
            f'none of {class_names} in {sorted(name_by_id.values())}')
        self.images = coco['images']
        self.anns_by_img = {}
        for a in coco['annotations']:
            if a['category_id'] in self.catid_to_idx:
                self.anns_by_img.setdefault(a['image_id'], []).append(a)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.images)

    def items(self):
        """[(coco_image_id, path, W, H)] — for pycocotools eval loops."""
        return [(im['id'],
                 os.path.join(self.root, self.split, im['file_name']),
                 im['width'], im['height']) for im in self.images]

    def __getitem__(self, idx):
        im  = self.images[idx]
        img = self.PILImage.open(os.path.join(
            self.root, self.split, im['file_name'])).convert('RGB')
        orig_w, orig_h = img.size

        rows = []
        for a in self.anns_by_img.get(im['id'], []):
            x, y, w, h = a['bbox']
            if w > 1 and h > 1:
                rows.append([x, y, x + w, y + h,
                             float(self.catid_to_idx[a['category_id']])])
        b5 = (torch.tensor(rows, dtype=torch.float32) if rows
              else torch.zeros((0, 5), dtype=torch.float32))

        # _augment only touches columns 0-3 and row-filters, so carrying the
        # label as a 5th column keeps boxes/labels aligned through box drops.
        if self.augment and len(b5):
            img, b5, _ = self._augment(img, b5)

        # Exact per-axis letterbox scales (audit P1-8)
        img_lb, sx, sy, pl, pt = letterbox_pil(img, self.img_size)
        boxes, labels = b5[:, :4].clone(), b5[:, 4].long()
        if len(boxes):
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] * sx + pl).clamp(0, self.img_size)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] * sy + pt).clamp(0, self.img_size)
            keep   = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
            boxes, labels = boxes[keep], labels[keep]

        return self.transform(img_lb), {
            'boxes' : boxes,
            'labels': labels,
            'orig_h': orig_h,
            'orig_w': orig_w,
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    imgs = torch.stack([b[0] for b in batch])
    tgts = [b[1] for b in batch]
    return imgs, tgts


def _resolve_batch_size(config, key, *, fallback_key=None, default=2):
    """Resolve a positive batch size without passing None to DataLoader.

    PyTorch interprets ``batch_size=None`` as "disable automatic batching".
    That mode is incompatible with :func:`collate_fn`, which expects a list of
    samples, and turns a configuration sentinel into a delayed validation
    crash.
    """
    value = config.get(key)
    if value is None and fallback_key is not None:
        value = config.get(fallback_key)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value

# ===========================================================================
# SECTION 8 — TRAINING UTILITIES
# ===========================================================================


def set_seed(seed=42, deterministic=False):
    """Set Python/NumPy/PyTorch RNG seeds for reproducible training runs."""
    import random
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelEMA:
    """Exponential Moving Average wrapper for model weights."""
    def __init__(self, model, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])

class WarmupMultiStepLR:
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

    def state_dict(self):
        return {'iter': self._iter, 'base_lrs': list(self._base_lrs)}

    def load_state_dict(self, state):
        self._iter = int(state.get('iter', 0))
        base = state.get('base_lrs')
        if base:
            self._base_lrs = list(base)


# --- checkpoint / metrics plumbing -----------------------------------------
# Everything below exists because a training run's *artifacts* were previously
# unusable off the training machine and a completed run was indistinguishable
# from a crashed one. See docs/HANDOFF.md.

_CONFIG_DROP_KEYS = ('train_dataset', 'val_dataset', 'backbone_init')


def sanitize_config_for_save(config):
    """Return a JSON-safe copy of ``config`` for embedding in a checkpoint.

    Live objects (the injected datasets, the ``backbone_init`` callable) are
    replaced by a ``repr``-style provenance string instead of being pickled.
    Pickling them made every checkpoint depend on importable project internals:
    because training runs as ``python -m agrinav.training.weeddet_train``, the
    ``_RicesegBackboneInit`` class pickled with ``__module__ == '__main__'`` and
    loading the checkpoint in any other process raised
    ``AttributeError: Can't get attribute '_RicesegBackboneInit'``.
    """
    safe = {}
    for key, value in config.items():
        if key in _CONFIG_DROP_KEYS:
            if value is not None:
                safe[f'{key}_repr'] = repr(value)
                ann = getattr(value, 'ann_file', None)
                if ann is not None:
                    safe[f'{key}_ann_file'] = str(ann)
                ckpt = getattr(value, 'ckpt_path', None)
                if ckpt is not None:
                    safe[f'{key}_ckpt_path'] = str(ckpt)
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = [v for v in value
                         if isinstance(v, (str, int, float, bool, type(None)))]
        elif isinstance(value, dict):
            safe[key] = {str(k): v for k, v in value.items()
                         if isinstance(v, (str, int, float, bool, type(None)))}
        else:
            safe[f'{key}_repr'] = repr(value)
    return safe


def atomic_torch_save(obj, path):
    """``torch.save`` via a temp file in the same directory, then ``os.replace``.

    Checkpoints are written straight onto a Google Drive FUSE mount; an
    interrupted in-place write previously truncated the only good checkpoint.
    ``os.replace`` is atomic within a filesystem, so the old file survives until
    the new one is complete.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f'{path}.tmp'
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def append_metrics_row(ckpt_dir, row):
    """Append one JSON object to ``<ckpt_dir>/metrics.jsonl`` and flush to disk.

    The per-epoch loss curve previously existed only in a notebook cell; Colab
    truncates long cell output, which destroyed the early epochs of a completed
    run. Flushing + fsync per epoch means a killed run still leaves every epoch
    it finished.
    """
    path = os.path.join(ckpt_dir, 'metrics.jsonl')
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return path


@torch.no_grad()
def evaluate_val_loss(model, loader, device):
    """Mean loss over ``loader`` in eval mode. Returns ``{}`` for an empty loader.

    Calls ``_get_logits`` + ``model.criterion`` directly rather than
    ``model(images, targets)``, because ``forward`` returns decoded detections
    whenever ``model.training`` is False. Running under ``eval()`` means BN uses
    its running statistics and — critically — does not *update* them from
    validation data.
    """
    was_training = model.training
    model.eval()
    totals, batches = {}, 0
    try:
        for imgs, tgts in loader:
            if not isinstance(imgs, torch.Tensor) or imgs.numel() == 0:
                continue
            imgs = imgs.to(device)
            tgts = [{k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in t.items()} for t in tgts]
            cls_logits, regs, anchors, _ = model._get_logits(imgs)
            losses = model.criterion(
                cls_logits, regs, anchors, tgts,
                getattr(model.anchor_gen, 'num_anchors_per_level', None))
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
            batches += 1
    finally:
        if was_training:
            model.train()
    if batches == 0:
        return {}
    return {k: v / batches for k, v in totals.items()}


def train_with_progress(config):
    """Training loop with tqdm progress bars — preferred for Colab/Kaggle."""
    device = torch.device(
        config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    if 'seed' in config:
        set_seed(config.get('seed', 42), config.get('deterministic', False))

    model = WeedDet(
        num_classes=config.get('num_classes', 1),
        anchor_base_scale=config.get('anchor_base_scale', 3),
        lsc_k=config.get('lsc_k', 7),
        use_atss=config.get('use_atss', True),
    ).to(device)

    # v6 (F1/F2): pretrained backbone, then BN policy — BEFORE optimizer
    # creation so requires_grad filtering sees the final flags.
    #
    # Phase-2 seam: an optional config['backbone_init'] callable injects an
    # in-domain backbone (e.g. the RiceSEG-pretrained weights) in place of
    # ImageNet. It owns its own fail-closed key/shape check, and apply_bn_policy
    # is deliberately skipped for it — the loaded BN running stats are in-domain
    # and must stay trainable (see load_riceseg_backbone: "never apply_bn_policy").
    # When the key is absent the else branch is byte-identical to the original.
    #
    # `bn_policy` makes the BN treatment an explicit, recordable factor instead
    # of a side effect of which backbone was chosen. 'auto' reproduces the
    # historical behaviour exactly: freeze BN where ImageNet stats exist, leave
    # it trainable for an injected in-domain backbone. That coupling meant the
    # ImageNet-vs-RiceSEG A/B varied *two* factors at once (backbone weights AND
    # BN trainability), so a delta could not be attributed. Set 'trainable' on
    # the ImageNet arm for a matched single-factor comparison.
    backbone_init = config.get('backbone_init')
    bn_policy = config.get('bn_policy', 'auto')
    if bn_policy not in ('auto', 'freeze_pretrained', 'trainable'):
        raise ValueError(
            f"bn_policy must be 'auto', 'freeze_pretrained' or 'trainable', "
            f"got {bn_policy!r}")
    if bn_policy == 'auto':
        bn_freeze = backbone_init is None
    else:
        bn_freeze = bn_policy == 'freeze_pretrained'
    config['bn_policy_resolved'] = 'freeze_pretrained' if bn_freeze else 'trainable'

    if backbone_init is not None:
        backbone_init(model)
    elif config.get('pretrained_backbone', True):
        load_imagenet_backbone(model)
    if bn_freeze:
        apply_bn_policy(model,
                        pretrained_loaded=config.get('pretrained_backbone', True),
                        verbose=True)
    print(f"[bn-policy] {bn_policy} -> {config['bn_policy_resolved']}")

    img_size = config.get('img_size', 512)
    train_batch_size = _resolve_batch_size(config, 'batch_size', default=2)
    config['batch_size'] = train_batch_size
    train_ds = (config['train_dataset'] if config.get('train_dataset') is not None
                else WeedDataset(config['data_root'], 'train', img_size, augment=True))
    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=config.get('num_workers', 2), pin_memory=True)
    print(f"Train: {len(train_ds)} images | {len(train_loader)} batches/epoch")

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get('base_lr', 0.001),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 0.0001))

    num_epochs = config.get('num_epochs', 24)
    save_every = config.get('save_every', 4)
    repeat     = config.get('repeat_factor', 2)
    total_steps = max(1, num_epochs * len(train_loader) * repeat)
    warmup_iters = config.get('warmup_iters', max(1, int(0.05 * total_steps)))

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_iters),
        eta_min=config.get('min_lr', 0.00001))

    warmup = WarmupMultiStepLR(
        optimizer,
        warmup_iters=warmup_iters,
        warmup_factor=config.get('warmup_factor', 0.001))

    use_amp = config.get('use_amp', torch.cuda.is_available())
    # torch.cuda.amp.* is deprecated from torch 2.4; the device-qualified API
    # is the supported spelling and stops the FutureWarning pair from being
    # printed into every run's log.
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = ModelEMA(model, decay=config.get('ema_decay', 0.999)) if config.get('use_ema', True) else None

    ckpt_dir   = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_loss  = float('inf')

    # Optional held-out validation. When a val dataset is injected, the epoch
    # metric that selects `best` is the val loss of *the network that actually
    # gets saved* (the EMA when EMA is on). Without this the selection number
    # came from the raw online model in train() mode on augmented data while the
    # saved weights were the EMA — the metric never scored the artifact.
    val_ds = config.get('val_dataset')
    val_loader = None
    val_batch_size = None
    if val_ds is not None:
        val_batch_size = _resolve_batch_size(
            config, 'val_batch_size', fallback_key='batch_size', default=2)
        config['val_batch_size'] = val_batch_size
        val_loader = DataLoader(
            val_ds, batch_size=val_batch_size,
            shuffle=False, collate_fn=collate_fn,
            num_workers=config.get('num_workers', 2), pin_memory=True)
        print(f"Val  : {len(val_ds)} images | {len(val_loader)} batches")

    # Validation *AP* is the real selection metric when it is available: loss is
    # a proxy that a detector can improve while its detections get worse, and
    # the July-29 audit found selection running on training loss alone. AP needs
    # the canonical decode, so it is opt-in via `val_ap_interval` (epochs between
    # evaluations; 0 disables) and requires the val dataset to expose the COCO
    # file it came from.
    val_ap_interval = int(config.get('val_ap_interval', 0) or 0)
    val_ap_ann_file = getattr(val_ds, 'ann_file', None) if val_ds is not None else None
    use_val_ap = bool(val_ap_interval > 0 and val_ap_ann_file)
    if val_ap_interval > 0 and not use_val_ap:
        raise ValueError(
            "val_ap_interval > 0 requires a val_dataset that records the COCO file it "
            "was built from (its `ann_file` attribute). Pass --val-ann-file/"
            "--val-images-root, or set val_ap_interval: 0 to select on val loss.")

    if use_val_ap:
        select_metric, higher_is_better = 'val/AP', True
    elif val_loader is not None:
        select_metric, higher_is_better = 'val/total_loss', False
    else:
        select_metric, higher_is_better = 'train/total_loss', False
    best_loss = float('-inf') if higher_is_better else float('inf')
    last_val_ap: dict = {}
    print(f"Checkpoint selection metric: {select_metric} "
          f"({'higher' if higher_is_better else 'lower'} is better)")

    # tqdm floods the notebook otherwise: 225 batches x 18 epochs x a forced
    # redraw per batch produced ~424k-496k characters in one Colab cell, which
    # made the notebook unscrollable and got the early epochs truncated away.
    bar_interval = float(config.get('progress_interval', 2.0))
    show_bars = TQDM_AVAILABLE and not config.get('no_progress', False)

    def _log(message):
        """Epoch summaries must not interleave with the bar's stderr writes."""
        if show_bars:
            tqdm.write(message)
        else:
            print(message, flush=True)

    saved_config = sanitize_config_for_save(config)
    history = []
    best_epoch = 0
    global_step = 0
    skipped_batches = 0
    amp_skipped_steps = 0
    completed_epochs = 0

    epoch_bar = (tqdm(range(1, num_epochs + 1), desc='Epochs', mininterval=bar_interval)
                 if show_bars else range(1, num_epochs + 1))

    for epoch in epoch_bar:
        model.train()
        # model.train() re-enables train mode on every BN, so the freeze has to
        # be re-applied each epoch (see apply_bn_policy's docstring).
        if bn_freeze:
            apply_bn_policy(model,
                            pretrained_loaded=config.get('pretrained_backbone', True))

        epoch_loss, n_batches = 0.0, 0
        comp_totals = {}
        epoch_started = time.time()
        clipped_steps = 0
        grad_clip = config.get('grad_clip', 0.5)

        for r in range(repeat):  # 2× dataset repeat per epoch (paper)
            batch_iter = (tqdm(enumerate(train_loader),
                               total=len(train_loader),
                               desc=f'E{epoch:02d} R{r+1}/{repeat}',
                               leave=False, mininterval=bar_interval)
                          if show_bars else enumerate(train_loader))

            for i, (imgs, tgts) in batch_iter:
                if not isinstance(imgs, torch.Tensor) or imgs.numel() == 0:
                    skipped_batches += 1
                    continue
                imgs = imgs.to(device)
                tgts = [{k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in t.items()} for t in tgts]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    losses = model(imgs, tgts)
                    loss   = losses.get('total_loss', sum(losses.values()))

                # A non-finite loss makes avg_loss NaN; `NaN < best_loss` is
                # False forever, so the run would silently burn every remaining
                # epoch while never updating `best` again.
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite training loss at epoch {epoch}, batch {i}: "
                        f"{ {k: float(v.detach()) for k, v in losses.items()} }. "
                        "Lower base_lr, check the annotations for degenerate boxes, "
                        "or disable AMP (use_amp: false) and re-run.")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if float(total_norm) > grad_clip:
                    clipped_steps += 1
                prev_scale = scaler.get_scale() if use_amp else None
                scaler.step(optimizer)
                scaler.update()
                # GradScaler silently skips a step whose grads are non-finite;
                # a run where most updates never applied otherwise looks healthy.
                if use_amp and scaler.get_scale() < prev_scale:
                    amp_skipped_steps += 1
                if ema is not None:
                    ema.update(model)
                if not warmup.warmup_done:
                    warmup.step_iter()
                else:
                    lr_scheduler.step()

                epoch_loss += loss.item()
                n_batches  += 1
                global_step += 1
                for key, value in losses.items():
                    comp_totals[key] = comp_totals.get(key, 0.0) + float(value.item())

                if show_bars:
                    batch_iter.set_postfix(
                        loss=f'{loss.item():.4f}',
                        cls=f'{losses["cls_loss"].item():.4f}',
                        reg=f'{losses["reg_loss"].item():.4f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.5f}')

        if n_batches == 0:
            raise RuntimeError(
                f"epoch {epoch} completed zero usable batches ({skipped_batches} "
                "skipped). Every batch collated empty — check that images_root "
                "resolves and that the annotations are readable.")

        avg_loss = epoch_loss / n_batches
        components = {f'train/{k}': v / n_batches for k, v in comp_totals.items()}

        # Held-out metrics, on the network that will actually be saved.
        val_metrics = {}
        if val_loader is not None:
            val_raw = evaluate_val_loss(model, val_loader, device)
            for key, value in val_raw.items():
                val_metrics[f'val_raw/{key}'] = value
            if ema is not None:
                val_ema = evaluate_val_loss(ema.ema, val_loader, device)
                for key, value in val_ema.items():
                    val_metrics[f'val_ema/{key}'] = value
                selected = val_ema
            else:
                selected = val_raw
            for key, value in selected.items():
                val_metrics[f'val/{key}'] = value

        # Validation AP on the weights that will actually be saved. Run at
        # `val_ap_interval` because a full decode over the val split costs real
        # time; on an off-epoch the previous AP is carried forward so `best`
        # never compares an AP epoch against a no-AP epoch.
        if use_val_ap and (epoch % val_ap_interval == 0 or epoch == num_epochs):
            from agrinav.evaluation.runner import evaluate_split
            scored = ema.ema if ema is not None else model
            ap_result, _dets, ap_protocol = evaluate_split(
                scored, val_ds, val_ap_ann_file, device=str(device),
                img_size=config.get('img_size', 512),
                batch_size=val_batch_size)
            val_ap = {
                'val/AP': ap_result.ap, 'val/AP50': ap_result.ap50,
                'val/AP75': ap_result.ap75, 'val/AP_small': ap_result.ap_small,
                'val/AR100': ap_result.ar_100,
                'val/eval_seconds': ap_protocol['predict_seconds'],
            }
            for cat_id, value in ap_result.per_category_ap.items():
                name = ap_result.category_names.get(cat_id, str(cat_id))
                val_ap[f'val/AP[{name}]'] = value
            val_metrics.update(val_ap)
            last_val_ap = val_ap

        if use_val_ap:
            epoch_metric = val_metrics.get('val/AP', last_val_ap.get('val/AP', float('-inf')))
        elif val_loader is not None:
            epoch_metric = val_metrics.get('val/total_loss', float('inf'))
        else:
            epoch_metric = avg_loss

        row = {'epoch': epoch, 'num_epochs': num_epochs,
               'train/total_loss': avg_loss, 'n_batches': n_batches,
               'skipped_batches': skipped_batches,
               'clipped_steps': clipped_steps, 'amp_skipped_steps': amp_skipped_steps,
               'lr': optimizer.param_groups[0]['lr'],
               'wall_s': round(time.time() - epoch_started, 2),
               'select_metric': select_metric, 'select_value': epoch_metric}
        row.update(components)
        row.update(val_metrics)
        append_metrics_row(ckpt_dir, row)
        history.append(row)

        is_best = (epoch_metric > best_loss if higher_is_better
                   else epoch_metric < best_loss)
        if is_best:
            best_loss, best_epoch = epoch_metric, epoch

        if show_bars:
            epoch_bar.set_postfix(avg_loss=f'{avg_loss:.4f}', best=f'{best_loss:.4f}')
        summary = (f"Epoch {epoch:02d}/{num_epochs}  avg_loss={avg_loss:.4f}  "
                   f"{select_metric}={epoch_metric:.4f}  best={best_loss:.4f}  "
                   f"lr={optimizer.param_groups[0]['lr']:.6f}")
        if 'val_raw/total_loss' in val_metrics and 'val_ema/total_loss' in val_metrics:
            summary += (f"  [val raw={val_metrics['val_raw/total_loss']:.4f} "
                        f"ema={val_metrics['val_ema/total_loss']:.4f}]")
        _log(summary)

        def _payload(loss_value):
            return {'epoch': epoch, 'global_step': global_step,
                    'state_dict': (ema.ema.state_dict() if ema is not None
                                   else model.state_dict()),
                    'raw_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'warmup': warmup.state_dict(),
                    'loss': loss_value,
                    'best_metric_name': select_metric,
                    'best_metric_value': best_loss,
                    'class_names': list(config.get('class_names', [])),
                    'num_classes': config.get('num_classes'),
                    'config': saved_config,
                    'scaler': scaler.state_dict() if use_amp else None}

        # Always write a terminal artifact. Previously the only per-epoch files
        # came from `epoch % save_every`, and 18 % 4 == 2 meant a *successful*
        # 18-epoch run left `weeddet_epoch16.pth` as its newest file — identical
        # on disk to a run killed at 16, which is exactly how four completed runs
        # were misread as crashes.
        atomic_torch_save(_payload(avg_loss), os.path.join(ckpt_dir, 'weeddet_last.pth'))

        if is_best:
            path = os.path.join(ckpt_dir, 'weeddet_best.pth')
            atomic_torch_save(_payload(best_loss), path)
            _log(f"  ★ Best checkpoint ({select_metric}={best_loss:.4f}) -> {path}")

        if epoch % save_every == 0:
            path = os.path.join(ckpt_dir, f'weeddet_epoch{epoch}.pth')
            atomic_torch_save(_payload(avg_loss), path)
            _log(f"  Checkpoint saved  -> {path}")

        completed_epochs = epoch

    status = {
        'completed': completed_epochs == num_epochs,
        'epochs_planned': num_epochs,
        'epochs_completed': completed_epochs,
        'best_epoch': best_epoch,
        'best_metric_name': select_metric,
        'best_metric_value': best_loss,
        'final_train_loss': history[-1]['train/total_loss'] if history else None,
        'skipped_batches': skipped_batches,
        'amp_skipped_steps': amp_skipped_steps,
        'global_step': global_step,
        'checkpoint_dir': os.path.abspath(ckpt_dir),
    }
    status_path = os.path.join(ckpt_dir, 'status.json')
    with open(status_path, 'w', encoding='utf-8') as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
    _log(f"\nTraining complete. best_epoch={best_epoch} "
         f"({select_metric}={best_loss:.4f}) -> {status_path}")
    return model
