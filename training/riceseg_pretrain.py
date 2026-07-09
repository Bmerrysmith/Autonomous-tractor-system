"""
riceseg_pretrain.py — pretrain WeedDet's Det-ResNet-50 on RiceSEG segmentation.

Why: ImageNet can only fill ~92% of Det-ResNet-50 (custom stem + layer1.0 stay
random), and freezing ImageNet BN behind that random stem regressed detection
(v6b AP@50 0.017 vs scratch 0.166). Training the WHOLE backbone on in-domain
rice imagery (RiceSEG, arXiv:2504.02880 — has an explicit `weeds` class) makes
every backbone tensor + BN stat coherent, including the stem.

Exports a checkpoint whose keys are byte-for-byte `WeedDet.backbone.*`
(same DetResNet50 class -> same keys), optionally `fpn.*` too.

Integration into WeedDet (see training/USING_RICESEG_BACKBONE.md):
    from riceseg_pretrain import load_riceseg_backbone, freeze_backbone_bn
    load_riceseg_backbone(weeddet_model, 'riceseg_backbone.pth')
    # BN option A (recommended, matches v5GPT): do nothing — all BN trainable
    # BN option B: freeze_backbone_bn(weeddet_model)  (re-apply after EVERY .train())
    # NEVER wd.apply_bn_policy() on a riceseg backbone — that is the F1+F2 regression.

CLI:
    python riceseg_pretrain.py --self-test                      # no data needed
    python riceseg_pretrain.py --data-root RiceSEG --overfit 8 --batch-size 4
    python riceseg_pretrain.py --data-root RiceSEG --epochs 30 --out riceseg_backbone.pth
Options: --img-size 512 --lr 3e-4 --val-ratio 0.1 --holdout-country NAME
         --no-imagenet --include-fpn
"""

import argparse
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------- import WeedDet
def _import_wd():
    here = Path(__file__).resolve().parent
    for p in (str(here.parent), str(here.parent / 'models'), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for name in ('models.weeddet_v6b', 'weeddet_v6b', 'models.weeddet_v6', 'weeddet_v6'):
        try:
            mod = __import__(name, fromlist=['x'])
            return mod
        except ImportError:
            continue
    raise ImportError('weeddet_v6b.py not importable — put it in models/ or on sys.path')

wd = _import_wd()

NUM_CLASSES  = 6      # 0 bg, 1 green veg, 2 senescent, 3 panicle, 4 weeds, 5 duckweed
CLASS_NAMES  = ['background', 'green_veg', 'senescent', 'panicle', 'weeds', 'duckweed']
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ================================================================ model
class RiceSegModel(nn.Module):
    """DetResNet50 + eFPN (names match WeedDet -> identical state-dict keys)
    + a light segmentation decoder. Only backbone(+fpn) get exported."""

    def __init__(self, num_classes=NUM_CLASSES, feat=256):
        super().__init__()
        self.backbone = wd.DetResNet50()
        self.fpn      = wd.eFPN(512, 1024, 2048, feat)
        self.decoder  = nn.Sequential(
            nn.Conv2d(feat, feat, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(feat, num_classes, 1)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)          # strides 4/8/16
        f = (p3
             + F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
             + F.interpolate(p5, size=p3.shape[-2:], mode='bilinear', align_corners=False))
        y = self.classifier(self.decoder(f))
        return F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)


# ================================================================ export / load
def export_backbone(model, path, include_fpn=False):
    keep = ('backbone.', 'fpn.') if include_fpn else ('backbone.',)
    sd = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith(keep)}
    torch.save(sd, path)
    print(f'[export] {len(sd)} tensors -> {path}')
    return sd


def load_riceseg_backbone(weeddet_model, ckpt_path, verbose=True):
    """Drop-in replacement for wd.load_imagenet_backbone — fills the WHOLE
    backbone (stem + layer1.0 included). Keep all BN trainable afterwards
    (or freeze ALL with freeze_backbone_bn) — never wd.apply_bn_policy."""
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    model_sd = weeddet_model.state_dict()
    loadable = {k: v for k, v in sd.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    weeddet_model.load_state_dict(loadable, strict=False)
    n_backbone = sum(1 for k in model_sd if k.startswith('backbone.'))
    if verbose:
        print(f'[riceseg] loaded {len(loadable)} tensors '
              f'({sum(1 for k in loadable if k.startswith("backbone."))}/{n_backbone} backbone)')
    assert len(loadable) > 150, (
        f'riceseg load looks broken — only {len(loadable)} tensors matched. '
        'Do NOT train from this state.')
    return len(loadable)


def freeze_backbone_bn(weeddet_model):
    """BN option B: freeze ALL backbone BN with the (now in-domain) RiceSEG
    stats. Re-apply after EVERY model.train() call."""
    n = 0
    for name, m in weeddet_model.named_modules():
        if name.startswith('backbone.') and isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            n += 1
    return n


# ================================================================ data
def scan_riceseg(data_root):
    """Find (rgb, label, country, source_id) tuples. Layout: any depth of
    <country>/.../rgb/<tile>.png with sibling label/ dir holding same-name mask."""
    root = Path(data_root)
    pairs = []
    for rgb in sorted(root.glob('**/rgb/*')):
        if rgb.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        lab_dir = rgb.parent.parent / 'label'
        lab = None
        for cand in (lab_dir / rgb.name, lab_dir / (rgb.stem + '.png')):
            if cand.exists():
                lab = cand
                break
        if lab is None:
            continue
        rel = rgb.relative_to(root)
        country = rel.parts[0] if len(rel.parts) > 3 else 'unknown'
        m = re.split(r'_subset', rgb.stem, maxsplit=1)
        source_id = f'{country}/{m[0]}'
        pairs.append({'rgb': str(rgb), 'label': str(lab),
                      'country': country, 'source': source_id})
    if not pairs:
        raise FileNotFoundError(f'no rgb/label pairs under {data_root}')
    return pairs


def split_pairs(pairs, val_ratio=0.1, seed=42, holdout_country=None):
    """Group-aware split: all tiles of one source photo stay in one split
    (tiles overlap -> random split would leak). Optional country holdout."""
    if holdout_country:
        train = [p for p in pairs if p['country'].lower() != holdout_country.lower()]
        val   = [p for p in pairs if p['country'].lower() == holdout_country.lower()]
        assert val, f'no tiles for country {holdout_country}'
        return train, val
    groups = defaultdict(list)
    for p in pairs:
        groups[p['source']].append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = val_ratio * len(pairs)
    val, n = [], 0
    for k in keys:
        if n >= target:
            break
        val += groups[k]; n += len(groups[k])
    val_set = {id(p) for p in val}
    train = [p for p in pairs if id(p) not in val_set]
    assert not ({p['source'] for p in train} & {p['source'] for p in val})
    return train, val


def compute_class_weights(pairs, num_classes=NUM_CLASSES, max_samples=400,
                          clamp=(0.5, 8.0)):
    """Median-frequency balancing, clamped. (Effective-number saturates on
    pixel counts in the millions -> degenerate weights; median-freq verified
    sane on RiceSEG: green_veg -> 0.5, weeds/duckweed -> 8.0.)"""
    from PIL import Image
    counts = np.zeros(num_classes, dtype=np.float64)
    step = max(1, len(pairs) // max_samples)
    for p in pairs[::step]:
        m = np.asarray(Image.open(p['label']))
        for c in range(num_classes):
            counts[c] += (m == c).sum()
    freq = counts / max(counts.sum(), 1)
    med  = np.median(freq[freq > 0])
    w = np.where(freq > 0, med / np.maximum(freq, 1e-12), 0.0)
    w = np.clip(w, clamp[0], clamp[1])
    print('[weights]', {n: round(float(x), 2) for n, x in zip(CLASS_NAMES, w)})
    return torch.tensor(w, dtype=torch.float32)


class RiceSegDataset(Dataset):
    def __init__(self, pairs, img_size=512, augment=False):
        self.pairs, self.img_size, self.augment = pairs, img_size, augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        from PIL import Image
        p = self.pairs[i]
        img = Image.open(p['rgb']).convert('RGB')
        lab = Image.open(p['label'])
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size,) * 2, Image.BILINEAR)
            lab = lab.resize((self.img_size,) * 2, Image.NEAREST)
        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
        y = torch.from_numpy(np.asarray(lab, dtype=np.int64).clip(0, NUM_CLASSES - 1))
        return x, y


# ================================================================ loss / metrics
class SegLoss(nn.Module):
    """class-weighted CE + soft Dice."""

    def __init__(self, class_weights=None, dice_w=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_w = dice_w

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        prob = logits.softmax(1)
        oh = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
        inter = (prob * oh).sum((0, 2, 3))
        card  = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        dice  = 1.0 - ((2 * inter + 1.0) / (card + 1.0)).mean()
        return ce + self.dice_w * dice


class ConfMat:
    def __init__(self, n=NUM_CLASSES):
        self.n, self.m = n, np.zeros((n, n), dtype=np.int64)

    def update(self, pred, gt):
        p, g = pred.flatten(), gt.flatten()
        self.m += np.bincount(g * self.n + p, minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        tp = np.diag(self.m).astype(np.float64)
        denom = self.m.sum(0) + self.m.sum(1) - tp
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        return iou, np.nanmean(iou)


# ================================================================ gates
def self_test():
    """No data needed. Asserts seg-model backbone keys == WeedDet backbone
    keys and round-trips export -> load_riceseg_backbone."""
    import tempfile
    seg = RiceSegModel()
    det = wd.WeedDet(num_classes=1, anchor_base_scale=3, lsc_k=7, use_atss=True)
    seg_bk = {k for k in seg.state_dict() if k.startswith('backbone.')}
    det_bk = {k for k in det.state_dict() if k.startswith('backbone.')}
    assert seg_bk == det_bk, (
        f'KEY MISMATCH: {len(seg_bk - det_bk)} only-seg, {len(det_bk - seg_bk)} only-det')
    print(f'[self-test] backbone keys identical: {len(seg_bk)} tensors')
    with torch.no_grad():
        y = seg(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, NUM_CLASSES, 64, 64), y.shape
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, 'bk.pth')
        export_backbone(seg, pth)
        n = load_riceseg_backbone(det, pth)
    print(f'[self-test] export->load round-trip OK ({n} tensors). PASS')


# ================================================================ train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root')
    ap.add_argument('--out', default='riceseg_backbone.pth')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=12)
    ap.add_argument('--img-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--holdout-country', default=None)
    ap.add_argument('--no-imagenet', action='store_true',
                    help='skip ImageNet warm-start of the compatible 92%')
    ap.add_argument('--include-fpn', action='store_true')
    ap.add_argument('--overfit', type=int, default=0,
                    help='sanity gate: train on N tiles, expect high mIoU '
                         '(use --batch-size 4 with --overfit 8)')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    assert args.data_root, '--data-root required (or use --self-test)'

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}')

    pairs = scan_riceseg(args.data_root)
    print(f'[data] {len(pairs)} tiles | countries: '
          f'{sorted({p["country"] for p in pairs})}')

    if args.overfit:
        train_p = val_p = pairs[:args.overfit]
        epochs = max(args.epochs, 60)
        print(f'[gate] OVERFIT-{args.overfit}: must reach high mIoU or something is broken')
    else:
        train_p, val_p = split_pairs(pairs, args.val_ratio, args.seed,
                                     args.holdout_country)
        epochs = args.epochs
        print(f'[split] train {len(train_p)} / val {len(val_p)} '
              f'(group-aware by source photo; 0 source overlap)')

    weights = compute_class_weights(train_p).to(device)
    tl = DataLoader(RiceSegDataset(train_p, args.img_size, augment=True),
                    batch_size=args.batch_size, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=len(train_p) > args.batch_size)
    vl = DataLoader(RiceSegDataset(val_p, args.img_size),
                    batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RiceSegModel().to(device)
    if not args.no_imagenet:
        wd.load_imagenet_backbone(model)   # warm-start the compatible 92%
        # NOTE: no BN freezing here — the whole point is retraining ALL BN in-domain.
    crit = SegLoss(weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); n += 1
        sched.step()

        model.eval()
        cm = ConfMat()
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                cm.update(pred, y.numpy())
        iou, miou = cm.iou()
        line = (f'epoch {ep:02d}/{epochs}  loss={tot/max(n,1):.4f}  mIoU={miou:.4f}  '
                + ' '.join(f'{nm}={v:.2f}' for nm, v in zip(CLASS_NAMES, iou)))
        if miou > best:
            best = miou
            export_backbone(model, args.out, include_fpn=args.include_fpn)
            line += '  * exported'
        print(line, flush=True)

    print(f'\nDone. Best mIoU {best:.4f} -> {args.out}')
    print('Load into WeedDet with load_riceseg_backbone(); keep BN trainable '
          '(or freeze_backbone_bn) — NEVER apply_bn_policy.')


if __name__ == '__main__':
    main()
