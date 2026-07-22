"""
riceseg_pretrain.py — pretrain WeedDet's Det-ResNet-50 on RiceSEG segmentation.

Why: ImageNet can only fill ~92% of Det-ResNet-50 (custom stem + layer1.0 stay
random), and freezing ImageNet BN behind that random stem regressed detection
(v6b AP@50 0.017 vs scratch 0.166). Training the WHOLE backbone on in-domain
rice imagery (RiceSEG, arXiv:2504.02880 — has an explicit `weeds` class) makes
every backbone tensor + BN stat coherent, including the stem.

Exports a checkpoint whose keys are byte-for-byte `WeedDet.backbone.*`
(same DetResNet50 class -> same keys), optionally `fpn.*` too.

Historical integration API (research only; read the July 20 audit before reuse):
    from agrinav.training.riceseg_pretrain import load_riceseg_backbone, freeze_backbone_bn
    load_riceseg_backbone(weeddet_model, 'riceseg_backbone.pth')
    # BN option A (recommended, matches v5GPT): do nothing — all BN trainable
    # BN option B: freeze_backbone_bn(weeddet_model)  (re-apply after EVERY .train())
    # NEVER wd.apply_bn_policy() on a riceseg backbone — that is the F1+F2 regression.

CLI:
    python -m agrinav.training.riceseg_pretrain --self-test                      # no data needed
    python -m agrinav.training.riceseg_pretrain --data-root RiceSEG --overfit 8 --batch-size 4
    python -m agrinav.training.riceseg_pretrain --data-root RiceSEG --epochs 30 --out riceseg_backbone.pth
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
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------- import WeedDet
def _import_wd():
    here = Path(__file__).resolve().parent
    for p in (str(here.parent), str(here.parent / "models"), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for name in ("agrinav.models.weeddet_v6b", "models.weeddet_v6b", "weeddet_v6b"):
        try:
            mod = __import__(name, fromlist=["x"])
            return mod
        except ImportError:
            continue
    raise ImportError("weeddet_v6b.py not importable — put it in models/ or on sys.path")


wd = _import_wd()

NUM_CLASSES = 6  # 0 bg, 1 green veg, 2 senescent, 3 panicle, 4 weeds, 5 duckweed
CLASS_NAMES = ["background", "green_veg", "senescent", "panicle", "weeds", "duckweed"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ================================================================ model
class RiceSegModel(nn.Module):
    """DetResNet50 + eFPN (names match WeedDet -> identical state-dict keys)
    + a light segmentation decoder. Only backbone(+fpn) get exported."""

    def __init__(self, num_classes=NUM_CLASSES, feat=256):
        super().__init__()
        self.backbone = wd.DetResNet50()
        self.fpn = wd.eFPN(512, 1024, 2048, feat)
        self.decoder = nn.Sequential(
            nn.Conv2d(feat, feat, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(feat, num_classes, 1)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)  # strides 4/8/16
        f = (
            p3
            + F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
            + F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        )
        y = self.classifier(self.decoder(f))
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


# ================================================================ export / load
def export_backbone(model, path, include_fpn=False):
    keep = ("backbone.", "fpn.") if include_fpn else ("backbone.",)
    sd = {k: v.cpu() for k, v in model.state_dict().items() if k.startswith(keep)}
    torch.save(sd, path)
    print(f"[export] {len(sd)} tensors -> {path}")
    return sd


def sha256_file(path, chunk_size=1024 * 1024):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_hint):
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "-C", str(repo_hint), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _env_versions():
    import platform

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except Exception:
        pass
    return versions


def write_run_manifest(path, manifest):
    """Emit one immutable JSON run record. The audit found runs were identified
    only by mutable notebook/Drive paths with no code/data/env hashes
    (deployment roadmap Gate 1, "Every run must record")."""
    import json

    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[manifest] {path}")


def save_full_checkpoint(path, model, optimizer, scheduler, epoch, best_miou, config, history):
    """Save a resumable segmentation checkpoint alongside the exported backbone
    (deployment roadmap Gate 4, RiceSEG repair #7). Distinct from the
    backbone-only export consumed by WeedDet."""
    torch.save(
        {
            "schema_version": "agrinav.riceseg_pretrain.fullckpt.v1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
            "config": config,
            "history": history,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        },
        path,
    )
    print(f"[checkpoint] full resumable state -> {path}")


def load_riceseg_backbone(weeddet_model, ckpt_path, verbose=True, require_fpn=False):
    """Drop-in replacement for wd.load_imagenet_backbone — fills the WHOLE
    backbone (stem + layer1.0 included). Keep all BN trainable afterwards
    (or freeze ALL with freeze_backbone_bn) — never wd.apply_bn_policy.

    Fails closed: the checkpoint must cover EVERY expected backbone tensor
    (optionally fpn.* too) with a matching shape. A partial or shape-mismatched
    load raises instead of silently training from a mostly-random backbone
    (deep audit 2026-07-20, Section 9.4, "partial validation of loaded weights").
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    model_sd = weeddet_model.state_dict()
    expected = {k for k in model_sd if k.startswith("backbone.")}
    if require_fpn:
        expected |= {k for k in model_sd if k.startswith("fpn.")}
    loadable, shape_mismatch = {}, []
    for k in sorted(expected):
        if k in sd:
            if tuple(sd[k].shape) == tuple(model_sd[k].shape):
                loadable[k] = sd[k]
            else:
                shape_mismatch.append((k, tuple(sd[k].shape), tuple(model_sd[k].shape)))
    missing = sorted(expected - set(loadable) - {k for k, *_ in shape_mismatch})
    if missing or shape_mismatch:
        raise ValueError(
            f"riceseg load is incomplete: {len(loadable)}/{len(expected)} expected "
            f"tensors matched; missing={missing[:6]}"
            f"{'...' if len(missing) > 6 else ''}; "
            f"shape_mismatch={shape_mismatch[:3]}. Do NOT train from this state."
        )
    weeddet_model.load_state_dict(loadable, strict=False)
    unexpected = sorted(set(sd) - set(model_sd))
    if verbose:
        n_backbone = sum(1 for k in loadable if k.startswith("backbone."))
        print(
            f"[riceseg] loaded {len(loadable)}/{len(expected)} expected tensors "
            f"({n_backbone} backbone); {len(unexpected)} checkpoint tensors ignored"
        )
    return len(loadable)


def freeze_backbone_bn(weeddet_model):
    """BN option B: freeze ALL backbone BN with the (now in-domain) RiceSEG
    stats. Re-apply after EVERY model.train() call."""
    n = 0
    for name, m in weeddet_model.named_modules():
        if name.startswith("backbone.") and isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            n += 1
    return n


# ================================================================ data
def parse_country_site(rgb_path, root):
    """Country/site for a RiceSEG rgb (or label) path.

    The supplied archive extracts to a single dataset wrapper directory that
    then holds one directory per country, e.g.

        <root>/global rice segmentation/China/GD/rgb/<tile>.jpg     (has site)
        <root>/global rice segmentation/India/rgb/<tile>.jpg        (no site)

    so relative to ``root`` the wrapper is ``parts[0]`` and the country is the
    component immediately after it. The previous implementation used
    ``parts[0]`` as the country and therefore labelled every tile
    "global rice segmentation", breaking country holdout and country reporting
    (deep audit 2026-07-20, Section 9.4). This anchors on the rgb/label kind
    directory instead — mirroring scripts/build_annotation_pilot so both tools
    agree on country/site — and returns ``(country, site_or_None)``.
    """
    parts = Path(rgb_path).relative_to(root).parts
    kind_index = next(
        (i for i, part in enumerate(parts) if part.lower() in {"rgb", "label"}),
        None,
    )
    if kind_index is None or kind_index < 2:
        return "unknown", None
    if kind_index == 2:
        return parts[1], None
    return parts[kind_index - 2], parts[kind_index - 1]


def scan_riceseg(data_root):
    """Find (rgb, label, country, site, source) tuples. Layout: any depth of
    <wrapper>/<country>[/<site>]/rgb/<tile> with a sibling label/ dir holding
    the same-name mask. Country/site come from ``parse_country_site``."""
    root = Path(data_root)
    pairs = []
    for rgb in sorted(root.glob("**/rgb/*")):
        if rgb.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        lab_dir = rgb.parent.parent / "label"
        lab = None
        for cand in (lab_dir / rgb.name, lab_dir / (rgb.stem + ".png")):
            if cand.exists():
                lab = cand
                break
        if lab is None:
            continue
        country, site = parse_country_site(rgb, root)
        m = re.split(r"_subset", rgb.stem, maxsplit=1)
        # Group by country/site/source-photo so overlapping tiles from one
        # photo never cross a split even when two sites reuse a photo stem.
        group_prefix = f"{country}/{site}" if site else country
        source_id = f"{group_prefix}/{m[0]}"
        pairs.append(
            {
                "rgb": str(rgb),
                "label": str(lab),
                "country": country,
                "site": site,
                "source": source_id,
            }
        )
    if not pairs:
        raise FileNotFoundError(f"no rgb/label pairs under {data_root}")
    return pairs


def split_pairs(pairs, val_ratio=0.1, seed=42, holdout_country=None):
    """Group-aware split: all tiles of one source photo stay in one split
    (tiles overlap -> random split would leak). Optional country holdout."""
    if holdout_country:
        train = [p for p in pairs if p["country"].lower() != holdout_country.lower()]
        val = [p for p in pairs if p["country"].lower() == holdout_country.lower()]
        assert val, f"no tiles for country {holdout_country}"
        return train, val
    groups = defaultdict(list)
    for p in pairs:
        groups[p["source"]].append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = val_ratio * len(pairs)
    val, n = [], 0
    for k in keys:
        if n >= target:
            break
        val += groups[k]
        n += len(groups[k])
    val_set = {id(p) for p in val}
    train = [p for p in pairs if id(p) not in val_set]
    assert not ({p["source"] for p in train} & {p["source"] for p in val})
    return train, val


def compute_class_weights(pairs, num_classes=NUM_CLASSES, max_samples=400, clamp=(0.5, 8.0)):
    """Median-frequency balancing, clamped. (Effective-number saturates on
    pixel counts in the millions -> degenerate weights; median-freq verified
    sane on RiceSEG: green_veg -> 0.5, weeds/duckweed -> 8.0.)"""
    from PIL import Image

    counts = np.zeros(num_classes, dtype=np.float64)
    step = max(1, len(pairs) // max_samples)
    for p in pairs[::step]:
        m = np.asarray(Image.open(p["label"]))
        for c in range(num_classes):
            counts[c] += (m == c).sum()
    freq = counts / max(counts.sum(), 1)
    med = np.median(freq[freq > 0])
    w = np.where(freq > 0, med / np.maximum(freq, 1e-12), 0.0)
    w = np.clip(w, clamp[0], clamp[1])
    print("[weights]", {n: round(float(x), 2) for n, x in zip(CLASS_NAMES, w)})
    return torch.tensor(w, dtype=torch.float32)


class RiceSegDataset(Dataset):
    def __init__(self, pairs, img_size=512, augment=False, ignore_index=None):
        self.pairs, self.img_size, self.augment = pairs, img_size, augment
        # ignore_index: an out-of-range value (e.g. 255 void) that is permitted
        # and preserved rather than rejected. Default None = strict.
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        from PIL import Image

        p = self.pairs[i]
        img = Image.open(p["rgb"]).convert("RGB")
        lab = Image.open(p["label"])
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size,) * 2, Image.BILINEAR)
            lab = lab.resize((self.img_size,) * 2, Image.NEAREST)
        if self.augment and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[
            :, None, None
        ]
        # Validate rather than clip. The old unconditional clip silently mapped
        # any unexpected value (e.g. a 255 void pixel) to duckweed (deep audit
        # 2026-07-20, Section 9.4). Unknown values now fail loudly.
        arr = np.asarray(lab, dtype=np.int64)
        valid = (arr >= 0) & (arr < NUM_CLASSES)
        if self.ignore_index is not None:
            valid |= arr == self.ignore_index
        if not valid.all():
            bad = np.unique(arr[~valid]).tolist()
            allowed = f"[0, {NUM_CLASSES - 1}]"
            if self.ignore_index is not None:
                allowed += f" or ignore_index={self.ignore_index}"
            raise ValueError(f"mask {p['label']} has values outside {allowed}: {bad}")
        y = torch.from_numpy(arr)
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
        card = prob.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        dice = 1.0 - ((2 * inter + 1.0) / (card + 1.0)).mean()
        return ce + self.dice_w * dice


class ConfMat:
    def __init__(self, n=NUM_CLASSES):
        self.n, self.m = n, np.zeros((n, n), dtype=np.int64)

    def update(self, pred, gt):
        p, g = pred.flatten(), gt.flatten()
        self.m += np.bincount(g * self.n + p, minlength=self.n**2).reshape(self.n, self.n)

    def iou(self):
        """Per-class IoU (NaN where a class never appears in gt or pred), the
        mean over PRESENT classes, and the explicit list of absent class names.
        The absent list is returned so callers report dropped classes rather
        than silently inflating mIoU via np.nanmean (deployment roadmap Gate 4,
        RiceSEG repair #4)."""
        tp = np.diag(self.m).astype(np.float64)
        denom = self.m.sum(0) + self.m.sum(1) - tp
        iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
        absent = [CLASS_NAMES[i] for i in range(self.n) if np.isnan(iou[i])]
        miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan")
        return iou, miou, absent


# ================================================================ overfit gate
def _overfit_output_path(base_out):
    """Return an isolated temp path for the overfit sanity run so it can never
    overwrite the production backbone (`base_out`). The audit found the notebook
    passed the production path into the gate, so a gate-only or failed run could
    leave an 8-image-overfit backbone at the real path (Section 9.4)."""
    import tempfile

    fd, path = tempfile.mkstemp(prefix="riceseg_overfit_", suffix=".pth")
    os.close(fd)
    return Path(path)


def _enforce_overfit_gate(best_miou, threshold):
    """Fail closed if the overfit sanity run did not reach its preregistered
    mIoU. The old gate always continued regardless of score (Section 9.4)."""
    if best_miou < threshold:
        raise SystemExit(
            f"[gate] OVERFIT FAILED: best mIoU {best_miou:.4f} < threshold "
            f"{threshold:.4f}. The preprocessing/model/loss path is broken; "
            "fix it before any full pretraining run."
        )
    print(f"[gate] OVERFIT PASSED: best mIoU {best_miou:.4f} >= {threshold:.4f}")


def _stratified_overfit_subset(pairs, n, num_classes=NUM_CLASSES, scan_cap=512):
    """Pick ~n tiles that together cover all classes so the overfit mIoU is a
    stable six-class score. The audit noted the first 8 sorted tiles contain no
    duckweed, so np.nanmean silently drops that class and inflates the gate
    (Section 9.4). Best-effort within a bounded scan."""
    from PIL import Image

    chosen, seen = [], set()
    for p in pairs[:scan_cap]:
        if len(chosen) >= n and seen >= set(range(num_classes)):
            break
        present = set(np.unique(np.asarray(Image.open(p["label"]))).tolist())
        if (present - seen) or len(chosen) < n:
            chosen.append(p)
            seen |= present
        if len(chosen) >= n and seen >= set(range(num_classes)):
            break
    for p in pairs:
        if len(chosen) >= n:
            break
        if p not in chosen:
            chosen.append(p)
    return chosen[: max(n, len(chosen)) if len(chosen) < n else n]


# ================================================================ gates
def self_test():
    """No data needed. Asserts seg-model backbone keys == WeedDet backbone
    keys and round-trips export -> load_riceseg_backbone."""
    import tempfile

    seg = RiceSegModel()
    det = wd.WeedDet(num_classes=1, anchor_base_scale=3, lsc_k=7, use_atss=True)
    seg_bk = {k for k in seg.state_dict() if k.startswith("backbone.")}
    det_bk = {k for k in det.state_dict() if k.startswith("backbone.")}
    assert (
        seg_bk == det_bk
    ), f"KEY MISMATCH: {len(seg_bk - det_bk)} only-seg, {len(det_bk - seg_bk)} only-det"
    print(f"[self-test] backbone keys identical: {len(seg_bk)} tensors")
    with torch.no_grad():
        y = seg(torch.randn(1, 3, 64, 64))
    assert y.shape == (1, NUM_CLASSES, 64, 64), y.shape
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "bk.pth")
        export_backbone(seg, pth)
        n = load_riceseg_backbone(det, pth)
    print(f"[self-test] export->load round-trip OK ({n} tensors). PASS")


# ================================================================ train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root")
    ap.add_argument("--out", default="riceseg_backbone.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--holdout-country", default=None)
    ap.add_argument(
        "--no-imagenet", action="store_true", help="skip ImageNet warm-start of the compatible 92%"
    )
    ap.add_argument("--include-fpn", action="store_true")
    ap.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="sanity gate: train on N tiles, expect high mIoU "
        "(use --batch-size 4 with --overfit 8)",
    )
    ap.add_argument(
        "--overfit-min-miou",
        type=float,
        default=0.80,
        help="preregistered mIoU the overfit gate must reach or the " "run fails (exit 2)",
    )
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    assert args.data_root, "--data-root required (or use --self-test)"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    pairs = scan_riceseg(args.data_root)
    countries = sorted({p["country"] for p in pairs})
    print(f"[data] {len(pairs)} tiles | countries: {countries}")

    # Fail loudly if a country holdout names a country the archive does not
    # contain (roadmap Gate 4, RiceSEG repair #2). The old parser labelled every
    # tile "global rice segmentation", so any real country silently held out 0.
    if args.holdout_country and args.holdout_country.lower() not in {c.lower() for c in countries}:
        raise SystemExit(
            f"--holdout-country {args.holdout_country!r} is not present; "
            f"available countries: {countries}"
        )

    if args.overfit:
        # Stratified so all six classes appear (else duckweed NaN inflates mIoU),
        # and exported to an isolated temp path that can never clobber args.out.
        train_p = val_p = _stratified_overfit_subset(pairs, args.overfit)
        epochs = max(args.epochs, 60)
        out_path = _overfit_output_path(args.out)
        print(
            f"[gate] OVERFIT-{args.overfit}: must reach mIoU >= "
            f"{args.overfit_min_miou:.2f}; isolated export -> {out_path} "
            f"(production path {args.out} untouched)"
        )
    else:
        train_p, val_p = split_pairs(pairs, args.val_ratio, args.seed, args.holdout_country)
        epochs = args.epochs
        out_path = Path(args.out)
        print(
            f"[split] train {len(train_p)} / val {len(val_p)} "
            f"(group-aware by source photo; 0 source overlap)"
        )

    weights = compute_class_weights(train_p).to(device)
    tl = DataLoader(
        RiceSegDataset(train_p, args.img_size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=len(train_p) > args.batch_size,
    )
    vl = DataLoader(
        RiceSegDataset(val_p, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = RiceSegModel().to(device)
    imagenet_coverage = None
    if not args.no_imagenet:
        # Fail closed on a silent ImageNet miss: a torchvision fetch failure
        # returns (0, 0) and would otherwise turn a labelled "ImageNet->RiceSEG"
        # run into an unlabelled scratch run (audit P1-9 / roadmap repair #8).
        n_loaded, n_missed = wd.load_imagenet_backbone(model)
        n_backbone = sum(1 for k in model.state_dict() if k.startswith("backbone."))
        imagenet_coverage = {
            "loaded": int(n_loaded),
            "missed": int(n_missed),
            "backbone_total": int(n_backbone),
        }
        if n_loaded == 0:
            raise SystemExit(
                "[imagenet] load returned 0 tensors — refusing to run a scratch "
                "job mislabelled as ImageNet->RiceSEG. Use --no-imagenet to run "
                "true scratch, or fix connectivity/weights."
            )
        print(
            f"[imagenet] warm-started {n_loaded}/{n_backbone} backbone tensors "
            f"(the custom stem + layer1.0 stay random by design)"
        )
        # NOTE: no BN freezing here — the whole point is retraining ALL BN in-domain.
    else:
        print("[imagenet] skipped (--no-imagenet): true random->RiceSEG condition")
    crit = SegLoss(weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best = -1.0
    best_epoch, best_iou_vec, best_absent = 0, None, []
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n += 1
        sched.step()

        model.eval()
        cm = ConfMat()
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                cm.update(pred, y.numpy())
        iou, miou, absent = cm.iou()
        avg_loss = tot / max(n, 1)
        history.append(
            {
                "epoch": ep,
                "loss": avg_loss,
                "miou": miou,
                "per_class_iou": {
                    nm: (None if np.isnan(v) else float(v)) for nm, v in zip(CLASS_NAMES, iou)
                },
                "absent_classes": absent,
            }
        )
        line = f"epoch {ep:02d}/{epochs}  loss={avg_loss:.4f}  mIoU={miou:.4f}  " + " ".join(
            f"{nm}={v:.2f}" for nm, v in zip(CLASS_NAMES, iou)
        )
        if absent:
            # Explicit, not silent: mIoU above is the mean over PRESENT classes.
            line += f'  [absent: {",".join(absent)}]'
        if miou > best:
            best, best_epoch, best_iou_vec, best_absent = miou, ep, iou, absent
            export_backbone(model, out_path, include_fpn=args.include_fpn)
            if not args.overfit:
                save_full_checkpoint(
                    Path(str(out_path) + ".fullckpt.pth"),
                    model,
                    opt,
                    sched,
                    ep,
                    best,
                    vars(args),
                    history,
                )
            line += "  * exported"
        print(line, flush=True)

    if args.overfit:
        _enforce_overfit_gate(best, args.overfit_min_miou)
        print(f"\nOverfit sanity artifact (not for reuse): {out_path}")
        return

    # Immutable run record beside the exported backbone (roadmap Gate 1/Gate 4 #7).
    manifest = {
        "schema_version": "agrinav.riceseg_pretrain.run.v1",
        "seed": args.seed,
        "git_commit": _git_commit(Path(__file__).resolve().parent),
        "environment": _env_versions(),
        "device": str(device),
        "config": vars(args),
        "data": {
            "data_root": str(args.data_root),
            "tiles": len(pairs),
            "countries": countries,
            "train_tiles": len(train_p),
            "val_tiles": len(val_p),
            "holdout_country": args.holdout_country,
        },
        "imagenet_coverage": imagenet_coverage,
        "best": {
            "epoch": best_epoch,
            "miou_present_classes": best,
            "per_class_iou": {
                nm: (None if best_iou_vec is None or np.isnan(v) else float(v))
                for nm, v in zip(
                    CLASS_NAMES,
                    best_iou_vec if best_iou_vec is not None else [np.nan] * NUM_CLASSES,
                )
            },
            "absent_classes": best_absent,
        },
        "history": history,
        "backbone_export": {
            "path": str(out_path),
            "sha256": sha256_file(out_path),
            "include_fpn": args.include_fpn,
        },
        "full_checkpoint": str(out_path) + ".fullckpt.pth",
    }
    write_run_manifest(str(out_path) + ".manifest.json", manifest)

    print(f"\nDone. Best mIoU {best:.4f} at epoch {best_epoch} -> {out_path}")
    if best_absent:
        print(
            f"[warn] classes absent from validation at best epoch: {best_absent} "
            "(mIoU is over present classes only)"
        )
    print(
        "Load into WeedDet with load_riceseg_backbone(); keep BN trainable "
        "(or freeze_backbone_bn) — NEVER apply_bn_policy."
    )


if __name__ == "__main__":
    main()
