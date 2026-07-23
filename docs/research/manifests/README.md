# Run manifests

Raw immutable run records for the RiceSEG pretraining and the baseline
architecture control. Committed here because these are the primary evidence
behind [`../RICESEG_PRETRAIN_RESULTS.md`](../RICESEG_PRETRAIN_RESULTS.md), and
Google Drive (their only other home) reverted repository files mid-session on
2026-07-21 — git is the store that has not lost data.

Each file carries `git_commit`, `environment`, model id/revision, full per-epoch
`history`, best epoch, and (for the baselines) the `stability` block used for the
headline comparison. Regenerate the stable-window numbers with
`baseline_seg_control.stability_stats(manifest['history'], window)`.

| File | Run | Stable weeds IoU (last 5 ep) |
|---|---|---|
| `riceseg_backbone.pth.manifest.json` | custom Det-ResNet-50, ImageNet→RiceSEG (the exported backbone) | 0.202 ± 0.053 |
| `baseline_deeplabv3_resnet50.json` | control: torchvision DeepLabV3-ResNet50 (COCO/VOC) | 0.483 ± 0.006 |
| `baseline_segformer_b2.json` | control: SegFormer-B2 (ADE20K) | 0.522 ± 0.002 |

The two baselines were run at commit `917991b` (a Colab autosave of
`58855b9`); the pretraining at `77c438e`. Same split/weights/loss/seed across all
three — only the model differs.
