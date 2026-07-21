# Phase 1 Fix Log — RiceSEG Transfer-Learning Correctness

**Date:** 2026-07-20
**Scope:** `training/riceseg_pretrain.py` only (the RiceSEG → WeedDet backbone pretraining, i.e. the project's **first batch of training**).
**Basis:** the July 20 deep audit (§9.4, §17.1, §18) and the deployment roadmap (Gate 4 → "RiceSEG-specific repairs", lines 722–730; Gate 1 run-record requirements).
**Method:** test-first, per `active/ACTIVE_NOTES.md` "Do next" item 2.

This log records what was changed to make the first training run correct and reproducible. It does **not** touch the detector (`models/weeddet_v6b.py`), inference, or the contaminated COCO split — those are separate roadmap gates and remain open (see "Not in scope" below).

---

## Completed tasks

| # | Roadmap repair (Gate 4, lines 722–730) | What changed in `riceseg_pretrain.py` | Proof |
|---|---|---|---|
| 1 | Parse country/source-photo IDs from the real layout; assert counts for all five countries | New `parse_country_site()` anchors on the `rgb`/`label` directory instead of `rel.parts[0]` (which returned the wrapper `"global rice segmentation"` for every tile). `scan_riceseg()` now records `country` and `site`. | `test_scan_riceseg_returns_all_five_countries`; verified on the real `RiceSEG.zip` → China 1120 / India 600 / Japan 704 / Philippines 600 / Tanzania 54 (matches audit §9.2) |
| 2 | Country/site holdout fails if the requested group is empty or unknown | `main()` raises `SystemExit` if `--holdout-country` is not among the parsed countries; `split_pairs` still asserts a non-empty holdout | `test_country_holdout_split_is_now_reachable`; CLI check with `--holdout-country Brazil` → clear error |
| 3 | Overfit fixture must contain every evaluated class (esp. weeds, duckweed) | New `_stratified_overfit_subset()` greedily covers all six classes within a bounded scan | exercised by the overfit path; smoke run shows all six classes present |
| 4 | Fixed-class metric; report absent classes instead of silently dropping them | `ConfMat.iou()` now returns `(iou, miou_over_present, absent_class_names)`; the epoch line and manifest print `[absent: …]` explicitly | `test_absent_classes_are_reported_not_silently_dropped`, `test_all_classes_present_reports_no_absent` |
| 5 | Diagnostic checkpoints to a temp path, never the production backbone name | New `_overfit_output_path()`; the overfit branch exports only to an isolated temp file | `test_overfit_output_is_isolated_from_production` |
| 6 | Enforce the overfit threshold; non-zero exit on failure | New `_enforce_overfit_gate()` raises `SystemExit` below `--overfit-min-miou` (default 0.80) | `test_overfit_gate_fails_below_threshold`, `test_overfit_gate_passes_at_or_above_threshold` |
| 7 | Full segmentation checkpoint + separately hashed backbone export | New `save_full_checkpoint()` (model/optimizer/scheduler/epoch/RNG/config/history) and `write_run_manifest()` with a `sha256_file()` of the backbone export, git commit, env, and per-epoch history | smoke run writes `bk.pth`, `bk.pth.fullckpt.pth`, `bk.pth.manifest.json` |
| 8 | Log exact ImageNet coverage and the incremental RiceSEG stage | `main()` captures `load_imagenet_backbone()`'s return, records coverage in the manifest, and **fails closed** if it loaded 0 tensors (audit P1-9: a torchvision fetch failure otherwise becomes a silent scratch run) | CLI: `--no-imagenet` vs. warm-start paths print distinct coverage lines |

**Also fixed (audit §9.4, prerequisite correctness):**

- **Mask validation instead of clip** — `RiceSegDataset` no longer does `.clip(0, NUM_CLASSES-1)` (which silently mapped any 255/void pixel to duckweed). Unknown values raise `ValueError`; an optional `ignore_index` is supported. (`test_unknown_mask_value_raises`, `test_valid_mask_values_pass`, `test_ignore_index_is_allowed_when_configured`)
- **Strict backbone loading** — `load_riceseg_backbone()` now requires the full expected `backbone.*` key set (optionally `fpn.*`) with matching shapes and raises on any gap, instead of accepting "any >150 shape-compatible tensors". (`test_incomplete_backbone_is_rejected`, `test_complete_backbone_roundtrips`)

---

## Test status

- New file: `tests/test_riceseg_pretrain.py` — 15 tests.
- Full suite: **52 tests pass** (`python -m unittest discover -s tests`).
- `python training/riceseg_pretrain.py --self-test` passes (342/342 backbone tensors round-trip).
- End-to-end smoke run on a synthetic RiceSEG tree (CPU, `--no-imagenet`) produces backbone + full checkpoint + manifest with all five countries and all six classes.

---

## Not in scope (still open per the roadmap)

The first *batch of training* enabled here is the RiceSEG backbone pretraining. The following are separate gates and were intentionally not touched:

- **Gate 0** — safety containment (retire spray-by-default `inference/inference_rice.py`, decision schema, hazard log).
- **Gate 2/3** — rebuild the capture-series-contaminated COCO split; annotation ontology/QA (the annotation pilot tooling already landed separately).
- **Gate 4 (detector half)** — `models/weeddet_v6b.py`: translation-augmentation label corruption (P0-1), ATSS spatial-diversity collapse (P1-2), orphaned-GT assignment (P1-3), "Varifocal Loss" misnaming (P1-1), letterbox scale (P1-8), Python NMS (P1-5), mixed evaluator protocol.
- **Gate 6** — the controlled scratch / ImageNet / random→RiceSEG / ImageNet→RiceSEG factorial and maintained baselines.

The detector cannot produce a defensible result until its Gate-4 items are fixed; the RiceSEG backbone, however, can now be trained correctly and its provenance captured.
