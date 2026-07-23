# AgriNav — Session Handoff

**Durable, cross-session status pointer.** Read this first when resuming work.
It is intentionally short: a *pointer*, not a log. Detailed history lives in git
(`git log`), the audit (`docs/audits/2026-07-20/`), and the ADRs (`docs/adr/`).

## The convention (how to maintain this file)

- **On resume:** read this file, then `git log --oneline -10` and `git status`.
- **At the end of a working session:** update **Current status**, **Done this
  session**, and **Open items**. Move finished open items into Done; add new
  ones. Keep each entry to a line or two.
- **Do not** duplicate git history, ADR content, or audit findings here — link to
  them instead.
- Commits remain **explicit-ask** (no auto-commit); this file is updated in the
  working tree and committed with the rest of a session's work.

---

## Current status — 2026-07-23

**Phase-2 detector training pipeline (exploratory) is built, smoke-validated, and
independently reviewed and committed.** The `weeddet_v6b` detector can now be
trained on the grouped, leakage-free RiceSEG split from a CLI (`agrinav
train-detector`) or the new Colab notebook. The first GPU run is **exploratory**
(loss convergence + qualitative val predictions); a defensible mAP still needs the
**deferred Gate-4 remainder** (COCO evaluator + canonical postprocess P1-6 + perf
P1-4/P1-5). Test split is **sealed** everywhere; selection is on val.

**Merged to `master` 2026-07-23:** `chore/gate1-packaging-ci`,
`feat/riceseg-training-optimization`, and `feat/phase2-detector-pipeline` are all
merged into `master` and pushed. The 3 stale pre-recovery local `master` commits
(V7 era) were preserved on `archive/pre-recovery-master` before master was reset
to the recovered origin lineage.

## 2026-07-23 — RiceSEG seg training fixes (items 1–3)

Branch `feat/riceseg-training-optimization` off `chore/gate1-packaging-ci`.
Diagnosed the pretrain-vs-baseline seg A/B, then landed:

1. **Recipe parity** in `riceseg_pretrain.py` — new-API AMP, `--num-workers` +
   persistent workers (matches `baseline_seg_control`); noted the unresolved
   **batch-size confound** (12 vs 8) in the recorded runs.
2. **Robust checkpoint selection** — export by EMA-smoothed minority-class IoU
   (`--select-metric minority`, new default) instead of background-dominated
   single-epoch mIoU; `miou` kept for repro; overfit gate unchanged. `stability`
   moved to `riceseg_pretrain` (single source; baseline re-imports).
3. **Loss/LR levers (opt-in)** — `--loss focal_tversky`, `--dice-weighted`,
   `--dice-ignore-bg`, `--warmup-epochs`, `--backbone-lr-mult`.

Verified: `pytest` 128 passed + 16 subtests; ruff/black clean. CPU smoke of both
loss paths + both selection metrics on synthetic tiles, **and a reduced real run
on `RiceSEG/Tanzania` (46 train tiles, 3 ep, 128px, focal_tversky + warmup +
backbone-mult + minority) — ran clean end-to-end in 46s** (real median-freq
weights, masks validated 0–5; weeds/duckweed absent from that val split so the
presence-guarded minority selector scored on present targets). Metrics are not a
baseline by design. See `docs/research/RICESEG_PRETRAIN_RESULTS.md`.

**Pushed to origin 2026-07-23:** `chore/gate1-packaging-ci` and
`feat/riceseg-training-optimization`. PR opened manually (no `gh`/connector here);
recommended base `chore/gate1-packaging-ci` ← `feat/riceseg-training-optimization`.

Full deployable run (30 ep / 512px / batch 8 / ImageNet on) still pending on a GPU.

## Done this session (2026-07-23 — Phase-2 detector pipeline)

- New CLI `agrinav train-detector` (`src/agrinav/training/weeddet_train.py`):
  `_CocoSplitDataset` (COCO split + `images_root + file_name` layout; contiguous
  class remap of the non-contiguous cat ids `{1,2,4}`), plus `--self-test`,
  `--overfit N`, and full modes. It **injects** its dataset into the byte-stable
  `train_with_progress` rather than reimplementing the loop.
- **One additive line** in `weeddet_v6b.py` `train_with_progress` (optional
  `config['train_dataset']`; absent key → identical old behavior). The
  byte-stability freeze is intentionally released for **exactly this line** now
  that Phase-2 / Gate-4 work has begun.
- `configs/training/detector_{smoke,default,gpu}.yaml`;
  `notebooks/weeddet_detector_colab.ipynb` (Drive mount, RiceSEG extract, GPU
  train, run manifest, qualitative-val cell — **test json never loaded**);
  `tests/test_weeddet_train.py` (5 tests).
- **Verified (this tree):** `--self-test` PASS (finite grads); **overfit-8 on real
  RiceSEG PASS — avg_loss 7.90 → 5.81, monotone** (wiring gate, not a quality
  benchmark); `pytest tests/test_weeddet_train.py` 5 passed; full suite 128 passed
  (builder); ruff/black/mypy clean. Independent review of the 7 pipeline files —
  sealed-test leak, class map, monolith back-compat, pickle round-trip, config
  precedence — **no findings**.
- Data re-validated: split counts match `detector_split_v1.json`, **0 group
  leakage**, human-GT provenance (`riceseg_human_semantic_mask`).

## Done this session (Gate 1)

- `src/agrinav/` package layout; all modules moved with history preserved
  (`models/ training/ scripts/* inference/` → `src/agrinav/...`). See ADR 0001.
- `pyproject.toml` packaging + dependency extras (`train`/`inference`/`docs`/
  `dev`); `requirements.txt` reduced to a pointer. See ADR 0002.
- Tooling: ruff + black + mypy config, `.pre-commit-config.yaml`, `Makefile`,
  `.python-version` (3.12).
- Fast CI: `.github/workflows/ci.yml` (lint → typecheck → test on 3.11/3.12 →
  package). Qodana workflow retained.
- Unified `agrinav` CLI (`src/agrinav/cli.py`, `python -m agrinav`).
- Docs: README Setup section, START_HERE paths, ADRs 0001/0002, `.env.example`.
- `weeddet_v6b.py` held **byte-stable** (recorded as an R100 rename; excluded
  from formatters) — its Gate-4 line references stay valid.

**Verified:** 105 tests + 16 subtests pass; ruff/black/pre-commit green;
`pip install -e ".[dev]"`, `agrinav --help`, and
`python -m agrinav.training.riceseg_pretrain --self-test` all work.

## Verify (resume checks)

```bash
pip install -e ".[dev]"                                   # or use existing .venv/
python -m agrinav.training.riceseg_pretrain --self-test   # no data needed
pytest                                                    # 105 passed + 16 subtests
ruff check . && black --check .
```

## Open items (needs owner / decision)

- [ ] **Run the Colab GPU notebook** `notebooks/weeddet_detector_colab.ipynb` for
      the first real detector convergence + qualitative-val pass. Needs
      `RiceSEG.zip` in `MyDrive/agrinav_data/` and the `detector_v1/split_v1` json
      (run `detector_data_prep_colab.ipynb` first if absent).
- [ ] **For a defensible mAP (deferred Gate-4 remainder):** COCO AP evaluator +
      one canonical decode/postprocess (P1-6) + perf P1-4 (anchor count) /
      P1-5 (`batched_nms`). The current pipeline is exploratory until these land.
- [ ] **Run `/code-review ultra`** on the branch (user-triggered; Claude cannot
      launch it).
- [ ] **Delete stray `~` dir** at `Downloads/agrinav_full/~` (outside the repo;
      Claude-tooling cruft) — needs confirmation before removal.

## Deferred / backlog (documented in ADR 0002)

- Committed dependency lockfile once the supported torch/CUDA matrix is defined.
- Enable stricter ruff rule families (`B`, `UP`, `SIM`, `PTH`) and make mypy
  blocking — including formatting `weeddet_v6b.py` — during the **Gate-4**
  detector rewrite.

## Key decisions & where they live

| Topic | Location |
|-------|----------|
| src/ layout | `docs/adr/0001-src-layout.md` |
| Packaging, tooling, CI, torch pinning, formatter exclusions | `docs/adr/0002-packaging-and-ci.md` |
| Gated research→deployment roadmap | `docs/audits/2026-07-20/AGRINAV_DEPLOYMENT_ROADMAP_2026-07-20.md` |
| Detector open correctness items (Gate 4) | `docs/audits/2026-07-20/PHASE2_DETECTOR_FIXLOG_2026-07-21.md` |
| What to do next / stop conditions | `START_HERE.md`, `active/ACTIVE_NOTES.md` |

## Environment notes

- Python 3.12 local; `.venv/` (gitignored) holds a verification env with CPU
  torch installed. CI installs CPU torch from the PyTorch CPU wheel index.
- Local RiceSEG images are extracted at
  `~/agrinav_data/incoming/extracted/riceseg/` (the dir that **contains** `global
  rice segmentation/`) — that path is the `--images-root` for the local detector
  smoke; the split json is `artifacts/detector_v1/split_v1/train.coco.json`.
