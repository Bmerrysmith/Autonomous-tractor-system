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

## Current status — 2026-07-27

**Detector can now warm-start from the RiceSEG backbone, and phase-2 (real RICE
data, not RiceSEG) is wired end to end.** `weeddet_train.py` gained
`--riceseg-backbone PATH`: a picklable `_RicesegBackboneInit` callable (module-level,
survives checkpoint pickling) that loads the phase-1 backbone instead of ImageNet
and leaves BN trainable (in-domain running stats). Omit the flag for the
ImageNet-only control — same command, one flag, that's the whole A/B. Merged to
`master` `e16ddaa`.

New: `configs/training/detector_rice_phase2.yaml` (2-class:
`rice_protect`/`weed_target`, num_classes derived from `class_names`) +
`notebooks/weeddet_rice_phase2_colab.ipynb` (21 cells) targeting the **curated,
grouped RICE split**, not RiceSEG and not Roboflow's leaky native split. Merged
`master` `8cd1832`.

**Blocking on the user:** upload `RICE_curated_phase2.zip` (932 MB, built from
`Downloads/agrinav_intake_2026-07-21/deliverable/detection/RICE/`, sha256
`2161e069…`, test split verifiably absent) to
`MyDrive/agrinav_data/rice_phase2/`, then run the notebook. Nothing else is
required to start a real phase-2 training run.

Git LFS was installed locally but is **not used and should not be** — nothing
tracked, `.gitattributes` unchanged, `artifacts/`/`*.pth` stay gitignored. GitHub
free tier is 1 GiB storage + 1 GiB bandwidth/month; the notebook clones the repo
fresh every Colab session, so committing the 932 MB image set via LFS would burn
the monthly bandwidth quota on the first run. Drive stays correct for data;
DVC remains the CLAUDE.md-prescribed answer if this needs proper versioning later
(still not set up).

The detector pipeline (either backbone) is still **exploratory** — checkpoint
selection is train-loss only, there's no mAP cell, because the canonical decode
(Gate-4 remainder, P1-6) doesn't exist yet. Test split is **sealed** everywhere.

Phase-2 detector training pipeline (loader/CLI/smoke) is built, smoke-validated,
and independently reviewed and committed — see the 2026-07-23 entry below for
that work.

**PHASE 1 (seg pretraining) IS CLOSED.** The overfit gate passed for the first
time (best mIoU 0.8631 >= 0.8000), and the full `ImageNet->RiceSEG` production run
then completed: **best mIoU 0.5827 @ epoch 30**, backbone at
`MyDrive/agrinav_data/out/riceseg_backbone.pth`. It **reproduces the 2026-07-21
run within 0.001 mIoU** (same split/seed/recipe, best epochs 11 apart) — a
reproducibility check, not an improvement. **Decision 2026-07-23: not training it
further.** The plateau is not a step-budget problem (30 ep here = ~6,900 steps vs
1,000 for the whole 500-epoch gate; val flat since ep19 while train loss kept
falling; senescent is a data ceiling across both architectures). Minority levers
(`focal_tversky`, `dice_weighted`, `warmup`/`backbone-lr-mult`) are built but
deliberately untried on full data. Full analysis:
`docs/research/RICESEG_PRETRAIN_RESULTS.md`.

**Next phase: the detector.** The open question this backbone exists to answer is
whether it beats ImageNet-only on detector mAP — no further seg training answers
that.

The gate had been failing on a pipeline that was
never broken — two defects in the *gate* were fixed (class-coverage truncation +
epoch-floored instead of step-floored budget). Threshold 0.80 is **validated, not
lowered**. Details and evidence: `docs/research/RICESEG_PRETRAIN_RESULTS.md`.

**Merged to `master` 2026-07-23:** `chore/gate1-packaging-ci`,
`feat/riceseg-training-optimization`, `feat/phase2-detector-pipeline`,
`fix/checkpoint-portability`, `chore/black-security-bump`, the two Colab-branch
chores, and `fix/overfit-gate` are all merged into `master` and pushed. The 3
stale pre-recovery local `master` commits (V7 era) were preserved on
`archive/pre-recovery-master` before master was reset to the recovered origin
lineage.

## 2026-07-23 — Overfit-gate fix (`fix/overfit-gate`, master `8946981`)

1. **`_stratified_overfit_subset` dropped the class it scanned for.** It appended
   the duckweed tile, then returned `chosen[:n]` and truncated it off (the only
   in-range duckweed tile is at scan index 360). duckweed was absent from the GT,
   so its IoU was `nan` or `0.00` per epoch and `np.nanmean` divided by **5 or 6
   depending on the epoch** — the failing run's epoch 43 (0.6223) and epoch 45
   (0.5010) have *identical per-class IoUs*. Now greedy set-cover, selected first
   and never truncated; uncoverable classes warn instead of skewing the mean.
2. **The gate floored epochs (60), not steps** — 8 tiles at batch 4 with
   `drop_last` is 120 AdamW steps. `_overfit_epochs` floors the step budget
   (1000); the cosine still spans exactly `epochs`. Evidence this mattered: the
   passing run was at **0.6405 at epoch 60** (where the old cap stopped) and did
   not cross 0.80 until ~epoch 160.

Also landed this session: `weights_only` checkpoint-load portability for torch
>= 2.6 (`fix/checkpoint-portability`, was going to break the Colab qualitative-val
cell), `black` 24.10.0 -> 26.3.1 for the Dependabot cache-path advisory, and both
Colab notebooks repointed from stale branches to `master` + the src layout.

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

## Done this session (2026-07-27 — RiceSEG backbone hookup + phase-2 RICE data)

- `weeddet_train.py`: `_RicesegBackboneInit` (module-level class, not a closure —
  configs get pickled into checkpoints, so the callable must survive
  `torch.load`), `_make_riceseg_backbone_init()` (raises `FileNotFoundError` on a
  bad path before training starts), `--riceseg-backbone` CLI flag,
  `"riceseg_backbone": None` in `_HARD_DEFAULTS` + `_CLI_TO_CONFIG`.
  `build_config` sets `pretrained_backbone = False` when the flag is present, so
  the saved config is unambiguous about which init path ran.
- `weeddet_v6b.py`: two guarded seams in `train_with_progress`, both
  `if key is None: <original behavior>` — `config['backbone_init']` callable
  overrides the ImageNet load + BN-policy call at init and skips the recurring
  BN-policy call in the per-epoch loop. Absent key is byte-identical to before;
  confirmed the existing audit's only line reference into this file
  (`weeddet_v6b.py:1076`) still lands before both edits.
- Data: built `RICE_curated_phase2.zip` from the curated/grouped RICE detection
  set (2,318 imgs: 1,799 train + 519 valid; both annotation JSONs;
  `grouped_split.json`; `filter_decisions.csv`) — sha256 `2161e069…`, test split
  asserted absent at build time.
  `configs/training/detector_rice_phase2.yaml` (2-class head) +
  `notebooks/weeddet_rice_phase2_colab.ipynb` — clone cell greps for
  `--riceseg-backbone` to fail loudly on a stale cached clone, extract cell
  asserts no `/test/` member in the archive, self-test + overfit-8 gate on real
  RICE images before the full run, run manifest records the backbone sha256 for
  traceability back to the exact phase-1 file.
- **Verified:** full suite 158 tests + 16 subtests pass; ruff/black clean;
  notebook JSON validated, every code cell parses, greped for stray test-split
  refs (none). Checked git/LFS state before advising on it (nothing tracked).
- Merged + pushed: `master` `e16ddaa` (backbone hookup), `master` `8cd1832`
  (phase-2 config + notebook). `git status` clean, `origin/master` in sync as of
  this session.

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
pytest                                                    # 136 passed + 16 subtests
ruff check . && black --check .
```

## Open items (needs owner / decision)

- [x] ~~Run full `ImageNet->RiceSEG` pretraining~~ — **DONE 2026-07-23**, best
      mIoU 0.5827 @ ep30. Phase closed; see results log for why it was not
      extended.
- [ ] **Paste the run manifest fields** (`git_commit`, backbone `sha256`, env
      versions) from `MyDrive/agrinav_data/out/riceseg_backbone.pth.manifest.json`
      into the `2026-07-23` run entry in
      `docs/research/RICESEG_PRETRAIN_RESULTS.md` — they are marked *not
      captured* there, so the run currently lacks full provenance.
- [x] ~~Phase-2b data decision — `rice-weed-seg` (2,579 imgs)~~ — **DECIDED
      2026-07-27**: use the curated/grouped RICE set (2,318 imgs after filtering),
      not either Roboflow export and not Roboflow's leaky native split. Packaged
      as `RICE_curated_phase2.zip`; config + notebook built. Still 2-class
      (`rice_protect`/`weed_target`) vs the 3-class RiceSEG segmentation head —
      no merge attempted, they're separate tasks (detector vs. segmentation).
- [ ] **User: upload `RICE_curated_phase2.zip`** (932 MB, at
      `C:\Users\Benny Merr\Downloads\`) to `MyDrive/agrinav_data/rice_phase2/`,
      then run `notebooks/weeddet_rice_phase2_colab.ipynb` (Runtime → GPU → Run
      all). This is the first real phase-2 training run, optionally warm-started
      from the phase-1 backbone via `--riceseg-backbone`.
- [ ] **Paste the phase-2 run manifest** into
      `docs/research/` (or wherever the phase-2 results log lands) once the
      Colab run finishes — `run_manifest.json` in the checkpoint dir already
      captures backbone sha256 + git commit.
- [ ] **Run the Colab GPU notebook** `notebooks/weeddet_detector_colab.ipynb` for
      the first real detector convergence + qualitative-val pass on RiceSEG data
      (separate from the RICE phase-2 notebook above). Needs `RiceSEG.zip` in
      `MyDrive/agrinav_data/` and the `detector_v1/split_v1` json (run
      `detector_data_prep_colab.ipynb` first if absent).
- [ ] **For a defensible mAP (deferred Gate-4 remainder):** COCO AP evaluator +
      one canonical decode/postprocess (P1-6) + perf P1-4 (anchor count) /
      P1-5 (`batched_nms`). The current pipeline is exploratory until these land.
- [ ] **Run `/code-review ultra`** on the branch (user-triggered; Claude cannot
      launch it).
- [ ] **Delete stray `~` dir** at `Downloads/agrinav_full/~` (outside the repo;
      Claude-tooling cruft) — needs confirmation before removal.
- [ ] **Stale Drive `code_snapshot_2783b8e/` folder** has only `pyproject.toml`,
      `__init__.py` files, and 3 configs — the large `.py` modules never
      transferred, and it's now several commits behind `master`. Recommend
      deleting it rather than risk running stale code from Drive; the Colab
      notebooks clone `master` fresh instead, which is the supported path.

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
