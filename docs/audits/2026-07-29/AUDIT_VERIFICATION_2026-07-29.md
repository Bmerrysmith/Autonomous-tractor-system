# Verification of the two 2026-07-29 AgriNav audits

**Date:** 2026-07-29
**Scope:** independent re-verification of
`AGRINAV_DEEP_TECHNICAL_AUDIT_2026-07-29.md` and
`AGRINAV_PHASE2_DETECTOR_FEASIBILITY_AUDIT_2026-07-29.md`, plus a fresh audit of
the local repository, the connected Google Drive, and the GitHub remote.
**Method:** every quantitative claim was recomputed from the primary artifacts
(the ZIP, the source deliverable, the working tree, the Drive metadata/manifests).
Nothing in this document is carried over from the audits on trust.
**Nothing was modified:** read-only on Drive and GitHub; the only write in the
repository is this file.

---

## 1. Headline

Both audits are **substantially correct on the data facts** — every checkable
number reproduces, most of them to the exact digit. Their code findings that I
could re-check are also real.

Two things they get materially wrong, both of which change what to do next:

1. **They audited a commit that is no longer HEAD.** Both audits pin
   `06c95e0`. The local working tree is at `2161fc9`, four commits ahead, on
   branch `feat/training-observability`, **and none of it is pushed**.
   `origin/master` is still `06c95e0`. Several of their findings (checkpoint
   resumability, checkpoint portability, BN-policy confound, validation loader)
   are partly or wholly addressed in the working tree.
2. **The "NO-GO, do not spend the GPU budget" decision is moot — the run already
   happened, twice.** Google Drive holds two completed phase-2 run directories
   from 2026-07-28, and `run_manifest.json` records
   `git_commit: 06c95e052a5d2d6097eb34ed707168bb4284711f`. So both existing
   detector checkpoints were trained **on the contaminated archive**, **with
   training-loss checkpoint selection**, by the exact code the audits condemn.
   The question is no longer whether to start; it is what to do with two
   artifacts that are already scientifically void.

And one finding neither audit made, which changes the repair plan:

3. **The archive is not merely mis-assigned, it is incomplete.** 263 of the
   2,579 images named in `grouped_split.json` are **absent** from the ZIP — 188
   of them intended *train*, 45 intended *valid*. The archive therefore cannot
   be repaired by re-sorting its own contents. Root cause identified below.

---

## 2. Claim-by-claim verification

### 2.1 Data claims — all confirmed, exactly

Recomputed from `C:\Users\Benny Merr\Downloads\RICE_curated_phase2.zip`
(931,860,119 bytes) against its own bundled `grouped_split.json`.

| Claim (both audits) | Audited value | Recomputed | Verdict |
|---|---:|---:|---|
| physical train images | 1,798 | 1,798 | ✅ |
| physical valid images | 518 | 518 | ✅ |
| intended split (manifest) | 1800 / 518 / 261 | 1800 / 518 / 261 | ✅ |
| split mismatches | 940 | **940** | ✅ |
| intended-test images in train/valid | 231 | **231** | ✅ |
| intended train → exported train | 1,261 | 1,261 | ✅ |
| intended train → exported valid | 351 | 351 | ✅ |
| intended valid → exported train | 358 | 358 | ✅ |
| intended valid → exported valid | 115 | 115 | ✅ |
| intended test → exported train | 179 | 179 | ✅ |
| intended test → exported valid | 52 | 52 | ✅ |
| exact-hash overlap train↔valid | none | **0 duplicate hashes at all** (2,316 unique) | ✅ |
| train images with raw dims ≠ COCO dims | 148 / 1798 | **148** | ✅ |
| valid images with raw dims ≠ COCO dims | 41 / 518 | **41** | ✅ |
| `exif_transpose` reconciles every case | yes | **189/189** | ✅ |
| annotations on affected train images | 2,412 (328 weed) | **2,412 (328 weed)** | ✅ |
| annotations on affected valid images | 681 (102 weed) | **681 (102 weed)** | ✅ |
| COCO-small fraction after resize to 512 | 56% | **56.0% train / 56.6% valid** | ✅ |
| min side < 16 px | "roughly one third" | **32.6% / 32.5%** | ✅ |
| median box at 512 | ~21 × 42 | **20.8 × 41.7** | ✅ |
| median objects per image | 31 | **31 train / 33 valid** | ✅ |
| rice:weed instance ratio | 6.8 : 1 | **6.77 : 1 train, 6.82 : 1 valid** | ✅ |
| degenerate/boundary annotations | ~111, ≥2 zero-width | **109 boundary cases; exactly 2 zero-width** | ✅ |
| grouped-split expected counts (2,579 imgs / 81,204 boxes) | as tabled | **matches the source annotations exactly** (70,752 rice + 10,452 weed) | ✅ |
| local Roboflow export has 1800/518/261 and +47 annotations | +47 | **81,251 vs 81,204 = +47** | ✅ |
| SAM polygons unreviewed, prompts were the human boxes | as stated | provenance: `review_status: unreviewed`, `human_edit_state: none`, 81,204 boxes → 81,204 polygons | ✅ |
| Drive `RICE_curated_phase2.zip` | 931,860,119 B | 931,860,119 B | ✅ |
| Drive `riceseg_backbone.pth` | 94,344,719 B | 94,344,719 B | ✅ |
| anchors per image at 512 px | ~258,048 | 12 anchors × (128²+64²+32²) = **258,048** | ✅ |
| test suite at `06c95e0` | 158 passed + 16 subtests | (at HEAD: **185 passed + 16 subtests**, 84.5 s) | ✅ consistent |

The EXIF cases are true 90° rotations, not sloppy metadata: e.g.
`1a_image (101)_jpg.rf.wsK5hY9BQFovCsSw896D.jpg` has raw size `3648×2048`,
EXIF orientation tag `8`, and COCO dims `2048×3648`. The loader letterboxes the
**unrotated** pixels while the boxes live in **rotated** coordinates, so on
those 189 images the annotations are effectively transposed relative to the
image content. Affected mass: 3,093 of 73,089 boxes (4.2%), including 430 of
9,396 weed boxes (4.6%).

### 2.2 Code claims — status at HEAD (`2161fc9`), not at the audited commit

Still open, verified in the working tree:

| Finding | Location at HEAD | Status |
|---|---|---|
| class-agnostic decode (`sigmoid().max(dim=1)` then one NMS over all labels) | `weeddet_v6b.py:1049`, `:1077-1083` | **open** |
| EXIF ignored on load | `weeddet_v6b.py:1234`, `:1319` | **open** |
| no model→COCO adapter; selection is not detection quality | `weeddet_v6b.py:1789-1806` | **partly fixed** — see below |
| notebook gates via `get_ipython().system(...)`, exit codes ignored | notebook cells 8, 12, 14, 17 | **open, and now worse** |
| sealed-test check is a `'/test/'` substring test; `extractall` unvalidated | notebook cell 10 | **open** |
| requested CUDA silently downgrades to CPU | `weeddet_train.py:246-252` | **open** |
| config validation checks keys, not values or cross-field invariants | `weeddet_train.py:255-277` | **open** |
| partial category-name matching accepted | `weeddet_train.py:189-193` | **open** (latent, see §3.5) |
| tiny/invalid boxes dropped silently (`w > 1 and h > 1`) | `weeddet_v6b.py:1325` | **open** |
| `assert` guards split safety (vanishes under `python -O`) | `build_detector_split.py:39`, `:128` | **open** |
| `unpad_boxes` does not clip to original bounds | `weeddet_v6b.py:277-285` | **open** |
| negatives never augmented (`if self.augment and len(b5)`) | `weeddet_v6b.py:1338` | **open** |
| broad `except Exception` around `torchvision.transforms` | `weeddet_v6b.py:57-58` | **open** |
| stale `CLASS_NAMES = ['Rice']` | `weeddet_v6b.py:979`, `:1109` | **open** |
| warmup applied after the first optimizer step | `weeddet_v6b.py:1738-1749` vs `:1402-1417` | **open** (see §3.6) |
| monolith excluded from ruff/black | `pyproject.toml:73-78`, `:96-105` | **open**, and the stated rationale is now void (§3.7) |
| `PyYAML` required but undeclared | `pyproject.toml:28-36` vs `weeddet_train.py:261` | **open** |
| no project license | `pyproject.toml:20-23` | **open** |
| Black 24.10.0 in CI vs 26.3.1 in `[dev]`; MyPy `continue-on-error`; no coverage gate; no wheel-install smoke | `.github/workflows/ci.yml:32`, `:53-56`, `:75-88` | **open** |
| Makefile help says "ruff-format + black", target runs `ruff check --fix` + black; `clean` is Unix-only | `Makefile:19`, `:35-37`, `:58-60` | **open** |
| no `--resume` | — | **open, and acknowledged** in `docs/HANDOFF.md` |

Fixed since the audited commit (`bf14514`, `2a6d063`, `89c24b9`, `2161fc9` — all
local-only):

- **Checkpoint contents.** `_payload` now carries `epoch`, `global_step`,
  EMA `state_dict`, `raw_state_dict`, `optimizer`, `scheduler`, `warmup`,
  `scaler`, `best_metric_name`/`value`, `class_names`, `num_classes`, and a
  provenance-string-only config. New checkpoints load under
  `weights_only=True`; legacy ones are repaired by
  `_alias_legacy_pickle_names`. Periodic checkpoints no longer omit optimizer
  state. **Still missing:** RNG state and any data-version/manifest hash — so
  "exact resume" and full provenance remain unmet.
- **Held-out validation exists** (`--val-ann-file` / `--val-images-root`), scores
  both raw and EMA weights, selects on the EMA — the network actually saved —
  and refuses a `test*` annotation file. **But the metric is validation
  *loss*, not AP.** The audits' core point (selection is not evidence of
  detection quality) still stands; the fix moved from "wrong quantity" to
  "proxy quantity".
- **Terminal artifacts:** `weeddet_last.pth` every epoch, `status.json`,
  `metrics.jsonl` (fsynced per epoch), atomic checkpoint writes,
  NaN-loss and zero-usable-batch hard failures.
- **`bn_policy` is now an explicit flag** (`auto|freeze_pretrained|trainable`),
  which is exactly the fix both audits asked for on the ImageNet-vs-RiceSEG
  confound. The default (`auto`) still reproduces the confounded behaviour, so
  the ImageNet control must be run with `--bn-policy trainable`.
- **Anchor-coverage audit added** (`agrinav data-anchor-audit`), which refuted a
  standing hypothesis: only 0.48% of train boxes fall below IoU 0.5 against the
  best anchor.

### 2.3 One claim to downgrade

Deep audit finding 27 (`max_dets` path is "misleading") — the failure mode is
louder than described. Measured directly:

```
max_dets=100 -> is_standard=True   ap=1.0000  ap50=1.0000
max_dets=300 -> is_standard=False  ap=-1.0    ap50=1.0000
```

A nonstandard `max_dets` makes the primary `ap` return the pycocotools
sentinel `-1.0`, not a plausible-but-wrong number, and the result object
already exposes `is_standard_maxdets=False`. Real, worth fixing, but not a
silent-corruption risk. Severity: Medium, not High.

Deep audit finding 34 (`.gitignore` misses `.venv-audit*`) describes the
auditor's own scratch environments; none exist in the worktree now. Informational
at most.

---

## 3. Findings the audits missed

### 3.1 The archive is the native split minus its test folder — `grouped_split.json` was never applied

The source deliverable
`Downloads\agrinav_intake_2026-07-21\deliverable\detection\RICE\` contains
`images/{train,valid,test}` with **1,798 / 518 / 263** images and matching COCO
files. The ZIP contains exactly the train and valid folders and nothing else.

Cross-tabulating `filter_decisions.csv` (whose `split` column is the *native*
split) against `grouped_split.json`:

| intended → native | test | train | valid |
|---|---:|---:|---:|
| **train** | 188 | 1,261 | 351 |
| **valid** | 45 | 358 | 115 |
| **test** | 30 | 179 | 52 |

The 263 manifest images absent from the ZIP are precisely the native `test/`
folder: 188 intended-train + 45 intended-valid + 30 intended-test. So the build
step was "drop the folder called test", not "apply the grouped map". That single
mistake produces both halves of the damage:

- 231 intended-test images **leaked in** (the audits found this), and
- 233 intended train/valid images **left out** (the audits did not).

Consequence for the repair plan: the deep audit's instruction to "rebuild the
archive from the canonical grouped membership map" is right, but it cannot be
done from the ZIP. It must be done from the source deliverable — which is
present locally and complete (all 2,579 images, all 81,204 boxes, plus a
`test/` folder). No re-download or re-annotation is needed.

### 3.2 Two contaminated checkpoints already exist on Drive

`MyDrive/agrinav_data/rice_phase2/runs/` holds two completed runs:

| Run | Contents |
|---|---|
| `weeddet_rice_20260728_190213` | `run_manifest.json`, `weeddet_epoch{4,8,12,16}.pth`, `weeddet_best.pth` (364 MB) |
| `weeddet_rice_20260728_203701` | same shape; `best` last written 21:00:23, ~23 min after start |

`run_manifest.json` (20:37 run) records `git_commit: 06c95e05…`,
`git_branch: master`, `git_dirty: false`, `torch 2.11.0+cu128`,
A100-SXM4-80GB, `backbone_sha256: 25a6f6d7…7bb`, seed 42. Neither directory
contains `status.json`, `metrics.jsonl`, `weeddet_last.pth`, or `train.log`,
and `weeddet_best.pth` (364 MB) is larger than the periodic checkpoints
(258 MB) — both signatures of the pre-`2a6d063` payload. Conclusion:

- these runs used **training-loss** selection (no validation loader existed at
  that commit), and
- they trained on the archive with 231 intended-test images inside.

The manifest's own note — *"Test split sealed and absent from the archive"* — is
false. That sentence is the single most dangerous artifact in the project right
now, because it will be believed later.

Roughly 88.5% (231/261) of the intended sealed test set has now been trained on.
Only the 30 intended-test images that happened to sit in the native test folder
are clean. Treat the intended test split as burned, as the deep audit
recommends, and build a fresh grouped test from the 2,579-image source.

### 3.3 The local fixes are unpushed, and the notebook clones `master`

`origin/master` = `06c95e0`. `feat/training-observability` (4 commits, all of
this session's fixes) has **no upstream**, as do `feat/coco-evaluator`,
`fix/checkpoint-portability`, `fix/overfit-gate`,
`feat/detector-riceseg-backbone`, and three `chore/*` branches. Effects:

- CI has never run on any of that work (CI triggers on `master`/`main` push and
  PRs);
- the Colab notebook's cell 6 clones `BRANCH = 'master'`, so a run started today
  gets `06c95e0` — which does **not** accept `--val-ann-file`. Cell 17 as
  written in the working tree would fail immediately against that clone. Either
  push the branch or the notebook cannot deliver the val-based selection its own
  comments advertise.

### 3.4 The phase-2 dataset has no dataset card

`docs/detector_dataset_card.md` documents a *different* dataset: the
RiceSEG-derived `detector_split_v1` (245 images, 8,200 weed polygons, 70/15/15,
seed 20260720). The deep audit cites its lines 54-55 as asserting leakage-free
grouping "as current truth" for the phase-2 archive — that is a mis-attribution.
The real gap is worse: the curated RICE phase-2 dataset is mentioned **only** in
`docs/HANDOFF.md`. No data card, no license record, no source attribution, no
counts, no known-limitations section. `docs/EXTERNAL_DATASETS_AUDIT.md` contains
no licensing entry for the Roboflow RICE export at all. (By contrast, the
RiceSEG licensing posture in `data/rice_training_curated/README.md` is documented
carefully and honestly — the feasibility audit's "records disagree between
unknown, CC BY 4.0 and public-domain" overstates that part.)

### 3.5 The archive ships two disagreeing category schemas

| File | Categories |
|---|---|
| `annotations/instances_{train,valid}.coco.json` (85 MB / 24 MB, carries SAM polygons) | `1: rice_protect`, `2: weed_target` |
| `images/{train,valid}/_annotations.coco.json` (7 MB / 2 MB, boxes only) | `0: rice-kF4t-rice-kF4t-MYY1-rice`, `1: rice`, `2: weed` |

Box counts are identical in both (56,502 / 16,587), so the boxes agree; only the
naming differs. The config and notebook point at the outer files, which match
`class_names: [rice_protect, weed_target]`. But `_CocoSplitDataset` accepts a
*partial* name match, so pointing `--ann-file` at the inner file with
`--class-names rice,weed` would train happily, and pointing it there with only
one matching name would silently train a one-class detector. For this particular
pair the mismatch fails loudly (zero names match), so the risk is latent rather
than active — but it is one flag away.

Useful for the repair: every COCO `images[]` record already carries a `sha256`
field (per `MANIFEST/provenance.json`), so a membership-and-integrity preflight
can be written without rehashing anything.

### 3.6 The intended grouped split has residual block-boundary leakage

`grouped_split.json`'s method is "40-frame contiguous blocks". Blocks cut
sequences, so frames adjacent in time can land in different splits. Measured on
the manifest (crude family parsing — treat as an order-of-magnitude figure):
**35 adjacent-frame pairs out of 1,866 (1.9%) cross a split boundary**, and 8 of
9 detected families span more than one split by design. This is far milder than
the 231-image leak, but it means "leakage-free" is the wrong word even for the
*correct* map. Either drop a guard band at block boundaries or state the residual
explicitly.

### 3.7 The `weeddet_v6b.py` lint exemption rationale is void

`pyproject.toml` excludes the monolith from ruff and black because it "is held
byte-stable pending its Gate-4 correctness rewrite: the July-20 audit references
its exact line numbers". The file has since grown by 292 net lines (1,579 →
1,871) across `bf14514`/`2a6d063`, and the July-29 audits' own line references
are already stale by ~2-20 lines. The exemption now buys nothing and costs lint
coverage on the highest-risk file in the repository.

### 3.8 The overfit gate confirms as described

`weeddet_train.py:528` — pass condition is exactly
`not (final_loss < initial_loss)` → fail. Loss direction only; no decode, no
ranking, no AP. Notebook cell 14 runs it with `--no-pretrained-backbone`, so the
production warm-start path is not what the gate exercises. Both audits correct.

---

## 4. Repository, Drive, and GitHub state

**Local** — `C:\Users\Benny Merr\Downloads\agrinav_full\agrinav_full`
HEAD `2161fc9` on `feat/training-observability`, working tree clean.
`185 passed, 16 subtests, 10 warnings in 84.51s` (venv Python 3.12.13,
torch 2.13.0+cpu). `.env` holds `ROBOFLOW_API_KEY` and is correctly ignored;
no tracked secrets, no tracked large binaries.

**GitHub** — `Bmerrysmith/Autonomous-tractor-system`, 18 remote heads.
`origin/master` = `06c95e0`, `origin/main` = `aba64a4` (divergent legacy).
Six local branches carry unpushed work. Two workflows: `ci.yml`,
`qodana_code_quality.yml`. No issues track any of the blockers (consistent with
the feasibility audit).

**Drive** — both files the earlier "missing file" claim disputed are present and
byte-size-exact:
`agrinav_data/rice_phase2/RICE_curated_phase2.zip` (931,860,119 B, uploaded
2026-07-27 15:35) and `agrinav_data/out/riceseg_backbone.pth` (94,344,719 B),
alongside `riceseg_backbone.pth.fullckpt.pth` (329 MB) and a manifest JSON.
Plus the two phase-2 run directories in §3.2 and a user-edited
`Copy of weeddet_rice_phase2_colab.ipynb` (modified 2026-07-28 21:01).

**Doc coherence** — `START_HERE.md:52` still says *"Do **not** train the detector
yet"* while `docs/HANDOFF.md` reports four completed phase-2 training runs, and
`README.md:98` carries a "Quick status (2026-07-20)" table. Three documents, three
different answers. Both audits flagged this; it is still true and now has a
concrete cost, because the thing the docs forbid has already been done twice.

---

## 5. Recommended order, revised

The audits' repair orders are sound. Two changes, given what actually happened:

**P0 — before anything else**

1. **Annotate the two Drive runs as void-for-evaluation.** Do not delete them;
   drop a `VOID.md` in each run directory recording *why* (leaked archive,
   train-loss selection, commit `06c95e0`) and correct the false
   "test split sealed and absent" line in each `run_manifest.json`. Provenance
   damage compounds silently; a wrong manifest outlives the memory of the run.
2. **Rebuild the dataset from `deliverable/detection/RICE`**, not from the ZIP:
   apply `grouped_split.json` to all 2,579 images, use the human boxes (not the
   SAM polygons), normalize EXIF at export and strip the tag, emit
   train/valid/test COCO files plus SHA256 for the archive and every JSON, and
   add a preflight that fails on any membership mismatch (the per-image `sha256`
   already in the COCO records makes this cheap). Expected: 1800/518/261 and
   59,694 / 15,226 / 6,284 boxes. Then build a **fresh** grouped test — the
   intended one is 88.5% burned.
3. **Push `feat/training-observability`** (or repoint the notebook at a pinned
   SHA). Right now the notebook and the code it clones disagree, and CI has never
   seen the fixes.
4. **Reconcile `START_HERE.md`, `README.md`, and `HANDOFF.md`** onto one gate
   statement.

**P1 — before any run whose number will be quoted**

5. Class-aware decode: expand `(anchor, class)` candidates, then
   `torchvision.ops.batched_nms` or validated per-class Soft-NMS. One canonical
   postprocessor.
6. One model→COCO adapter (inverse letterbox with clipping, class-id mapping,
   thresholds, NMS, formatting), wired into training so selection uses AP50 /
   AP@[.50:.95] / per-class AP — replacing validation *loss*.
7. Replace the loss-direction overfit gate with a decoded-AP gate on the
   production construction path.
8. Fail closed on `--device cuda` without CUDA; typed config validation;
   preflight rejection report for dropped annotations.
9. Add RNG state and dataset hashes to resume checkpoints; implement `--resume`.
10. Drop the `weeddet_v6b.py` lint exemption, unify Black versions, add a
    wheel-install smoke, declare `PyYAML`.

**P2** — baselines under the identical split/evaluator (FCOS or RetinaNet with a
small-object level; a YOLO P2 variant; an RT-DETR-family model), and the
scratch / ImageNet / RiceSEG comparison under matched BN policy. The audits'
literature framing here is reasonable and I have no correction to it.

---

## 6. Acted on, same day

Everything in P0 above was done in this session.

**Provenance cleanup (Google Drive, read-then-write; nothing deleted).**
`VOID.md` and `run_manifest.CORRECTED.json` written into both
`runs/weeddet_rice_20260728_190213/` and `runs/weeddet_rice_20260728_203701/`. The
Drive connector cannot rewrite a file in place, so the false
`"Test split sealed and absent from the archive"` note is corrected in the
`.CORRECTED.json` sibling, which also records the split-integrity numbers, the
selection metric actually used, and the code defects present at `06c95e0`. Each
`VOID.md` keeps the distinction the checkpoints deserve: void as *scientific*
evidence, still valid as *engineering* evidence for the `save_every` finding.
`DO_NOT_TRAIN_ON_RICE_curated_phase2.md` written next to the archive itself, since
that is the file a future session would otherwise reach for.

**Dataset rebuilt.** New module `src/agrinav/data/build_rice_phase2.py`
(`agrinav data-build-rice-phase2 build|preflight|package`, 38 tests). Run against
the source deliverable, it produced exactly the manifest's split:

| Split | Images | Boxes | rice | weed | EXIF-normalized |
|---|---:|---:|---:|---:|---:|
| train | 1,800 | 59,691 | 52,194 | 7,497 | 35 |
| valid | 518 | 15,226 | 13,201 | 2,025 | 80 |
| test (burned) | 261 | 6,284 | 5,355 | 929 | 99 |

214 of 2,579 images had EXIF orientation applied and stripped; only those were
re-encoded. 3 annotations rejected (all out-of-bounds by >1 px), 115 clipped, every
case itemized. Zero duplicate image hashes. Reconciled against the manifest's own
`per_split` totals: emitted + rejected == source-assigned, exactly, per class per
split. Packaged to `RICE_phase2_rebuild.zip` (644,891,718 B, sha256
`57484b9d…f190db27`, test excluded), then extracted and re-verified from the
archive: preflight clean, `splits_absent: ["test"]`.

The never-trained-on pool for a replacement test split is **781 images** (539
currently-train, 160 currently-valid, 82 currently-test), flagged per image in
`manifests/split_membership.json`.

**Branch pushed.** `feat/training-observability` is now on `origin`, so CI can see
it and Colab can clone it.

**Notebook made fail-closed and pinned.** Cells 6/8/10/12/14 use
`subprocess.run(..., check=True)`; the checkout is pinned to
`2161fc98fe71916a373a3afeb46c1d3f7e26bb86` and asserts the trainer flags it needs
exist; extraction validates `PurePosixPath(member).parts` and refuses the banned
archive by name; the dataset preflight runs as a gate; the detached launch is
verified by liveness; the run manifest records the archive, annotation, and config
hashes plus the dataset provenance block.

**Docs reconciled.** New `docs/GATE_STATUS.md` is the single authoritative gate
file; `START_HERE.md` and `README.md` link to it instead of restating verdicts, and
`START_HERE.md` no longer conflates the RiceSEG-derived dataset card with the
curated-RICE dataset. New `docs/rice_phase2_dataset_card.md` gives the phase-2
dataset the card it never had.

Not done, deliberately: the replacement test split. `grouped_split.json` stores
`num_groups: 68` and `block_size: 40` but no per-file group ids, so the original
grouping cannot be reproduced from it. `derive_group_id()` is an explicit
re-derivation (142 groups across the three splits) and is labelled as such;
choosing the rule for a new split belongs in an ADR, not a silent default.

## 7. Bottom line

The audits are trustworthy on facts and mostly trustworthy on prescriptions. I
reproduced their data findings independently and to the digit. But they describe
a repository state that is four commits stale and a decision — "don't start the
run" — that had already been overtaken by two completed runs on the contaminated
archive.

So the actual position is worse than "not ready to train" and better than
"nothing works": there is now a **provenance cleanup** to do (two void
checkpoints and a false manifest line), a **dataset rebuild** that is fully
achievable from local files with no re-annotation, and a **correctness sprint**
of which roughly a third has already landed in an unpushed branch.
