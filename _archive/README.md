# `_archive/` — Quarantined (do not use as active code)

Everything here is retained for provenance only. It is **not** part of the active
pipeline, must **not** be imported, and in several cases is known-unsafe or
known-buggy per the July 20 audit (`docs/audits/2026-07-20/`). Nothing was
deleted — it is moved here (Git history is intact) and can be restored with
`git mv` if ever needed.

| Folder | What | Why quarantined |
|---|---|---|
| `legacy_code/archive/` | v1–v5 era code + notebooks, VOC-era scripts | superseded; audit §12 flags invalid eval/training logic |
| `legacy_code/weeddet_trainingV7_1_T1.ipynb` | the old T7d detector notebook | mutable-notebook run identity + the translation-augmentation bug (audit P0-1) |
| `unsafe_inference/inference_rice.py` | historical inference entry point | **unsafe**: spray-by-default, incompatible with the active model (audit P0-2/P0-3) |
| `unsafe_inference/auto_annotate.py` | box→label auto-annotator | normalizes every plant to rice; not trusted ground truth (audit §11) |
| `historical_docs/` | stale trackers, handoffs, migration logs, Drive inventory, plus (added 2026-07-21) 5 pre-recovery analysis docs pulled forward from the older `Claude\Projects\Autonomous driving tractor` checkout | superseded by the audit/roadmap and current docs (audit §13). The accompanying binaries (5 figures under `images/`, 2 course-deliverable decks under `reports/`, and the research-analysis `.docx`) are **git-ignored** — retained locally / in the artifact store per the no-bulk-in-Git policy, not committed |

**Active code lives in:** `training/ models/ scripts/ tests/ configs/ data/ docs/ notebooks/`.
Start at `../START_HERE.md`.
