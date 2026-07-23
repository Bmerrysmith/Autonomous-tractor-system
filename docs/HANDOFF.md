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

## Current status — 2026-07-22

**Gate 1 (reproducible, installable project) is complete and verified.** The
project is now an installable `agrinav` package with packaging, fast CI, and a
unified CLI. Not yet pushed. The next real engineering gate is **Gate 4**
(detector correctness in `weeddet_v6b.py`) per `START_HERE.md` and the roadmap.

**Active branch:** `chore/gate1-packaging-ci` (4 commits) off
`codex/repository-recovery`. **Unpushed.**

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

- [ ] **Push + open PR:** `git push -u origin chore/gate1-packaging-ci`.
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
