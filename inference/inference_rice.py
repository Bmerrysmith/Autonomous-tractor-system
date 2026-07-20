"""Disabled historical inference entry point.

The previous implementation treated every pixel outside a predicted rice box as
a weed/spray target. That spray-by-default policy is unsafe, and the code was
also incompatible with the active model and preprocessing pipeline.

This module intentionally provides no inference or treatment command path.
Build a replacement only after the fail-closed safety, geometry, checkpoint,
and export gates in the July 20 audit are implemented and tested.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


DISABLED_REASON = (
    "Disabled unsafe historical prototype: absence of a rice detection must "
    "never be interpreted as weed evidence or permission to spray. See "
    "docs/audits/2026-07-20/ before implementing a replacement."
)


def build_parser() -> argparse.ArgumentParser:
    """Return the quarantine CLI parser without loading ML dependencies."""
    return argparse.ArgumentParser(
        description="Quarantined AgriNav inference prototype (disabled)",
        epilog=DISABLED_REASON,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed for every invocation."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.error(DISABLED_REASON)
    return 2  # pragma: no cover - argparse.error exits


if __name__ == "__main__":
    main()
