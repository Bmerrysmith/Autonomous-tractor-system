"""Run all standalone research visualizations.

This script is intentionally self-contained. It does not import the cloned
training or inference modules from the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .synthetic_data import (
        generate_localization_sample,
        generate_planning_tradeoff_sample,
        generate_row_geometry_sample,
    )
    from .visualize import (
        plot_confidence,
        plot_localization,
        plot_localization_components,
        plot_planning_tradeoffs,
        plot_row_geometry,
        plot_row_quality,
        plot_system_overview,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from synthetic_data import (  # type: ignore
        generate_localization_sample,
        generate_planning_tradeoff_sample,
        generate_row_geometry_sample,
    )
    from visualize import (  # type: ignore
        plot_confidence,
        plot_localization,
        plot_localization_components,
        plot_planning_tradeoffs,
        plot_row_geometry,
        plot_row_quality,
        plot_system_overview,
    )


def build_figures(output_dir: Path, seed: int = 7) -> list[Path]:
    row_sample = generate_row_geometry_sample(seed=seed)
    localization_sample = generate_localization_sample(seed=seed + 4)
    planning_sample = generate_planning_tradeoff_sample()

    generated = [
        plot_system_overview(output_dir),
        plot_row_geometry(row_sample, output_dir),
        plot_row_quality(row_sample, output_dir),
        plot_localization(localization_sample, output_dir),
        plot_localization_components(localization_sample, output_dir),
        plot_confidence(localization_sample, output_dir),
        plot_planning_tradeoffs(planning_sample, output_dir),
    ]

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone research figures")
    parser.add_argument("--output-dir", default="research_outputs", help="Directory to save figures")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created = build_figures(output_dir, seed=args.seed)
    print("Generated figures:")
    for path in created:
        print(f"- {path}")


if __name__ == "__main__":
    main()
