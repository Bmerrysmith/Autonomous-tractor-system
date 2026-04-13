"""Run all standalone research visualizations.

This script is intentionally self-contained. It does not import the cloned
training or inference modules from the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .model_training import train_lane_model
    from .synthetic_data import (
        generate_localization_sample,
        generate_planning_tradeoff_sample,
        generate_row_geometry_sample,
    )
    from .visualize import (
        plot_confidence,
        plot_localization,
        plot_localization_components,
        plot_prediction_scatter,
        plot_planning_tradeoffs,
        plot_row_geometry,
        plot_row_quality,
        plot_system_overview,
        plot_training_curves,
        plot_error_histograms,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from model_training import train_lane_model  # type: ignore
    from synthetic_data import (  # type: ignore
        generate_localization_sample,
        generate_planning_tradeoff_sample,
        generate_row_geometry_sample,
    )
    from visualize import (  # type: ignore
        plot_confidence,
        plot_localization,
        plot_localization_components,
        plot_prediction_scatter,
        plot_planning_tradeoffs,
        plot_row_geometry,
        plot_row_quality,
        plot_system_overview,
        plot_training_curves,
        plot_error_histograms,
    )


def build_figures(output_dir: Path, seed: int = 7) -> tuple[list[Path], dict[str, list[float]] | None, dict[str, list[float]] | None]:
    row_sample = generate_row_geometry_sample(seed=seed)
    localization_sample = generate_localization_sample(seed=seed + 4)
    planning_sample = generate_planning_tradeoff_sample()
    artifacts = train_lane_model(output_dir=output_dir, seed=seed + 10)

    generated = [
        plot_system_overview(output_dir),
        plot_row_geometry(row_sample, output_dir),
        plot_row_quality(row_sample, output_dir),
        plot_localization(localization_sample, output_dir),
        plot_localization_components(localization_sample, output_dir),
        plot_confidence(localization_sample, output_dir),
        plot_planning_tradeoffs(planning_sample, output_dir),
        plot_training_curves(artifacts.history, output_dir),
        plot_prediction_scatter(artifacts.y_true_test, artifacts.y_pred_test, artifacts.target_names, output_dir),
        plot_error_histograms(artifacts.y_true_test, artifacts.y_pred_test, artifacts.target_names, output_dir),
    ]
    generated.extend(artifacts.files)

    model_metrics = {
        "rmse": [float(x) for x in artifacts.rmse_per_target],
        "mae": [float(x) for x in artifacts.mae_per_target],
        "r2": [float(x) for x in artifacts.r2_per_target],
    }
    return generated, artifacts.history, model_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone research figures")
    parser.add_argument("--output-dir", default="research_outputs", help="Directory to save figures")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created, history, model_metrics = build_figures(output_dir, seed=args.seed)
    print("Generated figures:")
    for path in created:
        print(f"- {path}")

    if history is not None and model_metrics is not None:
        print("\nModel results:")
        print(f"- Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"- Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"- RMSE: {model_metrics['rmse']}")
        print(f"- MAE: {model_metrics['mae']}")
        print(f"- R2: {model_metrics['r2']}")


if __name__ == "__main__":
    main()
