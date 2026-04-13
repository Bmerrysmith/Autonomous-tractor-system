"""Publication-style visualizations for the standalone research toolkit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

try:
    from .synthetic_data import (
        LocalizationSample,
        PlanningTradeoffSample,
        RowGeometrySample,
        generate_sensor_fusion_signal,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from synthetic_data import (  # type: ignore
        LocalizationSample,
        PlanningTradeoffSample,
        RowGeometrySample,
        generate_sensor_fusion_signal,
    )


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "figure.facecolor": "#0f1117",
            "axes.facecolor": "#131722",
            "axes.edgecolor": "#8d93a1",
            "axes.labelcolor": "#e9edf5",
            "xtick.color": "#c6ccd8",
            "ytick.color": "#c6ccd8",
            "text.color": "#e9edf5",
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "grid.color": "#2f3747",
            "grid.alpha": 0.5,
            "axes.grid": True,
            "legend.frameon": False,
        }
    )


def _finalize(fig: plt.Figure, output_path: Path, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_row_geometry(sample: RowGeometrySample, output_dir: Path) -> Path:
    set_style()
    fig, ax = plt.subplots(figsize=(11, 5.8))

    ax.plot(sample.x, sample.left_row, color="#79d7ff", linewidth=2.2, label="Left row boundary")
    ax.plot(sample.x, sample.right_row, color="#ff9f7a", linewidth=2.2, label="Right row boundary")
    ax.plot(sample.x, sample.centerline, color="#f4e285", linewidth=2.4, linestyle="--", label="Centerline")
    ax.fill_between(sample.x, sample.roi_left, sample.roi_right, color="#58d68d", alpha=0.16, label="ROI corridor")

    ax.set_xlabel("Forward distance along row (m)")
    ax.set_ylabel("Lateral position (m)")
    ax.set_title("LiDAR crop-row geometry and ROI corridor")
    ax.legend(loc="upper right")
    ax.set_xlim(sample.x.min(), sample.x.max())

    return _finalize(fig, output_dir / "row_geometry.png", "LiDAR Row Geometry")


def plot_row_quality(sample: RowGeometrySample, output_dir: Path) -> Path:
    set_style()
    fig, ax1 = plt.subplots(figsize=(11, 5.2))

    ax1.plot(sample.x, sample.roi_width, color="#7cc6ff", linewidth=2.3, label="ROI width")
    ax1.set_xlabel("Forward distance along row (m)")
    ax1.set_ylabel("ROI width (m)")

    ax2 = ax1.twinx()
    ax2.plot(sample.x, sample.confidence, color="#ffcc66", linewidth=2.0, label="Detection confidence")
    ax2.set_ylabel("Confidence score")
    ax2.set_ylim(0.5, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("ROI width and geometry confidence over distance")

    return _finalize(fig, output_dir / "row_quality.png", "Row Quality")


def plot_localization(sample: LocalizationSample, output_dir: Path) -> Path:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    axes[0].plot(sample.ground_truth_x, sample.ground_truth_y, color="#f4e285", linewidth=2.4, label="Reference path")
    axes[0].plot(sample.gnss_x, sample.gnss_y, color="#ff8a80", alpha=0.72, linewidth=1.6, label="Raw GNSS")
    axes[0].plot(sample.ekf_x, sample.ekf_y, color="#79d7ff", linewidth=2.2, label="EKF fused")
    axes[0].set_xlabel("X position (m)")
    axes[0].set_ylabel("Y position (m)")
    axes[0].set_title("Trajectory comparison")
    axes[0].legend(loc="best")
    axes[0].axis("equal")

    axes[1].plot(sample.t, sample.position_error_gnss, color="#ff8a80", linewidth=1.8, label="GNSS error")
    axes[1].plot(sample.t, sample.position_error_ekf, color="#79d7ff", linewidth=2.0, label="EKF error")
    axes[1].plot(sample.t, sample.yaw_error_imu, color="#ffcc66", linewidth=1.5, alpha=0.85, label="IMU yaw error")
    axes[1].plot(sample.t, sample.yaw_error_ekf, color="#58d68d", linewidth=1.8, label="EKF yaw error")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Error magnitude")
    axes[1].set_title("Localization error over time")
    axes[1].legend(loc="upper right")

    return _finalize(fig, output_dir / "localization.png", "Localization")


def plot_localization_components(sample: LocalizationSample, output_dir: Path) -> Path:
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)

    axes[0].plot(sample.t, sample.yaw_truth, color="#f4e285", linewidth=2.0, label="Reference yaw")
    axes[0].plot(sample.t, sample.yaw_imu, color="#ff8a80", alpha=0.75, linewidth=1.4, label="IMU yaw")
    axes[0].plot(sample.t, sample.yaw_ekf, color="#79d7ff", linewidth=2.1, label="EKF yaw")
    axes[0].set_ylabel("Yaw (deg)")
    axes[0].set_title("Orientation fusion")
    axes[0].legend(loc="upper right")

    axes[1].plot(sample.t, sample.position_error_gnss, color="#ff8a80", linewidth=1.8, label="GNSS position error")
    axes[1].plot(sample.t, sample.position_error_ekf, color="#79d7ff", linewidth=2.0, label="EKF position error")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position error (m)")
    axes[1].set_title("Position fusion")
    axes[1].legend(loc="upper right")

    return _finalize(fig, output_dir / "localization_components.png", "Localization Components")


def plot_confidence(sample: LocalizationSample, output_dir: Path) -> Path:
    set_style()
    fusion = generate_sensor_fusion_signal(sample)
    fig, ax = plt.subplots(figsize=(11, 5.6))

    ax.plot(fusion["time"], fusion["confidence"], color="#79d7ff", linewidth=2.0, label="Fusion confidence")
    ax.plot(fusion["time"], fusion["corroboration"], color="#ffcc66", linewidth=1.8, label="Sensor corroboration")
    ax.plot(fusion["time"], fusion["fused"], color="#58d68d", linewidth=2.2, label="Final gate score")
    ax.fill_between(fusion["time"], 0.0, fusion["fused"], color="#58d68d", alpha=0.12)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Detection confidence and sensor corroboration")
    ax.legend(loc="upper right")

    return _finalize(fig, output_dir / "confidence_gate.png", "Confidence Gate")


def plot_planning_tradeoffs(sample: PlanningTradeoffSample, output_dir: Path) -> Path:
    set_style()
    fig, ax = plt.subplots(figsize=(11, 6.0))

    index = np.arange(len(sample.algorithms))
    width = 0.18

    ax.bar(index - 1.5 * width, sample.score_realtime, width, label="Real-time", color="#79d7ff")
    ax.bar(index - 0.5 * width, sample.score_constraint, width, label="Constraint handling", color="#ffcc66")
    ax.bar(index + 0.5 * width, sample.score_obstacle, width, label="Obstacle routing", color="#58d68d")
    ax.bar(index + 1.5 * width, sample.score_ros_maturity, width, label="ROS maturity", color="#ff8a80")

    ax.set_xticks(index)
    ax.set_xticklabels(sample.algorithms)
    ax.set_ylabel("Relative score (0-10)")
    ax.set_title("Path-planning trade-off comparison")
    ax.legend(loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(index, sample.response_time_ms, color="#f4e285", linewidth=2.3, marker="o", label="Response time")
    ax2.set_ylabel("Response time (ms)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=3)

    return _finalize(fig, output_dir / "planning_tradeoffs.png", "Planning Tradeoffs")


def plot_system_overview(output_dir: Path) -> Path:
    set_style()
    fig, ax = plt.subplots(figsize=(13, 7.0))
    ax.set_axis_off()

    nodes = [
        (0.05, 0.72, 0.22, 0.15, "LiDAR row\ndetection"),
        (0.33, 0.72, 0.22, 0.15, "ROI corridor\nconstraint"),
        (0.61, 0.72, 0.22, 0.15, "Weed detector\n& gate"),
        (0.33, 0.36, 0.22, 0.15, "GNSS/IMU\nEKF fusion"),
        (0.61, 0.36, 0.22, 0.15, "Mission\ncontroller"),
        (0.05, 0.36, 0.22, 0.15, "Field logs\n& plots"),
    ]

    for x, y, w, h, label in nodes:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.6,
            edgecolor="#9ed6ff",
            facecolor="#182030",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=12)

    arrows = [
        ((0.27, 0.795), (0.33, 0.795)),
        ((0.55, 0.795), (0.61, 0.795)),
        ((0.44, 0.72), (0.44, 0.51)),
        ((0.72, 0.72), (0.72, 0.51)),
        ((0.27, 0.435), (0.33, 0.435)),
        ((0.16, 0.51), (0.16, 0.72)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#f4e285"),
        )

    ax.text(
        0.5,
        0.92,
        "Standalone research workflow for lane detection, localization, and planning evaluation",
        ha="center",
        fontsize=14,
        weight="bold",
    )

    return _finalize(fig, output_dir / "system_overview.png", "System Overview")
