"""Synthetic datasets for research visualizations.

These generators are intentionally independent from the cloned repository's
training or inference code. They create publication-style demo data that can be
replaced later with measured ROS logs, CSV exports, or notebook outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class RowGeometrySample:
    x: np.ndarray
    left_row: np.ndarray
    right_row: np.ndarray
    centerline: np.ndarray
    roi_left: np.ndarray
    roi_right: np.ndarray
    roi_width: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class LocalizationSample:
    t: np.ndarray
    gnss_x: np.ndarray
    gnss_y: np.ndarray
    ekf_x: np.ndarray
    ekf_y: np.ndarray
    ground_truth_x: np.ndarray
    ground_truth_y: np.ndarray
    yaw_truth: np.ndarray
    yaw_imu: np.ndarray
    yaw_ekf: np.ndarray
    position_error_gnss: np.ndarray
    position_error_ekf: np.ndarray
    yaw_error_imu: np.ndarray
    yaw_error_ekf: np.ndarray


@dataclass(frozen=True)
class PlanningTradeoffSample:
    algorithms: list[str]
    score_realtime: np.ndarray
    score_constraint: np.ndarray
    score_obstacle: np.ndarray
    score_ros_maturity: np.ndarray
    response_time_ms: np.ndarray


def generate_row_geometry_sample(num_points: int = 120, seed: int = 7) -> RowGeometrySample:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 25.0, num_points)

    center = 0.15 * np.sin(x / 3.2) + 0.05 * np.cos(x / 1.8)
    row_half_width = 1.55 + 0.08 * np.sin(x / 5.0)

    left_row = center - row_half_width + 0.03 * rng.normal(size=num_points)
    right_row = center + row_half_width + 0.03 * rng.normal(size=num_points)

    centerline = 0.5 * (left_row + right_row)
    roi_width = right_row - left_row
    roi_left = left_row + 0.12 * roi_width
    roi_right = right_row - 0.12 * roi_width

    confidence = np.clip(0.93 - 0.015 * np.abs(np.gradient(row_half_width)) + 0.02 * rng.normal(size=num_points), 0.55, 0.99)

    return RowGeometrySample(
        x=x,
        left_row=left_row,
        right_row=right_row,
        centerline=centerline,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_width=roi_width,
        confidence=confidence,
    )


def generate_localization_sample(num_points: int = 220, seed: int = 11) -> LocalizationSample:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 110.0, num_points)

    ground_truth_x = 0.14 * t + 1.9 * np.sin(t / 17.0)
    ground_truth_y = 0.35 * np.sin(t / 9.0) + 0.08 * np.cos(t / 4.8)
    yaw_truth = 5.0 * np.sin(t / 20.0) + 1.5 * np.cos(t / 8.0)

    gnss_x = ground_truth_x + rng.normal(0.0, 0.75, num_points)
    gnss_y = ground_truth_y + rng.normal(0.0, 0.65, num_points)
    yaw_imu = yaw_truth + rng.normal(0.0, 2.8, num_points)

    ekf_x = ground_truth_x + rng.normal(0.0, 0.18, num_points)
    ekf_y = ground_truth_y + rng.normal(0.0, 0.16, num_points)
    yaw_ekf = yaw_truth + rng.normal(0.0, 0.8, num_points)

    position_error_gnss = np.sqrt((gnss_x - ground_truth_x) ** 2 + (gnss_y - ground_truth_y) ** 2)
    position_error_ekf = np.sqrt((ekf_x - ground_truth_x) ** 2 + (ekf_y - ground_truth_y) ** 2)
    yaw_error_imu = np.abs(yaw_imu - yaw_truth)
    yaw_error_ekf = np.abs(yaw_ekf - yaw_truth)

    return LocalizationSample(
        t=t,
        gnss_x=gnss_x,
        gnss_y=gnss_y,
        ekf_x=ekf_x,
        ekf_y=ekf_y,
        ground_truth_x=ground_truth_x,
        ground_truth_y=ground_truth_y,
        yaw_truth=yaw_truth,
        yaw_imu=yaw_imu,
        yaw_ekf=yaw_ekf,
        position_error_gnss=position_error_gnss,
        position_error_ekf=position_error_ekf,
        yaw_error_imu=yaw_error_imu,
        yaw_error_ekf=yaw_error_ekf,
    )


def generate_planning_tradeoff_sample() -> PlanningTradeoffSample:
    algorithms = ["DWA", "TEB", "RRT*"]
    score_realtime = np.array([9.2, 7.8, 4.1])
    score_constraint = np.array([6.0, 8.7, 5.4])
    score_obstacle = np.array([5.8, 7.3, 9.0])
    score_ros_maturity = np.array([8.9, 8.4, 6.7])
    response_time_ms = np.array([42.0, 71.0, 168.0])

    return PlanningTradeoffSample(
        algorithms=algorithms,
        score_realtime=score_realtime,
        score_constraint=score_constraint,
        score_obstacle=score_obstacle,
        score_ros_maturity=score_ros_maturity,
        response_time_ms=response_time_ms,
    )


def generate_sensor_fusion_signal(sample: LocalizationSample) -> Dict[str, np.ndarray]:
    """Convenience helper for confidence-style plots.

    Returns a simple time-series dictionary that can be replaced with actual ROS
    output or CSV logs later.
    """

    confidence = np.clip(1.0 - 0.55 * (sample.position_error_ekf / (sample.position_error_gnss + 1e-6)), 0.0, 1.0)
    corroboration = np.clip(0.5 + 0.5 * np.cos(sample.t / 14.0), 0.0, 1.0)
    fused = np.clip(0.35 * confidence + 0.65 * corroboration, 0.0, 1.0)

    return {
        "time": sample.t,
        "confidence": confidence,
        "corroboration": corroboration,
        "fused": fused,
    }
