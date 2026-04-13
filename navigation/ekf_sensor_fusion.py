"""
ekf_sensor_fusion.py
====================
Extended Kalman Filter for autonomous crop-row navigation.

Fuses LiDAR range measurements with IMU/encoder dead-reckoning to maintain
a robust estimate of the tractor's lateral position and heading within a
crop row.  Research shows this fusion increases reliable autonomous
navigation distance from ~51.6 m (single sensor) to ~400 m (8× improvement).

State vector  x = [dL, dR, phi]^T
──────────────────────────────────
  dL   – distance to the left row boundary   (m)
  dR   – distance to the right row boundary  (m)
  phi  – heading angle relative to the row   (rad, 0 = perfectly aligned)

Predict step  (IMU / wheel encoder)
────────────────────────────────────
  dL'  = dL  + v·sin(phi)·dt
  dR'  = dR  - v·sin(phi)·dt
  phi' = phi + omega·dt

  where v = forward speed (m/s), omega = yaw rate (rad/s), dt = timestep.

Update step  (LiDAR)
─────────────────────
  z = [dL_meas, dR_meas, phi_meas]^T   (row-boundary distances + heading
                                          extracted from LiDAR scan)

Usage
─────
  from navigation import RowEKF

  ekf = RowEKF(row_width=0.76)
  ekf.predict(v=1.2, omega=0.01, dt=0.05)
  ekf.update(z_dL=0.38, z_dR=0.38, z_phi=0.0)
  print(ekf.state)   # [dL, dR, phi]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# EKF CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EKFConfig:
    """Tunable noise / initialisation parameters for the row-tracking EKF."""

    # Process-noise standard deviations (predict step)
    sigma_dL:  float = 0.02     # lateral drift noise (m)
    sigma_dR:  float = 0.02
    sigma_phi: float = 0.005    # heading drift noise (rad)

    # Measurement-noise standard deviations (update step)
    sigma_z_dL:  float = 0.05   # LiDAR range noise (m)
    sigma_z_dR:  float = 0.05
    sigma_z_phi: float = 0.01   # LiDAR heading noise (rad)

    # Initial uncertainty (diagonal of P₀)
    init_P_dL:  float = 0.1
    init_P_dR:  float = 0.1
    init_P_phi: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# EXTENDED KALMAN FILTER
# ═══════════════════════════════════════════════════════════════════════════

class RowEKF:
    """
    3-state EKF tracking [dL, dR, phi] for crop-row navigation.

    Parameters
    ──────────
    row_width : float
        Nominal distance between left and right crop rows (m).
    config : EKFConfig | None
        Noise / init parameters.  Uses defaults when *None*.
    """

    N_STATES = 3   # dL, dR, phi

    def __init__(
        self,
        row_width: float = 0.76,
        config: Optional[EKFConfig] = None,
    ):
        self.row_width = row_width
        self.cfg = config or EKFConfig()

        # State vector  x = [dL, dR, phi]
        half = row_width / 2.0
        self.x = np.array([half, half, 0.0], dtype=np.float64)

        # Covariance matrix  P
        self.P = np.diag([
            self.cfg.init_P_dL,
            self.cfg.init_P_dR,
            self.cfg.init_P_phi,
        ])

        # Process noise Q  (rebuilt each predict call to incorporate dt)
        self._Q_diag = np.array([
            self.cfg.sigma_dL ** 2,
            self.cfg.sigma_dR ** 2,
            self.cfg.sigma_phi ** 2,
        ])

        # Measurement noise R  (constant)
        self.R = np.diag([
            self.cfg.sigma_z_dL ** 2,
            self.cfg.sigma_z_dR ** 2,
            self.cfg.sigma_z_phi ** 2,
        ])

        self._step = 0

    # ── public read-only properties ──────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        """Current state estimate [dL, dR, phi]."""
        return self.x.copy()

    @property
    def dL(self) -> float:
        return float(self.x[0])

    @property
    def dR(self) -> float:
        return float(self.x[1])

    @property
    def phi(self) -> float:
        return float(self.x[2])

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    @property
    def lateral_offset(self) -> float:
        """Signed offset from row centre (positive = shifted right)."""
        return (self.dL - self.dR) / 2.0

    # ── PREDICT ──────────────────────────────────────────────────────────

    def predict(self, v: float, omega: float, dt: float) -> np.ndarray:
        """
        Propagate the state forward using IMU / encoder odometry.

        Parameters
        ──────────
        v     : forward speed (m/s)
        omega : yaw rate from IMU (rad/s)
        dt    : time step (s)

        Returns
        ───────
        Predicted state [dL, dR, phi].
        """
        dL, dR, phi = self.x

        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)

        # Non-linear state transition
        dL_new  = dL + v * sin_phi * dt
        dR_new  = dR - v * sin_phi * dt
        phi_new = phi + omega * dt

        self.x = np.array([dL_new, dR_new, phi_new])

        # Jacobian F = ∂f/∂x
        F = np.array([
            [1.0,  0.0,  v * cos_phi * dt],
            [0.0,  1.0, -v * cos_phi * dt],
            [0.0,  0.0,  1.0             ],
        ])

        # Process noise scaled by dt
        Q = np.diag(self._Q_diag * dt)

        self.P = F @ self.P @ F.T + Q
        self._step += 1

        return self.state

    # ── UPDATE ───────────────────────────────────────────────────────────

    def update(
        self,
        z_dL: float,
        z_dR: float,
        z_phi: Optional[float] = None,
    ) -> np.ndarray:
        """
        Correct the state using a LiDAR measurement.

        Parameters
        ──────────
        z_dL  : measured distance to left row boundary (m)
        z_dR  : measured distance to right row boundary (m)
        z_phi : measured heading angle (rad).  If *None*, only the
                two range measurements are fused (partial update).

        Returns
        ───────
        Updated (corrected) state [dL, dR, phi].
        """
        if z_phi is not None:
            # Full 3-measurement update
            z = np.array([z_dL, z_dR, z_phi])
            H = np.eye(self.N_STATES)
            R = self.R
        else:
            # Partial update — only dL and dR observed
            z = np.array([z_dL, z_dR])
            H = np.eye(self.N_STATES)[:2, :]      # first two rows of I₃
            R = self.R[:2, :2]

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)

        # Kalman gain
        K = self.P @ H.T @ S_inv

        # State correction
        self.x = self.x + K @ y

        # Covariance correction (Joseph form for numerical stability)
        I = np.eye(self.N_STATES)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

        return self.state

    # ── RESET ────────────────────────────────────────────────────────────

    def reset(self, dL: Optional[float] = None, dR: Optional[float] = None,
              phi: float = 0.0) -> None:
        """Re-initialise the filter (e.g. at the start of a new row)."""
        half = self.row_width / 2.0
        self.x = np.array([
            dL if dL is not None else half,
            dR if dR is not None else half,
            phi,
        ])
        self.P = np.diag([
            self.cfg.init_P_dL,
            self.cfg.init_P_dR,
            self.cfg.init_P_phi,
        ])
        self._step = 0

    # ── helpers ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RowEKF(dL={self.dL:.3f}, dR={self.dR:.3f}, "
            f"phi={math.degrees(self.phi):.2f}°, step={self._step})"
        )
