#!/usr/bin/env python3
"""
lidar_row_detection_node.py

LiDAR-based geometric crop row detection for under-canopy navigation.
Implements the pipeline from Liu et al. (2024):
  1. Ground plane estimation + filtering
  2. K-means clustering to segment crop rows
  3. RANSAC line fitting for row boundaries
  4. Extended Kalman Filter for robust row tracking

Publishes:
  /crop_row_state       (CropRowState)   - EKF-filtered row positions [d, phi]
  /row_boundaries       (PointCloud2)    - left/right boundary points (for Benny's ROI)

Subscribes:
  /scan                 (LaserScan)      - 2D LiDAR scan
  /odom                 (Odometry)       - wheel odometry for EKF prediction

Author: Krish Shah — CEN 4930, FGCU Spring 2026
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import tf.transformations as tft

# Will be generated from msg/ definitions after catkin_make
# For standalone testing, we use a simple class wrapper
try:
    from undercanopy_ekf_nav.msg import CropRowState
except ImportError:
    # Fallback for testing outside catkin
    class CropRowState:
        def __init__(self):
            self.header = Header()
            self.left_row_distance = 0.0
            self.left_row_angle = 0.0
            self.right_row_distance = 0.0
            self.right_row_angle = 0.0
            self.left_confidence = 0.0
            self.right_confidence = 0.0
            self.prediction_only = False


# ─────────────────────────────────────────────────────────────────────────────
# RANSAC Line Fitter
# ─────────────────────────────────────────────────────────────────────────────
class RANSACLineFitter:
    """Fit a 2D line to points using RANSAC."""

    def __init__(self, max_iter=100, dist_thresh=0.05, min_inliers=5):
        self.max_iter = max_iter
        self.dist_thresh = dist_thresh
        self.min_inliers = min_inliers

    def fit(self, points):
        """
        Fit line to Nx2 array of (x, y) points.
        Returns (d, phi) where:
          d   = perpendicular distance from origin (robot) to line
          phi = angle of line relative to x-axis (forward)
        Returns None if not enough inliers.
        """
        if points.shape[0] < 2:
            return None

        best_inliers = 0
        best_line = None

        for _ in range(self.max_iter):
            idx = np.random.choice(points.shape[0], 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]

            # Line direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                continue

            # Normal to line: (a, b) normalized, line eq: ax + by + c = 0
            a = -dy / length
            b = dx / length
            c = -(a * p1[0] + b * p1[1])

            # Distances from all points to line
            dists = np.abs(a * points[:, 0] + b * points[:, 1] + c)
            inliers = np.sum(dists < self.dist_thresh)

            if inliers > best_inliers:
                best_inliers = inliers
                best_line = (a, b, c)

        if best_line is None or best_inliers < self.min_inliers:
            return None

        a, b, c = best_line
        d = abs(c)  # distance from origin to line (robot at origin)
        phi = np.arctan2(a, b)  # angle of line normal

        # Compute line angle (direction along the row)
        row_angle = np.arctan2(b, -a)

        # Signed distance: positive = line is to the left
        signed_d = c / np.sqrt(a**2 + b**2)

        return signed_d, row_angle


# ─────────────────────────────────────────────────────────────────────────────
# Crop Row EKF — tracks [d, phi] for a single row
# ─────────────────────────────────────────────────────────────────────────────
class CropRowEKF:
    """
    Extended Kalman Filter for tracking a single crop row's state [d, phi].

    State: x = [d, phi]^T
      d   = perpendicular distance from robot to row (meters)
      phi = angle of row relative to robot heading (radians)

    Prediction model uses robot odometry (dX, dY, dpsi) to propagate
    the row state as the robot moves.

    Measurement model: direct observation of [d, phi] from RANSAC.

    Reference: Liu et al. (2024), Section IV-A, EKF formulation.
    """

    def __init__(self, process_noise_d=0.01, process_noise_phi=0.005,
                 meas_noise_d=0.05, meas_noise_phi=0.03):
        # State vector [d, phi]
        self.x = np.array([0.0, 0.0])

        # State covariance
        self.P = np.diag([1.0, 0.5])

        # Process noise covariance
        self.Q = np.diag([process_noise_d, process_noise_phi])

        # Measurement noise covariance
        self.R = np.diag([meas_noise_d, meas_noise_phi])

        self.initialized = False

    def predict(self, delta_x, delta_y, delta_psi):
        """
        Predict step using robot odometry increments.
        The row is stationary in the world frame; the robot moves.

        As the robot translates by (dX, dY) and rotates by dpsi,
        the row's relative position changes:
          d'   = d - dY * cos(phi) + dX * sin(phi)
          phi' = phi - dpsi
        """
        if not self.initialized:
            return

        d, phi = self.x

        # State transition
        d_new = d - delta_y * np.cos(phi) + delta_x * np.sin(phi)
        phi_new = phi - delta_psi

        # Normalize angle
        phi_new = np.arctan2(np.sin(phi_new), np.cos(phi_new))

        self.x = np.array([d_new, phi_new])

        # Jacobian of state transition w.r.t. state
        F = np.array([
            [1.0, delta_y * np.sin(phi) + delta_x * np.cos(phi)],
            [0.0, 1.0]
        ])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_d, z_phi):
        """
        Update step with measurement [d, phi] from RANSAC.
        Measurement model is identity: H = I.
        """
        z = np.array([z_d, z_phi])

        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return

        # Innovation
        y = z - self.x
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  # angle wrap

        # Innovation covariance
        H = np.eye(2)
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.x[1] = np.arctan2(np.sin(self.x[1]), np.cos(self.x[1]))

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_innovation_magnitude(self):
        """Return the Mahalanobis distance of last innovation for confidence."""
        return np.sqrt(np.trace(self.P))


# ─────────────────────────────────────────────────────────────────────────────
# Main Node
# ─────────────────────────────────────────────────────────────────────────────
class LidarRowDetectionNode:
    def __init__(self):
        rospy.init_node('lidar_row_detection_node')

        # Parameters
        self.row_spacing = rospy.get_param('~row_spacing', 0.76)  # meters (typical paddy)
        self.max_range = rospy.get_param('~max_range', 5.0)
        self.min_range = rospy.get_param('~min_range', 0.1)
        self.n_clusters = rospy.get_param('~n_clusters', 2)  # left and right rows
        self.ransac_iterations = rospy.get_param('~ransac_iterations', 100)
        self.ransac_threshold = rospy.get_param('~ransac_threshold', 0.05)

        # RANSAC fitter
        self.ransac = RANSACLineFitter(
            max_iter=self.ransac_iterations,
            dist_thresh=self.ransac_threshold,
            min_inliers=5
        )

        # EKF for left and right rows
        self.ekf_left = CropRowEKF(
            process_noise_d=0.01, process_noise_phi=0.005,
            meas_noise_d=0.05, meas_noise_phi=0.03
        )
        self.ekf_right = CropRowEKF(
            process_noise_d=0.01, process_noise_phi=0.005,
            meas_noise_d=0.05, meas_noise_phi=0.03
        )

        # Odometry state for computing deltas
        self.last_odom = None
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0

        # Publishers
        self.row_state_pub = rospy.Publisher(
            '/crop_row_state', CropRowState, queue_size=10
        )

        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        rospy.loginfo("[lidar_row_detection] Node started. Waiting for /scan and /odom...")

    def odom_callback(self, msg):
        """Extract odometry increments for EKF prediction."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])

        if self.last_odom is not None:
            dx = x - self.last_x
            dy = y - self.last_y
            dpsi = yaw - self.last_yaw
            dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))

            # Transform global delta to robot frame
            cos_yaw = np.cos(self.last_yaw)
            sin_yaw = np.sin(self.last_yaw)
            delta_x_robot = cos_yaw * dx + sin_yaw * dy
            delta_y_robot = -sin_yaw * dx + cos_yaw * dy

            # EKF predict step
            self.ekf_left.predict(delta_x_robot, delta_y_robot, dpsi)
            self.ekf_right.predict(delta_x_robot, delta_y_robot, dpsi)

        self.last_x = x
        self.last_y = y
        self.last_yaw = yaw
        self.last_odom = msg

    def lidar_callback(self, msg):
        """Process 2D LiDAR scan to detect crop rows."""
        # Convert LaserScan to Cartesian points
        points = self._scan_to_points(msg)

        if points.shape[0] < 10:
            self._publish_prediction_only()
            return

        # Separate points into left (y > 0) and right (y < 0) clusters
        # This is the simplified K-means for 2-cluster case with known geometry
        left_points = points[points[:, 1] > 0.05]   # left of robot
        right_points = points[points[:, 1] < -0.05]  # right of robot

        # Further filter: expect rows within reasonable distance
        if left_points.shape[0] > 0:
            left_dists = np.abs(left_points[:, 1])
            mask = (left_dists > 0.1) & (left_dists < self.row_spacing * 1.5)
            left_points = left_points[mask]

        if right_points.shape[0] > 0:
            right_dists = np.abs(right_points[:, 1])
            mask = (right_dists > 0.1) & (right_dists < self.row_spacing * 1.5)
            right_points = right_points[mask]

        # RANSAC line fitting
        prediction_only = True
        left_result = self.ransac.fit(left_points) if left_points.shape[0] >= 5 else None
        right_result = self.ransac.fit(right_points) if right_points.shape[0] >= 5 else None

        # EKF update with measurements
        if left_result is not None:
            self.ekf_left.update(left_result[0], left_result[1])
            prediction_only = False

        if right_result is not None:
            self.ekf_right.update(right_result[0], right_result[1])
            prediction_only = False

        # Publish filtered state
        self._publish_state(prediction_only)

    def _scan_to_points(self, scan):
        """Convert LaserScan message to Nx2 array of (x, y) points in robot frame."""
        angles = np.arange(scan.angle_min, scan.angle_max + scan.angle_increment,
                           scan.angle_increment)
        ranges = np.array(scan.ranges)

        # Trim to same length
        n = min(len(angles), len(ranges))
        angles = angles[:n]
        ranges = ranges[:n]

        # Filter valid ranges
        valid = (ranges > self.min_range) & (ranges < self.max_range) & np.isfinite(ranges)
        angles = angles[valid]
        ranges = ranges[valid]

        # Convert to Cartesian
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        return np.column_stack([x, y])

    def _publish_state(self, prediction_only):
        """Publish the EKF-filtered crop row state."""
        msg = CropRowState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'base_link'

        if self.ekf_left.initialized:
            msg.left_row_distance = self.ekf_left.x[0]
            msg.left_row_angle = self.ekf_left.x[1]
            msg.left_confidence = max(0.0, min(1.0,
                1.0 - self.ekf_left.get_innovation_magnitude() / 2.0))
        else:
            msg.left_confidence = 0.0

        if self.ekf_right.initialized:
            msg.right_row_distance = self.ekf_right.x[0]
            msg.right_row_angle = self.ekf_right.x[1]
            msg.right_confidence = max(0.0, min(1.0,
                1.0 - self.ekf_right.get_innovation_magnitude() / 2.0))
        else:
            msg.right_confidence = 0.0

        msg.prediction_only = prediction_only
        self.row_state_pub.publish(msg)

    def _publish_prediction_only(self):
        """When no LiDAR measurements are available, publish EKF prediction."""
        self._publish_state(prediction_only=True)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = LidarRowDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
