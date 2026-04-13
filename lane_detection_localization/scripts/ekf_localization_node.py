#!/usr/bin/env python3
"""
ekf_localization_node.py

EKF-based GNSS-IMU-Odometry sensor fusion for global localization,
with GNSS outage detection and bridging using IMU + wheel odometry.

This node implements two operating modes:
  1. FULL FUSION: GNSS + IMU + wheel odometry (normal operation)
  2. DEAD RECKONING: IMU + wheel odometry only (GNSS outage / under canopy)

The transition is automatic — when GNSS signal is lost (timeout or high DOP),
the EKF seamlessly switches to dead-reckoning using IMU and odometry,
with process noise inflation to reflect growing uncertainty.

State vector: [x, y, yaw, vx, vy, yaw_rate]^T

Publishes:
  /localization/pose     (Odometry)        - fused global pose estimate
  /localization/status   (String)          - current fusion mode

Subscribes:
  /gps/fix               (NavSatFix)       - GNSS position (simulated)
  /imu/data              (Imu)             - IMU orientation + angular vel + accel
  /odom                  (Odometry)        - wheel encoder odometry

Author: Krish Shah — CEN 4930, FGCU Spring 2026
"""

import rospy
import numpy as np
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Header
from geometry_msgs.msg import Quaternion, Point, Pose, Twist, Vector3
import tf.transformations as tft


class SensorFusionEKF:
    """
    6-state EKF for 2D planar localization.
    State: x = [x, y, yaw, vx, vy, yaw_rate]^T

    The filter uses a constant-velocity-turn-rate (CVTR) motion model
    and fuses three sensor inputs with different update rates:
      - IMU:  orientation (yaw), angular velocity, linear acceleration
      - Odom: linear velocity (vx, vy from wheel encoders)
      - GNSS: absolute position (x, y) — when available

    GNSS outage bridging:
      When GNSS drops out, the filter continues with IMU+odom predictions.
      Process noise Q is inflated over time to reflect the growing
      uncertainty of dead-reckoning-only state estimation.
    """

    def __init__(self):
        # State: [x, y, yaw, vx, vy, yaw_rate]
        self.x = np.zeros(6)
        self.P = np.diag([1.0, 1.0, 0.1, 0.5, 0.5, 0.05])

        # Base process noise (tuned for agricultural robot ~0.5 m/s)
        self.Q_base = np.diag([0.01, 0.01, 0.005, 0.05, 0.05, 0.01])

        # Current Q (inflated during GNSS outage)
        self.Q = self.Q_base.copy()

        # Measurement noise matrices
        self.R_gnss = np.diag([0.5, 0.5])          # GPS: ~0.7m CEP
        self.R_imu = np.diag([0.01, 0.02])          # IMU: yaw, yaw_rate
        self.R_odom = np.diag([0.05, 0.05])          # Odom: vx, vy

        # Timing
        self.last_predict_time = None
        self.last_gnss_time = None
        self.gnss_timeout = 3.0  # seconds before declaring outage

        # GNSS origin (first fix becomes origin)
        self.gnss_origin = None
        self.gnss_available = True

        # Dead reckoning drift counter
        self.dr_elapsed = 0.0

    def predict(self, dt):
        """
        Predict step using CVTR motion model.
        x_new = x + vx*cos(yaw)*dt - vy*sin(yaw)*dt
        y_new = y + vx*sin(yaw)*dt + vy*cos(yaw)*dt
        yaw_new = yaw + yaw_rate * dt
        velocities persist (constant velocity assumption)
        """
        if dt <= 0 or dt > 1.0:
            return

        x, y, yaw, vx, vy, yr = self.x

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)

        # State prediction
        x_pred = np.array([
            x + (vx * cos_y - vy * sin_y) * dt,
            y + (vx * sin_y + vy * cos_y) * dt,
            yaw + yr * dt,
            vx,
            vy,
            yr
        ])
        x_pred[2] = np.arctan2(np.sin(x_pred[2]), np.cos(x_pred[2]))

        # Jacobian of state transition
        F = np.eye(6)
        F[0, 2] = (-vx * sin_y - vy * cos_y) * dt
        F[0, 3] = cos_y * dt
        F[0, 4] = -sin_y * dt
        F[1, 2] = (vx * cos_y - vy * sin_y) * dt
        F[1, 3] = sin_y * dt
        F[1, 4] = cos_y * dt
        F[2, 5] = dt

        # Inflate Q during GNSS outage (linear growth)
        if not self.gnss_available:
            self.dr_elapsed += dt
            inflation = 1.0 + 0.5 * self.dr_elapsed  # grows over time
            self.Q = self.Q_base * inflation
        else:
            self.Q = self.Q_base.copy()
            self.dr_elapsed = 0.0

        self.x = x_pred
        self.P = F @ self.P @ F.T + self.Q * dt

    def update_gnss(self, gps_x, gps_y):
        """Update with GNSS position measurement."""
        z = np.array([gps_x, gps_y])

        # Observation matrix: we observe x, y
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        y_innov = z - H @ self.x
        S = H @ self.P @ H.T + self.R_gnss
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_innov
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_gnss @ K.T

        self.gnss_available = True
        self.last_gnss_time = rospy.Time.now()

    def update_imu(self, yaw_meas, yaw_rate_meas):
        """Update with IMU yaw and yaw rate."""
        z = np.array([yaw_meas, yaw_rate_meas])

        H = np.zeros((2, 6))
        H[0, 2] = 1.0  # yaw
        H[1, 5] = 1.0  # yaw_rate

        y_innov = z - H @ self.x
        y_innov[0] = np.arctan2(np.sin(y_innov[0]), np.cos(y_innov[0]))

        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_innov
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_imu @ K.T

    def update_odom(self, vx_meas, vy_meas):
        """Update with wheel odometry velocities."""
        z = np.array([vx_meas, vy_meas])

        H = np.zeros((2, 6))
        H[0, 3] = 1.0  # vx
        H[1, 4] = 1.0  # vy

        y_innov = z - H @ self.x
        S = H @ self.P @ H.T + self.R_odom
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_innov
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_odom @ K.T

    def check_gnss_status(self):
        """Check if GNSS has timed out."""
        if self.last_gnss_time is None:
            self.gnss_available = False
            return

        elapsed = (rospy.Time.now() - self.last_gnss_time).to_sec()
        if elapsed > self.gnss_timeout:
            self.gnss_available = False


# ─────────────────────────────────────────────────────────────────────────────
# ROS Node Wrapper
# ─────────────────────────────────────────────────────────────────────────────
class EKFLocalizationNode:
    def __init__(self):
        rospy.init_node('ekf_localization_node')

        self.ekf = SensorFusionEKF()

        # GNSS origin conversion (simple flat-Earth for small areas)
        self.gnss_origin_lat = None
        self.gnss_origin_lon = None
        self.METERS_PER_DEG_LAT = 111320.0
        self.meters_per_deg_lon = 0.0

        # Publishers
        self.pose_pub = rospy.Publisher('/localization/pose', Odometry, queue_size=10)
        self.status_pub = rospy.Publisher('/localization/status', String, queue_size=10)

        # Subscribers
        rospy.Subscriber('/gps/fix', NavSatFix, self.gnss_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Timer for prediction and publishing at 50 Hz
        self.last_time = rospy.Time.now()
        self.timer = rospy.Timer(rospy.Duration(1.0 / 50.0), self.timer_callback)

        rospy.loginfo("[ekf_localization] Node started. Fusing GNSS + IMU + Odom.")

    def gnss_callback(self, msg):
        """Process GNSS fix — convert lat/lon to local XY."""
        if msg.status.status < 0:
            # No fix
            return

        lat = msg.latitude
        lon = msg.longitude

        if self.gnss_origin_lat is None:
            self.gnss_origin_lat = lat
            self.gnss_origin_lon = lon
            self.meters_per_deg_lon = self.METERS_PER_DEG_LAT * np.cos(np.radians(lat))
            rospy.loginfo(f"[ekf_localization] GNSS origin set: ({lat:.6f}, {lon:.6f})")

        # Convert to local ENU coordinates
        gps_x = (lat - self.gnss_origin_lat) * self.METERS_PER_DEG_LAT
        gps_y = (lon - self.gnss_origin_lon) * self.meters_per_deg_lon

        self.ekf.update_gnss(gps_x, gps_y)

    def imu_callback(self, msg):
        """Process IMU data — extract yaw and yaw rate."""
        q = msg.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_rate = msg.angular_velocity.z

        self.ekf.update_imu(yaw, yaw_rate)

    def odom_callback(self, msg):
        """Process wheel odometry — extract velocities."""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        self.ekf.update_odom(vx, vy)

    def timer_callback(self, event):
        """Periodic predict + publish cycle."""
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        self.last_time = now

        # Predict
        self.ekf.predict(dt)

        # Check GNSS status
        self.ekf.check_gnss_status()

        # Publish pose
        self._publish_pose(now)

        # Publish status
        if self.ekf.gnss_available:
            mode = "gnss_imu_odom"
        else:
            mode = f"imu_odom_only (DR: {self.ekf.dr_elapsed:.1f}s)"
        self.status_pub.publish(String(data=mode))

    def _publish_pose(self, stamp):
        """Build and publish Odometry message from EKF state."""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.child_frame_id = 'base_link'

        x, y, yaw, vx, vy, yr = self.ekf.x

        # Position
        msg.pose.pose.position = Point(x=x, y=y, z=0.0)
        q = tft.quaternion_from_euler(0, 0, yaw)
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Pose covariance (6x6, row-major, only fill position + yaw)
        cov = np.zeros(36)
        cov[0] = self.ekf.P[0, 0]   # x variance
        cov[7] = self.ekf.P[1, 1]   # y variance
        cov[35] = self.ekf.P[2, 2]  # yaw variance
        msg.pose.covariance = cov.tolist()

        # Velocity
        msg.twist.twist.linear = Vector3(x=vx, y=vy, z=0.0)
        msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=yr)

        self.pose_pub.publish(msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = EKFLocalizationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
