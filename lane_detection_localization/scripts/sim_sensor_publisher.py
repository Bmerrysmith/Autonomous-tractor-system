#!/usr/bin/env python3
"""
sim_sensor_publisher.py

Simulated sensor data publisher for testing the navigation stack
without a full Gazebo setup. Generates synthetic:
  - 2D LiDAR scans (crop rows at configurable spacing)
  - IMU data (orientation + angular velocity + acceleration)
  - Wheel odometry (with configurable noise)
  - GNSS fixes (with configurable outage windows)

Usage:
  rosrun undercanopy_ekf_nav sim_sensor_publisher.py

Parameters:
  ~row_spacing       (float)  Row spacing in meters (default: 0.76)
  ~robot_speed       (float)  Forward speed in m/s (default: 0.3)
  ~gnss_outage_start (float)  Time in seconds when GNSS drops (default: 15.0)
  ~gnss_outage_end   (float)  Time in seconds when GNSS returns (default: 35.0)
  ~imu_noise_std     (float)  IMU noise standard deviation (default: 0.01)
  ~odom_noise_std    (float)  Odometry noise standard deviation (default: 0.02)

Author: Krish Shah — CEN 4930, FGCU Spring 2026
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan, Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point, Vector3, Pose, Twist
import tf.transformations as tft


class SimSensorPublisher:
    def __init__(self):
        rospy.init_node('sim_sensor_publisher')

        # Parameters
        self.row_spacing = rospy.get_param('~row_spacing', 0.76)
        self.robot_speed = rospy.get_param('~robot_speed', 0.3)
        self.gnss_outage_start = rospy.get_param('~gnss_outage_start', 15.0)
        self.gnss_outage_end = rospy.get_param('~gnss_outage_end', 35.0)
        self.imu_noise_std = rospy.get_param('~imu_noise_std', 0.01)
        self.odom_noise_std = rospy.get_param('~odom_noise_std', 0.02)

        # Simulated robot state (ground truth)
        self.gt_x = 0.0
        self.gt_y = 0.0
        self.gt_yaw = 0.0  # heading along row
        self.gt_vx = self.robot_speed

        # Add slight lateral drift to make it interesting
        self.lateral_drift_freq = 0.1  # Hz
        self.lateral_drift_amp = 0.05  # meters

        # GNSS origin (arbitrary lat/lon for Fort Myers area)
        self.origin_lat = 26.4615
        self.origin_lon = -81.7800

        # Publishers
        self.scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=10)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.gps_pub = rospy.Publisher('/gps/fix', NavSatFix, queue_size=10)

        # Start time
        self.start_time = rospy.Time.now()

        # Timer at 20 Hz (LiDAR rate)
        self.timer = rospy.Timer(rospy.Duration(1.0 / 20.0), self.publish_all)

        rospy.loginfo("[sim_sensors] Publishing simulated sensor data.")
        rospy.loginfo(f"  Row spacing: {self.row_spacing}m")
        rospy.loginfo(f"  Robot speed: {self.robot_speed} m/s")
        rospy.loginfo(f"  GNSS outage: {self.gnss_outage_start}s - {self.gnss_outage_end}s")

    def publish_all(self, event):
        now = rospy.Time.now()
        t = (now - self.start_time).to_sec()
        dt = 1.0 / 20.0

        # Update ground truth robot state
        # Robot drives forward with slight lateral oscillation
        self.gt_x += self.gt_vx * dt
        self.gt_y = self.lateral_drift_amp * np.sin(2 * np.pi * self.lateral_drift_freq * t)
        self.gt_yaw = 0.02 * np.sin(2 * np.pi * 0.05 * t)  # slight heading wobble

        # Publish each sensor
        self._publish_scan(now, t)
        self._publish_imu(now, t)
        self._publish_odom(now, t)
        self._publish_gnss(now, t)

    def _publish_scan(self, stamp, t):
        """Generate synthetic 2D LiDAR scan with crop rows."""
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'
        msg.angle_min = -np.pi / 2
        msg.angle_max = np.pi / 2
        msg.angle_increment = np.radians(1.0)
        msg.range_min = 0.1
        msg.range_max = 5.0
        msg.time_increment = 0.0

        angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment,
                           msg.angle_increment)
        ranges = np.full_like(angles, msg.range_max)

        # Simulate crop rows as vertical lines at +/- row_spacing/2 from center
        half_spacing = self.row_spacing / 2.0
        left_row_y = half_spacing - self.gt_y   # distance to left row
        right_row_y = -half_spacing - self.gt_y  # distance to right row

        for i, angle in enumerate(angles):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            if abs(sin_a) > 0.01:
                # Check intersection with left row (y = left_row_y)
                r_left = left_row_y / sin_a
                if 0.1 < r_left < 5.0:
                    # Add noise to simulate plant irregularity
                    r_left += np.random.normal(0, 0.03)
                    ranges[i] = min(ranges[i], max(0.1, r_left))

                # Check intersection with right row (y = right_row_y)
                r_right = right_row_y / sin_a
                if 0.1 < r_right < 5.0:
                    r_right += np.random.normal(0, 0.03)
                    ranges[i] = min(ranges[i], max(0.1, r_right))

            # Add random noise points (weeds, ground clutter)
            if np.random.random() < 0.02:
                ranges[i] = np.random.uniform(0.3, 3.0)

        msg.ranges = ranges.tolist()
        self.scan_pub.publish(msg)

    def _publish_imu(self, stamp, t):
        """Generate synthetic IMU data."""
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'

        # Orientation (with noise)
        yaw = self.gt_yaw + np.random.normal(0, self.imu_noise_std)
        q = tft.quaternion_from_euler(0, 0, yaw)
        msg.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Angular velocity
        msg.angular_velocity = Vector3(
            x=np.random.normal(0, self.imu_noise_std * 0.5),
            y=np.random.normal(0, self.imu_noise_std * 0.5),
            z=0.02 * np.cos(2 * np.pi * 0.05 * t) * 2 * np.pi * 0.05
              + np.random.normal(0, self.imu_noise_std)
        )

        # Linear acceleration
        msg.linear_acceleration = Vector3(
            x=np.random.normal(0, 0.05),
            y=np.random.normal(0, 0.05),
            z=9.81 + np.random.normal(0, 0.02)
        )

        self.imu_pub.publish(msg)

    def _publish_odom(self, stamp, t):
        """Generate synthetic wheel odometry."""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        # Position (with cumulative drift noise)
        noise_x = np.random.normal(0, self.odom_noise_std)
        noise_y = np.random.normal(0, self.odom_noise_std)

        msg.pose.pose.position = Point(
            x=self.gt_x + noise_x * t * 0.01,  # small drift
            y=self.gt_y + noise_y * t * 0.01,
            z=0.0
        )

        q = tft.quaternion_from_euler(0, 0, self.gt_yaw)
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Velocity (with noise)
        msg.twist.twist.linear = Vector3(
            x=self.gt_vx + np.random.normal(0, self.odom_noise_std),
            y=np.random.normal(0, self.odom_noise_std * 0.5),
            z=0.0
        )

        self.odom_pub.publish(msg)

    def _publish_gnss(self, stamp, t):
        """Generate synthetic GNSS fix with outage window."""
        # Check if in outage window
        if self.gnss_outage_start <= t <= self.gnss_outage_end:
            # No GNSS during outage
            return

        msg = NavSatFix()
        msg.header.stamp = stamp
        msg.header.frame_id = 'gps'

        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS

        # Convert ground truth position to lat/lon
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(self.origin_lat))

        gps_noise = 0.5  # meters
        msg.latitude = self.origin_lat + (self.gt_x + np.random.normal(0, gps_noise)) / meters_per_deg_lat
        msg.longitude = self.origin_lon + (self.gt_y + np.random.normal(0, gps_noise)) / meters_per_deg_lon
        msg.altitude = 5.0

        # Position covariance
        cov = [gps_noise**2, 0, 0,
               0, gps_noise**2, 0,
               0, 0, 1.0]
        msg.position_covariance = cov
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.gps_pub.publish(msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = SimSensorPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
