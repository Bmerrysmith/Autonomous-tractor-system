#!/usr/bin/env python3
"""
gnss_outage_bridge_node.py

GNSS outage detection and bridging for under-canopy navigation.
Monitors GNSS signal quality and manages the transition between:
  - Full GNSS+IMU+Odom fusion (open field)
  - Dead-reckoning with IMU+Odom (under canopy / GNSS denied)

When GNSS drops out, this node:
  1. Detects the outage via timeout or quality degradation
  2. Inflates uncertainty estimates in the localization EKF
  3. Uses IMU heading + wheel odometry to maintain position tracking
  4. Re-acquires GNSS when signal returns, with jump mitigation

This node works alongside ekf_localization_node.py — it publishes
diagnostic information and can command the EKF to switch modes.

Publishes:
  /gnss/status           (String)     - "AVAILABLE" | "DEGRADED" | "OUTAGE"
  /gnss/outage_duration  (Float64)    - seconds since last valid fix

Subscribes:
  /gps/fix               (NavSatFix)  - raw GNSS fix
  /imu/data              (Imu)        - IMU for heading during outage

Author: Krish Shah — CEN 4930, FGCU Spring 2026
"""

import rospy
import numpy as np
from sensor_msgs.msg import NavSatFix, NavSatStatus, Imu
from std_msgs.msg import String, Float64
import tf.transformations as tft


class GNSSOutageBridgeNode:
    """
    Monitors GNSS health and implements outage bridging strategy.

    GNSS status levels:
      AVAILABLE  - fix_type >= 0, DOP acceptable, recent update
      DEGRADED   - fix_type >= 0 but high covariance or stale (>1s)
      OUTAGE     - no fix for > timeout threshold

    During OUTAGE, the node tracks elapsed dead-reckoning time and
    monitors IMU heading drift for diagnostics.
    """

    # Status constants
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    OUTAGE = "OUTAGE"

    def __init__(self):
        rospy.init_node('gnss_outage_bridge_node')

        # Parameters
        self.gnss_timeout = rospy.get_param('~gnss_timeout', 2.0)
        self.degraded_threshold = rospy.get_param('~degraded_threshold', 1.0)
        self.max_position_cov = rospy.get_param('~max_position_covariance', 10.0)

        # State
        self.last_fix_time = None
        self.last_fix_valid = False
        self.current_status = self.OUTAGE
        self.outage_start_time = rospy.Time.now()
        self.outage_duration = 0.0

        # IMU heading tracking during outage
        self.imu_heading_at_outage_start = None
        self.current_imu_heading = 0.0
        self.heading_drift_estimate = 0.0

        # Publishers
        self.status_pub = rospy.Publisher('/gnss/status', String, queue_size=10)
        self.outage_dur_pub = rospy.Publisher('/gnss/outage_duration', Float64, queue_size=10)

        # Subscribers
        rospy.Subscriber('/gps/fix', NavSatFix, self.gnss_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        # Timer at 10 Hz
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        rospy.loginfo("[gnss_outage_bridge] Node started. Monitoring GNSS health.")

    def gnss_callback(self, msg):
        """Evaluate incoming GNSS fix quality."""
        now = rospy.Time.now()

        # Check fix validity
        has_fix = msg.status.status >= NavSatStatus.STATUS_FIX

        # Check position covariance (diagonal elements)
        high_cov = False
        if len(msg.position_covariance) >= 5:
            pos_cov = max(msg.position_covariance[0], msg.position_covariance[4])
            high_cov = pos_cov > self.max_position_cov

        if has_fix and not high_cov:
            # Good fix
            self.last_fix_time = now
            self.last_fix_valid = True

            if self.current_status == self.OUTAGE:
                duration = self.outage_duration
                rospy.loginfo(
                    f"[gnss_outage_bridge] GNSS re-acquired after "
                    f"{duration:.1f}s outage. Heading drift: "
                    f"{np.degrees(self.heading_drift_estimate):.1f} deg"
                )
                self.imu_heading_at_outage_start = None

            self.current_status = self.AVAILABLE

        elif has_fix and high_cov:
            # Fix but unreliable
            self.last_fix_time = now
            self.current_status = self.DEGRADED

    def imu_callback(self, msg):
        """Track IMU heading for drift estimation during outage."""
        q = msg.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_imu_heading = yaw

        if self.current_status == self.OUTAGE:
            if self.imu_heading_at_outage_start is not None:
                drift = yaw - self.imu_heading_at_outage_start
                self.heading_drift_estimate = np.arctan2(
                    np.sin(drift), np.cos(drift)
                )

    def timer_callback(self, event):
        """Periodic status check and publish."""
        now = rospy.Time.now()

        # Check for timeout
        if self.last_fix_time is not None:
            elapsed = (now - self.last_fix_time).to_sec()

            if elapsed > self.gnss_timeout:
                if self.current_status != self.OUTAGE:
                    # Transition to OUTAGE
                    rospy.logwarn(
                        f"[gnss_outage_bridge] GNSS OUTAGE detected. "
                        f"No fix for {elapsed:.1f}s. Switching to DR mode."
                    )
                    self.outage_start_time = now
                    self.imu_heading_at_outage_start = self.current_imu_heading
                    self.current_status = self.OUTAGE

                self.outage_duration = (now - self.outage_start_time).to_sec()

            elif elapsed > self.degraded_threshold:
                if self.current_status == self.AVAILABLE:
                    self.current_status = self.DEGRADED
        else:
            # Never received a fix
            self.current_status = self.OUTAGE
            self.outage_duration = 0.0

        # Publish
        self.status_pub.publish(String(data=self.current_status))

        if self.current_status == self.OUTAGE:
            self.outage_dur_pub.publish(Float64(data=self.outage_duration))
        else:
            self.outage_dur_pub.publish(Float64(data=0.0))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = GNSSOutageBridgeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
