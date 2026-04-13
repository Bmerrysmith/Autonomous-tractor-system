#!/usr/bin/env python3
"""
row_centerline_node.py

Computes the row centerline from the detected crop row boundaries
and outputs it for:
  1. Lane-following controller (lateral offset + heading error)
  2. Path planner return target (WEED_INTERCEPT -> LANE_FOLLOW)
  3. Benny's LiDAR ROI subscriber (row width for detection mask)

The centerline is the midpoint between the left and right crop rows.
When only one row is detected, it estimates the other using known
row spacing.

Publishes:
  /row_centerline        (RowCenterline)  - centerline for lane following
  /cmd_vel               (Twist)          - velocity commands (pure pursuit)

Subscribes:
  /crop_row_state        (CropRowState)   - from lidar_row_detection_node
  /localization/status   (String)         - from ekf_localization_node
  /gnss/status           (String)         - from gnss_outage_bridge_node

Author: Krish Shah — CEN 4930, FGCU Spring 2026
"""

import rospy
import numpy as np
from std_msgs.msg import String, Header
from geometry_msgs.msg import Twist, Point

try:
    from undercanopy_ekf_nav.msg import CropRowState, RowCenterline
except ImportError:
    class CropRowState:
        pass
    class RowCenterline:
        def __init__(self):
            self.header = Header()
            self.lateral_offset = 0.0
            self.heading_error = 0.0
            self.row_width = 0.0
            self.centerline_waypoint = Point()
            self.localization_source = ""
            self.gnss_available = True
            self.lidar_rows_detected = True


class RowCenterlineNode:
    """
    Computes and publishes the row centerline for downstream consumers.

    Pure pursuit controller:
      Uses a lookahead distance to compute curvature for smooth following.
      Based on the approach in Liu et al. (2024): lookahead = 0.6 * row_length.
    """

    def __init__(self):
        rospy.init_node('row_centerline_node')

        # Parameters
        self.row_spacing = rospy.get_param('~row_spacing', 0.76)
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 1.5)
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 0.5)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 0.5)
        self.enable_cmd_vel = rospy.get_param('~enable_cmd_vel', True)

        # State
        self.localization_source = "unknown"
        self.gnss_status = "OUTAGE"
        self.last_row_state = None

        # Publishers
        self.centerline_pub = rospy.Publisher(
            '/row_centerline', RowCenterline, queue_size=10
        )
        if self.enable_cmd_vel:
            self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscribers
        rospy.Subscriber('/crop_row_state', CropRowState, self.row_state_callback)
        rospy.Subscriber('/localization/status', String, self.loc_status_callback)
        rospy.Subscriber('/gnss/status', String, self.gnss_status_callback)

        rospy.loginfo("[row_centerline] Node started. Computing centerline from crop rows.")

    def row_state_callback(self, msg):
        """Process crop row state and compute centerline."""
        self.last_row_state = msg

        # Extract row parameters
        d_left = msg.left_row_distance
        phi_left = msg.left_row_angle
        d_right = msg.right_row_distance
        phi_right = msg.right_row_angle
        conf_left = msg.left_confidence
        conf_right = msg.right_confidence

        # Determine which rows are reliably detected
        left_valid = conf_left > 0.3
        right_valid = conf_right > 0.3

        if left_valid and right_valid:
            # Both rows detected — compute true centerline
            lateral_offset = (d_left + d_right) / 2.0
            heading_error = (phi_left + phi_right) / 2.0
            row_width = abs(d_left - d_right)
        elif left_valid:
            # Only left row — estimate right using known spacing
            lateral_offset = d_left - self.row_spacing / 2.0
            heading_error = phi_left
            row_width = self.row_spacing
        elif right_valid:
            # Only right row — estimate left using known spacing
            lateral_offset = d_right + self.row_spacing / 2.0
            heading_error = phi_right
            row_width = self.row_spacing
        else:
            # No rows detected — hold last estimate or stop
            rospy.logwarn_throttle(5.0, "[row_centerline] No crop rows detected!")
            lateral_offset = 0.0
            heading_error = 0.0
            row_width = self.row_spacing

        # Compute centerline waypoint in robot frame (for path planner)
        # The waypoint is at lookahead distance along the centerline direction
        wp_x = self.lookahead_distance * np.cos(heading_error)
        wp_y = self.lookahead_distance * np.sin(heading_error) + lateral_offset

        # Build and publish RowCenterline message
        cl_msg = RowCenterline()
        cl_msg.header.stamp = rospy.Time.now()
        cl_msg.header.frame_id = 'base_link'
        cl_msg.lateral_offset = lateral_offset
        cl_msg.heading_error = heading_error
        cl_msg.row_width = row_width
        cl_msg.centerline_waypoint = Point(x=wp_x, y=wp_y, z=0.0)
        cl_msg.localization_source = self.localization_source
        cl_msg.gnss_available = (self.gnss_status == "AVAILABLE")
        cl_msg.lidar_rows_detected = (left_valid or right_valid)

        self.centerline_pub.publish(cl_msg)

        # Pure pursuit controller for velocity commands
        if self.enable_cmd_vel:
            self._pure_pursuit(lateral_offset, heading_error, left_valid or right_valid)

    def _pure_pursuit(self, lateral_offset, heading_error, rows_detected):
        """
        Pure pursuit controller for lane following.
        Computes curvature from the lookahead waypoint and converts
        to (linear_vel, angular_vel) commands.

        Reference: Liu et al. (2024) use pure pursuit with
        lookahead = 60% of predicted line length.
        """
        cmd = Twist()

        if not rows_detected:
            # Stop if no rows detected
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Target point on centerline at lookahead distance
            target_x = self.lookahead_distance
            target_y = lateral_offset + self.lookahead_distance * np.sin(heading_error)

            # Pure pursuit curvature: kappa = 2 * y / L^2
            L_sq = target_x**2 + target_y**2
            if L_sq > 0.01:
                curvature = 2.0 * target_y / L_sq
            else:
                curvature = 0.0

            # Convert to velocities
            cmd.linear.x = min(self.max_linear_vel,
                               self.max_linear_vel * (1.0 - 0.5 * abs(curvature)))
            cmd.angular.z = np.clip(
                cmd.linear.x * curvature,
                -self.max_angular_vel,
                self.max_angular_vel
            )

        if self.enable_cmd_vel:
            self.cmd_vel_pub.publish(cmd)

    def loc_status_callback(self, msg):
        self.localization_source = msg.data

    def gnss_status_callback(self, msg):
        self.gnss_status = msg.data

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = RowCenterlineNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
