#!/usr/bin/env python3
"""
record_and_plot.py

Records data from all navigation topics for 60 seconds, then generates
publication-quality plots for the IEEE paper and EagleX poster.

Usage:
  1. Make sure the system is running:
       roslaunch undercanopy_ekf_nav undercanopy_nav.launch use_sim:=true
  2. In another terminal:
       source ~/catkin_ws/devel/setup.bash
       python3 ~/catkin_ws/src/undercanopy_ekf_nav/scripts/record_and_plot.py

Outputs saved to ~/catkin_ws/plots/

Author: Krish Shah — CEN 4930, FGCU Spring 2026
"""

import rospy
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # non-GUI backend for WSL
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Twist

# Try importing custom messages
try:
    from undercanopy_ekf_nav.msg import CropRowState, RowCenterline
    HAS_CUSTOM_MSGS = True
except ImportError:
    HAS_CUSTOM_MSGS = False
    print("[WARN] Custom messages not found. Using generic subscribers.")


class DataRecorder:
    def __init__(self, duration=60.0):
        rospy.init_node('data_recorder', anonymous=True)
        self.duration = duration

        # Storage
        self.times = []
        self.gnss_status = []
        self.gnss_outage_dur = []
        self.loc_status = []
        self.pose_x = []
        self.pose_y = []
        self.pose_times = []
        self.left_row_d = []
        self.right_row_d = []
        self.left_conf = []
        self.right_conf = []
        self.row_times = []
        self.lateral_offset = []
        self.heading_error = []
        self.row_width = []
        self.cl_times = []
        self.cmd_vx = []
        self.cmd_wz = []
        self.cmd_times = []
        self.outage_times = []
        self.outage_vals = []

        # Subscribers
        rospy.Subscriber('/gnss/status', String, self.gnss_status_cb)
        rospy.Subscriber('/gnss/outage_duration', Float64, self.outage_cb)
        rospy.Subscriber('/localization/status', String, self.loc_status_cb)
        rospy.Subscriber('/localization/pose', Odometry, self.pose_cb)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb)

        if HAS_CUSTOM_MSGS:
            rospy.Subscriber('/crop_row_state', CropRowState, self.row_cb)
            rospy.Subscriber('/row_centerline', RowCenterline, self.cl_cb)

        self.start_time = None
        self.output_dir = os.path.expanduser('~/catkin_ws/plots')
        os.makedirs(self.output_dir, exist_ok=True)

    def _t(self):
        if self.start_time is None:
            self.start_time = rospy.Time.now()
        return (rospy.Time.now() - self.start_time).to_sec()

    def gnss_status_cb(self, msg):
        t = self._t()
        self.times.append(t)
        self.gnss_status.append(msg.data)

    def outage_cb(self, msg):
        t = self._t()
        self.outage_times.append(t)
        self.outage_vals.append(msg.data)

    def loc_status_cb(self, msg):
        self.loc_status.append((self._t(), msg.data))

    def pose_cb(self, msg):
        t = self._t()
        self.pose_times.append(t)
        self.pose_x.append(msg.pose.pose.position.x)
        self.pose_y.append(msg.pose.pose.position.y)

    def row_cb(self, msg):
        t = self._t()
        self.row_times.append(t)
        self.left_row_d.append(msg.left_row_distance)
        self.right_row_d.append(msg.right_row_distance)
        self.left_conf.append(msg.left_confidence)
        self.right_conf.append(msg.right_confidence)

    def cl_cb(self, msg):
        t = self._t()
        self.cl_times.append(t)
        self.lateral_offset.append(msg.lateral_offset)
        self.heading_error.append(msg.heading_error)
        self.row_width.append(msg.row_width)

    def cmd_cb(self, msg):
        t = self._t()
        self.cmd_times.append(t)
        self.cmd_vx.append(msg.linear.x)
        self.cmd_wz.append(msg.angular.z)

    def record(self):
        print(f"[recorder] Recording for {self.duration} seconds...")
        print("[recorder] GNSS outage occurs at t=15s, returns at t=35s")
        rate = rospy.Rate(10)
        self._t()  # init start time
        while not rospy.is_shutdown() and self._t() < self.duration:
            elapsed = self._t()
            if int(elapsed) % 10 == 0 and int(elapsed * 10) % 100 == 0:
                print(f"  ... {int(elapsed)}s / {int(self.duration)}s")
            rate.sleep()
        print("[recorder] Recording complete. Generating plots...")

    def plot_all(self):
        plt.style.use('seaborn-whitegrid')
        fig_params = {'figsize': (10, 4), 'dpi': 150}

        # ──────────────────────────────────────────────────────────────
        # Plot 1: GNSS Outage Duration Over Time
        # ──────────────────────────────────────────────────────────────
        if self.outage_times:
            fig, ax = plt.subplots(**fig_params)
            ax.plot(self.outage_times, self.outage_vals, 'r-', linewidth=1.5,
                    label='GNSS Outage Duration')
            ax.axvspan(15, 35, alpha=0.15, color='red', label='GNSS Denied Window')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Outage Duration (s)', fontsize=12)
            ax.set_title('GNSS Outage Detection and Dead-Reckoning Bridging', fontsize=13)
            ax.legend(fontsize=10)
            ax.set_xlim(0, self.duration)
            plt.tight_layout()
            path = os.path.join(self.output_dir, 'gnss_outage_duration.png')
            plt.savefig(path)
            plt.close()
            print(f"  Saved: {path}")

        # ──────────────────────────────────────────────────────────────
        # Plot 2: Robot Position (X,Y) with GNSS outage highlighted
        # ──────────────────────────────────────────────────────────────
        if self.pose_times:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150,
                                            sharex=True)
            ax1.plot(self.pose_times, self.pose_x, 'b-', linewidth=1.2,
                     label='X Position (forward)')
            ax1.axvspan(15, 35, alpha=0.15, color='red', label='GNSS Denied')
            ax1.set_ylabel('X Position (m)', fontsize=11)
            ax1.set_title('EKF-Fused Position Estimate During GNSS Outage', fontsize=13)
            ax1.legend(fontsize=9)

            ax2.plot(self.pose_times, self.pose_y, 'g-', linewidth=1.2,
                     label='Y Position (lateral)')
            ax2.axvspan(15, 35, alpha=0.15, color='red', label='GNSS Denied')
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.set_ylabel('Y Position (m)', fontsize=11)
            ax2.legend(fontsize=9)

            plt.tight_layout()
            path = os.path.join(self.output_dir, 'position_tracking.png')
            plt.savefig(path)
            plt.close()
            print(f"  Saved: {path}")

        # ──────────────────────────────────────────────────────────────
        # Plot 3: Crop Row Detection — EKF-filtered distances
        # ──────────────────────────────────────────────────────────────
        if self.row_times:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150,
                                            sharex=True)
            ax1.plot(self.row_times, self.left_row_d, 'b-', linewidth=1,
                     label='Left Row Distance', alpha=0.8)
            ax1.plot(self.row_times, self.right_row_d, 'r-', linewidth=1,
                     label='Right Row Distance', alpha=0.8)
            ax1.axvspan(15, 35, alpha=0.1, color='red')
            ax1.set_ylabel('Distance (m)', fontsize=11)
            ax1.set_title('EKF-Filtered Crop Row Detection [d, \u03c6]', fontsize=13)
            ax1.legend(fontsize=9)

            ax2.plot(self.row_times, self.left_conf, 'b-', linewidth=1,
                     label='Left Row Confidence', alpha=0.8)
            ax2.plot(self.row_times, self.right_conf, 'r-', linewidth=1,
                     label='Right Row Confidence', alpha=0.8)
            ax2.axvspan(15, 35, alpha=0.1, color='red')
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.set_ylabel('Confidence (0-1)', fontsize=11)
            ax2.set_ylim(-0.1, 1.1)
            ax2.legend(fontsize=9)

            plt.tight_layout()
            path = os.path.join(self.output_dir, 'crop_row_detection.png')
            plt.savefig(path)
            plt.close()
            print(f"  Saved: {path}")

        # ──────────────────────────────────────────────────────────────
        # Plot 4: Lane Following — Lateral Offset and Heading Error
        # ──────────────────────────────────────────────────────────────
        if self.cl_times:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), dpi=150,
                                                  sharex=True)
            ax1.plot(self.cl_times, self.lateral_offset, 'b-', linewidth=1)
            ax1.axvspan(15, 35, alpha=0.1, color='red')
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.set_ylabel('Lateral Offset (m)', fontsize=11)
            ax1.set_title('Row Centerline Tracking Performance', fontsize=13)

            ax2.plot(self.cl_times, np.degrees(self.heading_error), 'r-', linewidth=1)
            ax2.axvspan(15, 35, alpha=0.1, color='red')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.set_ylabel('Heading Error (\u00b0)', fontsize=11)

            ax3.plot(self.cl_times, self.row_width, 'g-', linewidth=1)
            ax3.axvspan(15, 35, alpha=0.1, color='red')
            ax3.set_xlabel('Time (s)', fontsize=11)
            ax3.set_ylabel('Row Width (m)', fontsize=11)

            plt.tight_layout()
            path = os.path.join(self.output_dir, 'centerline_tracking.png')
            plt.savefig(path)
            plt.close()
            print(f"  Saved: {path}")

        # ──────────────────────────────────────────────────────────────
        # Plot 5: Pure Pursuit Controller Output
        # ──────────────────────────────────────────────────────────────
        if self.cmd_times:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), dpi=150,
                                            sharex=True)
            ax1.plot(self.cmd_times, self.cmd_vx, 'b-', linewidth=1)
            ax1.axvspan(15, 35, alpha=0.1, color='red')
            ax1.set_ylabel('Linear Vel (m/s)', fontsize=11)
            ax1.set_title('Pure Pursuit Controller Output', fontsize=13)

            ax2.plot(self.cmd_times, self.cmd_wz, 'r-', linewidth=1)
            ax2.axvspan(15, 35, alpha=0.1, color='red')
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.set_ylabel('Angular Vel (rad/s)', fontsize=11)

            plt.tight_layout()
            path = os.path.join(self.output_dir, 'controller_output.png')
            plt.savefig(path)
            plt.close()
            print(f"  Saved: {path}")

        # ──────────────────────────────────────────────────────────────
        # Plot 6: GNSS Status Timeline
        # ──────────────────────────────────────────────────────────────
        if self.times and self.gnss_status:
            fig, ax = plt.subplots(figsize=(10, 2.5), dpi=150)
            status_map = {'AVAILABLE': 2, 'DEGRADED': 1, 'OUTAGE': 0}
            status_vals = [status_map.get(s, -1) for s in self.gnss_status]

            ax.plot(self.times, status_vals, 'k-', linewidth=2, drawstyle='steps-post')
            ax.axvspan(15, 35, alpha=0.15, color='red', label='GNSS Denied Window')
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['OUTAGE', 'DEGRADED', 'AVAILABLE'], fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_title('GNSS Signal Status Over Time', fontsize=13)
            ax.set_xlim(0, self.duration)
            ax.legend(fontsize=9)

            plt.tight_layout()
            path = os.path.join(self.output_dir, 'gnss_status_timeline.png')
            plt.savefig(path)
            plt.close()
            print(f"  Saved: {path}")

        print(f"\n[recorder] All plots saved to {self.output_dir}/")
        print("[recorder] Copy to Windows: cp ~/catkin_ws/plots/*.png /mnt/c/Users/krock/Downloads/")


if __name__ == '__main__':
    try:
        recorder = DataRecorder(duration=60.0)
        recorder.record()
        recorder.plot_all()
    except rospy.ROSInterruptException:
        pass
