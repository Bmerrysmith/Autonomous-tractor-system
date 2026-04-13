# CropRowState.msg
# EKF-filtered state of detected crop rows relative to robot
# Based on the [d, phi] state representation from Liu et al. (2024)
# "LiDAR-Based Crop Row Detection for Over-Canopy Autonomous Navigation"

std_msgs/Header header

# Left crop row
float64 left_row_distance      # perpendicular distance to left row (meters)
float64 left_row_angle          # angle of left row relative to robot heading (radians)

# Right crop row
float64 right_row_distance     # perpendicular distance to right row (meters)
float64 right_row_angle         # angle of right row relative to robot heading (radians)

# Confidence metrics
float64 left_confidence         # 0.0 to 1.0, based on EKF innovation
float64 right_confidence        # 0.0 to 1.0, based on EKF innovation

# Whether EKF is in prediction-only mode (no new LiDAR measurements)
bool prediction_only
