# RowCenterline.msg
# Computed centerline between crop rows — output for lane following
# and path planner return target (for WEED_INTERCEPT -> LANE_FOLLOW)

std_msgs/Header header

# Centerline offset: signed distance from robot to centerline (meters)
# Positive = robot is left of center, Negative = robot is right of center
float64 lateral_offset

# Heading error: angle between robot heading and row direction (radians)
# Positive = robot pointing left of row, Negative = robot pointing right
float64 heading_error

# Row width: distance between left and right rows (meters)
float64 row_width

# Centerline waypoint in robot frame (for path planner return target)
geometry_msgs/Point centerline_waypoint

# Localization source currently in use
string localization_source    # "gnss_imu_odom" | "imu_odom_only" | "lidar_imu"

# Overall system health
bool gnss_available
bool lidar_rows_detected
