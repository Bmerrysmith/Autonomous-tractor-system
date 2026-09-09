cmake_minimum_required(VERSION 3.0.2)
project(undercanopy_ekf_nav)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  message_generation
)

# Custom messages
add_message_files(
  FILES
  CropRowState.msg
  RowCenterline.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs nav_msgs geometry_msgs message_runtime
)

include_directories(${catkin_INCLUDE_DIRS})

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/lidar_row_detection_node.py
  scripts/ekf_localization_node.py
  scripts/row_centerline_node.py
  scripts/gnss_outage_bridge_node.py
  scripts/sim_sensor_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch and config
install(DIRECTORY launch config worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
