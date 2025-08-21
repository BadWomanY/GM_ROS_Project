cmake_minimum_required(VERSION 3.0.2)
project(isaac_ros_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  geometry_msgs
  trajectory_msgs
  moveit_msgs
  message_generation
)

add_service_files(
  FILES
  PlanPose.srv
  AttachPart.srv
  DetachPart.srv
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  trajectory_msgs
)

catkin_package(CATKIN_DEPENDS
  message_runtime
  std_msgs
  geometry_msgs
  trajectory_msgs
  moveit_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# catkin_install_python(PROGRAMS
#   scripts/gym_test.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
