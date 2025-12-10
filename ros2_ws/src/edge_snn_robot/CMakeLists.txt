cmake_minimum_required(VERSION 3.8)
project(edge_snn_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Intent.msg"
  "msg/JointCmd.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

# Install Python package
# ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  edge_snn_robot/emg_intent_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME emg_intent_node
)

install(PROGRAMS
  edge_snn_robot/servo_cmd_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME servo_cmd_node
)

install(PROGRAMS
  edge_snn_robot/fake_hardware.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fake_hardware_node
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()