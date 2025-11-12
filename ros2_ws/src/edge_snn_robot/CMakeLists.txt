cmake_minimum_required(VERSION 3.5)
project(edge_snn_robot)
find_package(ament_cmake REQUIRED)
install(DIRECTORY scripts/
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
