#!/bin/bash
set -e

# 1) ROS2 기본 환경
if [ -f "/opt/ros/humble/setup.bash" ]; then
  source /opt/ros/humble/setup.bash
fi

# 2) 워크스페이스 오버레이
if [ -f "/ros2_ws/install/setup.bash" ]; then
  echo "[ros_entrypoint] sourcing /ros2_ws/install/setup.bash"
  source /ros2_ws/install/setup.bash
else
  echo "[ros_entrypoint] WARNING: /ros2_ws/install/setup.bash not found"
  ls -R /ros2_ws || true
fi

echo "[ros_entrypoint] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[ros_entrypoint] running: $@"

exec "$@"
