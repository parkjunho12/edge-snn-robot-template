#!/bin/bash
# Setup Script for Edge SNN Robot ROS2 Package
# 
# This script automatically:
# 1. Checks ROS2 installation
# 2. Installs dependencies
# 3. Builds the package
# 4. Provides next steps
#
# Usage: ./setup.sh

set -e

echo "========================================"
echo "Edge SNN Robot - ROS2 Setup"
echo "Version: 0.3.0"
echo "========================================"
echo ""

# Check ROS2 installation
if ! command -v ros2 &> /dev/null; then
    echo "❌ ROS2 not found!"
    echo ""
    echo "Please install ROS2 Humble:"
    echo "  https://docs.ros.org/en/humble/Installation.html"
    echo ""
    echo "Quick install (Ubuntu 22.04):"
    echo "  sudo apt update"
    echo "  sudo apt install ros-humble-desktop"
    echo ""
    exit 1
fi

echo "✓ ROS2 found"

# Source ROS2
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "✓ ROS2 Humble sourced"
elif [ -f "/opt/ros/foxy/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
    echo "✓ ROS2 Foxy sourced"
else
    echo "❌ No ROS2 installation found in /opt/ros/"
    exit 1
fi

# Check colcon
if ! command -v colcon &> /dev/null; then
    echo "❌ colcon not found!"
    echo "Installing colcon..."
    sudo apt update
    sudo apt install -y python3-colcon-common-extensions
fi

echo "✓ colcon found"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --user requests numpy || true

# Install ROS2 dependencies
echo ""
echo "Installing ROS2 dependencies..."
cd "$(dirname "$0")"
rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || echo "⚠ Some dependencies may be missing"

# Build package
echo ""
echo "Building edge_snn_robot package..."
colcon build --packages-select edge_snn_robot

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Setup Complete!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Source the workspace:"
    echo "   source install/setup.bash"
    echo ""
    echo "2. Start FastAPI server (in another terminal):"
    echo "   cd /path/to/project"
    echo "   ./start_server.sh"
    echo ""
    echo "3. Launch ROS2 system:"
    echo "   ros2 launch edge_snn_robot robot_control.launch.py"
    echo ""
    echo "4. Monitor topics:"
    echo "   ros2 topic list"
    echo "   ros2 topic echo /emg_intent"
    echo "   ros2 topic echo /joint_cmd"
    echo ""
    echo "5. (Optional) Run MATLAB simulator:"
    echo "   cd matlab_simulator"
    echo "   run_hand_simulator"
    echo ""
    echo "========================================"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    echo "Please check error messages above"
    exit 1
fi