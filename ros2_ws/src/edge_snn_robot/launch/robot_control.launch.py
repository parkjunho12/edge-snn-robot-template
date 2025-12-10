#!/usr/bin/env python3
"""
Robot Control Launch File

Launches all nodes for EMG-controlled robot system:
1. emg_intent_node - connects to inference server
2. servo_cmd_node - converts intent to joint commands
3. fake_hardware_node - simulates robot hardware

Usage:
    ros2 launch edge_snn_robot robot_control.launch.py

    # Custom server:
    ros2 launch edge_snn_robot robot_control.launch.py server_url:=http://192.168.1.100:8000

    # Higher confidence threshold:
    ros2 launch edge_snn_robot robot_control.launch.py min_confidence:=0.85

    # Without hardware (visualization only):
    ros2 launch edge_snn_robot robot_control.launch.py enable_hardware:=false

Author: EMG Robot Team
Date: 2025-12
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description with all nodes"""

    # Declare launch arguments
    server_url_arg = DeclareLaunchArgument(
        "server_url",
        default_value="http://localhost:8000",
        description="FastAPI inference server URL",
    )

    poll_rate_arg = DeclareLaunchArgument(
        "poll_rate_hz", default_value="30.0", description="Inference polling rate (Hz)"
    )

    min_confidence_arg = DeclareLaunchArgument(
        "min_confidence",
        default_value="0.7",
        description="Minimum prediction confidence threshold",
    )

    enable_hardware_arg = DeclareLaunchArgument(
        "enable_hardware",
        default_value="true",
        description="Enable hardware node (false for visualization only)",
    )

    # Get configurations
    server_url = LaunchConfiguration("server_url")
    poll_rate_hz = LaunchConfiguration("poll_rate_hz")
    min_confidence = LaunchConfiguration("min_confidence")

    # Node 1: EMG Intent Publisher
    emg_intent_node = Node(
        package="edge_snn_robot",
        executable="emg_intent_node",
        name="emg_intent_node",
        output="screen",
        parameters=[
            {
                "server_url": server_url,
                "poll_rate_hz": poll_rate_hz,
                "min_confidence": min_confidence,
                "encoding_type": "rate",
                "model_prefix": "tcn",
                "device": "cpu",
            }
        ],
    )

    # Node 2: Servo Command Generator
    servo_cmd_node = Node(
        package="edge_snn_robot",
        executable="servo_cmd_node",
        name="servo_cmd_node",
        output="screen",
        parameters=[
            {
                "min_confidence": min_confidence,
                "filter_alpha": 0.3,
                "enable_safety_limits": True,
            }
        ],
    )

    # Node 3: Fake Hardware
    fake_hardware_node = Node(
        package="edge_snn_robot",
        executable="fake_hardware_node",
        name="fake_hardware_node",
        output="screen",
        parameters=[
            {
                "enable_feedback": True,
                "feedback_rate_hz": 10.0,
                "enable_logging": True,
                "verbose": False,
            }
        ],
    )

    return LaunchDescription(
        [
            # Arguments
            server_url_arg,
            poll_rate_arg,
            min_confidence_arg,
            enable_hardware_arg,
            # Nodes
            emg_intent_node,
            servo_cmd_node,
            fake_hardware_node,
        ]
    )
