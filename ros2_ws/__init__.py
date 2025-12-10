"""
ROS2 Interface Package
Bridges FastAPI EMG inference server with ROS2 robot control

Components:
- ros2_bridge: Main bridge connecting inference to robot commands
- publishers: ROS2 publishers for joint and PWM commands
- config: Configuration management for ROS2 settings
"""

from .config import ROS2Config, DEFAULT_CONFIG, create_default_config_file
from .ros2_bridge import ROS2Bridge, get_bridge

# Check if ROS2 is available
try:
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

__all__ = [
    'ROS2Config',
    'DEFAULT_CONFIG',
    'ROS2Bridge',
    'get_bridge',
    'create_default_config_file',
    'ROS2_AVAILABLE'
]

__version__ = '0.3.0'
