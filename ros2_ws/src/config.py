"""
ROS2 Configuration for EMG Inference System
Centralized configuration for ROS2 topics, QoS, and robot parameters
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path

try:
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Dummy QoS for when ROS2 is not available
    class QoSProfile:
        def __init__(self, *args, **kwargs):
            pass


@dataclass
class ROS2Config:
    """
    Configuration for ROS2 bridge and publishers
    
    This class holds all ROS2-related configuration including:
    - Topic names
    - QoS settings
    - Joint/servo parameters
    - Safety limits
    """
    
    # Topic names
    joint_topic: str = '/cmd_joint'
    pwm_topic: str = '/cmd_pwm'
    emg_topic: str = '/emg_data'  # For receiving EMG from ROS2
    status_topic: str = '/inference_status'
    
    # Message types
    use_joint_state_msg: bool = True  # Use JointState vs Float64MultiArray
    
    # Joint configuration
    joint_names: List[str] = field(default_factory=lambda: [
        'wrist_rotate',
        'finger_1',
        'finger_2',
        'finger_3',
        'thumb'
    ])
    
    # Servo/PWM configuration
    servo_names: List[str] = field(default_factory=lambda: [
        'servo_0',
        'servo_1',
        'servo_2',
        'servo_3',
        'servo_4'
    ])
    
    # Safety limits
    max_joint_velocity: float = 1.0  # rad/s
    max_joint_acceleration: float = 2.0  # rad/s²
    min_pwm_value: int = 1000  # Minimum safe PWM (typically 1000 for servos)
    max_pwm_value: int = 2000  # Maximum safe PWM (typically 2000 for servos)
    
    # Inference settings
    min_confidence_threshold: float = 0.7  # Minimum confidence to execute command
    prediction_smoothing: int = 3  # Number of consecutive predictions needed
    
    # Publishing settings
    enable_joint_commands: bool = True
    enable_pwm_commands: bool = True
    publish_rate_hz: float = 30.0  # Hz
    
    # QoS Profile
    qos_reliability: str = 'reliable'  # 'reliable' or 'best_effort'
    qos_history_depth: int = 10
    
    def __post_init__(self):
        """Initialize QoS profile after dataclass initialization"""
        if ROS2_AVAILABLE:
            # Create QoS profile based on settings
            reliability = (
                ReliabilityPolicy.RELIABLE 
                if self.qos_reliability == 'reliable' 
                else ReliabilityPolicy.BEST_EFFORT
            )
            
            self.qos_profile = QoSProfile(
                reliability=reliability,
                history=HistoryPolicy.KEEP_LAST,
                depth=self.qos_history_depth
            )
        else:
            self.qos_profile = QoSProfile()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ROS2Config':
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ROS2Config instance
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract ROS2 section
        ros2_config = config_dict.get('ros2', {})
        
        return cls(**ros2_config)
    
    def to_yaml(self, output_path: str):
        """
        Save configuration to YAML file
        
        Args:
            output_path: Path to save YAML configuration
        """
        config_dict = {
            'ros2': {
                'joint_topic': self.joint_topic,
                'pwm_topic': self.pwm_topic,
                'emg_topic': self.emg_topic,
                'status_topic': self.status_topic,
                'use_joint_state_msg': self.use_joint_state_msg,
                'joint_names': self.joint_names,
                'servo_names': self.servo_names,
                'max_joint_velocity': self.max_joint_velocity,
                'max_joint_acceleration': self.max_joint_acceleration,
                'min_pwm_value': self.min_pwm_value,
                'max_pwm_value': self.max_pwm_value,
                'min_confidence_threshold': self.min_confidence_threshold,
                'prediction_smoothing': self.prediction_smoothing,
                'enable_joint_commands': self.enable_joint_commands,
                'enable_pwm_commands': self.enable_pwm_commands,
                'publish_rate_hz': self.publish_rate_hz,
                'qos_reliability': self.qos_reliability,
                'qos_history_depth': self.qos_history_depth
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return (
            f"ROS2Config(\n"
            f"  Topics: {self.joint_topic}, {self.pwm_topic}\n"
            f"  Joints: {len(self.joint_names)}\n"
            f"  Servos: {len(self.servo_names)}\n"
            f"  PWM Range: [{self.min_pwm_value}, {self.max_pwm_value}]\n"
            f"  Min Confidence: {self.min_confidence_threshold}\n"
            f"  Publish Rate: {self.publish_rate_hz} Hz\n"
            f")"
        )


# Default configuration instance
DEFAULT_CONFIG = ROS2Config()


def create_default_config_file(output_path: str = 'config/ros2_config.yaml'):
    """
    Create a default configuration file
    
    Args:
        output_path: Path to save the configuration file
    """
    from pathlib import Path
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save default configuration
    DEFAULT_CONFIG.to_yaml(output_path)
    
    print(f"Created default ROS2 configuration at: {output_path}")


if __name__ == '__main__':
    # Create default config file when run as script
    create_default_config_file()
    print("\nDefault Configuration:")
    print(DEFAULT_CONFIG)