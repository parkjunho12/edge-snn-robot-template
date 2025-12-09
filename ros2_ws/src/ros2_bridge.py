"""
ROS2 Bridge for EMG Inference System
Connects FastAPI inference server to ROS2 robot control system
Publishes joint commands and PWM signals based on EMG predictions
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray, Int32MultiArray
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("ROS2 (rclpy) not available. ROS2 bridge will run in simulation mode.")

from .publishers import JointCommandPublisher, PWMCommandPublisher
from .config import ROS2Config

logger = logging.getLogger(__name__)


class ROS2Bridge:
    """
    Bridge between FastAPI inference server and ROS2 robot control
    
    This class handles:
    - Converting EMG predictions to robot commands
    - Publishing joint angles to /cmd_joint
    - Publishing PWM signals to /cmd_pwm
    - Rate limiting and safety checks
    """
    
    def __init__(self, config: Optional[ROS2Config] = None):
        """
        Initialize ROS2 bridge
        
        Args:
            config: ROS2 configuration object
        """
        self.config = config or ROS2Config()
        self.ros2_available = ROS2_AVAILABLE
        
        # ROS2 node and publishers
        self.node: Optional[Node] = None
        self.joint_publisher: Optional[JointCommandPublisher] = None
        self.pwm_publisher: Optional[PWMCommandPublisher] = None
        
        # State tracking
        self.last_prediction: Optional[int] = None
        self.prediction_count: int = 0
        self.is_running: bool = False
        
        # Safety limits
        self.max_joint_velocity = self.config.max_joint_velocity
        self.max_pwm_value = self.config.max_pwm_value
        self.min_pwm_value = self.config.min_pwm_value
        
        # Prediction to command mapping
        self.prediction_to_joint = self._create_prediction_mapping()
        
        logger.info(f"ROS2 Bridge initialized (ROS2 available: {self.ros2_available})")
    
    def initialize(self):
        """Initialize ROS2 node and publishers"""
        if not self.ros2_available:
            logger.warning("ROS2 not available, running in simulation mode")
            return
        
        try:
            # Initialize ROS2
            if not rclpy.ok():
                rclpy.init()
            
            # Create node
            self.node = rclpy.create_node('emg_inference_bridge')
            logger.info("ROS2 node 'emg_inference_bridge' created")
            
            # Create publishers
            self.joint_publisher = JointCommandPublisher(
                self.node,
                self.config
            )
            
            self.pwm_publisher = PWMCommandPublisher(
                self.node,
                self.config
            )
            
            self.is_running = True
            logger.info("ROS2 publishers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ROS2: {e}")
            self.ros2_available = False
    
    def _create_prediction_mapping(self) -> Dict[int, Dict[str, Any]]:
        """
        Create mapping from prediction class to robot commands
        
        Returns:
            Dictionary mapping prediction to joint angles and PWM values
        """
        # Example mapping - customize based on your robot and gestures
        mapping = {
            0: {  # Rest/No gesture
                'joint_angles': [0.0, 0.0, 0.0, 0.0, 0.0],  # Neutral position
                'pwm_values': [1500, 1500, 1500, 1500, 1500],  # Servo center
                'description': 'Rest position'
            },
            1: {  # Gesture 1: Hand open
                'joint_angles': [0.0, 0.5, 0.5, 0.5, 0.5],
                'pwm_values': [1500, 1800, 1800, 1800, 1800],
                'description': 'Hand open'
            },
            2: {  # Gesture 2: Hand close/Grasp
                'joint_angles': [0.0, -0.5, -0.5, -0.5, -0.5],
                'pwm_values': [1500, 1200, 1200, 1200, 1200],
                'description': 'Hand close'
            },
            3: {  # Gesture 3: Wrist flex
                'joint_angles': [0.5, 0.0, 0.0, 0.0, 0.0],
                'pwm_values': [1800, 1500, 1500, 1500, 1500],
                'description': 'Wrist flex'
            },
            4: {  # Gesture 4: Wrist extend
                'joint_angles': [-0.5, 0.0, 0.0, 0.0, 0.0],
                'pwm_values': [1200, 1500, 1500, 1500, 1500],
                'description': 'Wrist extend'
            },
            5: {  # Gesture 5: Pinch
                'joint_angles': [0.0, -0.3, -0.3, 0.5, 0.5],
                'pwm_values': [1500, 1300, 1300, 1800, 1800],
                'description': 'Pinch grasp'
            },
            6: {  # Gesture 6: Point
                'joint_angles': [0.0, 0.5, -0.5, -0.5, -0.5],
                'pwm_values': [1500, 1800, 1200, 1200, 1200],
                'description': 'Point gesture'
            }
        }
        
        return mapping
    
    def process_prediction(
        self,
        prediction: int,
        confidence: float,
        timestamp: float
    ) -> bool:
        """
        Process EMG prediction and publish robot commands
        
        Args:
            prediction: Predicted gesture class (0-6)
            confidence: Prediction confidence (0.0-1.0)
            timestamp: Prediction timestamp
            
        Returns:
            True if command was published successfully
        """
        # Confidence threshold check
        if confidence < self.config.min_confidence_threshold:
            logger.debug(f"Confidence {confidence:.2f} below threshold, ignoring prediction")
            return False
        
        # Get command mapping
        if prediction not in self.prediction_to_joint:
            logger.warning(f"Unknown prediction class: {prediction}")
            return False
        
        command = self.prediction_to_joint[prediction]
        
        # Update state
        self.last_prediction = prediction
        self.prediction_count += 1
        
        # Publish commands
        success = True
        
        if self.config.enable_joint_commands:
            success &= self._publish_joint_command(
                command['joint_angles'],
                command['description']
            )
        
        if self.config.enable_pwm_commands:
            success &= self._publish_pwm_command(
                command['pwm_values'],
                command['description']
            )
        
        return success
    
    def _publish_joint_command(
        self,
        joint_angles: list,
        description: str
    ) -> bool:
        """
        Publish joint command to ROS2
        
        Args:
            joint_angles: List of joint angles in radians
            description: Command description for logging
            
        Returns:
            True if published successfully
        """
        if not self.is_running or self.joint_publisher is None:
            if self.ros2_available:
                logger.warning("Joint publisher not initialized")
            return False
        
        try:
            # Apply safety limits
            safe_angles = self._apply_joint_limits(joint_angles)
            
            # Publish
            self.joint_publisher.publish(safe_angles)
            logger.debug(f"Published joint command: {description}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish joint command: {e}")
            return False
    
    def _publish_pwm_command(
        self,
        pwm_values: list,
        description: str
    ) -> bool:
        """
        Publish PWM command to ROS2
        
        Args:
            pwm_values: List of PWM values (typically 1000-2000)
            description: Command description for logging
            
        Returns:
            True if published successfully
        """
        if not self.is_running or self.pwm_publisher is None:
            if self.ros2_available:
                logger.warning("PWM publisher not initialized")
            return False
        
        try:
            # Apply safety limits
            safe_pwm = self._apply_pwm_limits(pwm_values)
            
            # Publish
            self.pwm_publisher.publish(safe_pwm)
            logger.debug(f"Published PWM command: {description}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish PWM command: {e}")
            return False
    
    def _apply_joint_limits(self, angles: list) -> list:
        """Apply safety limits to joint angles"""
        limited = []
        for angle in angles:
            # Clamp to ±π radians
            clamped = np.clip(angle, -np.pi, np.pi)
            limited.append(float(clamped))
        return limited
    
    def _apply_pwm_limits(self, pwm_values: list) -> list:
        """Apply safety limits to PWM values"""
        limited = []
        for pwm in pwm_values:
            # Clamp to configured min/max
            clamped = np.clip(
                pwm,
                self.min_pwm_value,
                self.max_pwm_value
            )
            limited.append(int(clamped))
        return limited
    
    def stop_robot(self):
        """Send stop command (all joints to neutral position)"""
        neutral_command = self.prediction_to_joint[0]  # Use rest position
        
        if self.config.enable_joint_commands:
            self._publish_joint_command(
                neutral_command['joint_angles'],
                'STOP - Neutral position'
            )
        
        if self.config.enable_pwm_commands:
            self._publish_pwm_command(
                neutral_command['pwm_values'],
                'STOP - Neutral PWM'
            )
        
        logger.info("Robot stopped - neutral position commanded")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current bridge status
        
        Returns:
            Dictionary with bridge status information
        """
        return {
            'ros2_available': self.ros2_available,
            'is_running': self.is_running,
            'last_prediction': self.last_prediction,
            'prediction_count': self.prediction_count,
            'joint_commands_enabled': self.config.enable_joint_commands,
            'pwm_commands_enabled': self.config.enable_pwm_commands
        }
    
    def shutdown(self):
        """Shutdown ROS2 bridge"""
        logger.info("Shutting down ROS2 bridge...")
        
        # Stop robot
        self.stop_robot()
        
        # Mark as stopped
        self.is_running = False
        
        # Cleanup ROS2
        if self.ros2_available and self.node is not None:
            try:
                self.node.destroy_node()
                rclpy.shutdown()
                logger.info("ROS2 node destroyed")
            except Exception as e:
                logger.error(f"Error shutting down ROS2: {e}")
    
    def __del__(self):
        """Destructor - ensure clean shutdown"""
        if self.is_running:
            self.shutdown()


# Singleton instance for FastAPI integration
_bridge_instance: Optional[ROS2Bridge] = None


def get_bridge(config: Optional[ROS2Config] = None) -> ROS2Bridge:
    """
    Get or create ROS2 bridge singleton
    
    Args:
        config: ROS2 configuration (only used on first call)
        
    Returns:
        ROS2Bridge instance
    """
    global _bridge_instance
    
    if _bridge_instance is None:
        _bridge_instance = ROS2Bridge(config)
        _bridge_instance.initialize()
    
    return _bridge_instance