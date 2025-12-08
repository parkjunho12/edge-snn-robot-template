"""
Robot Control Abstraction Layer
Unified interface that works with both MATLAB simulator and real ROS2 robot

This allows seamless switching between:
1. MATLAB simulation (development/testing)
2. Real ROS2 robot (production)

Usage:
    # For MATLAB simulator
    controller = RobotController(mode='matlab')
    
    # For real ROS2 robot
    controller = RobotController(mode='ros2')
    
    # Then use the same API
    controller.send_command(prediction=2, confidence=0.85)
"""

import logging
from enum import Enum
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    """Robot control mode"""
    MATLAB = 'matlab'      # MATLAB simulator (HTTP-based)
    ROS2 = 'ros2'          # Real ROS2 robot
    SIMULATION = 'sim'     # Python-only simulation (no external connection)


class RobotController:
    """
    Unified robot control interface
    
    This class provides a single API that works with:
    - MATLAB simulator (via HTTP)
    - Real ROS2 robot (via ROS2 bridge)
    - Python simulation (standalone)
    
    The goal is to make it trivial to switch between simulation and real robot:
    Just change the mode parameter!
    """
    
    def __init__(
        self,
        mode: str = 'matlab',
        matlab_url: str = 'http://localhost:8000',
        ros2_config: Optional[Any] = None
    ):
        """
        Initialize robot controller
        
        Args:
            mode: Control mode ('matlab', 'ros2', or 'sim')
            matlab_url: URL for MATLAB simulator (if mode='matlab')
            ros2_config: ROS2 configuration (if mode='ros2')
        """
        self.mode = ControlMode(mode)
        self.matlab_url = matlab_url
        
        # Initialize based on mode
        if self.mode == ControlMode.MATLAB:
            self._init_matlab()
        elif self.mode == ControlMode.ROS2:
            self._init_ros2(ros2_config)
        elif self.mode == ControlMode.SIMULATION:
            self._init_simulation()
        
        # Statistics
        self.command_count = 0
        self.start_time = time.time()
        print(f"Robot controller initialized in {self.mode.value} mode")
        logger.info(f"Robot controller initialized in {self.mode.value} mode")
    
    def _init_matlab(self):
        """Initialize MATLAB simulator connection"""
        logger.info(f"Connecting to MATLAB simulator at {self.matlab_url}")
        
        print(f"Connecting to MATLAB simulator at {self.matlab_url}...")
        # MATLAB simulator doesn't need initialization
        # It receives commands via its own HTTP endpoint
        # or we can just rely on the inference server sending predictions
        
        self.matlab_connected = True
        logger.info("MATLAB mode: Commands will be sent via inference server")
    
    def _init_ros2(self, config):
        """Initialize ROS2 bridge"""
        try:
            from src.ros2_interface import get_bridge, ROS2Config
            
            if config is None:
                config = ROS2Config()
            
            self.ros2_bridge = get_bridge(config)
            logger.info("ROS2 bridge initialized successfully")
            
        except ImportError:
            logger.error("ROS2 not available. Install rclpy or use matlab/sim mode")
            raise
    
    def _init_simulation(self):
        """Initialize Python-only simulation"""
        logger.info("Running in pure simulation mode (no external connection)")
        self.sim_state = {
            'joint_angles': [0.0] * 5,
            'last_prediction': None,
            'last_confidence': 0.0
        }
    
    def send_command(
        self,
        prediction: int,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Send command to robot (unified interface)
        
        This method works the same regardless of mode!
        
        Args:
            prediction: Gesture class (0-6)
            confidence: Prediction confidence (0-1)
            timestamp: Command timestamp (optional)
            
        Returns:
            True if command sent successfully
        """
        if timestamp is None:
            timestamp = time.time()
        
        success = False
        
        try:
            if self.mode == ControlMode.MATLAB:
                success = self._send_matlab_command(prediction, confidence, timestamp)
            
            elif self.mode == ControlMode.ROS2:
                success = self._send_ros2_command(prediction, confidence, timestamp)
            
            elif self.mode == ControlMode.SIMULATION:
                success = self._send_sim_command(prediction, confidence, timestamp)
            
            if success:
                self.command_count += 1
                
                if self.command_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.command_count / elapsed
                    logger.info(f"Sent {self.command_count} commands ({rate:.1f} Hz)")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def _send_matlab_command(self, prediction, confidence, timestamp) -> bool:
        """Send command to MATLAB simulator"""
        # MATLAB simulator reads from the inference stream
        # So we don't need to send anything explicitly
        # Just log the command
        logger.debug(f"MATLAB: prediction={prediction}, confidence={confidence:.2f}")
        return True
    
    def _send_ros2_command(self, prediction, confidence, timestamp) -> bool:
        """Send command to ROS2 robot"""
        return self.ros2_bridge.process_prediction(
            prediction=prediction,
            confidence=confidence,
            timestamp=timestamp
        )
    
    def _send_sim_command(self, prediction, confidence, timestamp) -> bool:
        """Update simulation state"""
        # Update internal simulation state
        self.sim_state['last_prediction'] = prediction
        self.sim_state['last_confidence'] = confidence
        
        # Simple joint angle mapping (same as ROS2)
        angle_mapping = {
            0: [0.0, 0.0, 0.0, 0.0, 0.0],
            1: [0.0, 0.5, 0.5, 0.5, 0.5],
            2: [0.0, -0.5, -0.5, -0.5, -0.5],
            3: [0.5, 0.0, 0.0, 0.0, 0.0],
            4: [-0.5, 0.0, 0.0, 0.0, 0.0],
            5: [0.0, -0.3, -0.3, 0.5, 0.5],
            6: [0.0, 0.5, -0.5, -0.5, -0.5]
        }
        
        if prediction in angle_mapping:
            self.sim_state['joint_angles'] = angle_mapping[prediction]
        
        logger.debug(f"SIM: prediction={prediction}, angles={self.sim_state['joint_angles']}")
        return True
    
    def stop(self):
        """Stop robot (return to neutral position)"""
        logger.info("Stopping robot - returning to neutral position")
        
        if self.mode == ControlMode.ROS2:
            self.ros2_bridge.stop_robot()
        
        # Send rest position (class 0)
        self.send_command(prediction=0, confidence=1.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        elapsed = time.time() - self.start_time
        
        status = {
            'mode': self.mode.value,
            'command_count': self.command_count,
            'elapsed_time': elapsed,
            'command_rate_hz': self.command_count / elapsed if elapsed > 0 else 0
        }
        
        if self.mode == ControlMode.ROS2:
            status['ros2_status'] = self.ros2_bridge.get_status()
        elif self.mode == ControlMode.SIMULATION:
            status['sim_state'] = self.sim_state
        
        return status
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'mode'):
            if self.mode == ControlMode.ROS2 and hasattr(self, 'ros2_bridge'):
                self.ros2_bridge.shutdown()


def create_controller(mode: str = 'matlab', **kwargs) -> RobotController:
    """
    Factory function to create robot controller
    
    Args:
        mode: 'matlab', 'ros2', or 'sim'
        **kwargs: Additional arguments passed to RobotController
        
    Returns:
        RobotController instance
    
    Example:
        # For development with MATLAB
        controller = create_controller('matlab')
        
        # For production with real robot
        controller = create_controller('ros2', ros2_config=config)
    """
    return RobotController(mode=mode, **kwargs)


# Example usage
if __name__ == '__main__':
    import sys
    
    # Parse command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else 'matlab'
    
    print(f"Testing robot controller in {mode} mode...")
    
    # Create controller
    controller = create_controller(mode)
    
    # Test commands
    gestures = [
        (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
          (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
          (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
        (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
          (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
          (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
        (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
          (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
          (0, 'Rest'),
        (1, 'Hand Open'),
        (2, 'Hand Close'),
        (3, 'Wrist Flex'),
        (4, 'Wrist Extend'),
        (5, 'Pinch'),
        (6, 'Point'),
    ]
    
    import time
    for gesture_id, gesture_name in gestures:
        print(f"\nTesting {gesture_name}...")
        controller.send_command(
            prediction=gesture_id,
            confidence=0.9
        )
        time.sleep(1.0)
    
    # Return to rest
    print("\nReturning to rest position...")
    controller.stop()
    
    # Show statistics
    print("\nStatistics:")
    status = controller.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")