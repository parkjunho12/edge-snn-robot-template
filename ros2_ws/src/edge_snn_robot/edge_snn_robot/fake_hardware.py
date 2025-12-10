#!/usr/bin/env python3
"""
Fake Hardware Node

Simulates robot hardware for development and testing WITHOUT real motors/servos.
This allows full ROS2 development without physical robot hardware.

⚠️  REPLACE WITH REAL HARDWARE DRIVER FOR PRODUCTION!

Real driver should:
- Connect to ESP32 via Serial/WiFi
- Send PWM commands to PCA9685 servo driver
- Read encoder feedback
- Handle emergency stop

Author: EMG Robot Team
Date: 2025-12
"""

import rclpy
from rclpy.node import Node
from edge_snn_robot.msg import JointCmd
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import time
import numpy as np


class FakeHardwareNode(Node):
    """
    Fake Hardware Node
    
    Simulates robot hardware without actual motors/servos.
    Receives JointCmd and simulates execution.
    """
    
    def __init__(self):
        super().__init__('fake_hardware_node')
        
        # Declare parameters
        self.declare_parameter('enable_feedback', True)
        self.declare_parameter('feedback_rate_hz', 10.0)
        self.declare_parameter('enable_logging', True)
        self.declare_parameter('verbose', False)
        
        # Get parameters
        self.enable_feedback = self.get_parameter('enable_feedback').value
        feedback_rate = self.get_parameter('feedback_rate_hz').value
        self.enable_logging = self.get_parameter('enable_logging').value
        self.verbose = self.get_parameter('verbose').value
        
        # Subscriber
        self.cmd_sub = self.create_subscription(
            JointCmd,
            '/joint_cmd',
            self.command_callback,
            10
        )
        
        # Publisher (feedback)
        if self.enable_feedback:
            self.state_pub = self.create_publisher(JointState, '/joint_state', 10)
            self.feedback_timer = self.create_timer(
                1.0 / feedback_rate,
                self.publish_feedback
            )
        
        # Simulated hardware state
        self.current_angles = None
        self.current_pwm = None
        self.last_gesture_id = -1
        self.last_cmd_time = time.time()
        
        # Statistics
        self.command_count = 0
        self.gesture_counts = {}
        self.start_time = time.time()
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('⚠️  FAKE HARDWARE NODE STARTED')
        self.get_logger().info('=' * 50)
        self.get_logger().warn('THIS IS SIMULATION - NO REAL HARDWARE')
        self.get_logger().warn('Replace with real_hardware.py for production')
        self.get_logger().info(f'Feedback: {self.enable_feedback}')
        self.get_logger().info(f'Logging: {self.enable_logging}')
        self.get_logger().info('=' * 50)
    
    def command_callback(self, msg: JointCmd):
        """Receive joint command and simulate execution"""
        
        # Update state
        self.current_angles = np.array(msg.joint_angles)
        self.current_pwm = np.array(msg.pwm_values)
        self.last_gesture_id = msg.gesture_id
        self.last_cmd_time = time.time()
        
        self.command_count += 1
        
        # Track gesture statistics
        gesture_id = msg.gesture_id
        if gesture_id not in self.gesture_counts:
            self.gesture_counts[gesture_id] = 0
        self.gesture_counts[gesture_id] += 1
        
        # Log command (verbose mode)
        if self.enable_logging and self.verbose:
            self.get_logger().info(
                f'Cmd #{self.command_count}: '
                f'Gesture={gesture_id}, '
                f'Joints={len(msg.joint_angles)}, '
                f'PWM={len(msg.pwm_values)}, '
                f'Conf={msg.confidence:.2f}'
            )
        
        # Simulate hardware execution delay
        # Real hardware would:
        # 1. Send serial command to ESP32
        # 2. ESP32 writes to PCA9685 I2C
        # 3. Servos move to position
        # 4. Read encoder feedback
        time.sleep(0.001)  # 1ms simulated delay
        
        # Periodic summary
        if self.command_count % 100 == 0:
            self.log_statistics()
    
    def publish_feedback(self):
        """Publish simulated joint state feedback"""
        
        if self.current_angles is None:
            return
        
        # Create JointState message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'robot_hand'
        
        # In simulation, assume perfect tracking
        msg.name = [f'joint_{i}' for i in range(len(self.current_angles))]
        msg.position = self.current_angles.tolist()
        msg.velocity = [0.0] * len(self.current_angles)
        msg.effort = [0.0] * len(self.current_angles)
        
        self.state_pub.publish(msg)
    
    def log_statistics(self):
        """Log command statistics"""
        elapsed = time.time() - self.start_time
        rate = self.command_count / elapsed if elapsed > 0 else 0
        
        self.get_logger().info(
            f'Statistics: {self.command_count} commands '
            f'({rate:.1f} Hz, {elapsed:.1f}s)'
        )
        
        # Gesture breakdown
        for gesture_id in sorted(self.gesture_counts.keys()):
            count = self.gesture_counts[gesture_id]
            percentage = (count / self.command_count) * 100
            self.get_logger().info(
                f'  Gesture {gesture_id}: {count} ({percentage:.1f}%)'
            )


def main(args=None):
    rclpy.init(args=args)
    node = FakeHardwareNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Final statistics
        node.get_logger().info('=' * 60)
        node.get_logger().info('FAKE HARDWARE - FINAL STATISTICS')
        node.get_logger().info('=' * 60)
        node.log_statistics()
        node.get_logger().info('=' * 60)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()