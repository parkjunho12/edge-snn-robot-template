#!/usr/bin/env python3
"""
Servo Command Node

Converts Intent (gesture predictions) to JointCmd (robot control commands).
Maps 7 gestures to 17 DOF robot hand with smoothing filter and safety limits.

Author: EMG Robot Team
Date: 2025-12
"""
from edge_snn_robot.msg import Intent, JointCmd
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import numpy as np


class ServoCmdNode(Node):
    """
    Servo Command Node
    
    Maps EMG intent (gestures) to robot joint commands
    with filtering and safety limits.
    """
    
    def __init__(self):
        super().__init__('servo_cmd_node')
        
        # Declare parameters
        self.declare_parameter('min_confidence', 0.7)
        self.declare_parameter('filter_alpha', 0.3)
        self.declare_parameter('enable_safety_limits', True)
        
        # Get parameters
        self.min_confidence = self.get_parameter('min_confidence').value
        self.filter_alpha = self.get_parameter('filter_alpha').value
        self.enable_safety = self.get_parameter('enable_safety_limits').value
        
        # Subscriber
        self.intent_sub = self.create_subscription(
            Intent,
            '/emg_intent',
            self.intent_callback,
            10
        )
        
        # Publisher
        self.joint_cmd_pub = self.create_publisher(JointCmd, '/joint_cmd', 10)
        
        # Joint configuration (17 DOF: 2 wrist + 5 fingers × 3 joints)
        self.joint_names = [
            'wrist_rotation', 'wrist_flexion',
            'thumb_mcp', 'thumb_pip', 'thumb_dip',
            'index_mcp', 'index_pip', 'index_dip',
            'middle_mcp', 'middle_pip', 'middle_dip',
            'ring_mcp', 'ring_pip', 'ring_dip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip'
        ]
        
        # PWM configuration
        self.num_servos = 17
        self.pwm_min = 1000
        self.pwm_max = 2000
        self.pwm_center = 1500
        
        # Current state (for filtering)
        self.current_angles = np.zeros(len(self.joint_names))
        self.current_pwm = np.full(self.num_servos, self.pwm_center)
        
        # Gesture mappings
        self.gesture_mappings = self._init_gesture_mappings()
        
        # Statistics
        self.command_count = 0
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('Servo Command Node Started')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Joints: {len(self.joint_names)}')
        self.get_logger().info(f'Servos: {self.num_servos}')
        self.get_logger().info(f'Safety Limits: {self.enable_safety}')
        self.get_logger().info(f'Filter Alpha: {self.filter_alpha}')
        self.get_logger().info('=' * 50)
    
    def _init_gesture_mappings(self):
        """Initialize gesture to joint angle mappings (same as MATLAB!)"""
        mappings = {
            0: {  # Rest
                'wrist': [0.0, 0.0],
                'thumb': [0.0, 0.0, 0.0],
                'index': [0.0, 0.0, 0.0],
                'middle': [0.0, 0.0, 0.0],
                'ring': [0.0, 0.0, 0.0],
                'pinky': [0.0, 0.0, 0.0]
            },
            1: {  # Hand Open
                'wrist': [0.0, 0.0],
                'thumb': [0.8, 0.0, 0.0],
                'index': [0.0, 0.0, 0.0],
                'middle': [0.0, 0.0, 0.0],
                'ring': [0.0, 0.0, 0.0],
                'pinky': [0.0, 0.0, 0.0]
            },
            2: {  # Hand Close
                'wrist': [0.0, 0.0],
                'thumb': [0.3, 1.2, 1.0],
                'index': [1.0, 1.2, 1.0],
                'middle': [1.0, 1.2, 1.0],
                'ring': [1.0, 1.2, 1.0],
                'pinky': [1.0, 1.2, 1.0]
            },
            3: {  # Wrist Flex
                'wrist': [0.0, 0.5],
                'thumb': [0.3, 0.2, 0.1],
                'index': [0.2, 0.2, 0.1],
                'middle': [0.2, 0.2, 0.1],
                'ring': [0.2, 0.2, 0.1],
                'pinky': [0.2, 0.2, 0.1]
            },
            4: {  # Wrist Extend
                'wrist': [0.0, -0.5],
                'thumb': [0.3, 0.2, 0.1],
                'index': [0.2, 0.2, 0.1],
                'middle': [0.2, 0.2, 0.1],
                'ring': [0.2, 0.2, 0.1],
                'pinky': [0.2, 0.2, 0.1]
            },
            5: {  # Pinch
                'wrist': [0.0, 0.0],
                'thumb': [0.5, 0.8, 0.6],
                'index': [0.8, 1.0, 0.8],
                'middle': [0.2, 0.3, 0.2],
                'ring': [0.2, 0.3, 0.2],
                'pinky': [0.2, 0.3, 0.2]
            },
            6: {  # Point
                'wrist': [0.0, 0.0],
                'thumb': [0.3, 1.0, 0.8],
                'index': [0.0, 0.0, 0.0],
                'middle': [1.0, 1.2, 1.0],
                'ring': [1.0, 1.2, 1.0],
                'pinky': [1.0, 1.2, 1.0]
            }
        }
        return mappings
    
    def intent_callback(self, msg: Intent):
        """Process Intent message and publish JointCmd"""
        
        # Check confidence
        if msg.confidence < self.min_confidence:
            return
        
        # Get gesture mapping
        if msg.gesture_id not in self.gesture_mappings:
            self.get_logger().warn(f'Unknown gesture: {msg.gesture_id}')
            return
        
        mapping = self.gesture_mappings[msg.gesture_id]
        
        # Convert to joint angles array [17 values]
        target_angles = np.array(
            mapping['wrist'] +
            mapping['thumb'] +
            mapping['index'] +
            mapping['middle'] +
            mapping['ring'] +
            mapping['pinky']
        )
        
        # Apply exponential smoothing filter
        self.current_angles = (
            self.filter_alpha * target_angles +
            (1 - self.filter_alpha) * self.current_angles
        )
        
        # Apply safety limits
        if self.enable_safety:
            self.current_angles = np.clip(self.current_angles, -1.5, 1.5)
        
        # Convert angles to PWM (1000-2000 μs)
        angle_range = 3.0  # -1.5 to 1.5 radians
        pwm_range = self.pwm_max - self.pwm_min
        
        self.current_pwm = (
            self.pwm_center +
            (self.current_angles / angle_range) * (pwm_range / 2)
        ).astype(int)
        
        # Clip PWM values
        self.current_pwm = np.clip(self.current_pwm, self.pwm_min, self.pwm_max)
        
        # Create JointCmd message
        cmd = JointCmd()
        cmd.header = Header()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'robot_hand'
        
        # Joint angles
        cmd.joint_names = self.joint_names
        cmd.joint_angles = self.current_angles.tolist()
        
        # PWM values
        cmd.pwm_channels = list(range(self.num_servos))
        cmd.pwm_values = self.current_pwm.tolist()
        
        # Metadata
        cmd.gesture_id = msg.gesture_id
        cmd.confidence = msg.confidence
        
        # Publish
        self.joint_cmd_pub.publish(cmd)
        
        self.command_count += 1
        
        # Log periodically
        if self.command_count % 100 == 0:
            self.get_logger().info(
                f'Commands: {self.command_count} | '
                f'Gesture: {msg.gesture_name} | '
                f'Conf: {msg.confidence:.2f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ServoCmdNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Total commands published: {node.command_count}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()