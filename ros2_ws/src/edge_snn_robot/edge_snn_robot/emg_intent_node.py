#!/usr/bin/env python3
"""
EMG Intent Node

Connects to FastAPI inference server and publishes Intent messages.
Polls the /infer endpoint at configurable rate (default 30Hz).

Author: EMG Robot Team
Date: 2025-12
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import requests
import time

# Import custom messages
try:
    from edge_snn_robot.msg import Intent
except ImportError:
    print("ERROR: Custom messages not found. Did you build with: colcon build?")
    raise


class EMGIntentNode(Node):
    """
    EMG Intent Publisher Node
    
    Polls FastAPI inference server and publishes gesture predictions
    as Intent messages on /emg_intent topic.
    """
    
    def __init__(self):
        super().__init__('emg_intent_node')
        
        # Declare parameters
        self.declare_parameter('server_url', 'http://localhost:8000')
        self.declare_parameter('poll_rate_hz', 30.0)
        self.declare_parameter('min_confidence', 0.7)
        self.declare_parameter('encoding_type', 'rate')
        self.declare_parameter('model_prefix', 'tcn')
        self.declare_parameter('device', 'cpu')
        
        # Get parameters
        self.server_url = self.get_parameter('server_url').value
        poll_rate = self.get_parameter('poll_rate_hz').value
        self.min_confidence = self.get_parameter('min_confidence').value
        self.encoding_type = self.get_parameter('encoding_type').value
        self.model_prefix = self.get_parameter('model_prefix').value
        self.device = self.get_parameter('device').value
        
        # Publisher
        self.intent_pub = self.create_publisher(Intent, '/emg_intent', 10)
        
        # Timer for polling
        self.timer = self.create_timer(1.0 / poll_rate, self.poll_callback)
        
        # Statistics
        self.frame_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Gesture name mapping
        self.gesture_names = {
            0: 'Rest',
            1: 'Hand Open',
            2: 'Hand Close',
            3: 'Wrist Flex',
            4: 'Wrist Extend',
            5: 'Pinch',
            6: 'Point'
        }
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('EMG Intent Node Started')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Server URL: {self.server_url}')
        self.get_logger().info(f'Poll Rate: {poll_rate} Hz')
        self.get_logger().info(f'Min Confidence: {self.min_confidence}')
        self.get_logger().info(f'Model: {self.model_prefix}')
        
        # Test connection
        self.test_connection()
    
    def test_connection(self):
        """Test connection to inference server"""
        try:
            response = requests.get(f'{self.server_url}/health', timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.get_logger().info(f'✓ Server connected: {data.get("status")}')
            else:
                self.get_logger().warn(f'Server returned status {response.status_code}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to connect: {e}')
            self.get_logger().error('Please start FastAPI server: ./start_server.sh')
    
    def poll_callback(self):
        """Poll inference server and publish Intent"""
        try:
            # Prepare request
            request_data = {
                'encoding_type': self.encoding_type,
                'model_prefix': self.model_prefix,
                'device': self.device
            }
            
            # Call inference endpoint
            response = requests.post(
                f'{self.server_url}/infer',
                json=request_data,
                timeout=2.0
            )
            
            if response.status_code != 200:
                self.get_logger().warn(f'Server error: {response.status_code}')
                self.error_count += 1
                return
            
            # Parse response
            data = response.json()
            gesture_id = int(data['pred_idx'])
            confidence = float(data['conf'])
            latency_ms = float(data['latency_ms'])
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                return
            
            # Create Intent message
            msg = Intent()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'emg_sensor'
            
            msg.gesture_id = gesture_id
            msg.gesture_name = self.gesture_names.get(gesture_id, 'Unknown')
            msg.confidence = confidence
            msg.latency_ms = latency_ms
            msg.model_type = self.model_prefix
            msg.device = self.device
            
            # Publish
            self.intent_pub.publish(msg)
            
            self.frame_count += 1
            
            # Log periodically
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                rate = self.frame_count / elapsed
                self.get_logger().info(
                    f'Published {self.frame_count} intents '
                    f'({rate:.1f} Hz, {self.error_count} errors)'
                )
            
        except requests.exceptions.Timeout:
            self.error_count += 1
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            self.error_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = EMGIntentNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Statistics
        if node.frame_count > 0:
            elapsed = time.time() - node.start_time
            rate = node.frame_count / elapsed
            node.get_logger().info('=' * 50)
            node.get_logger().info('Final Statistics:')
            node.get_logger().info(f'  Total Frames: {node.frame_count}')
            node.get_logger().info(f'  Average Rate: {rate:.1f} Hz')
            node.get_logger().info(f'  Total Errors: {node.error_count}')
            node.get_logger().info(f'  Total Time: {elapsed:.1f}s')
            node.get_logger().info('=' * 50)
        
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()