#!/usr/bin/env python3
import rclpy
from rclpy.node import Node


class Minimal(Node):
    def __init__(self):
        super().__init__("edge_snn_minimal")
        self.timer = self.create_timer(1.0, self.tick)

    def tick(self):
        self.get_logger().info("Edge SNN Robot node alive")


def main():
    rclpy.init()
    n = Minimal()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
