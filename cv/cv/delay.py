import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.time import Time
import time

class Delay(Node):
    def __init__(self):
        super().__init__('image_time_diff')
        self.subscription = self.create_subscription(
            Image,
            '/flir_camera/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg: Image):
        img_stamp = Time.from_msg(msg.header.stamp)
        current_time = self.get_clock().now()
        time_diff = current_time - img_stamp
        time_diff_sec = time_diff.nanoseconds / 1e6
        self.get_logger().info(f"Time difference: {time_diff_sec:.6f} ms")

def main(args=None):
    rclpy.init(args=args)
    node = Delay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
