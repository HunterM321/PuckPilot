import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.raw_vid_publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        fps = 118
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()

        if not self.cap.isOpened():
            self.get_logger().error('Unable to open camera')
            rclpy.shutdown()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Publish the frame as a ROS2 Image message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

            # Set the timestamp in the message header
            msg.header.stamp = self.get_clock().now().to_msg()

            self.raw_vid_publisher.publish(msg)
            self.get_logger().info('Published a video frame')
        else:
            self.get_logger().error('Failed to capture frame')

    def destroy_node(self):
        self.cap.release()  # Release the camera
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)

    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
