import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class Tracker(Node):
    def __init__(self):
        super().__init__('puck_tracker')
        self.raw_vid_subscription = self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)
        self.puck_track_publisher = self.create_publisher(Image, '/camera/image_puck_track', 10)
        self.puck_filtered_publisher = self.create_publisher(Image, 'camera/image_puck_filtered', 10)
        self.bridge = CvBridge()
        self.first_frame = True
        self.prev_time = None
        self.prev_location = None
    
    def listener_callback(self, msg):
        # Retrieve the timestamp of each frame
        timestamp_sec = msg.header.stamp.sec
        timestamp_nanosec = msg.header.stamp.nanosec
        timestamp = timestamp_sec + timestamp_nanosec * 1e-9

        # Convert the ROS Image message to an OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Apply Gaussian Blur to simulate actual footage
        # frame = cv2.GaussianBlur(frame, (31, 31), 0)
        
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the range for "red" color in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2  # Combine the two masks

        # Publish the mask
        puck_filtered_msg = self.bridge.cv2_to_imgmsg(mask, encoding='passthrough')
        self.puck_filtered_publisher.publish(puck_filtered_msg)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M['m00'] != 0:
                # Calculate the center of the contour
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                self.get_logger().info(f'Position of puck: x={x}, y={y}')

                if self.first_frame:
                    self.first_frame = False
                else:
                    prev_x = self.prev_location[0]
                    prev_y = self.prev_location[1]
                    dx = x - prev_x
                    dy = y - prev_y
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    dt = timestamp - self.prev_time
                    velocity = (dx / dt, dy / dt)
                    speed = distance / dt
                    self.get_logger().info(f'Velocity: x={velocity[0]}, y={velocity[1]}')
                    self.get_logger().info(f'Speed: {speed}')
                    self.get_logger().info(f'Elapsed time: {dt}')
            else:
                self.get_logger().info('No valid red contour found')
        
        # Update the previous timestamp and location
        self.prev_time = timestamp
        self.prev_location = (x, y)
        
        # Publish the frame with the detected circle center
        puck_track_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.puck_track_publisher.publish(puck_track_msg)

        self.get_logger().info('=' * 40)

def main(args=None):
    rclpy.init(args=args)
    puck_tracker = Tracker()
    rclpy.spin(puck_tracker)

    puck_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
