import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import csv

import os
import sys
sys.path.append(os.path.abspath('/home/fizzer/ai_airhockey/src/PuckPilot/cv/cv'))
import extrinsic

class Tracker(Node):
    def __init__(self):
        super().__init__('puck_tracker')
        self.raw_vid_subscription = self.create_subscription(Image, '/flir_camera/image_raw', self.listener_callback, 10)
        self.track_publisher = self.create_publisher(Image, '/camera/image_track', 10)
        self.puck_mask_publisher = self.create_publisher(Image, '/camera/image_puck_mask', 10)
        self.player_mallet_mask_publisher = self.create_publisher(Image, '/camera/image_player_mallet_mask', 10)
        self.agent_mallet_mask_publisher = self.create_publisher(Image, '/camera/image_agent_mallet_mask', 10)
        self.bridge = CvBridge()
        self.first_frame = True
        self.prev_time = None
        self.prev_puck_location = None
        self.prev_player_mallet_location = None
        self.prev_agent_mallet_location = None

        self.extrinsic_count = 0

        # Camera parameters
        self.intrinsic_matrix = np.array([[1.64122926e+03, 0.0, 1.01740071e+03],
                                          [0.0, 1.64159345e+03, 7.67420885e+02],
                                          [0.0, 0.0, 1.0]])
        self.distortion_coeffs = np.array([-0.11734783, 0.11960238, 0.00017337, -0.00030401, -0.01158902])
        self.rotation_matrix = cv2.Rodrigues(np.array([-2.4881045, -2.43093864, 0.81342852]))[0]
        self.translation_vector = np.array([-0.54740303, -1.08125622,  2.45483598])
        # self.extrinsic_matrix = np.hstack((self.rotation_matrix, self.translation_vector))
    
    def create_mask(self, frame, target):
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if target == 'puck':
            # Define the range for dark red color in HSV
            lower1 = np.array([0, 100, 100])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([160, 100, 100])
            upper2 = np.array([179, 255, 255])
            # Create two masks and combine them
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif target == 'player mallet':
            # Define the range for black color in HSV
            lower = np.array([0, 0, 0])
            upper = np.array([179, 255, 50])
        else:
            # Define the range for green color in HSV
            lower = np.array([35, 100, 100])
            upper = np.array([85, 255, 255])

        # Create mask
        if target != 'puck':
            mask = cv2.inRange(hsv, lower, upper)
        return mask
    
    def log_csv(self, position):
        with open('/home/fizzer/ai_airhockey/src/PuckPilot/cv/cv/position.csv', mode='a') as file:
            writer = csv.writer(file)
            if position:
                writer.writerow([position[0], position[1]])
    
    def track(self, frame, timestamp, dt, target):
        if target not in ['puck', 'player mallet', 'agent mallet']:
            self.get_logger().error('Must select to track the puck or the mallets')
            
            position = None
            velocity = None

        # Creat mask for target
        mask = self.create_mask(frame, target)

        # Publish the mask
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='passthrough')
        if target == 'puck':
            # self.puck_mask_publisher.publish(mask_msg)
            pass
        elif target == 'player mallet':
            self.player_mallet_mask_publisher.publish(mask_msg)
        else:
            self.agent_mallet_mask_publisher.publish(mask_msg)

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
                img_pos = np.array([x, y])
                world_pos = extrinsic.reprojection(img_pos,
                                                   self.rotation_matrix,
                                                   self.translation_vector,
                                                   self.intrinsic_matrix,
                                                   self.distortion_coeffs)
                x = world_pos[0]
                y = world_pos[1]
                position = (x, y)

                # Draw the center of the target
                cv2.circle(frame, img_pos, 5, (0, 255, 255), -1)

                if self.first_frame:
                    velocity = None
                else:
                    if target == 'puck':
                        prev_x = self.prev_puck_location[0]
                        prev_y = self.prev_puck_location[1]
                    elif target == 'player mallet':
                        prev_x = self.prev_player_mallet_location[0]
                        prev_y = self.prev_player_mallet_location[1]
                    else:
                        prev_x = self.prev_agent_mallet_location[0]
                        prev_y = self.prev_agent_mallet_location[1]
                    dx = x - prev_x
                    dy = y - prev_y
                    velocity = (dx / dt, dy / dt)
                
                # Update the previous location of the target
                if target == 'puck':
                    self.prev_puck_location = (x, y)
                elif target == 'player mallet':
                    self.prev_player_mallet_location = (x, y)
                else:
                    self.prev_agent_mallet_location = (x, y)
            else:
                position = None
                velocity = None
                self.get_logger().info('No valid contour found')
        else:
            position = None
            velocity = None
            self.get_logger().info('No valid contour found')
        
        # Update the previous timestamp
        self.prev_time = timestamp

        self.log_csv(position)

        return position, velocity
    
    def log_data(self, position, velocity, target):
        self.get_logger().info('{}:'.format(target))
        if position:
            self.get_logger().info(f'Position: x={position[0]}, y={position[1]}')
        if velocity:
            speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
            self.get_logger().info(f'Velocity: x={velocity[0]}, y={velocity[1]}')
            self.get_logger().info(f'Speed: {speed}')
        self.get_logger().info('\n')
    
    def listener_callback(self, msg):
        # Retrieve the timestamp of each frame
        timestamp_sec = msg.header.stamp.sec
        timestamp_nanosec = msg.header.stamp.nanosec
        timestamp = timestamp_sec + timestamp_nanosec * 1e-9

        if not self.first_frame:
            dt = timestamp - self.prev_time
        else:
            dt = None

        # Convert the ROS Image message to an OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.extrinsic_count % 100 == 0:
            rvec, tvec = extrinsic.calibrate_extrinsic(frame)
            self.rotation_matrix = cv2.Rodrigues(rvec)[0]
            self.translation_vector = tvec
            self.get_logger().info('Extrinsic calibration successful')
        
        self.extrinsic_count += 1

        # Apply Gaussian Blur to simulate actual footage
        # frame = cv2.GaussianBlur(frame, (31, 31), 0)
        
        position_puck, velocity_puck = self.track(frame, timestamp, dt, target='puck')
        # position_player_mallet, velocity_player_mallet = self.track(frame, timestamp, dt, target='player mallet')
        # position_agent_mallet, velocity_agent_mallet = self.track(frame, timestamp, dt, target='agent mallet')

        self.log_data(position_puck, velocity_puck, target='puck')
        # self.log_data(position_player_mallet, velocity_player_mallet, target='player mallet')
        # self.log_data(position_agent_mallet, velocity_agent_mallet, target='agent mallet')
        
        if dt:
            self.get_logger().info(f'Elapsed time: {dt}')
        
        # Update the flag after the first frame
        if self.first_frame:
            self.first_frame = False
        
        # Publish the processed frame with the center of each target
        track_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        # self.track_publisher.publish(track_msg)

        self.get_logger().info('=' * 40)

def main(args=None):
    rclpy.init(args=args)
    puck_tracker = Tracker()
    rclpy.spin(puck_tracker)

    puck_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
