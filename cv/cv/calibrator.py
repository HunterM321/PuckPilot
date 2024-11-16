import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class CameraCalibrator(Node):
    def __init__(self):
        super().__init__('camera_calibrator')
        self.raw_vid_subscription = self.create_subscription(Image, '/camera/image_raw', self.listener_callback, 10)
        self.chess_publisher = self.create_publisher(Image, '/camera/image_chess', 10)
        self.bridge = CvBridge()

        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Prepare object points (3D points in real-world space) for a 5x5 chessboard
        self.objp = np.zeros((5 * 5, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:5, 0:5].T.reshape(-1, 2)

        # Arrays to store object points and image points from valid frames
        self.objpoints = []  # 3D points in real-world space
        self.imgpoints = []  # 2D points in image plane
        self.image_shape = None

    def listener_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Set image shape
        if self.image_shape is None:
            self.image_shape = gray.shape[::-1]

        # Find the chessboard corners for a 5x5 pattern
        ret_chess, corners = cv2.findChessboardCorners(gray, (5, 5), None)

        if ret_chess:
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)

            # Draw the corners on the frame
            cv2.drawChessboardCorners(frame, (5, 5), corners2, ret_chess)

            self.get_logger().info("Captured frame for calibration.")

            # Perform calibration if enough frames have been captured
            if len(self.objpoints) >= 5:
                self.calibrate_camera()
        
        # Publish the frame as a ROS2 Image message
        chess_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.chess_publisher.publish(chess_msg)

    def calibrate_camera(self):
        # Calibrate the camera using the collected points and image shape
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            np.array(self.objpoints, dtype=np.float32),
            np.array(self.imgpoints, dtype=np.float32),
            self.image_shape,
            None, None
        )

        # Extract and display key intrinsic parameters
        if ret:
            fx, fy = mtx[0, 0], mtx[1, 1]
            cx, cy = mtx[0, 2], mtx[1, 2]
            k1, k2, p1, p2, k3 = dist[0]
            self.get_logger().info(f'Calibration successful:\nMatrix: {mtx}\nDistortion: {dist}')
            self.get_logger().info(f"\nfx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
            self.get_logger().info(f"k1: {k1}, k2: {k2}, p1: {p1}, p2: {p2}, k3: {k3}")
        else:
            self.get_logger().error('Calibration failed.')

def main(args=None):
    rclpy.init(args=args)
    camera_calibrator = CameraCalibrator()
    rclpy.spin(camera_calibrator)

    camera_calibrator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
