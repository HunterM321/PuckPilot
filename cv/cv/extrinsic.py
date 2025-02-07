import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_aruco_markers(img):
    # Load the image
    # image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = np.array(gray, dtype=np.uint8)[1300:, 1200:1400]
    #gray = cv2.equalizeHist(gray)

    # Show resized image
    
    # Load the predefined ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Initialize the detector parameters
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    """
    output_img = image.copy()
    output_img = cv2.aruco.drawDetectedMarkers(output_img, corners, ids)
    #resized_image = np.array(output_img, dtype=np.uint8)[:200, 1000:1300]
    
    screen_width = 1280  # Adjust based on your screen resolution
    screen_height = 720  # Adjust based on your screen resolution

    # Get image dimensions
    height, width = output_img.shape[:2]

    # Compute scale factor while maintaining aspect ratio
    scale = min(screen_width / width, screen_height / height)

    # Resize the image
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(output_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    #print(corners)
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(100000)
    """
    
    # If markers are detected
    if ids is not None:
        top_right_corners = {}
        for i, marker_id in enumerate(ids.flatten()):
            # Extract top-right corner (third corner in OpenCV ArUco order)
            top_right = tuple(corners[i][0][0].astype(int))
            top_right_corners[marker_id] = top_right
            
            # Draw marker and top-right corner
            cv2.polylines(img, [corners[i].astype(int)], True, (0, 255, 0), 2)
            cv2.circle(img, top_right, 5, (0, 0, 255), -1)
        
        return top_right_corners, img
    
    return {}, img


def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Project 3D points to 2D pixel coordinates with distortion.
    
    Args:
        points_3d: np.array of shape (N, 3) containing 3D points in world coordinates
        rvec: rotation vector (3,) or (3, 1)
        tvec: translation vector (3,) or (3, 1)
        camera_matrix: 3x3 intrinsic camera matrix
        dist_coeffs: (k1, k2, p1, p2, k3) distortion coefficients
    
    Returns:
        points_2d: np.array of shape (N, 2) containing projected pixel coordinates
    """
    # Ensure points are float32
    points_3d = np.float32(points_3d)
    
    # Convert rotation vector to matrix
    R = cv2.Rodrigues(rvec)[0]
    
    # Transform points to camera coordinates
    points_cam = np.dot(R, points_3d.T).T + tvec
    
    # Project to normalized image coordinates
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]
    
    # Prepare for distortion
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    
    # Get distortion coefficients
    k1, k2, p1, p2, k3 = dist_coeffs
    
    # Apply radial distortion
    x_dist = x * (1 + k1*r2 + k2*r4 + k3*r6)
    y_dist = y * (1 + k1*r2 + k2*r4 + k3*r6)
    
    # Apply tangential distortion
    x_dist = x_dist + (2*p1*x*y + p2*(r2 + 2*x*x))
    y_dist = y_dist + (p1*(r2 + 2*y*y) + 2*p2*x*y)
    
    # Apply camera matrix
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]
    skew = camera_matrix[0,1]
    
    u = fx * x_dist + skew * y_dist + cx
    v = fy * y_dist + cy
    
    return np.column_stack((u, v))

def verify_projection(points_3d, points_2d, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Verify projection accuracy by comparing with given 2D points.
    
    Args:
        points_3d: np.array of shape (N, 3) containing 3D points
        points_2d: np.array of shape (N, 2) containing corresponding 2D points
        rvec, tvec, camera_matrix, dist_coeffs: camera parameters
    
    Returns:
        mean_error: average reprojection error in pixels
    """
    projected_points = project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(points_2d - projected_points, axis=1)
    mean_error = np.mean(errors)
    
    return mean_error

def random_rotation_vector():
    # Create a random 3x3 matrix
    random_matrix = np.random.randn(3, 3)
    
    # Use QR decomposition to get a random rotation matrix
    # Q will be a random rotation matrix
    Q, R = np.linalg.qr(random_matrix)
    
    # Ensure proper rotation matrix (determinant = 1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    # Convert rotation matrix to rotation vector using Rodrigues formula
    rvec = cv2.Rodrigues(Q)[0].ravel()
    
    return rvec

def visualize_points_comparison(projected_points, actual_points, image_size=(2048, 1536)):
    """
    Display projected points and actual image points on the same plot.
    
    Args:
        projected_points: np.array of shape (N, 2) containing projected pixel coordinates
        actual_points: np.array of shape (N, 2) containing actual pixel coordinates
        image_size: tuple of (width, height) for setting plot limits
    """
    plt.figure(figsize=(12, 8))
    
    # Plot projected points in red
    plt.scatter(projected_points[:, 0], projected_points[:, 1], 
               c='red', marker='o', label='Projected Points', s=100)
    
    # Plot actual points in blue
    plt.scatter(actual_points[:, 0], actual_points[:, 1], 
               c='blue', marker='x', label='Actual Points', s=100)
    
    # Draw lines between corresponding points
    for proj, act in zip(projected_points, actual_points):
        plt.plot([proj[0], act[0]], [proj[1], act[1]], 
                'g--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title('Comparison of Projected vs Actual Image Points')
    plt.legend()
    
    # Set axis limits to image dimensions
    plt.xlim(0, image_size[0])
    plt.ylim(0, image_size[1])
    
    # Invert y-axis to match image coordinates
    plt.gca().invert_yaxis()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.show()

def global_coordinate(pixel_point, rot_mat, tvec, camera_matrix, dist_coeffs, z_world):
    # Extract camera matrix parameters
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    cx = camera_matrix[0,2]
    cy = camera_matrix[1,2]
    skew = camera_matrix[0,1]
    
    # Get rotation matrix
    # R = cv2.Rodrigues(rvec)[0]
    # R_inv = np.linalg.inv(R)
    R_inv = rot_mat.T
    
    # Normalize pixel coordinates
    u, v = pixel_point
    x_distorted = (u - cx - skew * (v - cy)/fy) / fx
    y_distorted = (v - cy) / fy
    
    # Iteratively solve for undistorted coordinates
    x = x_distorted
    y = y_distorted
    
    # Newton's method to remove distortion
    for _ in range(10):
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        
        k1, k2, p1, p2, k3 = dist_coeffs
        
        x_calc = x * (1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_calc = y * (1 + k1*r2 + k2*r4 + k3*r6) + p1*(r2 + 2*y*y) + 2*p2*x*y
        
        if abs(x_calc - x_distorted) < 1e-6 and abs(y_calc - y_distorted) < 1e-6:
            print("Solved")
            break
            
        x = x - 1*(x_calc - x_distorted)
        y = y - 1*(y_calc - y_distorted)

    # Set up system of equations:
    # We know: point_world = R_inv @ (point_cam - tvec)
    # And point_cam = z_cam * [x, y, 1]
    # And point_world[2] = z_world

    # This means:
    # z_world = R_inv[2,0]*(z_cam*x - tvec[0]) + R_inv[2,1]*(z_cam*y - tvec[1]) + R_inv[2,2]*(z_cam - tvec[2])

    # Solve for z_cam:
    a = R_inv[2,0]*x + R_inv[2,1]*y + R_inv[2,2]
    b = -(R_inv[2,0]*tvec[0] + R_inv[2,1]*tvec[1] + R_inv[2,2]*tvec[2])
    z_cam = (z_world - b) / a

    # Now we can get the camera coordinates
    point_cam = z_cam * np.array([x, y, 1])
    
    # Convert to world coordinates
    point_world = R_inv @ (point_cam - tvec)
    
    return point_world

def calibrate_extrinsic(img: np.ndarray):
    top_left_corners, _ = detect_aruco_markers(img)
    # Given 3D points (in world coordinates)
    object_points = np.array([[-0.079, 0.1258, 0],
                              [-0.0775, 0.744, 0],
                              [0.405, 0.9995, 0],
                              [1.9962, 0.681, 0],
                              [1.997, 0.238, 0],
                              [0.3592, -0.0752, 0]], dtype=np.float32)
    image_points = np.array([top_left_corners[i] for i in range(6)], dtype=np.float32)
    # Given intrinsic matrix I (assumed known)
    intrinsic_matrix = np.array([[1.64122926e+03, 0.00000000e+00, 1.01740071e+03],
                                 [0.00000000e+00, 1.64159345e+03, 7.67420885e+02],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                                 dtype=np.float32)
    # Solve for rotation and translation (extrinsic parameters)
    dist_coeffs = np.array([-0.11734783, 0.11960238, 0.00017337, -0.00030401, -0.01158902], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points,
                                       image_points,
                                       intrinsic_matrix,
                                       dist_coeffs,
                                       flags=cv2.SOLVEPNP_ITERATIVE,
                                       useExtrinsicGuess=True,
                                       tvec=np.array([-0.54740303, -1.08125622, 2.45483598]),
                                       rvec=np.array([-2.4881045, -2.43093864, 0.81342852]))
    return rvec, tvec

def reprojection(img_point, rot_mat, tvec, intrinsic_matrix, dist_coeffs):
    z = -1.4e-2
    # z = 0
    object_point = global_coordinate(img_point, rot_mat, tvec, intrinsic_matrix, dist_coeffs, z)
    return object_point[0: 2]

'''
top_left_corners, image = detect_aruco_markers("air_hockey_aruco_markers3.png") #"aruco_markers_test.jpg")

if False: #show the image
    screen_width = 1280  # Adjust based on your screen resolution
    screen_height = 720  # Adjust based on your screen resolution

    # Get image dimensions
    height, width = image.shape[:2]

    # Compute scale factor while maintaining aspect ratio
    scale = min(screen_width / width, screen_height / height)

    # Resize the image
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    #print(corners)
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(100000)

# Given 3D points (in world coordinates)
object_points = np.array([
    [-0.079, 0.1258, 0], [-0.0775, 0.744, 0], [0.405, 0.9995, 0],
    [1.9962, 0.681, 0], [1.997, 0.238, 0], [0.3592, -0.0752, 0]
], dtype=np.float32)

image_points = np.array([top_left_corners[i] for i in range(6)], dtype=np.float32)

# Given intrinsic matrix I (assumed known)
intrinsic_matrix = np.array([
    [1.64122926e+03, 0.00000000e+00, 1.01740071e+03],
 [0.00000000e+00, 1.64159345e+03, 7.67420885e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

# Solve for rotation and translation (extrinsic parameters)
dist_coeffs = np.array([-0.11734783,  0.11960238,  0.00017337, -0.00030401, -0.01158902], dtype=np.float32)  # Replace with actual values
success, rvec, tvec = cv2.solvePnP(object_points, image_points, intrinsic_matrix, dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True,tvec=np.array([-0.54740303, -1.08125622,  2.45483598]),rvec=np.array([-2.4881045,  -2.43093864,  0.81342852]))

if False: #Visualize and get rvec, tvec
    print(rvec)
    print(tvec)
    projected_final = project_points(object_points, rvec, tvec, intrinsic_matrix, dist_coeffs)
    visualize_points_comparison(projected_final, image_points)

if False: #Get position of object given pixel and object z coordinate
    pixel_point = image_points[0]
    z = 0
    object_point = global_coordinate(pixel_point, rvec, tvec, intrinsic_matrix, dist_coeffs, z)
    print(object_point)
    print(object_points[0])
'''
