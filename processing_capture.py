import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import json
from datetime import datetime
from ArucoMarkers import ArucoMarkerDetector
from data_manager import DataManager
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D


def load_aruco_parameters(config_filename):
    """Load saved ArUco parameters from JSON file"""
    try:
        with open(config_filename, 'r') as f:
            config = json.load(f)

        # Create detector parameters from saved config
        params = cv2.aruco.DetectorParameters()
        param_data = config['parameters']

        params.adaptiveThreshWinSizeMin = param_data['adaptiveThreshWinSizeMin']
        params.adaptiveThreshWinSizeMax = param_data['adaptiveThreshWinSizeMax']
        params.adaptiveThreshWinSizeStep = param_data['adaptiveThreshWinSizeStep']
        params.adaptiveThreshConstant = param_data['adaptiveThreshConstant']
        params.minMarkerPerimeterRate = param_data['minMarkerPerimeterRate']
        params.maxMarkerPerimeterRate = param_data['maxMarkerPerimeterRate']
        params.polygonalApproxAccuracyRate = param_data['polygonalApproxAccuracyRate']
        params.cornerRefinementMethod = param_data['cornerRefinementMethod']
        params.cornerRefinementWinSize = param_data['cornerRefinementWinSize']
        params.cornerRefinementMaxIterations = param_data['cornerRefinementMaxIterations']
        params.errorCorrectionRate = param_data['errorCorrectionRate']

        print(f"✅ Loaded ArUco parameters from: {config_filename}")
        print(f"📋 Config: {config['best_config_name']}")
        print(f"🎯 Expected performance: {config['performance']['markers_detected']} markers per capture")

        return params, config

    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_filename}")
        print("💡 Run parameter optimization first to generate the config file")
        return None, None
    except Exception as e:
        print(f"❌ Error loading parameters: {e}")
        return None, None

###############################################
########## The main Processing Class ##########
###############################################

class Processor:
    """Manages all processing"""

    def __init__(self, capture, custom_detector=None):
        self.data = capture
        self.aruco_detector = custom_detector or ArucoMarkerDetector()
        self.grayscale = self.get_grayscale()


    def get_grayscale(self):
        """Convert RGB image to grayscale"""
        realsense_data = self.data.get("realsense_data", {})
        aligned_rgb = realsense_data.get("aligned_rgb_image")
        if aligned_rgb is not None:
            return cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)
        return None

    def get_rgb(self):
        """Get RGB image directly"""
        realsense_data = self.data.get("realsense_data", {})
        aligned_rgb = realsense_data.get("aligned_rgb_image")
        return aligned_rgb

    def get_histogram(self):
        """Get the histogram data"""
        return self.data.get("histogram")

    def get_position(self):
        """Get the position data"""
        return self.data.get("pos")

    def detect_aruco(self):
        """Detect ArUco markers in the image"""
        if self.grayscale is None:
            print("Error: Could not get grayscale image")
            return None, None, None

        corners, ids, rejected = self.aruco_detector.detect(self.grayscale)
        print(f'corners: {len(corners) if corners is not None else 0} detected')
        print(f'ids: {ids}')
        print(f'rejected: {len(rejected) if rejected is not None else 0}')

        return corners, ids, rejected

    def build_board(self):
        return self.aruco_detector.build_board()

    def draw_aruco(self):
        """Draw ArUco markers in the image with outline and center points"""

        if self.grayscale is None:
            print("Error: Could not get grayscale image for drawing")
            return None

        corners, ids, _ = self.detect_aruco()

        # Check if any markers were found (corners is a list, not None when empty)
        if corners is not None and len(corners) > 0:
            print(f"Drawing {len(corners)} ArUco markers with outlines and centers")

            # Convert grayscale to BGR for colored drawing
            image_bgr = cv2.cvtColor(self.grayscale, cv2.COLOR_GRAY2BGR)

            # Draw detected markers (outline)
            image_bgr = cv2.aruco.drawDetectedMarkers(image_bgr, corners, ids)

            # Draw center points and additional info
            for i, corner in enumerate(corners):
                # Calculate center point
                corner_points = corner[0]  # corner is shape (1, 4, 2)
                center_x = int(np.mean(corner_points[:, 0]))
                center_y = int(np.mean(corner_points[:, 1]))
                center = (center_x, center_y)
                cv2.circle(image_bgr, center, 3, (255, 0, 0), -1)

                # Draw ID text near center if IDs are available
                if ids is not None and i < len(ids):
                    marker_id = ids[i][0]
                    cv2.putText(image_bgr, f"ID: {marker_id}",
                               (center_x + 15, center_y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Draw corner numbers for debugging
                for j, point in enumerate(corner_points):
                    pt = (int(point[0]), int(point[1]))
                    cv2.circle(image_bgr, pt, 3, (255, 0, 0), -1)  # Blue corner points
                    cv2.putText(image_bgr, str(j), (pt[0] + 5, pt[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            return image_bgr
        else:
            print(f'No ArUco markers detected - returning original grayscale image')
            return self.grayscale

    def get_ground_plane_pose(self, corners, ids):
        return self.aruco_detector.board_pose(corners, ids)

    def draw_board(self, board_pose):
        """
        Draw the board on the rgb image
        """
        return cv2.drawFrameAxes(self.get_rgb(), self.aruco_detector.camera_matrix, self.aruco_detector.dist_coeffs, board_pose[1], board_pose[2], 0.1)


    def read_intrinsics(self, path):
        """
        Read the intrinsics from a file

        Args:
            path (str): The path to the intrinsics file

        Returns:
            dict: The intrinsics data
            None if error reading the file
        """
        try:
            with open(path, 'r') as f:
                intrinsics = json.load(f)
            print(f"✅ Intrinsics file found: {path}")
            return intrinsics
        except FileNotFoundError:
            print(f"❌ Intrinsics file not found: {path}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Error parsing intrinsics file: {path}")
            return None
        except:
            print(f'Issue reading intrinsics file: {path}')
            return None


    @staticmethod
    def show_image(image, window_name="ArUco Detection"):
        """Show the image"""
        if image is None:
            print("Error: No image to display")
            return

        cv2.imshow(window_name, image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()



######################################
########## HELPER FUNCTIONS ##########
######################################
@staticmethod
def show_image(image, window_name: str):
        """Show the image"""
        if image is None:
            print("Error: No image to display")
            return

        cv2.imshow(window_name, image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def plot_scene_3d(board, rvec_rel, tvec_rel, axis_length=0.5):
    """
    Visualize a planar ArUco board and a sensor frame in 3D (in board coordinates),
    plus the vector from the board origin to the sensor origin.

    Args:
        board      : cv2.aruco.Board or GridBoard with getObjPoints() method
        rvec_rel   : (3,1) Rodrigues vector of the sensor in board frame
        tvec_rel   : (3,1) translation vector of the sensor in board frame
        axis_length: length of the plotted axes (default 0.5 units)
    """
    # --- 1) Board outline at Z=0 ---
    all_pts = np.vstack(board.getObjPoints()).reshape(-1, 3)
    x_min, x_max = all_pts[:,0].min(), all_pts[:,0].max()
    y_min, y_max = all_pts[:,1].min(), all_pts[:,1].max()
    board_outline = np.array([
        [x_min, y_min, 0],
        [x_min, y_max, 0],
        [x_max, y_max, 0],
        [x_max, y_min, 0],
        [x_min, y_min, 0]
    ])

    # --- 2) Sensor rotation & origin ---
    R_s, _    = cv2.Rodrigues(rvec_rel)
    origin    = tvec_rel.flatten()  # [x_s, y_s, z_s]

    # sensor axes endpoints
    x_end = origin + R_s[:,0] * axis_length
    y_end = origin + R_s[:,1] * axis_length
    z_end = origin + R_s[:,2] * axis_length

    # --- 3) Plot setup ---
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')

    # Board perimeter
    ax.plot(board_outline[:,0], board_outline[:,1], board_outline[:,2],
            color='k', linewidth=2, label='Board perimeter')

    # Board axes at origin
    ax.quiver(0,0,0, axis_length,0,0, color='r', linewidth=1)
    ax.quiver(0,0,0, 0,axis_length,0, color='g', linewidth=1)
    ax.quiver(0,0,0, 0,0,axis_length, color='b', linewidth=1)

    # --- 4) Origin→Sensor vector ---
    ax.quiver(0,0,0,
              origin[0], origin[1], origin[2],
              color='m', linewidth=2, arrow_length_ratio=0.1,
              label='Origin→Sensor')

    # --- 5) Plot sensor point ---
    ax.scatter(*origin, color='m', s=50, label='Sensor origin')


    # --- 7) Formatting & view ---
    ax.set_xlabel('X (board frame)')
    ax.set_ylabel('Y (board frame)')
    ax.set_zlabel('Z (board frame)')
    ax.set_title('Board & Sensor Pose in Board Coordinates')
    ax.legend(loc='upper left')

    # make axes equal-ish
    max_range = np.array([x_max - x_min, y_max - y_min, axis_length]).max()
    mid_x = 0.5*(x_max + x_min)
    mid_y = 0.5*(y_max + y_min)
    mid_z = axis_length/2
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(0, max_range)

    plt.tight_layout()

    # Add key event handler to close on any key press
    def on_key(event):
        plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("3D scene plotted. Press any key to close the plot and continue...")
    plt.show(block=True)

def loop_through_captures(data, custom_detector=None):

    """Loop through all captures and return the maximum number of markers detected

    Args:
        data: DataManager instance with capture data
        custom_detector: Optional ArucoMarkerDetector with optimized parameters
    """
    # Track maximum and store all marker counts
    max_markers = 0
    all_marker_counts = []
    all_board_poses = []
    all_sensor_poses = []
    batch_processor = Processor(data.data_entries[0], custom_detector) # only used to draw average board and sensor

    # Loop through all captures
    for i in range(len(data.data_entries)):
        print(f"\n=== Processing Capture {i} ===")

        # Get the capture data directly
        selected_capture = data.data_entries[i]
        data.capture_idx = i  # Set the index for display purposes

        if selected_capture:
            # data.display_capture_data(selected_capture)
            # data.plot_capture_data(selected_capture)

            # Process ArUco detection with custom detector if provided
            # processed stores the data for the current capture
            processed = Processor(selected_capture, custom_detector)

            # Get marker count for this capture
            #corners, ids, rejected = processed.detect_aruco()
            detected_aruco, board_info, sensor_info = process_capture(processed)


            print(f'{detected_aruco['corners']=}')
            print(f'{type(detected_aruco['corners'])=}')
            print(f'{type(detected_aruco['corners'][0])=}')

            num_markers = len(detected_aruco['corners']) if detected_aruco['corners'] is not None else 0
            all_marker_counts.append(num_markers)

            if board_info['board_pose'][0] is not None:
                all_board_poses.append(board_info['board_pose'])
            if sensor_info['sensor_pose'][0] is not False:
                all_sensor_poses.append(sensor_info['sensor_pose'])


            # Track maximum
            if num_markers > max_markers:
                max_markers = num_markers

            aruco_image = processed.draw_aruco()
            # Show the image with capture index in window name
            # print(f'{board_pose=}')

            #processed.show_image(board_info['board_image'], f"ArUco Detection - Capture {i}")
            processed.show_image(aruco_image, f"ArUco Detection - Capture {i}")

            print(f"Capture {i} processed ({num_markers} markers). Press any key to continue to next capture...")

    print(f"\nAll captures processed!")
    print(f"Marker counts per capture: {all_marker_counts}")
    print(f"Maximum markers detected: {max_markers}")
    print(f'{len(all_board_poses)=}')
    print(f'{len(all_sensor_poses)=}')
    print(f"Board poses: {all_board_poses}")
    print(f"Sensor poses: {all_sensor_poses}")

    avg_board_pose, avg_sensor_pose = compute_average_poses(all_board_poses, all_sensor_poses)
    board_image = draw_board(batch_processor, avg_board_pose, batch_processor.get_rgb())
    combined_image = draw_board(batch_processor, avg_sensor_pose, board_image)
    show_image(combined_image, "Average Board and Sensor")

    rvec_rel, t_rel = localize_sensor(avg_board_pose, avg_sensor_pose)
    plot_scene_3d(batch_processor.build_board(), rvec_rel, t_rel)
    write_scene_to_json(batch_processor.build_board(), rvec_rel, t_rel, batch_processor.aruco_detector.size_map, "test_scene.json")

    return max_markers


def compute_average_poses(all_board_poses, all_sensor_poses):
    """
    Compute the average pose of the board and sensor
    """
    return average_poses(all_board_poses), average_poses(all_sensor_poses)


def process_capture(processor: Processor):
    """
    Process a single capture
    Returns:
        detected_aruco: info about detected aruco markers
        board_info: stores the board, its pose, and image
        sensor_info: stores the sensor, its pose, and image

    """


    corners, ids, rejected = processor.detect_aruco() # stores all of the detected corners and ids
    detected_aruco = {
        'corners': corners,
        'ids': ids,
        'rejected': rejected
    }

    # board must first be built before pose can be estimated
    board = processor.build_board()
    board_pose = processor.get_ground_plane_pose(corners, ids) # builds board with filtered ids
    board_image = processor.draw_board(board_pose)

    board_info = {
        'board': board,
        'board_pose': board_pose,
        'board_image': board_image
    }

    sensor_pose = processor.aruco_detector.sensor_pose(corners, ids)
    #object_image = processor.draw_board(obj_pose)
    sensor_info = {
        'sensor_pose': sensor_pose,
        #'object_image': object_image
    }

    return detected_aruco, board_info, sensor_info

def draw_board(processor, pose, image):
    """
    Draw the board on the image
    pose is (avg_rvec, avg_tvec) from average_poses()
    """
    return cv2.drawFrameAxes(image, processor.aruco_detector.camera_matrix, processor.aruco_detector.dist_coeffs, pose[0], pose[1], 0.1)

def average_poses(rt_list):
    """
    Average a list of (retval, rvec, tvec) tuples.

    Args:
        rt_list: List of tuples [(retval1, rvec1, tvec1), (retval2, rvec2, tvec2), ...]
                 - retval: number of markers used for pose estimation
                 - rvec: (3,1) Rodrigues rotation vector
                 - tvec: (3,1) translation vector
    Returns:
        avg_rvec: (3,1) average Rodrigues rotation vector
        avg_tvec: (3,1) average translation vector
    """
    # 1) Average translations directly
    t_stack = np.hstack([tvec.reshape(3,1) for _, _, tvec in rt_list])  # shape (3, N)
    t_avg = np.mean(t_stack, axis=1).reshape(3,1)

    # 2) Convert rvecs to Rotation objects and compute the mean rotation
    r_stack = np.vstack([rvec.flatten() for _, rvec, _ in rt_list])      # shape (N, 3)
    rotations = R.from_rotvec(r_stack)
    rot_mean = rotations.mean()                                        # Principal‐axis quaternion average
    rvec_avg = rot_mean.as_rotvec().reshape(3,1)

    return rvec_avg, t_avg

def localize_sensor(board_pose, sensor_pose):
    """
    Localize the sensor in the board coordinate system

    """
    # convert to rotation matrices
    Rb, _ = cv2.Rodrigues(board_pose[0])
    Rs, _ = cv2.Rodrigues(sensor_pose[0])

    # compute sensor frame in board coordinate system
    R_rel = Rb.T.dot(Rs)
    t_rel = Rb.T.dot((sensor_pose[1] - board_pose[1]).reshape(3,1))
    rvec_rel, _ = cv2.Rodrigues(R_rel)

    return rvec_rel, t_rel

def write_scene_to_json(board, rvec_rel, tvec_rel, size_map, filename):
    """
    Writes a JSON containing:
      - ground_plane: the four corner vertices of your board in board‐frame
      - sensor: position & orientation (as a quaternion) relative to the board origin
      - marker_sizes: sizes of all ArUco markers

    Args:
      board      : your cv2.aruco.Board or GridBoard
      rvec_rel   : (3×1) Rodrigues of sensor in board frame
      tvec_rel   : (3×1) translation of sensor in board frame
      size_map   : dict mapping marker IDs to their physical sizes
      filename   : path to write .json
    """
    # 1) extract board corners (Z=0)
    all_pts = np.vstack(board.getObjPoints()).reshape(-1, 3)
    x_min, x_max = float(all_pts[:,0].min()), float(all_pts[:,0].max())
    y_min, y_max = float(all_pts[:,1].min()), float(all_pts[:,1].max())
    board_corners = [
        [x_min, y_min, 0.0],
        [x_min, y_max, 0.0],
        [x_max, y_max, 0.0],
        [x_max, y_min, 0.0],
    ]

    # 2) sensor pose → quaternion
    # scipy’s as_quat() returns [x, y, z, w]
    rot = R.from_rotvec(rvec_rel.flatten())
    q_xyz_w = rot.as_quat()
    quat = [float(q_xyz_w[3]), float(q_xyz_w[0]), float(q_xyz_w[1]), float(q_xyz_w[2])]

    sensor_pos = tvec_rel.flatten().tolist()

    # Convert size_map to JSON-serializable format
    marker_sizes = {str(marker_id): float(size) for marker_id, size in size_map.items()}

    scene = {
        "ground_plane": {
            "corners": board_corners,
            "normal": [0.0, 0.0, 1.0]        # plane faces +Z in board frame
        },
        "sensor": {
            "position": sensor_pos,         # [x, y, z]
            "orientation_quat": quat        # [w, x, y, z]
        },
        "marker_sizes": marker_sizes        # mapping of marker ID to physical size
    }

    with open(filename, "w") as f:
        json.dump(scene, f, indent=2)

    print(f"✅ Wrote scene JSON → {filename}")

# --- Main Execution ---
if __name__ == "__main__":

    # --- Configuration ---
    # IMPORTANT: Replace this with the actual path to your .pkl file
    # Example: PKL_PATH = Path("out_data_captured/2025-06-30/my_object_20250630_0.pkl")
    PKL_PATH_INPUT = "test_20250709_1.pkl"
    #PKL_PATH_INPUT = "test_20250703.pkl"
    PKL_PATH = Path(PKL_PATH_INPUT)

    data = DataManager(PKL_PATH)
    data.display_metadata()

    # Load optimized ArUco detector parameters
    optimized_detector = ArucoMarkerDetector()
    optimized_detector.load_optimized_parameters()

    # Process all captures
    loop_through_captures(data, custom_detector=optimized_detector)
