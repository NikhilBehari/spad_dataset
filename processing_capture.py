import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import json
from datetime import datetime
from ArucoMarkers import ArucoMarkerDetector



class DataManager:
    '''
    This class is used to load and visualize the data from .pkl files.
    '''

    def __init__(self, pkl_path: str):
        '''
        Initializes the DataManager class.

        Args:
            pkl_path (str): The path to the .pkl file.
        '''
        self.pkl_path = pkl_path
        self.metadata_entry, self.data_entries = self.load_objects_and_metadata()
        self.capture_idx = 0

    def load_objects(self):
        """Load the objects from the .pkl file."""
        with open(self.pkl_path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def load_objects_and_metadata(self):
        """
        Load the objects and metadata from the .pkl file.
        Returns a tuple of (metadata_entry, data_entries).
        """
        try:
            # Convert generator to list to get all objects
            all_objects = list(self.load_objects())

            if not all_objects:
                print(f"ERROR: PKL file at '{self.pkl_path}' is empty or corrupted.")
                return None, []

            # First object is metadata, rest are data entries
            metadata_entry = all_objects[0]
            data_entries = all_objects[1:]

            return metadata_entry, data_entries

        except FileNotFoundError:
            print(f"ERROR: PKL file not found at '{self.pkl_path}'. Please double-check the path.")
            return None, []
        except Exception as e:
            print(f"ERROR: Failed to load PKL file at '{self.pkl_path}': {e}")
            return None, []

    def get_metadata(self):
        """Returns the metadata of the .pkl file."""
        if self.metadata_entry:
            return self.metadata_entry.get("metadata", {})
        return {}

    def display_metadata(self):
        """Displays the metadata of the .pkl file."""
        print("\n--- Metadata ---")
        metadata = self.get_metadata()
        for key, value in metadata.items():
            if key == "realsense_intrinsics":
                print(f"  {key}:")
                for stream, intrinsics in value.items():
                    print(f"    {stream}:")
                    for k, v in intrinsics.items():
                        print(f"      {k}: {v}")
            else:
                print(f"  {key}: {value}")

    def get_capture_data(self):
        """Returns the capture data of the .pkl file."""
        num_captures = len(self.data_entries)
        if num_captures == 0:
            print("No actual capture data entries found.")
            return None

        while True:
            try:
                idx_str = input(f"Enter capture index (0 to {num_captures - 1}): ").strip()
                self.capture_idx = int(idx_str)
                if 0 <= self.capture_idx < num_captures:
                    break
                else:
                    print(f"Invalid index. Please enter a number between 0 and {num_captures - 1}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        selected_capture = self.data_entries[self.capture_idx]
        return selected_capture

    def display_capture_data(self, selected_capture):
        """Displays selected capture data of the .pkl file."""
        print(f"\n--- Displaying Capture Index {self.capture_idx} ---")
        print(f"  Iteration: {selected_capture.get('iter')}")
        print(f"  Position: {selected_capture.get('pos')}")


    def plot_capture_data(self, selected_capture):
        """Plot Aligned RGB, Aligned Depth, and Histogram in a single row"""
        print(f"\n--- Plotting Capture Index {self.capture_idx} ---")
        realsense_data = selected_capture.get("realsense_data", {})
        aligned_rgb = realsense_data.get("aligned_rgb_image")
        print(f'type of aligned_rgb: {type(aligned_rgb)}')
        aligned_depth = realsense_data.get("aligned_depth_image")
        histogram = selected_capture.get("histogram")

        plt.figure(figsize=(20, 6)) # Adjusted figure size for 3 plots in a row

        # Plot Aligned RGB
        if aligned_rgb is not None:
            plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st plot
            plt.imshow(aligned_rgb[:, :, ::-1]) # Convert BGR to RGB for matplotlib
            plt.title(f"Aligned RGB (Idx {self.capture_idx})")
            plt.axis('off')
        else:
            print("Warning: Aligned RGB image not found.")
            plt.subplot(1, 3, 1)
            plt.text(0.5, 0.5, "Aligned RGB Not Found", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')

        # Plot Aligned Depth
        if aligned_depth is not None:
            plt.subplot(1, 3, 2) # 1 row, 3 columns, 2nd plot
            # Scale depth for visualization (adjust vmax for your depth sensor's range)
            plt.imshow(aligned_depth, cmap='jet', vmin=0, vmax=np.max(aligned_depth) * 0.5)
            plt.title(f"Aligned Depth (Idx {self.capture_idx})")
            plt.colorbar(label='Depth (units)')
            plt.axis('off')
        else:
            print("Warning: Aligned Depth image not found.")
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, "Aligned Depth Not Found", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')

        # Plot Histogram
        if histogram is not None:
            plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd plot
            if histogram.ndim > 1: # If multiple channels, average them for a single plot
                avg_hist = histogram.mean(axis=0)
                plt.bar(range(len(avg_hist)), avg_hist)
                plt.title(f"Averaged Histogram (Idx {self.capture_idx})")
            else:
                plt.bar(range(len(histogram)), histogram)
                plt.title(f"Histogram (Idx {self.capture_idx})")
            plt.xlabel("Bin")
            plt.ylabel("Count")
            plt.grid(axis='y', alpha=0.75)
        else:
            print("Warning: Histogram data not found.")
            plt.subplot(1, 3, 3)
            plt.text(0.5, 0.5, "Histogram Not Found", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')

        plt.tight_layout()
        plt.show()


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


class Processor:
    """Manages all processing"""

    def __init__(self, capture, custom_detector=None):
        self.data = capture
        self.aruco_detector = custom_detector or ArucoMarkerDetector()
        self._cached_detection = None  # Cache to avoid redundant detection
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

        # Use cached result if available
        if self._cached_detection is None:
            corners, ids, rejected = self.aruco_detector.detect(self.grayscale)
            self._cached_detection = (corners, ids, rejected)
            print(f'corners: {len(corners) if corners is not None else 0} detected')
            print(f'ids: {ids}')
            print(f'rejected: {len(rejected) if rejected is not None else 0}')

        return self._cached_detection

    def build_board(self):
        pass

    def draw_aruco(self):
        """Draw ArUco markers in the image with outline and center points"""

        if self.grayscale is None:
            print("Error: Could not get grayscale image for drawing")
            return None

        # Use cached detection (this won't re-run detection)
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

    def draw_board(self, board):
        """
        Draws the ground plane on the rgb image

        Args:
            board (cv2.aruco.Board): The board to draw

        Returns:
            image_bgr (np.ndarray): The image with the board drawn on it
        """

        if self.grayscale is None:
            print("Error: Could not get grayscale image for drawing")
            return None

        # Get the rgb image
        rgb_image = self.get_rgb()

        # Draw the board
        image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        image_bgr = cv2.aruco.drawPlanarBoard(image_bgr, board, 1, (0, 0, 255))

        return image_bgr


    def read_intrinsics(self, path):
        """
        Read the intrinsics from a file - delegates to ArucoMarkerDetector

        Args:
            path (str): The path to the intrinsics file

        Returns:
            dict: The intrinsics data
        """
        return self.aruco_detector.read_intrinsics(path)


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

def loop_through_captures(data, custom_detector=None):

    """Loop through all captures and return the maximum number of markers detected

    Args:
        data: DataManager instance with capture data
        custom_detector: Optional ArucoMarkerDetector with optimized parameters
    """
    num_captures = len(data.data_entries)

    if custom_detector:
        print(f"\n🎯 Using optimized ArUco detector parameters")
    else:
        print(f"\n📋 Using default ArUco detector parameters")

    print(f"Processing {num_captures} captures...")

    max_markers = 0
    all_marker_counts = []

    # Loop through all captures
    for i in range(num_captures):
        print(f"\n=== Processing Capture {i} ===")

        # Get the capture data directly
        selected_capture = data.data_entries[i]
        data.capture_idx = i  # Set the index for display purposes

        if selected_capture:
            # data.display_capture_data(selected_capture)
            # data.plot_capture_data(selected_capture)

            # Process ArUco detection with custom detector if provided
            processed = Processor(selected_capture, custom_detector)

            # Get marker count for this capture
            corners, ids, rejected = processed.detect_aruco()
            num_markers = len(corners) if corners is not None else 0
            all_marker_counts.append(num_markers)

            # Track maximum
            if num_markers > max_markers:
                max_markers = num_markers

            #aruco_image = processed.draw_aruco()
            # Show the image with capture index in window name
            #processed.show_image(aruco_image, f"ArUco Detection - Capture {i}")
            board_image = processed.draw_board(processed.aruco_detector.grid_board)

            print(f"Capture {i} processed ({num_markers} markers). Press any key to continue to next capture...")

    print(f"\nAll captures processed!")
    print(f"Marker counts per capture: {all_marker_counts}")
    print(f"Maximum markers detected: {max_markers}")

    return max_markers

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

    # Check if we have data entries
    if not data.data_entries:
        print("No capture data found. Exiting.")
        exit(1)

    # To test dictionaries:
    #detector = ArucoMarkerDetector();
    #detector.test_dictionaries(data)

    # Load optimized parameters (automatically uses latest config file)
    config_filename = 'best_aruco_params_20250710_224754.json'

    # Alternative: manually specify a config file
    # config_filename = "best_aruco_params_20250103_143022.json"
    # Alternative: list available configs and choose one
    # available_configs = list_aruco_configs()

    if config_filename:
        # Load the saved parameters
        params, config = load_aruco_parameters(config_filename)

        if params is not None:
            # Create detector with optimized parameters
            optimized_detector = ArucoMarkerDetector(cv2.aruco.DICT_ARUCO_ORIGINAL, params)
            loop_through_captures(data, custom_detector=optimized_detector)
        else:
            # Fallback to default parameters
            print("⚠️  Using default parameters due to loading error")
            loop_through_captures(data)
    else:
        # No config file found, use default parameters
        print("📋 No saved parameters found, using default detection")
        loop_through_captures(data)

    # To vary parameters: detector = ArucoMarkerDetector(); detector.vary_parameters(data)
