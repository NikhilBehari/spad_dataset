import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from .config import FILE_PATHS

class CameraManager:
    """
    This class is used to manage the camera

    Returns relevant information for Aruco localization
    """

    def __init__(self):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start the pipeline
        self.profile = self.pipeline.start(self.config)


        # Get intrinsics and save to JSON
        self._save_intrinsics_to_json()

    def _save_intrinsics_to_json(self):
        """Extract intrinsic parameters and save to JSON file."""
        # Get the active profile
        color_profile = self.profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        # Extract parameters
        intrinsics_data = {
            'camera_matrix': {
                'fx': color_intrinsics.fx,
                'fy': color_intrinsics.fy,
                'cx': color_intrinsics.ppx,
                'cy': color_intrinsics.ppy,
                'skew': 0.0
            },
            'distortion_coefficients': list(color_intrinsics.coeffs)
        }

        # Save to JSON file using config path
        filename = FILE_PATHS['camera_intrinsics']
        with open(filename, 'w') as f:
            json.dump(intrinsics_data, f, indent=2)

    def close(self):
        """Stop the camera pipeline."""
        self.pipeline.stop()

if __name__ == "__main__":
    camera = CameraManager()
    print(f"Camera intrinsics saved to {FILE_PATHS['camera_intrinsics']}")
    camera.close()
