"""
Process script for running ArUco marker detection and scene processing.

This script runs the main processing pipeline from the processing package.
"""

import sys
from pathlib import Path

# Add the processing package to path
sys.path.insert(0, str(Path(__file__).parent / "pkgs" / "processing"))
from cc_hardware.processing.processing_capture import *
from cc_hardware.processing import FILE_PATHS

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace this with the actual path to your .pkl file
    # Example: PKL_PATH = Path("out_data_captured/2025-06-30/my_object_20250630_0.pkl")
    PKL_PATH_INPUT = FILE_PATHS['default_pkl']
    #PKL_PATH_INPUT = "test_20250703.pkl"
    PKL_PATH = Path(PKL_PATH_INPUT)

    data = DataManager(PKL_PATH)
    data.display_metadata()

    # Load optimized ArUco detector parameters
    optimized_detector = ArucoMarkerDetector()
    optimized_detector.load_optimized_parameters()

    # Process all captures
    loop_through_captures(data, custom_detector=optimized_detector)
