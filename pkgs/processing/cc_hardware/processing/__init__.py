"""
Processing package for ArUco marker detection and scene processing.

This package contains modules for:
- ArUco marker detection and optimization
- Camera management and intrinsics
- Data loading and visualization
- Scene processing and localization
- Blender environment generation
"""

# Dynamic __all__ list that only includes successfully imported symbols
__all__ = []

# Import modules conditionally to handle missing dependencies gracefully
try:
    from .processing_capture import *
    # Add processing_capture exports to __all__
    __all__.extend(['Processor', 'load_aruco_parameters', 'loop_through_captures', 'plot_scene_3d'])
except ImportError as e:
    print(f"Warning: Could not import processing_capture: {e}")

try:
    from .aruco_markers import ArucoMarkerDetector
    __all__.append('ArucoMarkerDetector')
except ImportError as e:
    print(f"Warning: Could not import ArucoMarkerDetector: {e}")

try:
    from .data_manager import DataManager
    __all__.append('DataManager')
except ImportError as e:
    print(f"Warning: Could not import DataManager: {e}")

try:
    from .camera import CameraManager
    __all__.append('CameraManager')
except ImportError as e:
    print(f"Warning: Could not import CameraManager: {e}")

try:
    from .blender_env import BlenderEnvironmentBuilder
    __all__.append('BlenderEnvironmentBuilder')
except ImportError as e:
    print(f"Warning: Could not import BlenderEnvironmentBuilder: {e}")

# Always available - no optional dependencies
from .config import ARUCO_CONFIG, FILE_PATHS, VISUALIZATION, PROCESSING
__all__.extend(['ARUCO_CONFIG', 'FILE_PATHS', 'VISUALIZATION', 'PROCESSING'])

# Conditionally import utils (may have optional dependencies)
try:
    from .utils import read_intrinsics, show_image, setup_camera_matrix, load_json_config, save_json_config
    __all__.extend(['read_intrinsics', 'show_image', 'setup_camera_matrix',
                    'load_json_config', 'save_json_config'])
except ImportError as e:
    print(f"Warning: Could not import utils functions: {e}")
