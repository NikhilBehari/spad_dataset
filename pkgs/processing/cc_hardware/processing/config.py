"""
Centralized configuration for the processing pipeline.
"""

# ArUco Marker Configuration
ARUCO_CONFIG = {
    'dictionary': 'cv2.aruco.DICT_ARUCO_ORIGINAL',
    'ground_plane_ids': list(range(200, 204)),
    'object_id': 204,
    'sensor_id': 100,
    'marker_sizes': {
        'ground_marker_size': 0.1,
        'sensor_marker_size': 0.08,
        'object_marker_size': 0.08
    },
    # Physical coordinates of ground plane markers (in meters)
    'ground_plane_coordinates': {
        'marker_0': [0.0, 0.0, 0.0],          # Top-left origin
        'marker_1': [1.335, 0.0, 0.0],        # Top-right
        'marker_2': [0, -1.057002, 0.0],      # Bottom-left
        'marker_3': [1.3351002, -1.0493502, 0.0]  # Bottom-right
    }
}

# File Paths
FILE_PATHS = {
    'camera_intrinsics': 'camera_intrinsics.json',
    'optimized_params': 'best_aruco_params_20250710_224754.json',
    'default_pkl': 'test_20250709_1.pkl',
    'scene_output': 'test_scene.json'
}

# Visualization Settings
VISUALIZATION = {
    'window_timeout': 0,  # 0 = wait for key
    'marker_outline_color': (0, 255, 0),  # Green
    'marker_center_color': (255, 0, 0),   # Blue
    'marker_id_color': (255, 255, 0),     # Yellow
    'corner_color': (255, 0, 0),          # Blue
    'frame_axes_length': 0.1,             # Length of coordinate frame axes in meters
    'font_scale_large': 0.6,              # Font size for marker IDs
    'font_scale_small': 0.4,              # Font size for corner numbers
    'font_thickness': 2,                  # Font thickness
}

# Processing Settings
PROCESSING = {
    'detection_mode': 'grid',
    'axis_length': 0.5,  # For 3D plotting
    'ground_plane_thickness': 0.0508  # For Blender
}
