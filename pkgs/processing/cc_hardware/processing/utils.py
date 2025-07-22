"""
Shared utilities for the processing pipeline.
Consolidates common functions to eliminate code duplication.
"""

import json
import numpy as np
from typing import Tuple, Optional, Dict, Any
from .config import FILE_PATHS, VISUALIZATION

# Optional imports - blender does not have cv2
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def read_intrinsics(path: str = None) -> Tuple[Optional[Dict], Optional[Dict], Optional[np.ndarray]]:
    """
    Centralized intrinsics reader.

    Args:
        path: Path to intrinsics file. If None, uses default from config.

    Returns:
        tuple: (intrinsics_dict, camera_matrix_dict, distortion_coefficients)
               or (None, None, None) if error
    """
    if path is None:
        path = FILE_PATHS['camera_intrinsics']

    try:
        with open(path, 'r') as f:
            intrinsics = json.load(f)
        print(f"✅ Intrinsics loaded from: {path}")
        return intrinsics, intrinsics['camera_matrix'], intrinsics['distortion_coefficients']
    except FileNotFoundError:
        print(f"❌ Intrinsics file not found: {path}")
        return None, None, None
    except json.JSONDecodeError:
        print(f"❌ Error parsing intrinsics file: {path}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error reading intrinsics file {path}: {e}")
        return None, None, None


def show_image(image: np.ndarray, window_name: str = "Image") -> None:
    """
    Centralized image display function.

    Args:
        image: Image array to display
        window_name: Name for the window
    """
    if not _HAS_CV2:
        print(f"⚠️  Cannot display '{window_name}' - OpenCV not available")
        return

    if image is None:
        print(f"❌ Error: No image to display in window '{window_name}'")
        return

    cv2.imshow(window_name, image)
    if VISUALIZATION['window_timeout'] == 0:
        print(f"📷 Displaying '{window_name}' - Press any key to continue...")
        cv2.waitKey(0)
    else:
        cv2.waitKey(VISUALIZATION['window_timeout'])
    cv2.destroyAllWindows()


def setup_camera_matrix(camera_matrix_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert camera matrix dictionary to numpy array.

    Args:
        camera_matrix_dict: Dictionary with fx, fy, cx, cy

    Returns:
        3x3 camera matrix array
    """
    return np.array([
        [camera_matrix_dict['fx'], 0, camera_matrix_dict['cx']],
        [0, camera_matrix_dict['fy'], camera_matrix_dict['cy']],
        [0, 0, 1]
    ], dtype=np.float32)


def load_json_config(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Generic JSON config loader with error handling.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary from JSON or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in file: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading config {filepath}: {e}")
        return None


def save_json_config(data: Dict[str, Any], filepath: str) -> bool:
    """
    Generic JSON config saver with error handling.

    Args:
        data: Dictionary to save
        filepath: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Config saved to: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Error saving config to {filepath}: {e}")
        return False


def ensure_shape_consistency(corners: np.ndarray) -> np.ndarray:
    """
    Ensure corner arrays have consistent shape (1, 4, 2).

    Args:
        corners: Corner array that might have shape (4, 2) or (1, 4, 2)

    Returns:
        Consistently shaped corner array
    """
    if corners.ndim == 2 and corners.shape == (4, 2):
        return corners.reshape(1, 4, 2)
    return corners
