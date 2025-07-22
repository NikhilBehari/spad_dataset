# Processing Package

This package contains modules for ArUco marker detection and scene processing for the CC Hardware SPAD dataset.

## Modules

- `aruco_markers.py`: ArUco marker detection and optimization
- `camera.py`: Camera management and intrinsics
- `data_manager.py`: Data loading and visualization from .pkl files
- `processing_capture.py`: Main processing pipeline for captures
- `blender_env.py`: Blender environment generation from scene data

## Usage

```python
from cc_hardware.processing import (
    ArucoMarkerDetector,
    DataManager,
    CameraManager,
    BlenderEnvironmentBuilder,
    Processor
)

# Load data
data = DataManager("path/to/data.pkl")

# Create detector
detector = ArucoMarkerDetector()

# Process captures
processor = Processor(data.data_entries[0], detector)
```

## Dependencies

- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy
- PyRealSense2
- Blender (for environment generation)
