import os
os.environ["HYDRA_HYDRA_LOGGING__FILE"] = "false"
os.environ["HYDRA_JOB_LOGGING__FILE"] = "false"

import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple

from cc_hardware.drivers.spads import SPADSensor, SPADSensorConfig
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import SnakeStepperController
from cc_hardware.tools.dashboard import SPADDashboard, SPADDashboardConfig
from cc_hardware.utils import get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler
from cc_hardware.utils.manager import Manager
import serial.tools.list_ports

import sys
sys.path.insert(0, r"C:\Users\nb_cc\Desktop\UROPs\cc-hardware\librealsense\wrappers\python")
import pyrealsense2 as rs
import numpy as np

# RealSense Camera Configuration
REALSENSE_COLOR_WIDTH = 640
REALSENSE_COLOR_HEIGHT = 480
REALSENSE_DEPTH_WIDTH = 640
REALSENSE_DEPTH_HEIGHT = 480
REALSENSE_FPS = 30

# Configuration
DASHBOARD_FULLSCREEN = True
DASHBOARD_CONFIG = "PyQtGraphDashboardConfig"
GANTRY_CONFIG = "SingleDrive1AxisGantry"
GANTRY_AXES_KWARGS = {}
SENSOR_CONFIG = "TMF8828Config"

# Default sample values (will be overridden by 't' for test mode)
DEFAULT_X_SAMPLES = 10
DEFAULT_Y_SAMPLES = 10

# Map specific descriptions to spad + gantry port 
GANTRY_DESC = "USB-SERIAL CH340"
SENSOR_DESC = "USB Serial Device"
GANTRY_PORT = None
SENSOR_PORT = None
for port in serial.tools.list_ports.comports():
    if GANTRY_DESC in port.description:
        GANTRY_PORT = port.device
    elif SENSOR_DESC in port.description:
        SENSOR_PORT = port.device
print("GANTRY PORT: ", GANTRY_PORT, "    SPAD PORT: ", SENSOR_PORT)

if SENSOR_PORT is None:
    print("Continue code without sensor port? \033[1;31mRunning without sensor connected will save RGB images but no histogram data will be saved.\033[0m (y/n): ")
    choice = input().strip().lower()
    if choice != 'y':
        sys.exit("Operation cancelled by user.")

NOW = datetime.now()
LOG_BASE_DIR = Path("out_data_captured")
DAILY_LOG_DIR_NAME = NOW.strftime("%Y-%m-%d") 

def setup(
    manager: Manager,
    *,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    gantry: StepperMotorSystem,
    x_samples: int,
    y_samples: int,
    logdir: Path,
    object: str,
):
    logdir.mkdir(parents=True, exist_ok=True)

    if SENSOR_PORT is None:
        # spad = SPADSensor.create_from_config(sensor)
        # if not spad.is_okay:
        #     get_logger().fatal("Failed to initialize spad")
        #     return
        manager.add(spad=None)
    else: 
        spad = SPADSensor.create_from_config(sensor)
        if not spad.is_okay:
            get_logger().fatal("Failed to initialize spad")
            return
        manager.add(spad=spad)

    if SENSOR_PORT is None:
        manager.add(dashboard=None)
    else: 
        dashboard = SPADDashboard.create_from_config(dashboard, sensor=spad)
        dashboard.setup()
        manager.add(dashboard=dashboard)

    gantry_controller = SnakeStepperController(
        [
            dict(name="x", range=(0, 32), samples=x_samples),
            dict(name="y", range=(0, 32), samples=y_samples),
        ]
    )
    manager.add(gantry=gantry, controller=gantry_controller)

    # Initialize RealSense Camera
    realsense_pipeline = rs.pipeline()
    realsense_config = rs.config()
    realsense_config.enable_stream(rs.stream.depth, REALSENSE_DEPTH_WIDTH, REALSENSE_DEPTH_HEIGHT, rs.format.z16, REALSENSE_FPS)
    realsense_config.enable_stream(rs.stream.color, REALSENSE_COLOR_WIDTH, REALSENSE_COLOR_HEIGHT, rs.format.bgr8, REALSENSE_FPS)

    # Start streaming and get profile
    realsense_profile = realsense_pipeline.start(realsense_config)

    # Create an align object
    realsense_align = rs.align(rs.stream.color)

    # Get intrinsic parameters for depth and color streams
    depth_stream_profile = realsense_profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_stream_profile = realsense_profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_intrinsics = depth_stream_profile.get_intrinsics()
    color_intrinsics = color_stream_profile.get_intrinsics()

    manager.add(realsense_pipeline=realsense_pipeline)
    manager.add(realsense_align=realsense_align)


    # Generate unique filename with _N suffix if needed
    base_filename = f"{object}_{NOW.strftime('%Y%m%d')}"
    suffix = 0
    final_output_pkl_path = None
    while True:
        suffix_str = "" if suffix == 0 else f"_{suffix}"
        potential_filename = f"{base_filename}{suffix_str}.pkl"
        potential_path = logdir / potential_filename
        if not potential_path.exists():
            final_output_pkl_path = potential_path
            break
        suffix += 1

    pkl_writer = PklHandler(final_output_pkl_path)
    manager.add(writer=pkl_writer)
    manager.add(final_output_pkl_path_for_print=final_output_pkl_path)

    # Append RealSense config and intrinsics to metadata
    pkl_writer.append({
        "metadata": {
            "object": object,
            "start_time": NOW.isoformat(),
            "realsense_config": {
                "color_width": REALSENSE_COLOR_WIDTH,
                "color_height": REALSENSE_COLOR_HEIGHT,
                "depth_width": REALSENSE_DEPTH_WIDTH,
                "depth_height": REALSENSE_DEPTH_HEIGHT,
                "fps": REALSENSE_FPS,
            },
            "realsense_intrinsics": {
                "depth": {
                    "ppx": depth_intrinsics.ppx, "ppy": depth_intrinsics.ppy,
                    "fx": depth_intrinsics.fx, "fy": depth_intrinsics.fy,
                    "model": str(depth_intrinsics.model), # Convert enum to string
                    "coeffs": depth_intrinsics.coeffs # Convert tuple to list
                },
                "color": {
                    "ppx": color_intrinsics.ppx, "ppy": color_intrinsics.ppy,
                    "fx": color_intrinsics.fx, "fy": color_intrinsics.fy,
                    "model": str(color_intrinsics.model), # Convert enum to string
                    "coeffs": color_intrinsics.coeffs # Convert tuple to list
                },
            }
        }
    })

def loop(
    iter: int,
    manager: Manager,
    spad: SPADSensor,
    dashboard: SPADDashboard,
    controller: SnakeStepperController,
    gantry: StepperMotorSystem,
    writer: PklHandler,
    **kwargs,
) -> bool:
    get_logger().info(f"Starting iter {iter}...")

    if SENSOR_PORT is None:
        histogram = None 
    else:
        histogram = spad.accumulate()
        dashboard.update(iter, histograms=histogram)

    pos = controller.get_position(iter)
    if pos is None:
        return False

    gantry.move_to(pos["x"], pos["y"])

    # --- RealSense Capture ---
    realsense_pipeline = manager.components["realsense_pipeline"]
    realsense_align = manager.components["realsense_align"]
    frames = realsense_pipeline.wait_for_frames()
    # Align frames for aligned depth and color
    aligned_frames = realsense_align.process(frames)
    # Get aligned depth and color frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame() # This is the color frame aligned to the depth stream
    # Convert aligned frames to numpy arrays
    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
    aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
    writer.append(
        {
            "iter": iter,
            "pos": pos,
            "histogram": histogram,
            "realsense_data": { # New key for RealSense data
                "aligned_rgb_image": aligned_color_image,
                "aligned_depth_image": aligned_depth_image,
            }
        }
    )

    time.sleep(0.25)
    return True

def cleanup(gantry: StepperMotorSystem, manager: Manager, **kwargs): # Added 'manager: Manager' to signature
    get_logger().info("Cleaning up...")
    gantry.move_to(0, 0)
    gantry.close()

    # Stop RealSense pipeline
    realsense_pipeline = manager.components.get("realsense_pipeline")
    if realsense_pipeline:
        realsense_pipeline.stop()
        get_logger().info("RealSense pipeline stopped.")


@register_cli
def spad_gantry_capture_v2(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    gantry: StepperMotorSystem,
    object: str,
    logdir: Path = LOG_BASE_DIR / DAILY_LOG_DIR_NAME,
):
    _setup = partial(
        setup,
        sensor=sensor,
        dashboard=dashboard,
        gantry=gantry,
        logdir=logdir,
        x_samples=X_SAMPLES,
        y_samples=Y_SAMPLES,
        object=object,
    )

    with Manager() as manager:
        try:
            manager.run(setup=_setup, loop=loop, cleanup=cleanup)
        except KeyboardInterrupt:
            cleanup(gantry=manager.components["gantry"], manager=manager) # Pass manager to cleanup
        finally:
            # Retrieve the actual path that was used for saving from the manager
            final_pkl_path_for_print = manager.components.get("final_output_pkl_path_for_print")
            if final_pkl_path_for_print: # Check if the path was successfully stored
                print(
                    f"\033[1;32mPKL file saved to "
                    f"{final_pkl_path_for_print.resolve()}\033[0m"
                )
            else:
                get_logger().error("Could not retrieve final PKL path for printing.")


def main():
    import sys

    global X_SAMPLES, Y_SAMPLES # Declare X_SAMPLES and Y_SAMPLES as global to modify them

    object_name_confirmed = False
    current_object_name = ""

    while not object_name_confirmed:
        current_object_name = input("Please enter the object name: ").strip()
        if not current_object_name:
            print("Object name cannot be empty. Please try again.")
            continue

        print(f"\033[1;32mObject Name: {current_object_name}\033[0m")
        choice = input("Does this look OK? (y/n/t for test mode): ").strip().lower()

        if choice == 'y':
            X_SAMPLES = DEFAULT_X_SAMPLES
            Y_SAMPLES = DEFAULT_Y_SAMPLES
            print(f"\033[1;32mRunning in FULL CAPTURE MODE: {X_SAMPLES}x{Y_SAMPLES} samples.\033[0m") 
            object_name_confirmed = True
        elif choice == 't':
            X_SAMPLES = 2
            Y_SAMPLES = 2
            print(f"\033[1;33mRunning in TEST MODE: X_SAMPLES={X_SAMPLES}, Y_SAMPLES={Y_SAMPLES}\033[0m")
            object_name_confirmed = True
        elif choice == 'n':
            exit() 
        else:
            print("Invalid choice. Please enter 'y', 'n', or 't'.")
        print() 

    original_argv = sys.argv
    try:
        sys.argv = [
            "capture.py",
            f"sensor.port={SENSOR_PORT}",
            f"+gantry.port={GANTRY_PORT}",
            f"dashboard.fullscreen={str(DASHBOARD_FULLSCREEN).lower()}",
            f"dashboard={DASHBOARD_CONFIG}",
            f"gantry={GANTRY_CONFIG}",
            "+gantry.axes_kwargs={}",
            f"sensor={SENSOR_CONFIG}",
            f"+object={current_object_name}", # Use the dynamically entered object name
        ]
        run_cli(spad_gantry_capture_v2)
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()