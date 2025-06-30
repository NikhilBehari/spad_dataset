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

# Configuration
DASHBOARD_FULLSCREEN = True
DASHBOARD_CONFIG = "PyQtGraphDashboardConfig"
GANTRY_CONFIG = "SingleDrive1AxisGantry"
GANTRY_AXES_KWARGS = {}
SENSOR_CONFIG = "TMF8828Config"

# Default sample values (will be overridden by 't' for test mode)
DEFAULT_X_SAMPLES = 100
DEFAULT_Y_SAMPLES = 100


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

NOW = datetime.now()

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

    spad = SPADSensor.create_from_config(sensor)
    if not spad.is_okay:
        get_logger().fatal("Failed to initialize spad")
        return
    manager.add(spad=spad)

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

    output_pkl = logdir / f"{object}_{NOW.strftime('%Y%m%d_%H%M%S')}.pkl"
    assert not output_pkl.exists(), f"Output file {output_pkl} already exists"
    pkl_writer = PklHandler(output_pkl)
    manager.add(writer=pkl_writer)

    pkl_writer.append({
        "metadata": {
            "object": object,
            "start_time": NOW.isoformat(),
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

    histogram = spad.accumulate()
    dashboard.update(iter, histograms=histogram)

    pos = controller.get_position(iter)
    if pos is None:
        return False

    gantry.move_to(pos["x"], pos["y"])

    writer.append(
        {
            "iter": iter,
            "pos": pos,
            "histogram": histogram,
        }
    )

    time.sleep(0.25)
    return True


def cleanup(gantry: StepperMotorSystem, **kwargs):
    get_logger().info("Cleaning up...")
    gantry.move_to(0, 0)
    gantry.close()


@register_cli
def spad_gantry_capture_v2(
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig,
    gantry: StepperMotorSystem,
    object: str,
    logdir: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S"),
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
            cleanup(manager.components["gantry"])
        finally:
            final_pkl_path = logdir / f"{object}_{NOW.strftime('%Y%m%d_%H%M%S')}.pkl"
            print(
                f"\033[1;32mPKL file saved to "
                f"{final_pkl_path.resolve()}\033[0m"
            )


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