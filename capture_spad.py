#!/usr/bin/env python3
import sys
import serial.tools.list_ports
import subprocess
from pathlib import Path
from glob import glob
from datetime import datetime

# === Configuration ===
OBJECT_NAME = "arrow"
SPAD_POSITION = [0.1, 0.4, 0.5]
DASHBOARD_CONFIG = "PyQtGraphDashboardConfig"
SENSOR_CONFIG = "TMF8828Config"
DASHBOARD_FULLSCREEN = True
RGBD_CAPTURE = True

# === Log directory ===
LOGDIR = Path("logs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")

# === Script path ===
CAPTURE_SCRIPT = Path(__file__).parent / "examples" / "spad_capture" / "spad_capture.py"

SENSOR_DESC = "USB Serial Device"

def find_sensor_port():
    if sys.platform == "darwin":
        ports = glob("/dev/cu.*")
        matches = [p for p in ports if "usbmodem" in p]
        if not matches:
            raise RuntimeError("No serial port matching 'usbmodem' found")
        return sorted(matches)[0]
    elif sys.platform == "win32":
        for port in serial.tools.list_ports.comports():
            if SENSOR_DESC in port.description:
                return port.device
        raise RuntimeError(f"No serial port with description '{SENSOR_DESC}' found")
    elif sys.platform in ["linux", "wsl"]:
        ports = glob("/dev/ttyACM*")
        if not ports:
            raise RuntimeError("No serial port matching '/dev/ttyACM*' found")
        return sorted(ports)[0]
    else:
        raise RuntimeError("Unsupported platform")

SENSOR_PORT = find_sensor_port()

def build_command():
    spad_pos_str = "[" + ",".join(str(v) for v in SPAD_POSITION) + "]"

    cmd = [
        "python", str(CAPTURE_SCRIPT),
        f"sensor.port={SENSOR_PORT}",
        f"sensor={SENSOR_CONFIG}",
        f"dashboard={DASHBOARD_CONFIG}",
        f"dashboard.fullscreen={str(DASHBOARD_FULLSCREEN).lower()}",
        f"logdir={LOGDIR}",
        f"+object={OBJECT_NAME}",
        f"+spad_position={spad_pos_str}",
        f"use_realsense={RGBD_CAPTURE}"
    ]
    return cmd

def main():
    cmd = build_command()
    print("Running:", " \\\n  ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
