#!/usr/bin/env python3

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

# === Log directory ===
LOGDIR = Path("logs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")

# === Script path ===
CAPTURE_SCRIPT = Path(__file__).parent / "examples" / "spad_capture" / "spad_capture.py"

def find_port(prefix: str) -> str:
    ports = glob("/dev/cu.*")
    matches = [p for p in ports if prefix in p]
    if not matches:
        raise RuntimeError(f"No serial port matching '{prefix}' found under /dev/cu.*")
    return sorted(matches)[0]

SENSOR_PORT = find_port("usbmodem")

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
    ]
    return cmd

def main():
    cmd = build_command()
    print("Running:", " \\\n  ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
