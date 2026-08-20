#!/usr/bin/env python3
"""Send simulated crane vision result frames over serial for testing.

This script does not need cameras or models. It opens the serial port defined
in configs/vision.yaml and continuously sends 12-byte result frames
(A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A) until interrupted.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from models.devices.serial_port import open_serial_port


# Default simulated result:
# C1-C3 = pickup zone colors (1=yellow, 2=green, 3=white)
# N4-N8 = placement zone digits (1..5)
DEFAULT_FRAME = (2, 1, 3, 2, 1, 3, 4, 5)


def _load_serial_cfg(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "serial" not in cfg:
        raise RuntimeError(f"No 'serial' section in {config_path}")
    return cfg["serial"]


def main():
    parser = argparse.ArgumentParser(
        description="Send simulated crane vision result frame(s) over serial"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument("--port", help="Serial port (overrides config)")
    parser.add_argument("--baudrate", type=int, help="Baud rate (overrides config)")
    parser.add_argument(
        "--frame",
        nargs=8,
        type=int,
        metavar=("C1", "C2", "C3", "N4", "N5", "N6", "N7", "N8"),
        default=DEFAULT_FRAME,
        help="Result payload: C1-C3 colors (1-3) and N4-N8 digits (1-5)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of frames to send (0 = infinite, stop with Ctrl+C)",
    )
    parser.add_argument(
        "--interval", type=float, default=0.1, help="Seconds between frames"
    )
    args = parser.parse_args()

    if args.port and args.baudrate:
        serial_cfg = {"port": args.port, "baudrate": args.baudrate, "timeout": 0.05}
    else:
        serial_cfg = _load_serial_cfg(args.config)
        if args.port:
            serial_cfg["port"] = args.port
        if args.baudrate:
            serial_cfg["baudrate"] = args.baudrate

    port = open_serial_port(
        serial_cfg["port"],
        serial_cfg["baudrate"],
        timeout=serial_cfg.get("timeout", 0.05),
    )

    frame = tuple(args.frame)
    head = bytes([0xA5, 0x5A])
    tail = bytes([0x0D, 0x0A])
    raw_frame = head + bytes(frame) + tail

    try:
        mode = "infinite" if args.count == 0 else f"{args.count} frame(s)"
        print(f"Sending {mode} to {serial_cfg['port']} @ {serial_cfg['baudrate']}")
        print(f"Payload (C1-C3, N4-N8): {frame}")
        print(f"Full frame: {raw_frame.hex(' ').upper()}")
        print("Press Ctrl+C to stop\n")

        sent = 0
        while args.count == 0 or sent < args.count:
            port.send_result(*frame)
            sent += 1
            print(
                f"[{sent}] TX {raw_frame.hex(' ').upper()}  "
                f"C1-C3={frame[:3]} N4-N8={frame[3:]}"
            )
            if args.count == 0 or sent < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nStopped after {sent} frame(s)")
    finally:
        port.close()
        print("Serial port closed")


if __name__ == "__main__":
    main()
