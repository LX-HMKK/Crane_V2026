#!/usr/bin/env python3
"""Serial port loopback / protocol test for RDK X5.

Modes:
  loopback  -- TX and RX are shorted; send bytes and verify echo.
  protocol  -- Use SerialPort to send a result frame and wait for START.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import serial

from models.runtime import ROOT as RUNTIME_ROOT, init_logging, load_vision_cfg
from models.serial_port import START_COMMAND, SerialPort, open_serial_port


def _load_serial_cfg(config_path):
    cfg = load_vision_cfg(config_path)
    if "serial" not in cfg:
        raise RuntimeError(f"No 'serial' section in {config_path}")
    return cfg["serial"]


def _run_loopback(ser, count, interval):
    """Send random payloads and expect echo on a TX-RX shorted port."""
    import random

    errors = 0
    for i in range(count):
        payload = bytes(random.randint(0, 255) for _ in range(12))
        ser.write(payload)
        ser.flush()
        time.sleep(0.05)
        echo = ser.read(len(payload))
        if echo != payload:
            errors += 1
            logging.error(
                "Loopback mismatch at frame %d: sent=%s echo=%s",
                i + 1,
                payload.hex(),
                echo.hex(),
            )
        else:
            logging.info("Loopback frame %d OK", i + 1)
        if i + 1 < count:
            time.sleep(interval)
    return errors


def _run_protocol(port, count, interval):
    """Send a valid result frame and wait for START command in response."""
    errors = 0
    for i in range(count):
        try:
            port.send_result(1, 2, 3, 3, 1, 5, 2, 4)
            logging.info("Protocol frame %d sent", i + 1)
        except Exception as exc:
            errors += 1
            logging.error("Failed to send frame %d: %s", i + 1, exc)
            continue

        if port.wait_start(timeout=2.0):
            logging.info("Protocol frame %d: START received", i + 1)
        else:
            errors += 1
            logging.error("Protocol frame %d: START timeout", i + 1)

        if i + 1 < count:
            time.sleep(interval)
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Serial port loopback / protocol test"
    )
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["loopback", "protocol"],
        required=True,
        help="loopback: TX-RX shorted; protocol: send result and wait START",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of frames to send",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between frames",
    )
    args = parser.parse_args()

    init_logging()
    serial_cfg = _load_serial_cfg(args.config)

    logging.info(
        "Opening serial port %s @ %s for %s test",
        serial_cfg["port"],
        serial_cfg["baudrate"],
        args.mode,
    )

    if args.mode == "loopback":
        ser = serial.Serial(
            serial_cfg["port"],
            serial_cfg["baudrate"],
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
        )
        try:
            # Drain any stale data.
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            errors = _run_loopback(ser, args.count, args.interval)
        finally:
            ser.close()
    else:
        port = open_serial_port(
            serial_cfg["port"],
            serial_cfg["baudrate"],
            timeout=serial_cfg.get("timeout", 0.05),
        )
        try:
            errors = _run_protocol(port, args.count, args.interval)
        finally:
            port.close()

    if errors:
        logging.error("Test finished with %d error(s)", errors)
        sys.exit(1)
    logging.info("Test finished successfully")


if __name__ == "__main__":
    main()
