#!/usr/bin/env python3
"""Crane vision runtime entry point for RDK X5 (ARM64)."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add repo root to path so scripts/ can import models/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.runtime import (
    ROOT as RUNTIME_ROOT,
    create_task_generator,
    init_logging,
    list_cameras,
    load_models,
    load_vision_cfg,
    open_camera,
    release_cameras,
    scan_once,
    warmup_models,
)
from models.serial_port import SerialPort
from models.ui import CalibrationUI


def run_calibration(cfg, cfg_path, mock=False):
    cam_bean = open_camera(
        cfg["cameras"]["bean"], mock=mock, fallback_empty=True
    )
    cam_digit = open_camera(
        cfg["cameras"]["digit"], mock=mock, fallback_empty=True
    )

    try:
        bean_model = None
        digit_model = None
        try:
            bean_model, digit_model = load_models(cfg)
        except EnvironmentError:
            logging.warning("BPU not available; running UI in camera-only mode")

        ui = CalibrationUI(
            cfg_path,
            cam_bean,
            cam_digit,
            bean_model,
            digit_model,
        )
        ui.run()
    finally:
        release_cameras(cam_bean, cam_digit)


def run_competition(cfg):
    import serial

    init_logging()
    logging.info("Starting crane vision in competition mode (ARM64)")

    # Open cameras and models before acquiring the serial handle.
    # Use an outer try/finally so cameras are released even if model load fails.
    cam_bean = open_camera(cfg["cameras"]["bean"])
    cam_digit = open_camera(cfg["cameras"]["digit"])
    try:
        bean_model, digit_model = load_models(cfg)
        task_gen = create_task_generator(cfg, bean_model, digit_model)

        warmup_models(cam_bean, cam_digit, bean_model, digit_model)
        logging.info("Ready. Waiting for START frame...")

        port = SerialPort(
            serial.Serial(
                cfg["serial"]["port"],
                cfg["serial"]["baudrate"],
                timeout=cfg["serial"]["timeout"],
            )
        )

        try:
            while True:
                try:
                    if not port.wait_start():
                        continue
                    logging.info("START received, scanning...")

                    deadline = time.monotonic() + cfg["scan"].get("timeout_sec", 30)
                    result = None
                    reason = ""
                    max_attempts = cfg["scan"].get("max_retries", 0) + 1
                    for attempt in range(max_attempts):
                        if time.monotonic() >= deadline:
                            reason = "scan timeout"
                            break
                        result, reason = scan_once(
                            cam_bean, cam_digit, bean_model, digit_model, task_gen, cfg
                        )
                        if result is not None:
                            break
                        logging.warning(
                            "Scan attempt %d failed: %s; retrying...",
                            attempt + 1,
                            reason,
                        )
                        time.sleep(0.1)

                    if result is None:
                        logging.error("Failed to generate task: %s", reason)
                        continue

                    p1, p2, p3 = result
                    logging.info("Sending TASK: %d %d %d", p1, p2, p3)
                    port.send_task(p1, p2, p3)
                    logging.info("Done. Returning to WAIT_START.")
                except Exception:
                    logging.exception("Error during competition cycle, returning to WAIT_START")
                    continue
        except KeyboardInterrupt:
            logging.info("Shutdown requested")
        finally:
            port.close()
    finally:
        release_cameras(cam_bean, cam_digit)


def main():
    parser = argparse.ArgumentParser(description="Crane vision runtime for RDK X5")
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )

    ui_group = parser.add_mutually_exclusive_group()
    ui_group.add_argument(
        "--ui",
        action="store_true",
        help="Run calibration UI (default behavior, backward-compatible alias)",
    )
    ui_group.add_argument(
        "--no-ui",
        action="store_true",
        help="Run competition mode",
    )

    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    parser.add_argument(
        "--mock-cameras",
        action="store_true",
        help="Use synthetic cameras for UI testing",
    )
    args = parser.parse_args()

    init_logging()

    if args.list_cameras:
        list_cameras()
        return

    cfg = load_vision_cfg(args.config)

    if args.no_ui:
        run_competition(cfg)
    else:
        run_calibration(cfg, args.config, mock=args.mock_cameras)


if __name__ == "__main__":
    main()
