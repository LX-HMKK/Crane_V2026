#!/usr/bin/env python3
"""Crane vision calibration UI entry point."""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.devices.camera import StaticImageCamera
from models.application.runtime import (
    ROOT as RUNTIME_ROOT,
    init_logging,
    list_cameras,
    load_models,
    load_vision_cfg,
    open_camera,
    release_cameras,
)
from models.devices.serial_port import open_serial_port
from models.ui import CalibrationUI


def _open_image_camera(image_path, camera_cfg):
    """Open a StaticImageCamera for offline file-based testing."""
    return StaticImageCamera(
        image_path, camera_cfg["width"], camera_cfg["height"]
    )


def run_calibration(cfg, cfg_path, mock=False, bean_image=None, digit_image=None):
    if bean_image is not None:
        cam_bean = _open_image_camera(bean_image, cfg["cameras"]["bean"])
    else:
        cam_bean = open_camera(
            cfg["cameras"]["bean"], mock=mock, fallback_empty=True
        )

    if digit_image is not None:
        cam_digit = _open_image_camera(digit_image, cfg["cameras"]["digit"])
    else:
        cam_digit = open_camera(
            cfg["cameras"]["digit"], mock=mock, fallback_empty=True
        )

    port = None
    try:
        bean_model = None
        digit_model = None
        try:
            bean_model, digit_model = load_models(cfg)
        except EnvironmentError:
            logging.warning("BPU not available; running UI in camera-only mode")

        serial_cfg = cfg.get("serial")
        if serial_cfg:
            port = open_serial_port(
                serial_cfg["port"],
                serial_cfg["baudrate"],
                timeout=serial_cfg.get("timeout", 0.05),
            )
            logging.info(
                "Serial port opened for calibration: %s @ %s",
                serial_cfg["port"],
                serial_cfg["baudrate"],
            )
        else:
            logging.warning(
                "No serial configuration found; running UI without serial port"
            )

        ui = CalibrationUI(
            cfg_path,
            cam_bean,
            cam_digit,
            bean_model,
            digit_model,
        )
        ui.run()
    finally:
        if port is not None:
            port.close()
        release_cameras(cam_bean, cam_digit)


def main():
    parser = argparse.ArgumentParser(description="Crane vision calibration UI")
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--mock-cameras",
        action="store_true",
        help="Use synthetic cameras for UI testing",
    )
    parser.add_argument(
        "--bean-image",
        help="Path to a static bean camera image (overrides real/mock camera)",
    )
    parser.add_argument(
        "--digit-image",
        help="Path to a static digit camera image (overrides real/mock camera)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras",
    )
    args = parser.parse_args()

    init_logging()

    if args.list_cameras:
        list_cameras()
        return

    cfg = load_vision_cfg(args.config)
    run_calibration(
        cfg,
        args.config,
        mock=args.mock_cameras,
        bean_image=args.bean_image,
        digit_image=args.digit_image,
    )


if __name__ == "__main__":
    main()
