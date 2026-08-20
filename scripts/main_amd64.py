#!/usr/bin/env python3
"""Crane vision x86 test entry point (AMD64).

This entry point mirrors scripts/main_arm64.py but uses PyTorch `.pt` models
and no serial port. It is intended for development-host testing of the same
calibration UI and single-shot scan logic that runs on the RDK X5.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Add repo root to path so scripts/ can import models/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.devices.camera import StaticImageCamera
from models.application.runtime import (
    ROOT as RUNTIME_ROOT,
    create_task_generator,
    init_logging,
    list_cameras,
    load_pt_models,
    load_vision_cfg,
    open_camera,
    release_cameras,
    scan_once,
    warmup_models,
)
from models.ui import CompetitionUI


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

    try:
        bean_model = None
        digit_model = None
        task_gen = None
        try:
            bean_model, digit_model = load_pt_models(cfg)
            task_gen = create_task_generator(cfg, bean_model, digit_model)
        except EnvironmentError:
            logging.warning(
                "PyTorch models not available; running UI in camera-only mode"
            )

        ui = CompetitionUI(
            cfg_path,
            cam_bean,
            cam_digit,
            bean_model,
            digit_model,
            task_gen=task_gen,
            serial_thread=None,
        )
        ui.run()
    finally:
        release_cameras(cam_bean, cam_digit)


def run_once(cfg, mock=False, bean_image=None, digit_image=None):
    """Run a single scan and print the result, without opening the UI."""
    init_logging()
    logging.info("Starting crane vision single-scan test (AMD64)")

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

    try:
        bean_model = None
        digit_model = None
        try:
            bean_model, digit_model = load_pt_models(cfg)
        except EnvironmentError as exc:
            logging.warning(
                "PyTorch models not available: %s; running scan in camera-only mode",
                exc,
            )

        task_gen = create_task_generator(cfg, bean_model, digit_model)

        if bean_model is not None and digit_model is not None:
            warmup_models(cam_bean, cam_digit, bean_model, digit_model)

        result, reason = scan_once(
            cam_bean, cam_digit, bean_model, digit_model, task_gen, cfg
        )
        if result is None:
            logging.error("Scan failed: %s", reason)
            print(f"SCAN FAILED: {reason}")
        else:
            p1, p2, p3 = result
            logging.info("SCAN RESULT: %d %d %d", p1, p2, p3)
            print(f"SCAN RESULT: {p1} {p2} {p3}")
        return result, reason
    finally:
        release_cameras(cam_bean, cam_digit)


def main():
    parser = argparse.ArgumentParser(
        description="Crane vision test runtime for x86 (AMD64)"
    )
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
        "--once",
        action="store_true",
        help="Run a single scan without opening the UI",
    )

    parser.add_argument(
        "--list-cameras", action="store_true", help="List available cameras"
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
    args = parser.parse_args()

    init_logging()

    if args.list_cameras:
        list_cameras()
        return

    cfg = load_vision_cfg(args.config)

    if args.once:
        run_once(
            cfg,
            mock=args.mock_cameras,
            bean_image=args.bean_image,
            digit_image=args.digit_image,
        )
    else:
        run_calibration(
            cfg,
            args.config,
            mock=args.mock_cameras,
            bean_image=args.bean_image,
            digit_image=args.digit_image,
        )


if __name__ == "__main__":
    main()
