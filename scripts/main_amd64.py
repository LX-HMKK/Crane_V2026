#!/usr/bin/env python3
"""Crane vision x86 test entry point (AMD64).

This entry point exercises the full vision pipeline on a development host:
camera capture -> model inference -> task generation -> result output.
It does not use serial communication; scanning is triggered by a key press
and the result is printed to the console.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Add repo root to path so scripts/ can import models/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.runtime import (
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


def _trigger_scan(cam_bean, cam_digit, bean_model, digit_model, task_gen, cfg):
    """Run one scan and print the result to the console."""
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


def _run_preview_loop(
    cfg, cam_bean, cam_digit, bean_model, digit_model, task_gen, once=False
):
    """Show live preview windows and trigger scans on key press."""
    cv2.namedWindow("bean", cv2.WINDOW_NORMAL)
    cv2.namedWindow("digit", cv2.WINDOW_NORMAL)

    print("Press SPACE or 's' to scan, ESC to quit")

    try:
        while True:
            ok_b, frame_b = cam_bean.capture()
            ok_d, frame_d = cam_digit.capture()
            if ok_b:
                cv2.imshow("bean", frame_b)
            if ok_d:
                cv2.imshow("digit", frame_d)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            if key in (ord(" "), ord("s")):
                _trigger_scan(
                    cam_bean, cam_digit, bean_model, digit_model, task_gen, cfg
                )
                if once:
                    break
    finally:
        cv2.destroyAllWindows()


def run_amd64_test(cfg, mock_cameras=False, once=False):
    init_logging()
    logging.info("Starting crane vision test loop (AMD64)")

    cam_bean = open_camera(
        cfg["cameras"]["bean"], mock=mock_cameras, fallback_empty=True
    )
    cam_digit = open_camera(
        cfg["cameras"]["digit"], mock=mock_cameras, fallback_empty=True
    )

    try:
        bean_model = None
        digit_model = None
        try:
            bean_model, digit_model = load_pt_models(cfg)
        except EnvironmentError as exc:
            logging.warning(
                "PyTorch models not available: %s; running test in camera-only mode",
                exc,
            )

        task_gen = create_task_generator(cfg, bean_model, digit_model)

        if bean_model is not None and digit_model is not None:
            warmup_models(cam_bean, cam_digit, bean_model, digit_model)

        if once:
            _trigger_scan(
                cam_bean, cam_digit, bean_model, digit_model, task_gen, cfg
            )
        else:
            _run_preview_loop(
                cfg, cam_bean, cam_digit, bean_model, digit_model, task_gen
            )
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
    parser.add_argument(
        "--list-cameras", action="store_true", help="List available cameras"
    )
    parser.add_argument(
        "--mock-cameras",
        action="store_true",
        help="Use synthetic cameras for testing",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scan immediately and exit (no preview window)",
    )
    args = parser.parse_args()

    init_logging()

    if args.list_cameras:
        list_cameras()
        return

    cfg = load_vision_cfg(args.config)
    run_amd64_test(cfg, mock_cameras=args.mock_cameras, once=args.once)


if __name__ == "__main__":
    main()
