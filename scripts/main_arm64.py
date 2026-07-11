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

from models.camera import StaticImageCamera
from models.runtime import (
    ROOT as RUNTIME_ROOT,
    create_task_generator,
    init_logging,
    list_cameras,
    load_models,
    load_vision_cfg,
    open_camera,
    release_cameras,
    scan_once_dets,
    warmup_models,
)
from models.serial_port import SerialPort, open_serial_port
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


def _scan_for_frame(cam_bean, cam_digit, bean_model, digit_model, task_gen, cfg):
    """Run scan_once_dets with retries until a valid 8-byte result frame is produced.

    Returns (frame, reason) where frame is an 8-tuple on success or None.
    """
    deadline = time.monotonic() + cfg["scan"].get("timeout_sec", 30)
    max_attempts = cfg["scan"].get("max_retries", 0) + 1
    for attempt in range(max_attempts):
        if time.monotonic() >= deadline:
            return None, "scan timeout"
        bean_dets, digit_dets, reason = scan_once_dets(
            cam_bean, cam_digit, bean_model, digit_model, cfg
        )
        if reason != "ok":
            logging.warning(
                "Scan attempt %d failed: %s; retrying...",
                attempt + 1,
                reason,
            )
            time.sleep(0.1)
            continue
        frame, reason = task_gen.generate_frame(bean_dets, digit_dets)
        if frame is not None:
            return frame, "ok"
        logging.warning(
            "Scan attempt %d failed: %s; retrying...",
            attempt + 1,
            reason,
        )
        time.sleep(0.1)
    return None, reason


def run_competition(cfg):
    init_logging()
    logging.info("Starting crane vision in competition mode (ARM64)")

    # Open cameras and models before acquiring the serial handle.
    # Use an outer try/finally so cameras are released even if model load fails.
    cam_bean = open_camera(cfg["cameras"]["bean"])
    cam_digit = open_camera(cfg["cameras"]["digit"])
    try:
        bean_model, digit_model = load_models(cfg)
        task_gen = create_task_generator(cfg, bean_model, digit_model)
        if task_gen is None:
            raise RuntimeError(
                "Task generator unavailable; competition mode requires BPU models"
            )

        warmup_models(cam_bean, cam_digit, bean_model, digit_model)
        logging.info("Ready. Waiting for START command...")

        port = open_serial_port(
            cfg["serial"]["port"],
            cfg["serial"]["baudrate"],
            timeout=cfg["serial"].get("timeout", 0.05),
        )

        proto = cfg.get("protocol", {})
        start_timeout = proto.get("start_timeout_sec", 0.3)
        send_interval = proto.get("send_interval_sec", 0.1)
        poll_interval = proto.get("poll_interval_sec", 0.01)

        cached_frame = None
        last_send_time = 0.0
        last_scan_attempt = 0.0
        scan_retry_interval = proto.get("scan_retry_interval_sec", 0.5)

        try:
            while True:
                try:
                    port.poll_start()
                    active = port.is_start_active(start_timeout)

                    if not active:
                        cached_frame = None
                        time.sleep(poll_interval)
                        continue

                    if cached_frame is None:
                        now = time.monotonic()
                        if now - last_scan_attempt >= scan_retry_interval:
                            last_scan_attempt = now
                            frame, reason = _scan_for_frame(
                                cam_bean,
                                cam_digit,
                                bean_model,
                                digit_model,
                                task_gen,
                                cfg,
                            )
                            if frame is None:
                                logging.error("Failed to generate result: %s", reason)
                            else:
                                cached_frame = frame
                                last_send_time = 0.0
                                logging.info(
                                    "Result ready: C1-C3=%s, N4-N8=%s",
                                    cached_frame[:3],
                                    cached_frame[3:],
                                )
                        time.sleep(poll_interval)
                        continue

                    if time.monotonic() - last_send_time >= send_interval:
                        port.send_result(*cached_frame)
                        logging.info(
                            "Sent result frame: %s",
                            " ".join(f"{b:02X}" for b in cached_frame),
                        )
                        last_send_time = time.monotonic()

                    time.sleep(poll_interval)
                except Exception:
                    logging.exception("Error during competition cycle")
                    time.sleep(poll_interval)
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

    if args.no_ui:
        run_competition(cfg)
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
