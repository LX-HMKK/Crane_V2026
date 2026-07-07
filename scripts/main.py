#!/usr/bin/env python3
"""Crane vision runtime entry point."""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Add repo root to path so scripts/ can import models/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.bean_model import BeanModel
from models.camera import Camera
from models.digit_model import DigitModel
from models.mapping import DEFAULT_BEAN_TO_DIGIT
from models.serial_port import SerialPort
from models.task_generator import TaskGenerator
from models.ui import CalibrationUI


def load_vision_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def list_cameras():
    import cv2

    print("Available cameras:")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  /dev/video{i}: {w}x{h}")
            cap.release()


def _open_camera(camera_cfg):
    return Camera(
        camera_cfg["device"],
        camera_cfg["width"],
        camera_cfg["height"],
        camera_cfg.get("buffer_size", 1),
    )


def _load_models(cfg):
    """Load BPU models; returns (bean_model, digit_model)."""
    bean_model = BeanModel(
        cfg["models"]["bean"]["bin"],
        cfg["models"]["bean"]["cfg"],
        cfg["inference"],
    )
    digit_model = DigitModel(
        cfg["models"]["digit"]["bin"],
        cfg["models"]["digit"]["cfg"],
        cfg["inference"],
    )
    return bean_model, digit_model


def run_calibration(cfg, cfg_path):
    cam_bean = _open_camera(cfg["cameras"]["bean"])
    cam_digit = _open_camera(cfg["cameras"]["digit"])

    try:
        bean_model = None
        digit_model = None
        try:
            bean_model, digit_model = _load_models(cfg)
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
        cam_bean.release()
        cam_digit.release()


def scan_once(bean_cam, digit_cam, bean_model, digit_model, task_gen, cfg):
    frames = cfg["scan"]["frames"]
    all_bean_dets = []
    all_digit_dets = []

    for _ in range(frames):
        ok_b, frame_b = bean_cam.capture()
        ok_d, frame_d = digit_cam.capture()
        if ok_b and bean_model is not None:
            all_bean_dets.extend(bean_model.predict(frame_b))
        if ok_d and digit_model is not None:
            all_digit_dets.extend(digit_model.predict(frame_d))

    return task_gen.generate(all_bean_dets, all_digit_dets)


def run_competition(cfg):
    import serial

    init_logging()
    logging.info("Starting crane vision in competition mode")

    # Open cameras and models before acquiring the serial handle.
    cam_bean = _open_camera(cfg["cameras"]["bean"])
    cam_digit = _open_camera(cfg["cameras"]["digit"])
    bean_model, digit_model = _load_models(cfg)

    mapping = cfg.get("mapping", {}).get("bean_to_digit", DEFAULT_BEAN_TO_DIGIT)
    task_gen = TaskGenerator(
        mapping,
        cfg["lines"],
        bean_model.names,
        digit_model.names,
    )

    port = SerialPort(
        serial.Serial(
            cfg["serial"]["port"],
            cfg["serial"]["baudrate"],
            timeout=cfg["serial"]["timeout"],
        )
    )

    # Warm-up
    logging.info("Warming up models...")
    for _ in range(2):
        ok, frame = cam_bean.capture()
        if ok:
            bean_model.predict(frame)
        ok, frame = cam_digit.capture()
        if ok:
            digit_model.predict(frame)

    logging.info("Ready. Waiting for START frame...")

    try:
        while True:
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
                    "Scan attempt %d failed: %s; retrying...", attempt + 1, reason
                )
                time.sleep(0.1)

            if result is None:
                logging.error("Failed to generate task: %s", reason)
                continue

            p1, p2, p3 = result
            logging.info("Sending TASK: %d %d %d", p1, p2, p3)
            port.send_task(p1, p2, p3)
            logging.info("Done. Returning to WAIT_START.")
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        port.close()
        cam_bean.release()
        cam_digit.release()


def main():
    parser = argparse.ArgumentParser(description="Crane vision runtime")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "vision.yaml"), help="Path to vision.yaml"
    )
    parser.add_argument("--ui", action="store_true", help="Run calibration UI")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    args = parser.parse_args()

    init_logging()

    if args.list_cameras:
        list_cameras()
        return

    cfg = load_vision_cfg(args.config)

    if args.ui:
        run_calibration(cfg, args.config)
    else:
        run_competition(cfg)


if __name__ == "__main__":
    main()
