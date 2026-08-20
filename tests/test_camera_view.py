#!/usr/bin/env python3
"""Standalone dual-camera viewer for placement verification.

No model inference, no serial port, no sliders. Press 1/2 to switch cameras.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.devices.camera import StaticImageCamera
from models.ui.viewer import CameraViewer
from models.application.runtime import (
    ROOT as RUNTIME_ROOT,
    init_logging,
    load_vision_cfg,
    open_camera,
)


def _open_image_camera(image_path, camera_cfg):
    return StaticImageCamera(
        image_path, camera_cfg["width"], camera_cfg["height"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="View bean and digit cameras for placement check"
    )
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--mock-cameras",
        action="store_true",
        help="Use synthetic cameras",
    )
    parser.add_argument(
        "--bean-image",
        help="Path to a static bean camera image",
    )
    parser.add_argument(
        "--digit-image",
        help="Path to a static digit camera image",
    )
    args = parser.parse_args()

    init_logging()
    cfg = load_vision_cfg(args.config)

    if args.bean_image is not None:
        cam_bean = _open_image_camera(args.bean_image, cfg["cameras"]["bean"])
    else:
        cam_bean = open_camera(
            cfg["cameras"]["bean"], mock=args.mock_cameras, fallback_empty=True
        )

    if args.digit_image is not None:
        cam_digit = _open_image_camera(args.digit_image, cfg["cameras"]["digit"])
    else:
        cam_digit = open_camera(
            cfg["cameras"]["digit"], mock=args.mock_cameras, fallback_empty=True
        )

    viewer = CameraViewer(cfg, args.config, cam_bean, cam_digit)
    try:
        viewer.run()
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        cam_bean.release()
        cam_digit.release()


if __name__ == "__main__":
    main()
