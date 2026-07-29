#!/usr/bin/env python3
"""Crane vision competition runtime entry point for RDK X5 (ARM64)."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.runtime import (
    ROOT as RUNTIME_ROOT,
    create_task_generator,
    init_logging,
    load_models,
    load_vision_cfg,
    open_camera,
    release_cameras,
    warmup_models,
)
from models.serial_port import SerialThread
from models.ui import CompetitionUI


def run_competition(cfg, cfg_path):
    init_logging()
    logging.info("Starting crane vision in competition mode (ARM64)")

    cam_bean = open_camera(cfg["cameras"]["bean"])
    cam_digit = open_camera(cfg["cameras"]["digit"])
    try:
        bean_model, digit_model = load_models(cfg)
        warmup_models(cam_bean, cam_digit, bean_model, digit_model)

        task_gen = create_task_generator(cfg, bean_model, digit_model)
        if task_gen is None:
            raise RuntimeError(
                "Task generator unavailable; competition mode requires BPU models"
            )

        proto = cfg.get("protocol", {})
        serial = SerialThread(
            cfg["serial"]["port"],
            cfg["serial"]["baudrate"],
            timeout=cfg["serial"].get("timeout", 0.05),
            start_timeout=proto.get("start_timeout_sec", 0.3),
            send_interval=proto.get("send_interval_sec", 0.1),
        )

        logging.info("Ready. Starting competition UI...")
        ui = CompetitionUI(
            cfg_path, cam_bean, cam_digit,
            bean_model, digit_model,
            task_gen=task_gen, serial_thread=serial,
        )
        ui.run()
    finally:
        release_cameras(cam_bean, cam_digit)


def main():
    parser = argparse.ArgumentParser(
        description="Crane vision competition runtime for RDK X5"
    )
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    init_logging(debug=args.verbose)
    cfg = load_vision_cfg(args.config)
    run_competition(cfg, args.config)


if __name__ == "__main__":
    main()
