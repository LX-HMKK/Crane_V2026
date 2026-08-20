#!/usr/bin/env python3
"""Crane vision competition runtime entry point for RDK X5 (ARM64)."""

import argparse
import logging
import os
import subprocess
import sys
import time

_BOOTSTRAP_STARTED_AT = time.monotonic()
if __name__ == "__main__":
    # This precedes the native runtime imports, so boot logs separate process
    # scheduling/dynamic loading time from camera and BPU initialization.
    os.write(
        sys.stderr.fileno(),
        f"Startup: Python bootstrap entered at {_BOOTSTRAP_STARTED_AT:.3f}s\n".encode(),
    )

from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.application.runtime import (
    ROOT as RUNTIME_ROOT,
    create_task_generator,
    inference_gate_open,
    init_logging,
    load_models,
    load_vision_cfg,
    open_camera,
    release_cameras,
    scan_once_dets,
    serial_start_active,
    warmup_models,
)
from models.devices.serial_port import SerialThread

_RUNTIME_IMPORT_DURATION_SEC = time.monotonic() - _BOOTSTRAP_STARTED_AT


CAMERA_FAILURE_BACKOFF_SEC = 0.1
CAMERA_FAILURE_LOG_INTERVAL_SEC = 5.0
IDLE_POLL_INTERVAL_SEC = 0.01


def _startup_phase(started_at, phase):
    logging.info("Startup: %s at %.2fs", phase, time.monotonic() - started_at)


def _competition_device_paths(cfg):
    """Return the RDK device nodes required before cameras and BPU can open."""
    devices = ["/dev/bpu", cfg["serial"]["port"]]
    devices.extend(
        cfg["cameras"][name]["device"] for name in ("bean", "digit")
    )
    return [
        device for device in devices
        if isinstance(device, str) and device.startswith("/dev/")
    ]


def _wait_for_competition_devices(cfg, poll_interval=0.2):
    """Wait here so expensive imports can overlap early device initialization."""
    required = _competition_device_paths(cfg)
    missing = [device for device in required if not Path(device).exists()]
    if missing:
        logging.info("Startup: waiting for devices: %s", ", ".join(missing))
    while missing:
        time.sleep(poll_interval)
        missing = [device for device in required if not Path(device).exists()]


def _create_serial_thread(cfg):
    proto = cfg.get("protocol", {})
    reconnect_cfg = cfg.get("serial", {}).get("reconnect", {}) or {}
    return SerialThread(
        cfg["serial"]["port"],
        cfg["serial"]["baudrate"],
        timeout=cfg["serial"].get("timeout", 0.05),
        start_timeout=proto.get("start_timeout_sec", 0.3),
        send_interval=proto.get("send_interval_sec", 0.1),
        reconnect_enabled=reconnect_cfg.get("enabled", True),
        reconnect_after=reconnect_cfg.get("after_failures", 5),
        reconnect_interval=reconnect_cfg.get("interval_sec", 1.0),
    )


def build_completed_result(task_gen, bean_dets, digit_dets, cfg):
    """Build the serial payload using the same logical completion as the UI."""
    logic_cfg = cfg.get("logic", {})
    frame, _bean_results, _digit_results, reason = (
        task_gen.generate_frame_with_completion(
            bean_dets,
            digit_dets,
            confidence_threshold=logic_cfg.get("confidence_threshold", 0.5),
            max_inferred_beans=logic_cfg.get("max_inferred_beans", 1),
            max_inferred_digits=logic_cfg.get("max_inferred_digits", 2),
        )
    )
    return frame, reason


class HeadlessCompetitionLoop:
    """Continuous headless inference loop that can hand cameras to the UI."""

    def __init__(
        self, cfg, cam_bean, cam_digit, bean_model, digit_model, task_gen,
        serial, started_at,
    ):
        self.cfg = cfg
        self.cam_bean = cam_bean
        self.cam_digit = cam_digit
        self.bean_model = bean_model
        self.digit_model = digit_model
        self.task_gen = task_gen
        self.serial = serial
        self.started_at = started_at
        # Field debugging without a UI: force inference even when START is
        # inactive. Set by the UI DEBUG button and synced back on handoff.
        self.debug_force_inference = bool(
            cfg.get("debug", {}).get("force_inference", False)
        )
        self.first_result_at = None
        self.consecutive_failures = 0
        self.last_frame = None
        self._capture_failure_count = 0
        self._last_capture_failure_log_at = None
        self._capture_failed_this_scan = False

    def inference_enabled(self):
        """Whether the headless loop should run inference right now.

        Competition rules gate recognition on the lower machine's START
        heartbeat. The gate can be bypassed by config (inference.start_gated
        disabled) or by the DEBUG override.
        """
        return inference_gate_open(
            self.cfg,
            getattr(self.serial, "is_start_active", None),
            force=self.debug_force_inference,
        )

    def _debug_forced(self):
        """True when inference runs purely because debug_force forces it with
        START inactive. Results must stay out of the serial cache (headless
        equivalent of the UI DEBUG button's display-only behavior).
        """
        if not self.cfg.get("inference", {}).get("start_gated", True):
            return False
        return self.debug_force_inference and not serial_start_active(self.serial)

    def _clear_serial_cache(self):
        """Keep the serial cache empty while START is inactive so a pre-gap
        frame is never transmitted when START resumes. DEBUG-forced results
        are display-only and must not linger in the cache either.
        """
        if self.last_frame is not None:
            self.last_frame = None
        if self.consecutive_failures != 0:
            self.consecutive_failures = 0
        if self.serial is not None and self.serial.current_frame is not None:
            self.serial.current_frame = None

    def run(self, handoff_ready=None):
        """Run until a handoff callback returns true, or forever when omitted."""
        while True:
            if not serial_start_active(self.serial):
                # START inactive: keep the serial cache empty so a pre-gap
                # frame is never transmitted when START resumes. DEBUG-forced
                # inference may still run below but stays display-only.
                self._clear_serial_cache()
            if not self.inference_enabled():
                # No START yet: keep the models warm and poll without running
                # inference. Idle must never count toward failure counters,
                # and the UI handoff still needs to be honored.
                if handoff_ready is not None and handoff_ready():
                    return
                time.sleep(IDLE_POLL_INTERVAL_SEC)
                continue
            retry_after_capture_failure = self._scan_once()
            if handoff_ready is not None and handoff_ready():
                return
            if retry_after_capture_failure:
                time.sleep(CAMERA_FAILURE_BACKOFF_SEC)

    def _scan_once(self):
        self._capture_failed_this_scan = False
        bean_dets, digit_dets, capture_reason = scan_once_dets(
            self.cam_bean,
            self.cam_digit,
            self.bean_model,
            self.digit_model,
            self.cfg,
            on_capture_failure=self._on_capture_failure,
        )
        capture_failed = capture_reason == "too many camera capture failures"
        if not self._capture_failed_this_scan:
            self._capture_failure_count = 0
            self._last_capture_failure_log_at = None

        if capture_failed:
            # Camera outage: keep the last valid frame sticky (never clear it)
            # and let the caller back off. Outages are not logic failures, so
            # they must not count toward the clear-on-10 threshold — otherwise
            # a recovery that still has no valid arrangement could never clear
            # the stale frame again.
            return True

        debug_forced = self._debug_forced()

        if capture_reason == "ok":
            frame, reason = build_completed_result(
                self.task_gen, bean_dets, digit_dets, self.cfg
            )
        else:
            frame, reason = None, capture_reason

        if frame is not None:
            if not debug_forced:
                self.consecutive_failures = 0
                self.serial.current_frame = frame
                if self.first_result_at is None:
                    self.first_result_at = time.monotonic()
                    logging.info(
                        "Startup: first valid result ready at %.2fs",
                        self.first_result_at - self.started_at,
                    )
                if frame != self.last_frame:
                    logging.info(
                        "Result updated: C1-C3=%s, N4-N8=%s",
                        frame[:3], frame[3:],
                    )
                    self.last_frame = frame
            else:
                # DEBUG is display-only in headless too: never write the
                # serial cache or the failure counters.
                logging.debug(
                    "DEBUG inference: C1-C3=%s, N4-N8=%s", frame[:3], frame[3:],
                )
            return False

        if debug_forced:
            logging.warning("DEBUG inference produced no result: %s", reason)
            return False

        self.consecutive_failures += 1
        if self.consecutive_failures == 10:
            self.serial.current_frame = None
            logging.error(
                "No valid result after %d scans; clearing serial frame: %s",
                self.consecutive_failures, reason,
            )
        elif self.consecutive_failures < 10:
            logging.warning(
                "Result generation failed (%d/10): %s",
                self.consecutive_failures, reason,
            )
        return False

    def _on_capture_failure(self, bean_ok, digit_ok):
        self._capture_failed_this_scan = True
        self._capture_failure_count += 1
        now = time.monotonic()
        if (
            self._last_capture_failure_log_at is not None
            and now - self._last_capture_failure_log_at < CAMERA_FAILURE_LOG_INTERVAL_SEC
        ):
            return
        logging.warning(
            "Camera capture failure (bean_ok=%s, digit_ok=%s; %d consecutive failure(s))",
            bean_ok,
            digit_ok,
            self._capture_failure_count,
        )
        self._last_capture_failure_log_at = now


def _x11_ready():
    """Return true only after the sunrise X11 session accepts connections."""
    display = os.environ.setdefault("DISPLAY", ":0")
    xauthority = os.environ.setdefault("XAUTHORITY", "/home/sunrise/.Xauthority")
    display_number = display[1:].split(".", 1)[0] if display.startswith(":") else ""
    if not display_number:
        return False
    if not Path(f"/tmp/.X11-unix/X{display_number}").exists():
        return False
    if not Path(xauthority).is_file():
        return False
    try:
        result = subprocess.run(
            ["xdpyinfo", "-display", display],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def run_competition(cfg, cfg_path):
    """Run the touchscreen competition UI after a desktop session is available."""
    # Keep the desktop-only imports out of the headless boot path.
    from models.ui import CompetitionUI

    started_at = time.monotonic()
    logging.info("Starting crane vision in competition UI mode (ARM64)")

    _wait_for_competition_devices(cfg)

    cam_bean = None
    cam_digit = None
    try:
        cam_bean = open_camera(cfg["cameras"]["bean"])
        cam_digit = open_camera(cfg["cameras"]["digit"])
        _startup_phase(started_at, "cameras opened")
        bean_model, digit_model = load_models(cfg)
        _startup_phase(started_at, "models loaded")
        warmup_models(cam_bean, cam_digit, bean_model, digit_model, rounds=1)
        _startup_phase(started_at, "models warmed")

        task_gen = create_task_generator(cfg, bean_model, digit_model)
        if task_gen is None:
            raise RuntimeError(
                "Task generator unavailable; competition mode requires BPU models"
            )

        serial = _create_serial_thread(cfg)

        _startup_phase(started_at, "ready; starting competition UI")
        ui = CompetitionUI(
            cfg_path, cam_bean, cam_digit,
            bean_model, digit_model,
            task_gen=task_gen, serial_thread=serial,
        )
        ui.run()
    finally:
        release_cameras(cam_bean, cam_digit)


def _run_headless_then_ui(loop, cfg_path):
    """Show the UI when XFCE is ready, then resume headless work on close."""
    next_ui_check = 0.0

    def handoff_ready():
        nonlocal next_ui_check
        now = time.monotonic()
        if now < next_ui_check:
            return False
        next_ui_check = now + 1.0
        return _x11_ready()

    while True:
        loop.run(handoff_ready=handoff_ready)
        logging.info("X11 is ready; handing cameras and models to the competition UI")
        try:
            from models.ui import CompetitionUI

            ui = CompetitionUI(
                cfg_path,
                loop.cam_bean,
                loop.cam_digit,
                loop.bean_model,
                loop.digit_model,
                task_gen=loop.task_gen,
                serial_thread=loop.serial,
                cfg=loop.cfg,
                owns_cameras=False,
                owns_serial=False,
                start_serial=False,
            )
            if not ui._window_ok:
                logging.warning("Competition UI is not available yet; keeping headless mode")
                next_ui_check = time.monotonic() + 2.0
                continue
            ui.run()
            # The UI mutates the shared config object, including calibration lines.
            loop.cfg = ui.cfg
            # Carry the operator's DEBUG override back into the headless loop.
            loop.debug_force_inference = getattr(
                ui, "_debug_force_inference", False
            )
            logging.info("Competition UI closed; resuming headless competition mode")
            loop.run()
        except Exception:
            logging.exception("Competition UI failed; keeping headless mode")
            next_ui_check = time.monotonic() + 2.0


def run_headless_competition(cfg, cfg_path=None, ui_when_ready=False):
    """Run competition inference and serial output without a desktop session.

    This is the boot-time path used by the system service. It deliberately
    shares the UI's logical-completion rules so moving to headless operation
    does not change the 12-byte result protocol.
    """
    started_at = time.monotonic()
    serial = None
    cam_bean = None
    cam_digit = None

    logging.info("Starting crane vision in headless competition mode (ARM64)")
    try:
        _wait_for_competition_devices(cfg)
        serial = _create_serial_thread(cfg)
        serial.start()
        _startup_phase(started_at, "serial receiver started")
        cam_bean = open_camera(cfg["cameras"]["bean"])
        cam_digit = open_camera(cfg["cameras"]["digit"])
        _startup_phase(started_at, "cameras opened")
        bean_model, digit_model = load_models(cfg)
        _startup_phase(started_at, "models loaded")
        warmup_models(cam_bean, cam_digit, bean_model, digit_model, rounds=1)
        _startup_phase(started_at, "models warmed")

        task_gen = create_task_generator(cfg, bean_model, digit_model)
        if task_gen is None:
            raise RuntimeError(
                "Task generator unavailable; competition mode requires BPU models"
            )

        _startup_phase(started_at, "ready; waiting for first valid result")

        loop = HeadlessCompetitionLoop(
            cfg,
            cam_bean,
            cam_digit,
            bean_model,
            digit_model,
            task_gen,
            serial,
            started_at,
        )
        if ui_when_ready:
            _run_headless_then_ui(loop, cfg_path)
        else:
            loop.run()
    finally:
        if serial is not None:
            serial.stop()
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the competition inference and serial loop without Tk/XFCE",
    )
    parser.add_argument(
        "--ui-when-ready",
        action="store_true",
        help="In headless mode, show the competition UI when the X11 session is ready",
    )
    args = parser.parse_args()

    init_logging(debug=args.verbose)
    logging.info(
        "Startup: runtime imports completed in %.2fs",
        _RUNTIME_IMPORT_DURATION_SEC,
    )
    cfg = load_vision_cfg(args.config)
    if args.ui_when_ready and not args.headless:
        parser.error("--ui-when-ready requires --headless")
    if args.headless:
        run_headless_competition(
            cfg, args.config, ui_when_ready=args.ui_when_ready
        )
    else:
        run_competition(cfg, args.config)


if __name__ == "__main__":
    main()
