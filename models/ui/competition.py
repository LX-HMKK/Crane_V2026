import logging
import sys
import threading
import time
import tkinter as tk

import cv2

from models.application.competition_controller import CompetitionResultController
from models.application.runtime import inference_gate_open, serial_start_active
from .calibration import (
    CAPTURE_FAILURE_BACKOFF_SEC,
    CAPTURE_OFFLINE_AFTER_FAILURES,
    DEFAULT_PAIR_MAX_SKEW_SEC,
    DEFAULT_TASK_UPDATE_INTERVAL_SEC,
    STATUS_TITLES,
    CalibrationUI,
)

class CompetitionUI(CalibrationUI):
    """Competition-mode UI: calibration viewer + serial + task generation.

    Extends CalibrationUI with a SerialThread and a post-inference hook that
    runs TaskGenerator.generate_frame() and updates the serial thread.

    Unlike the calibration UI, both cameras are captured on every frame cycle
    so that generate_frame always has fresh detections from both views.
    """

    def __init__(
        self,
        cfg_path,
        cam_bean,
        cam_digit,
        bean_model,
        digit_model,
        task_gen=None,
        serial_thread=None,
        root=None,
        cfg=None,
        owns_cameras=True,
        owns_serial=True,
        start_serial=True,
    ):
        self._owns_serial = bool(owns_serial)
        # Assigned before super().__init__() because gating is part of the
        # shared inference lifecycle.
        self.serial = serial_thread
        # DEBUG forces inference regardless of serial START. Set on the Tk
        # thread, read-only from the capture/inference workers (GIL-atomic).
        self._debug_force_inference = False
        super().__init__(
            cfg_path,
            cam_bean,
            cam_digit,
            bean_model,
            digit_model,
            root=root,
            cfg=cfg,
            owns_cameras=owns_cameras,
        )
        if not self._window_ok:
            return
        self.task_gen = task_gen
        try:
            task_update_interval = max(
                0.0,
                float(
                    self.cfg.get("protocol", {}).get(
                        "send_interval_sec", DEFAULT_TASK_UPDATE_INTERVAL_SEC
                    )
                ),
            )
        except (TypeError, ValueError):
            task_update_interval = DEFAULT_TASK_UPDATE_INTERVAL_SEC
        try:
            pair_max_skew_sec = max(
                0.0,
                float(
                    self.cfg.get("protocol", {}).get(
                        "pair_max_skew_sec", DEFAULT_PAIR_MAX_SKEW_SEC
                    )
                ),
            )
        except (TypeError, ValueError):
            pair_max_skew_sec = DEFAULT_PAIR_MAX_SKEW_SEC
        self._result_controller = CompetitionResultController(
            task_gen,
            self.serial,
            self.cfg.get("logic", {}),
            update_interval=task_update_interval,
            pair_max_skew_sec=pair_max_skew_sec,
        )
        self._zone_results = {"bean": {}, "digit": {}}
        self._capture_lock = threading.Lock()
        self._capture_stop = threading.Event()
        self._capture_threads = {}
        self._captured_frames = {"bean": None, "digit": None}
        self._capture_ready = {"bean": False, "digit": False}
        self._capture_failure_logged = {"bean": False, "digit": False}
        self._capture_sequences = {"bean": 0, "digit": 0}
        self._capture_timestamps = {"bean": None, "digit": None}
        self._capture_failures = {"bean": 0, "digit": 0}
        if self.serial is not None and start_serial:
            self.serial.start()
        self._build_competition_controls()
        self._build_status_bar()
        # ESC and window close: hard exit so the watchdog restarts.
        # Only the EXIT button stops the watchdog (clean exit 0).
        self.root.protocol("WM_DELETE_WINDOW", lambda: sys.exit(1))
        self.root.bind("<Escape>", lambda _event: sys.exit(1))

    # ── dual-camera capture ────────────────────────────────────────────

    def _start_inference_worker(self):
        self._start_capture_workers()
        super()._start_inference_worker()

    def _start_capture_workers(self):
        """Start one persistent latest-frame reader for each camera."""
        if self._capture_threads:
            return

        self._capture_stop.clear()
        for mode, camera in (("bean", self.cam_bean), ("digit", self.cam_digit)):
            thread = threading.Thread(
                target=self._capture_worker,
                args=(mode, camera),
                name=f"crane-camera-{mode}",
                daemon=True,
            )
            self._capture_threads[mode] = thread
            thread.start()

    @staticmethod
    def _capture_interval(camera):
        fps = getattr(camera, "target_fps", None)
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            fps = getattr(camera, "fps", None)
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            fps = 30.0
        return 1.0 / float(fps)

    def _capture_worker(self, mode, camera):
        """Continuously replace one camera's raw-frame slot without queuing."""
        interval = self._capture_interval(camera)
        next_capture_at = time.monotonic()
        while not self._capture_stop.is_set():
            delay = next_capture_at - time.monotonic()
            if delay > 0 and self._capture_stop.wait(delay):
                break

            try:
                ok, frame = camera.capture()
            except Exception as exc:
                with self._capture_lock:
                    self._capture_ready[mode] = False
                    log_failure = not self._capture_failure_logged[mode]
                    self._capture_failure_logged[mode] = True
                    self._capture_failures[mode] += 1
                    invalidate = (
                        self._capture_failures[mode]
                        >= CAPTURE_OFFLINE_AFTER_FAILURES
                    )
                if invalidate:
                    self._invalidate_camera_detections(mode)
                if log_failure:
                    logging.warning("Camera capture failed (%s): %s", mode, exc)
                self._capture_stop.wait(CAPTURE_FAILURE_BACKOFF_SEC)
                next_capture_at = time.monotonic()
                continue

            if not ok or frame is None:
                with self._capture_lock:
                    self._capture_ready[mode] = False
                    log_failure = not self._capture_failure_logged[mode]
                    self._capture_failure_logged[mode] = True
                    self._capture_failures[mode] += 1
                    invalidate = (
                        self._capture_failures[mode]
                        >= CAPTURE_OFFLINE_AFTER_FAILURES
                    )
                if invalidate:
                    self._invalidate_camera_detections(mode)
                if log_failure:
                    logging.warning("Camera capture returned no frame (%s)", mode)
                self._capture_stop.wait(CAPTURE_FAILURE_BACKOFF_SEC)
                next_capture_at = time.monotonic()
                continue

            with self._capture_lock:
                self._captured_frames[mode] = frame
                self._capture_ready[mode] = True
                self._capture_failure_logged[mode] = False
                self._capture_failures[mode] = 0
                self._capture_sequences[mode] += 1
                capture_sequence = self._capture_sequences[mode]
                captured_at = time.monotonic()
                self._capture_timestamps[mode] = captured_at
            # Feed inference from the camera thread rather than the Tk event
            # loop. Rendering can run slower than capture without reducing
            # the rate at which each model receives its newest frame.
            self._submit_frame_for_inference(
                mode,
                frame,
                capture_sequence=capture_sequence,
                captured_at=captured_at,
            )
            next_capture_at += interval

    def _invalidate_camera_detections(self, mode):
        """Clear one camera's stale detections once it is confirmed offline."""
        with self._result_lock:
            if self._latest_dets[mode] is None and self._latest_names[mode] is None:
                return
            self._latest_dets[mode] = None
            self._latest_names[mode] = None
            self._latest_capture_meta[mode] = None
            logging.warning(
                "Camera %s offline; invalidating its last detections", mode
            )

    def _camera_capture_healthy(self, mode):
        """False once a camera is confirmed offline, so its stale detections
        are neither re-published nor merged into a new result frame."""
        with self._capture_lock:
            return self._capture_failures.get(mode, 0) < CAPTURE_OFFLINE_AFTER_FAILURES

    def _should_infer(self):
        """Gate inference on serial START / DEBUG / config.

        Competition mode infers only while the lower machine's START command
        is active or the DEBUG toggle forces it. The x86 path (no serial) and
        a config that disables the gate always infer, so development/testing
        workflows are unchanged.
        """
        if self.serial is None:
            return True
        return inference_gate_open(
            self.cfg,
            getattr(self.serial, "is_start_active", None),
            force=self._debug_force_inference,
        )

    def _debug_forced(self):
        """True when inference runs purely because DEBUG forces it (no START).

        In that state generated results must stay display-only: the serial
        cache and the competition failure counters are never touched.
        """
        if not self.cfg.get("inference", {}).get("start_gated", True):
            return False
        if self.serial is None:
            return False
        return self._debug_force_inference and not serial_start_active(self.serial)

    def _clear_serial_cache(self):
        """Drop serial/result state (idempotent; returns True if changed).

        Keeps the serial cache empty while START is inactive so a pre-gap
        frame is never transmitted when START resumes. Also called from the
        DEBUG path: DEBUG results are display-only and must not linger in the
        cache.
        """
        return self._result_controller.clear()

    def _on_inference_gate_closed(self):
        """Reset inference/display/serial state for a gate-off (idle) period.

        Called from the capture-worker submit path whenever the inference
        gate is observed closed. Idempotent under duplicate/concurrent
        invocation (both camera workers hit it each capture tick).
        """
        dirty = self._clear_serial_cache()
        with self._result_lock:
            if (
                self._latest_dets["bean"] is not None
                or self._latest_dets["digit"] is not None
                or self._latest_names["bean"] is not None
                or self._latest_names["digit"] is not None
                or self._zone_results["bean"]
                or self._zone_results["digit"]
            ):
                self._latest_dets = {"bean": None, "digit": None}
                self._latest_names = {"bean": None, "digit": None}
                self._latest_capture_meta = {"bean": None, "digit": None}
                self._zone_results = {"bean": {}, "digit": {}}
                dirty = True
        with self._infer_pending_cv:
            if (
                self._infer_pending["bean"] is not None
                or self._infer_pending["digit"] is not None
            ):
                self._infer_pending = {"bean": None, "digit": None}
                self._infer_pending_meta = {"bean": None, "digit": None}
                self._infer_pending_cv.notify_all()
                dirty = True
        if dirty:
            logging.info("Inference gate closed; idle inference state cleared")

    def _latest_capture_snapshot(self, mode):
        """Return a private copy of a camera's latest frame and sequence."""
        with self._capture_lock:
            frame = self._captured_frames[mode]
            if not self._capture_ready[mode] or frame is None:
                return False, None, self._capture_sequences[mode]
            return True, frame.copy(), self._capture_sequences[mode]

    def _latest_capture_metadata(self, mode):
        """Return the sequence and capture timestamp for one latest frame."""
        with self._capture_lock:
            return {
                "sequence": self._capture_sequences[mode],
                "captured_at": self._capture_timestamps[mode],
            }

    def _camera_state(self, mode):
        """Return (state, frame) for one camera: ok / grace / waiting / offline."""
        with self._capture_lock:
            ready = self._capture_ready.get(mode, False)
            failures = self._capture_failures.get(mode, 0)
            frame = self._captured_frames.get(mode)
        if failures >= CAPTURE_OFFLINE_AFTER_FAILURES:
            return ("offline", None)
        if ready and frame is not None:
            # The renderer draws boxes and guide lines in place. Returning the
            # capture buffer itself races with the next inference submission.
            return ("ok", frame.copy())
        if frame is not None:
            return ("grace", frame.copy())
        return ("waiting", None)

    @staticmethod
    def _inferred_label(mode, zone, value):
        """Return the on-frame label for a logically completed zone."""
        if mode == "bean":
            names = {1: "YELLOW", 2: "GREEN", 3: "WHITE"}
            value_text = names.get(value, str(value))
        else:
            value_text = str(value)
        return f"INFER Z{zone}: {value_text}"

    def _draw_inferred_zone_labels(self, frame, zone_results, mode):
        """Draw virtual labels for inferred zones that have no detection box."""
        if not zone_results:
            return

        height, width = frame.shape[:2]
        if mode == "bean":
            boundaries = (
                0,
                self.cfg["lines"]["bean"]["left_x"],
                self.cfg["lines"]["bean"]["right_x"],
                width,
            )
            first_zone = 1
        else:
            digit_lines = self.cfg["lines"]["digit"]
            boundaries = (
                0,
                digit_lines["x1"],
                digit_lines["x2"],
                digit_lines["x3"],
                digit_lines["x4"],
                width,
            )
            first_zone = 4

        for zone, result in sorted(zone_results.items()):
            value, source, _confidence = result
            if source != "inferred":
                continue

            index = zone - first_zone
            if not 0 <= index < len(boundaries) - 1:
                continue

            text = self._inferred_label(mode, zone, value)
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            center_x = (boundaries[index] + boundaries[index + 1]) // 2
            x = max(4, min(width - text_w - 4, center_x - text_w // 2))
            y = max(
                text_h + 4,
                min(height - baseline - 4, height // 2 + text_h // 2),
            )
            cv2.rectangle(
                frame,
                (x - 3, y - text_h - 5),
                (x + text_w + 3, y + baseline + 3),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 140, 255), 2,
            )

    def _build_competition_controls(self):
        """Extend the control bar with a DEBUG toggle on the right side."""
        font = ("Helvetica", int(12 * self.font_scale))
        self._debug_button = tk.Button(
            self._mode_frame,
            text="DEBUG",
            command=self._toggle_debug,
            bg="#555555",
            fg="white",
            font=font,
            width=8,
            height=max(1, self.button_height // 20),
        )
        # Packed after SAVE, so the row reads (right to left) ... DEBUG SAVE EXIT.
        self._debug_button.pack(side=tk.RIGHT, padx=1)
        if not self._debug_button_usable():
            self._debug_button.config(state=tk.DISABLED)
        self.root.bind("<KeyPress-d>", self._toggle_debug)
        self._info_label.config(
            text="1=BEAN  2=DIGIT  3=SETTINGS  d=DEBUG  f=fullscreen  [SAVE] [EXIT]"
        )

    def _debug_button_usable(self):
        """DEBUG is meaningful only when the START gate can actually be bypassed."""
        if not self.cfg.get("inference", {}).get("start_gated", True):
            return False
        return self.serial is not None

    def _toggle_debug(self, _event=None):
        """Toggle DEBUG mode: force inference regardless of serial START.

        DEBUG is display-only — it never bypasses the serial START gate, so
        results are shown on the UI but not transmitted to the lower machine.
        """
        self._debug_force_inference = not self._debug_force_inference
        btn = getattr(self, "_debug_button", None)
        if btn is not None and btn.winfo_exists():
            if self._debug_force_inference:
                btn.config(text="DEBUG:ON", bg="#cc6600", relief=tk.SUNKEN)
            else:
                btn.config(text="DEBUG", bg="#555555", relief=tk.RAISED)
        logging.info(
            "DEBUG mode %s", "enabled" if self._debug_force_inference else "disabled"
        )
        self._update_status()
        if self._debug_force_inference:
            # Kick both cameras' newest frames into inference immediately so
            # the first DEBUG result appears without waiting for the next
            # capture cycle.
            for mode in ("bean", "digit"):
                ok, frame, _seq = self._latest_capture_snapshot(mode)
                if ok:
                    meta = self._latest_capture_metadata(mode)
                    self._submit_frame_for_inference(
                        mode,
                        frame,
                        capture_sequence=meta["sequence"],
                        captured_at=meta["captured_at"],
                    )

    def _build_status_bar(self):
        """Add a status bar under the control panel for camera/serial state."""
        status_frame = tk.Frame(self._control_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        font = ("Helvetica", int(12 * self.font_scale))
        self._status_labels = {}
        self._status_states = {}
        for key, title in STATUS_TITLES.items():
            label = tk.Label(
                status_frame,
                text=f"{title}: N/A",
                font=font,
                fg="white",
                bg="#555555",
            )
            label.pack(side=tk.LEFT, padx=2)
            self._status_labels[key] = label
            self._status_states[key] = None

    def _set_status(self, key, ok):
        """Update one status label; ok True/False/None (None means N/A)."""
        if self._status_states.get(key) == ok:
            return
        title = STATUS_TITLES.get(key, key)
        if key == "start":
            # START reflects the lower machine's command, not a fault: an
            # absent heartbeat is idle (grey OFF), never an alarm-red OFFLINE.
            if ok is None:
                text, bg = f"{title}: N/A", "#666666"
            elif ok:
                text, bg = f"{title}: ON", "#006600"
            else:
                text, bg = f"{title}: OFF", "#555555"
        elif ok is None:
            text, bg = f"{title}: N/A", "#666666"
        elif ok:
            text, bg = f"{title}: OK", "#006600"
        else:
            text, bg = f"{title}: OFFLINE", "#aa0000"
        self._status_labels[key].config(text=text, bg=bg)
        self._status_states[key] = ok

    def _update_status(self):
        """Refresh the status bar; errors must never break the frame loop."""
        try:
            for mode in ("bean", "digit"):
                offline = (
                    self._capture_failures.get(mode, 0)
                    >= CAPTURE_OFFLINE_AFTER_FAILURES
                )
                self._set_status(mode, not offline)
            if self.serial is None:
                self._set_status("serial", None)
            else:
                self._set_status(
                    "serial", bool(getattr(self.serial, "connected", True))
                )
            if self.serial is None:
                self._set_status("start", None)
            else:
                self._set_status("start", serial_start_active(self.serial))
        except Exception:
            logging.exception("Failed to refresh status bar")

    def _update_frame(self):
        """Render the active camera while capture workers feed both models."""
        self._update_status()
        if not self._running:
            return

        if self.mode == "settings":
            # Capture workers keep detection and serial output live while the
            # settings page intentionally hides both camera previews.
            if self._running:
                self.root.after(30, self._update_frame)
            return

        try:
            avail_w = int(self._video_container.winfo_width())
            avail_h = int(self._video_container.winfo_height())
        except Exception:
            avail_w, avail_h = 0, 0
        if avail_w <= 1 or avail_h <= 1:
            avail_w, avail_h = self.window_width, self.video_max_height

        state, frame = self._camera_state(self.mode)

        if state == "offline":
            self._show_frame(
                self._render_placeholder(
                    f"CAMERA OFFLINE: {self.mode.upper()}", avail_w, avail_h
                )
            )
            if self._running:
                self.root.after(30, self._update_frame)
            return

        if state == "waiting":
            # First frame not delivered yet (e.g. startup) — this is not an
            # offline alarm, so show a neutral placeholder while the status bar
            # keeps reporting the live failure counters.
            self._show_frame(
                self._render_placeholder(
                    "WAITING FOR CAMERA...", avail_w, avail_h, color=(160, 160, 160)
                )
            )
            if self._running:
                self.root.after(30, self._update_frame)
            return

        with self._result_lock:
            zone_results = dict(self._zone_results.get(self.mode, {}))
            dets = self._latest_dets.get(self.mode)
            names = self._latest_names.get(self.mode)

        if self.mode == "bean":
            self._draw_bean_lines(frame)
            model = self.bean_model
        else:
            self._draw_digit_lines(frame)
            model = self.digit_model

        if self._should_infer():
            if model is not None and dets is not None:
                self._draw_detections(
                    frame, dets, names,
                    zone_results=zone_results, mode=self.mode,
                )
            self._draw_inferred_zone_labels(frame, zone_results, self.mode)
        # Idle (no START, no DEBUG): show the live feed and guide lines only,
        # with no stale detection or inference overlays.

        display_frame = self._resize_for_display(
            frame, max_w=avail_w, max_h=avail_h
        )
        self._show_frame(display_frame)

        if self._running:
            self.root.after(30, self._update_frame)

    # ── task generation hook ───────────────────────────────────────────

    def _on_inference_done(self, mode, dets):
        """After each inference, update the task state from both cameras."""
        del mode, dets
        self._update_task_from_latest_detections(
            debug_forced=self._debug_forced()
        )

    def _update_task_from_latest_detections(self, debug_forced=False):
        """Pass a coherent inference snapshot to the result controller."""

        with self._result_lock:
            bean_dets = self._latest_dets.get("bean")
            digit_dets = self._latest_dets.get("digit")
            bean_meta = self._latest_capture_meta.get("bean")
            digit_meta = self._latest_capture_meta.get("digit")

        result = self._result_controller.update(
            bean_dets,
            digit_dets,
            bean_meta=bean_meta,
            digit_meta=digit_meta,
            cameras_healthy=(
                self._camera_capture_healthy("bean")
                and self._camera_capture_healthy("digit")
            ),
            debug_forced=debug_forced,
        )
        if result is not None:
            with self._result_lock:
                self._zone_results["bean"] = result.bean_zones
                self._zone_results["digit"] = result.digit_zones

    # ── lifecycle ──────────────────────────────────────────────────────

    def _stop_capture_workers(self):
        """Stop capture before either camera can be released or handed off."""
        threads = list(getattr(self, "_capture_threads", {}).values())
        if not threads:
            return True

        self._capture_stop.set()
        stopped = True
        for thread in threads:
            thread.join(timeout=1.0)
            if thread.is_alive():
                stopped = False

        if stopped:
            self._capture_threads = {}
        else:
            logging.warning("Camera capture workers did not stop before shutdown")
        return stopped

    def _quit(self):
        self._running = False
        self._stop_inference_worker()
        self._stop_capture_workers()
        if self.serial is not None and self._owns_serial:
            self.serial.stop()
        self.root.destroy()
