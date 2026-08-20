import logging
import queue
import sys
import threading
import time
import tkinter as tk

import cv2
import yaml
from PIL import Image, ImageTk

from .viewer import CameraViewer, LINE_COLORS
from models.config.config_binding import ConfigBinding
from models.application.runtime import inference_gate_open, serial_start_active
from models.domain.zones import ZoneLayout

SLIDER_DEFS = {
    "bean": [
        ("bean_left_x", "Bean left x", "bean", "left_x", "max_x"),
        ("bean_right_x", "Bean right x", "bean", "right_x", "max_x"),
    ],
    "digit": [
        ("digit_x1", "Digit x1", "digit", "x1", "max_x"),
        ("digit_x2", "Digit x2", "digit", "x2", "max_x"),
        ("digit_x3", "Digit x3", "digit", "x3", "max_x"),
        ("digit_x4", "Digit x4", "digit", "x4", "max_x"),
    ],
}

DEFAULT_UI_CFG = {
    "window": {"fullscreen": False, "width": 1024, "height": 600},
    "video": {"max_height": 420, "fit_mode": "contain"},
    "controls": {
        "touch_mode": False,
        "slider_width": 16,
        "slider_height": 32,
        "button_height": 16,
        "padding": 1,
        "font_scale": 0.8,
        "inference_overlay": True,
    },
}

RESULT_FAILURE_LOG_INTERVAL_SEC = 5.0
RESULT_UPDATE_LOG_INTERVAL_SEC = 1.0
DEFAULT_TASK_UPDATE_INTERVAL_SEC = 0.05
DEFAULT_PAIR_MAX_SKEW_SEC = 0.15
CAPTURE_FAILURE_BACKOFF_SEC = 0.05
CAPTURE_OFFLINE_AFTER_FAILURES = 5
INFERENCE_RATE_LOG_INTERVAL_SEC = 5.0

STATUS_TITLES = {
    "bean": "BEAN",
    "digit": "DIGIT",
    "serial": "SERIAL",
    "start": "START",
}


class _InferenceRateMeter:
    """Count completed per-camera inferences over a bounded time window."""

    def __init__(self, interval=INFERENCE_RATE_LOG_INTERVAL_SEC, started_at=None):
        self.interval = float(interval)
        self._window_started_at = (
            time.monotonic() if started_at is None else float(started_at)
        )
        self._counts = {"bean": 0, "digit": 0}
        self._lock = threading.Lock()

    def record(self, mode, now=None):
        """Record one completed inference and return rates when the window ends."""
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            self._counts[mode] += 1
            elapsed = now - self._window_started_at
            if elapsed < self.interval:
                return None

            bean_fps = self._counts["bean"] / elapsed
            digit_fps = self._counts["digit"] / elapsed
            rates = {
                "bean_fps": bean_fps,
                "digit_fps": digit_fps,
                "total_fps": bean_fps + digit_fps,
                "window_sec": elapsed,
            }
            self._counts = {"bean": 0, "digit": 0}
            self._window_started_at = now
            return rates


class CalibrationUI(CameraViewer):
    """Tkinter-based calibration UI.

    Extends CameraViewer with sliders, config saving and inference overlay.
    The control panel lives outside the video frame so buttons and sliders do
    not obscure the camera image.
    """

    def __init__(
        self,
        cfg_path,
        cam_bean,
        cam_digit,
        bean_model=None,
        digit_model=None,
        root=None,
        cfg=None,
        owns_cameras=True,
    ):
        self._owns_cameras = bool(owns_cameras)
        super().__init__(cfg, cfg_path, cam_bean, cam_digit, root=root)
        if not self._window_ok:
            return

        self.bean_model = bean_model
        self.digit_model = digit_model

        # Keep one replaceable pending frame per camera. Each model has its
        # own worker, so an expensive inference can never block the other
        # camera while stale frames remain disposable.
        self._infer_pending = {"bean": None, "digit": None}
        self._infer_pending_meta = {"bean": None, "digit": None}
        self._infer_pending_cv = threading.Condition()
        self._result_lock = threading.Lock()
        self._latest_dets = {"bean": None, "digit": None}
        self._latest_names = {"bean": None, "digit": None}
        self._latest_capture_meta = {"bean": None, "digit": None}
        self._inference_rate_meter = _InferenceRateMeter()
        self._infer_threads = {}
        self._save_feedback_id = None
        self._settings_layout_active = False

        # Parent already created main_frame/video_container/video_label;
        # add the control panel below the video area.
        self._build_calibration_controls()

    def _apply_ui_cfg(self):
        """Merge UI config with defaults and pre-compute derived values."""
        super()._apply_ui_cfg()
        ctl = self.ui_cfg["controls"]
        self.slider_width = int(ctl.get("slider_width", 16))
        self.slider_height = int(ctl.get("slider_height", 32))
        self.button_height = int(ctl.get("button_height", 16))
        self.padding = int(ctl.get("padding", 1))
        self.inference_overlay = bool(ctl.get("inference_overlay", True))

    def _save_cfg(self):
        try:
            with open(self.cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(self.cfg, f, sort_keys=False, allow_unicode=True)
            return True
        except OSError as exc:
            logging.error("Failed to save config to %s: %s", self.cfg_path, exc)
            return False

    def _save_with_feedback(self):
        """Write config to disk and briefly show confirmation on the save button."""
        ok = self._save_cfg()
        btn = getattr(self, "_save_button", None)
        if btn is None or not btn.winfo_exists():
            return
        if ok:
            btn.config(text="SAVED ✓", bg="#006600")
        else:
            btn.config(text="ERROR", bg="#aa0000")
        # Cancel any previous restore timer so rapid clicks don't stack.
        if self._save_feedback_id is not None:
            self.root.after_cancel(self._save_feedback_id)
        self._save_feedback_id = self.root.after(1500, self._restore_save_button_text)

    def _restore_save_button_text(self):
        self._save_feedback_id = None
        btn = getattr(self, "_save_button", None)
        if btn is not None and btn.winfo_exists():
            btn.config(text="SAVE", bg="#0066cc")

    def _quit(self):
        """Destroy the window without saving config (use SAVE button first)."""
        super()._quit()

    def _start_inference_worker(self):
        """Start one background inference worker for each loaded model."""
        if not self.inference_overlay:
            return
        if self._infer_threads:
            return

        for mode, model in (("bean", self.bean_model), ("digit", self.digit_model)):
            if model is None:
                continue
            thread = threading.Thread(
                target=self._inference_worker,
                args=(mode,),
                name=f"crane-inference-{mode}",
                daemon=True,
            )
            self._infer_threads[mode] = thread
            thread.start()

    def _stop_inference_worker(self):
        """Signal all inference workers to stop and wait for them to finish."""
        threads = list(self._infer_threads.items())
        if not threads:
            return
        # Wake the worker if it is blocked waiting for a pending frame.
        with self._infer_pending_cv:
            self._infer_pending = {"bean": None, "digit": None}
            self._infer_pending_meta = {"bean": None, "digit": None}
            self._infer_pending_cv.notify_all()

        still_running = []
        for mode, thread in threads:
            thread.join(timeout=1.0)
            if thread.is_alive():
                still_running.append(mode)
        if still_running:
            logging.warning(
                "Inference workers did not stop before shutdown: %s",
                ", ".join(still_running),
            )
        else:
            self._infer_threads = {}

    def _stop_capture_workers(self):
        """Hook for subclasses that own background camera capture workers."""
        return True

    def _take_pending_inference_frame(self, mode):
        """Take this model's latest frame without looking at the other camera."""
        with self._infer_pending_cv:
            while self._running and self._infer_pending[mode] is None:
                self._infer_pending_cv.wait(timeout=0.1)
            if not self._running:
                return None

            frame = self._infer_pending[mode]
            capture_meta = self._infer_pending_meta[mode]
            self._infer_pending[mode] = None
            self._infer_pending_meta[mode] = None
            return frame, capture_meta

    def _camera_capture_healthy(self, mode):
        """Whether this camera is currently producing live frames.

        The calibration UI captures synchronously on the Tk thread, so any
        inference it schedules already ran on a live frame. CompetitionUI
        overrides this with a capture-failure check.
        """
        return True

    def _should_infer(self):
        """Whether captured frames should be fed to the model workers.

        Calibration always infers so guide lines can be tuned against live
        detections. CompetitionUI overrides this to gate inference on the
        serial START command and the DEBUG toggle.
        """
        return True

    def _inference_worker(self, mode):
        """Run inference on the latest pending frame for one camera/model."""
        model = self.bean_model if mode == "bean" else self.digit_model
        if model is None:
            return
        while self._running:
            pending = self._take_pending_inference_frame(mode)
            if pending is None:
                continue
            frame, capture_meta = pending
            try:
                dets = model.predict(frame)
                # Do not publish a result or update serial output after UI shutdown.
                if not self._running:
                    break
                if not self._camera_capture_healthy(mode):
                    # The camera is confirmed offline. Drop this inference so
                    # stale detections from the last pre-disconnect frame can
                    # never be re-published into a fresh result frame.
                    continue
                if not self._should_infer():
                    # The gate closed while this inference was running
                    # (START dropped or DEBUG off). Drop the result so an
                    # idle period can never publish stale detections.
                    continue
                with self._result_lock:
                    self._latest_dets[mode] = dets
                    self._latest_names[mode] = model.names
                    self._latest_capture_meta[mode] = capture_meta
                self._on_inference_done(mode, dets)
                rates = self._inference_rate_meter.record(mode)
                if rates is not None:
                    logging.info(
                        "Inference throughput: bean=%.2f fps, digit=%.2f fps, "
                        "total=%.2f fps (%.1fs window)",
                        rates["bean_fps"],
                        rates["digit_fps"],
                        rates["total_fps"],
                        rates["window_sec"],
                    )
            except Exception as exc:
                logging.error("Inference worker error (%s): %s", mode, exc)

    def _on_inference_done(self, mode, dets):
        """Hook called after each inference completes (in worker thread).

        Override in subclasses to trigger task generation, serial updates, etc.
        """

    def _submit_frame_for_inference(
        self, mode, frame, *, capture_sequence=None, captured_at=None
    ):
        """Hand the latest frame to its model worker without blocking the UI."""
        if not self.inference_overlay:
            return
        if not self._should_infer():
            # Inference is gated off (no START, no DEBUG in competition
            # mode): clear idle state and do not queue a frame. The model
            # worker stays blocked and warm until the gate reopens.
            self._on_inference_gate_closed()
            return
        if getattr(self, "serial", None) is not None and not serial_start_active(
            self.serial
        ):
            # Inference is running on DEBUG alone (START inactive). DEBUG is
            # display-only: keep the serial cache empty so a pre-gap frame is
            # never transmitted when START resumes (F2).
            self._clear_serial_cache()
        model = self.bean_model if mode == "bean" else self.digit_model
        if model is None:
            return
        with self._infer_pending_cv:
            self._infer_pending[mode] = frame.copy()
            self._infer_pending_meta[mode] = {
                "sequence": capture_sequence,
                "captured_at": captured_at,
            }
            self._infer_pending_cv.notify_all()

    def _build_calibration_controls(self):
        """Add control panel below the video area created by CameraViewer."""
        control_frame = tk.Frame(self._video_container.master)
        control_frame.grid(row=1, column=0, sticky="ew")
        self._control_frame = control_frame

        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 1))
        self._mode_frame = mode_frame

        font = ("Helvetica", int(12 * self.font_scale))

        self._bean_button = tk.Button(
            mode_frame,
            text="BEAN (1)",
            command=lambda: self._set_mode("bean"),
            bg="#00aa00",
            fg="white",
            font=font,
            width=10,
            height=max(1, self.button_height // 20),
        )
        self._bean_button.pack(side=tk.LEFT, padx=1)

        self._digit_button = tk.Button(
            mode_frame,
            text="DIGIT (2)",
            command=lambda: self._set_mode("digit"),
            bg="#555555",
            fg="white",
            font=font,
            width=10,
            height=max(1, self.button_height // 20),
        )
        self._digit_button.pack(side=tk.LEFT, padx=1)

        self._settings_button = tk.Button(
            mode_frame,
            text="SETTINGS (3)",
            command=lambda: self._set_mode("settings"),
            bg="#555555",
            fg="white",
            font=font,
            width=12,
            height=max(1, self.button_height // 20),
        )
        self._settings_button.pack(side=tk.LEFT, padx=1)

        # SAVE / EXIT buttons in the same row, right-aligned.
        # Pack EXIT first (far right), then SAVE (to its left).
        self._exit_button = tk.Button(
            mode_frame,
            text="EXIT",
            command=self._quit,
            bg="#aa3333",
            fg="white",
            font=font,
            width=8,
            height=max(1, self.button_height // 20),
        )
        self._exit_button.pack(side=tk.RIGHT, padx=1)

        self._save_button = tk.Button(
            mode_frame,
            text="SAVE",
            command=self._save_with_feedback,
            bg="#0066cc",
            fg="white",
            font=font,
            width=8,
            height=max(1, self.button_height // 20),
        )
        self._save_button.pack(side=tk.RIGHT, padx=1)

        self.sliders_frame = tk.Frame(control_frame)
        self.sliders_frame.pack(fill=tk.X)

        self._create_sliders(self.mode)

        # Update the info bar to mention the touch-friendly buttons.
        self._info_label.config(
            text="1=BEAN  2=DIGIT  3=SETTINGS  f=fullscreen  [SAVE] [EXIT]"
        )
        self.root.bind("<KeyPress-3>", lambda _event: self._set_mode("settings"))

        self.root.update_idletasks()

    def _create_sliders(self, mode):
        """Recreate sliders for the active view, sized for the touchscreen."""
        for widget in self.sliders_frame.winfo_children():
            widget.destroy()

        if mode == "settings":
            self._create_settings_controls()
            return

        font = ("Helvetica", int(12 * self.font_scale))
        # Use the video container's allocated width as the slider budget when the
        # widget has already been drawn; otherwise fall back to the window.
        try:
            video_w = int(self._video_container.winfo_width())
        except Exception:
            video_w = 0
        if video_w <= 1:
            video_w = self.window_width

        # Reserve horizontal space for labels and padding; keep slider usable.
        label_width = 10 if self.touch_mode else 6
        slider_length = max(
            100,
            video_w - label_width * int(8 * self.font_scale) - self.padding * 4,
        )

        for _slider_name, label_text, group, key, limit_attr in SLIDER_DEFS[mode]:
            row = tk.Frame(self.sliders_frame)
            row.pack(fill=tk.X, pady=0)

            tk.Label(
                row,
                text=label_text,
                width=label_width,
                anchor=tk.W,
                font=font,
            ).pack(side=tk.LEFT, padx=(0, 1))

            max_val = getattr(self, limit_attr)
            line_binding = ConfigBinding(self.cfg, "lines", group, key, type_=int)
            var = tk.IntVar(master=self.root, value=line_binding.read())
            tk.Scale(
                row,
                from_=0,
                to=max_val,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda value, b=line_binding: b.write(value),
                showvalue=False,
                length=slider_length,
                width=max(6, self.slider_width),
                sliderlength=max(8, self.slider_height),
                font=font,
                bd=0,
                highlightthickness=0,
                troughcolor="#cccccc",
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, pady=0)

    def _create_settings_controls(self):
        """Build controls for runtime camera and model settings."""
        font = ("Helvetica", int(12 * self.font_scale))
        self._auto_exposure_vars = {}
        self._exposure_scales = {}

        for camera_name, title in (("bean", "BEAN CAMERA"), ("digit", "DIGIT CAMERA")):
            row = tk.Frame(self.sliders_frame)
            row.pack(fill=tk.X, pady=1)

            tk.Label(row, text=title, width=14, anchor=tk.W, font=font).pack(
                side=tk.LEFT, padx=(0, 2)
            )

            auto_binding = ConfigBinding(
                self.cfg, "cameras", camera_name, "auto_exposure", type_=bool,
            ).on_change(lambda v, n=camera_name: self._apply_auto_exposure(n, v))

            auto_var = tk.BooleanVar(master=self.root, value=auto_binding.read())
            self._auto_exposure_vars[camera_name] = auto_var
            tk.Checkbutton(
                row,
                text="AUTO",
                variable=auto_var,
                command=lambda b=auto_binding, v=auto_var: b.write(v.get()),
                font=font,
            ).pack(side=tk.LEFT, padx=(0, 8))

            tk.Label(row, text="EXPOSURE", font=font).pack(side=tk.LEFT, padx=(0, 2))

            exp_binding = ConfigBinding(
                self.cfg, "cameras", camera_name, "exposure", type_=float,
            ).on_change(lambda v, n=camera_name: self._apply_exposure(n, v))

            exposure_var = tk.IntVar(
                master=self.root, value=int(exp_binding.read()),
            )
            scale = tk.Scale(
                row,
                from_=1,
                to=10000,
                resolution=1,
                orient=tk.HORIZONTAL,
                variable=exposure_var,
                font=font,
                command=lambda current, b=exp_binding: b.write(current),
                showvalue=True,
                length=max(180, self.window_width - 260),
                width=max(6, self.slider_width),
                sliderlength=max(8, self.slider_height),
                bd=0,
                highlightthickness=0,
                troughcolor="#cccccc",
            )
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._exposure_scales[camera_name] = scale

        for model_name, title in (("bean", "BEAN CONF"), ("digit", "DIGIT CONF")):
            row = tk.Frame(self.sliders_frame)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=title, width=14, anchor=tk.W, font=font).pack(
                side=tk.LEFT, padx=(0, 2)
            )

            conf_binding = ConfigBinding(
                self.cfg, "models", model_name, "conf_thres", default=0.7, type_=float,
            ).on_change(lambda v, n=model_name: self._apply_confidence(n, v))

            var = tk.DoubleVar(master=self.root, value=conf_binding.read())
            tk.Scale(
                row,
                from_=0.05,
                to=0.95,
                resolution=0.05,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda current, b=conf_binding: b.write(current),
                showvalue=True,
                length=max(180, self.window_width - 180),
                width=max(6, self.slider_width),
                sliderlength=max(8, self.slider_height),
                font=font,
                bd=0,
                highlightthickness=0,
                troughcolor="#cccccc",
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── 逻辑补全推断上限 ──────────────────────────────────────────
        for logic_key, title, max_val in (
            ("max_inferred_beans", "BEAN INFER MAX", 3),
            ("max_inferred_digits", "DIGIT INFER MAX", 5),
        ):
            row = tk.Frame(self.sliders_frame)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=title, width=14, anchor=tk.W, font=font).pack(
                side=tk.LEFT, padx=(0, 2)
            )

            logic_binding = ConfigBinding(
                self.cfg, "logic", logic_key, default=1, type_=int,
            )
            var = tk.IntVar(master=self.root, value=logic_binding.read())
            tk.Scale(
                row,
                from_=0,
                to=max_val,
                resolution=1,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda current, b=logic_binding: b.write(current),
                showvalue=True,
                length=max(180, self.window_width - 180),
                width=max(6, self.slider_width),
                sliderlength=max(8, self.slider_height),
                font=font,
                bd=0,
                highlightthickness=0,
                troughcolor="#cccccc",
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._refresh_exposure_control_states()

    def _camera_for_name(self, camera_name):
        return self.cam_bean if camera_name == "bean" else self.cam_digit

    # ── side-effect callbacks for ConfigBinding ──────────────────────

    def _apply_auto_exposure(self, camera_name, value):
        """Push auto-exposure to the live camera and refresh UI state."""
        setter = getattr(self._camera_for_name(camera_name), "set_auto_exposure", None)
        if setter is not None:
            setter(value)
        self._refresh_exposure_control_states()

    def _apply_exposure(self, camera_name, value):
        """Push exposure value to the live camera when auto is off."""
        if not self.cfg["cameras"][camera_name].get("auto_exposure", True):
            setter = getattr(self._camera_for_name(camera_name), "set_exposure", None)
            if setter is not None:
                setter(value)

    def _apply_confidence(self, model_name, value):
        """Push confidence threshold to the live model object."""
        model = self.bean_model if model_name == "bean" else self.digit_model
        if model is not None:
            model.conf_thres = round(float(value), 2)

    def _refresh_exposure_control_states(self):
        for camera_name, scale in self._exposure_scales.items():
            auto_var = self._auto_exposure_vars[camera_name]
            scale.config(state=tk.DISABLED if auto_var.get() else tk.NORMAL)

    def _set_settings_layout(self, enabled):
        if enabled == self._settings_layout_active:
            return
        self._settings_layout_active = enabled
        if enabled:
            self._video_container.grid_remove()
            self._control_frame.grid(row=0, column=0, sticky="nsew")
            self.sliders_frame.pack_configure(fill=tk.BOTH, expand=True)
        else:
            self._control_frame.grid(row=1, column=0, sticky="ew")
            self._video_container.grid()
            self.sliders_frame.pack_configure(fill=tk.X, expand=False)

    def _set_mode(self, mode):
        if mode == self.mode:
            return
        super()._set_mode(mode)
        # Clear stale detection results so the new view doesn't briefly show
        # overlay from a previous calibration session.
        if mode in self._latest_dets:
            with self._result_lock:
                self._latest_dets[mode] = None
                self._latest_names[mode] = None
                self._latest_capture_meta[mode] = None
        self._set_settings_layout(mode == "settings")
        self._update_button_appearance()
        self._create_sliders(mode)

    def _update_button_appearance(self):
        if self.mode == "bean":
            self._bean_button.config(bg="#00aa00", relief=tk.SUNKEN)
            self._digit_button.config(bg="#555555", relief=tk.RAISED)
            self._settings_button.config(bg="#555555", relief=tk.RAISED)
        elif self.mode == "digit":
            self._bean_button.config(bg="#555555", relief=tk.RAISED)
            self._digit_button.config(bg="#0066cc", relief=tk.SUNKEN)
            self._settings_button.config(bg="#555555", relief=tk.RAISED)
        else:
            self._bean_button.config(bg="#555555", relief=tk.RAISED)
            self._digit_button.config(bg="#555555", relief=tk.RAISED)
            self._settings_button.config(bg="#0066cc", relief=tk.SUNKEN)


    @staticmethod
    def _class_name(names, cls_id):
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, (list, tuple)):
            if 0 <= cls_id < len(names):
                return names[cls_id]
        return str(cls_id)

    def _select_zone_detections(self, dets, mode):
        """Keep the highest-confidence detection from each configured zone."""
        if mode not in ("bean", "digit"):
            return dets
        best = ZoneLayout(self.cfg["lines"]).select_best(
            dets, mode, ignore_invalid=True
        )
        return [det for det in best.values() if det is not None]

    def _draw_detections(self, frame, dets, names, zone_results=None, mode=None):
        color_detected = (0, 255, 255)   # yellow
        color_inferred = (0, 140, 255)   # orange
        for det in self._select_zone_detections(dets, mode):
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            name = self._class_name(names, cls_id)
            label = f"{name} {conf:.2f}"

            # Check if this detection's zone was inferred
            color = color_detected
            if zone_results and mode:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                zone = ZoneLayout(self.cfg["lines"]).zone_for(mode, cx, cy)
                if zone is not None and zone in zone_results:
                    if zone_results[zone][1] == "inferred":
                        color = color_inferred
                        label = f"~{name} {conf:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                frame, label, (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

    def _draw_model_overlay(self, frame, model, error_y=30):
        if model is None:
            return
        try:
            dets = model.predict(frame)
            self._draw_detections(frame, dets, model.names)
        except Exception as e:
            logging.error("Model overlay inference failed: %s", e)
            cv2.putText(
                frame, f"infer error: {e}", (10, error_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

    def _update_frame(self):
        if not self._running:
            return

        # All calibration logic runs on the native-resolution frame;
        # only the displayed copy is scaled to fit the available container.
        try:
            avail_w = int(self._video_container.winfo_width())
            avail_h = int(self._video_container.winfo_height())
        except Exception:
            avail_w, avail_h = 0, 0
        if avail_w <= 1 or avail_h <= 1:
            avail_w, avail_h = self.window_width, self.video_max_height

        if self.mode == "bean":
            ok, frame = self.cam_bean.capture()
            model = self.bean_model
        else:
            ok, frame = self.cam_digit.capture()
            model = self.digit_model

        if not ok or frame is None:
            # Calibration runs on the Tk main thread, so capture health is
            # checked synchronously here; show a placeholder instead of
            # leaving the last frame frozen on screen.
            self._show_frame(
                self._render_placeholder(
                    f"CAMERA OFFLINE: {self.mode.upper()}", avail_w, avail_h
                )
            )
        else:
            if self.mode == "bean":
                self._draw_bean_lines(frame)
            else:
                self._draw_digit_lines(frame)

            # Run inference in the background so the Tkinter event loop stays
            # responsive to slider drags and button clicks.
            if model is not None:
                self._submit_frame_for_inference(self.mode, frame)
                with self._result_lock:
                    latest_dets = self._latest_dets[self.mode]
                    latest_names = self._latest_names[self.mode]
                if latest_dets is not None:
                    self._draw_detections(
                        frame, latest_dets, latest_names, mode=self.mode
                    )

            display_frame = self._resize_for_display(
                frame, max_w=avail_w, max_h=avail_h
            )
            self._show_frame(display_frame)

        if self._running:
            self.root.after(30, self._update_frame)

    def run(self):
        if not self._window_ok:
            raise RuntimeError(
                "GUI is not available; cannot run calibration UI on this machine"
            )

        print("Calibration UI started. Keys: 1=bean, 2=digit, f=fullscreen | Buttons: SAVE, EXIT")
        self._running = True
        self._start_inference_worker()
        self._update_frame()
        try:
            self.root.mainloop()
        finally:
            self._running = False
            self._stop_inference_worker()
            capture_workers_stopped = self._stop_capture_workers()
            if self._owns_cameras and capture_workers_stopped:
                self.cam_bean.release()
                self.cam_digit.release()
