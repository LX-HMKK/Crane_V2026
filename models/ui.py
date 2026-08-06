import logging
import queue
import sys
import threading
import time
import tkinter as tk

import cv2
import yaml
from PIL import Image, ImageTk

from models.camera_viewer import CameraViewer, LINE_COLORS
from models.config_binding import ConfigBinding
from models.runtime import inference_gate_open, serial_start_active

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
        self._infer_pending_cv = threading.Condition()
        self._result_lock = threading.Lock()
        self._latest_dets = {"bean": None, "digit": None}
        self._latest_names = {"bean": None, "digit": None}
        self._inference_rate_meter = _InferenceRateMeter()
        self._infer_threads = {}
        self._save_feedback_id = None
        self._settings_layout_active = False

        # Parent already created main_frame/video_container/video_label;
        # add the control panel below the video area.
        self._build_controls()

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
            self._infer_pending[mode] = None
            return frame

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
            frame = self._take_pending_inference_frame(mode)
            if frame is None:
                continue
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

    def _submit_frame_for_inference(self, mode, frame):
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
            self._infer_pending_cv.notify_all()

    def _build_controls(self):
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

    def _draw_detections(self, frame, dets, names, zone_results=None, mode=None):
        color_detected = (0, 255, 255)   # yellow
        color_inferred = (0, 140, 255)   # orange
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            name = self._class_name(names, cls_id)
            label = f"{name} {conf:.2f}"

            # Check if this detection's zone was inferred
            color = color_detected
            if zone_results and mode:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if mode == "bean" and hasattr(self, 'task_gen') and self.task_gen:
                    zone = self.task_gen._bean_zone(cx)
                elif mode == "digit" and hasattr(self, 'task_gen') and self.task_gen:
                    zone = self.task_gen._digit_zone(cx, cy)
                else:
                    zone = None
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
                    self._draw_detections(frame, latest_dets, latest_names)

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
        # Assigned before super().__init__() so the DEBUG button built during
        # _build_controls() can decide its enabled state from the serial ref.
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
        self._last_frame = None
        self._consecutive_failures = 0
        self._last_result_failure_log_at = None
        self._last_result_update_log_at = None
        self._task_lock = threading.Lock()
        try:
            self._task_update_interval = max(
                0.0,
                float(
                    self.cfg.get("protocol", {}).get(
                        "send_interval_sec", DEFAULT_TASK_UPDATE_INTERVAL_SEC
                    )
                ),
            )
        except (TypeError, ValueError):
            self._task_update_interval = DEFAULT_TASK_UPDATE_INTERVAL_SEC
        self._last_task_update_at = None
        self._zone_results = {"bean": {}, "digit": {}}
        self._capture_lock = threading.Lock()
        self._capture_stop = threading.Event()
        self._capture_threads = {}
        self._captured_frames = {"bean": None, "digit": None}
        self._capture_ready = {"bean": False, "digit": False}
        self._capture_failure_logged = {"bean": False, "digit": False}
        self._capture_sequences = {"bean": 0, "digit": 0}
        self._capture_failures = {"bean": 0, "digit": 0}
        if self.serial is not None and start_serial:
            self.serial.start()
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
            # Feed inference from the camera thread rather than the Tk event
            # loop. Rendering can run slower than capture without reducing
            # the rate at which each model receives its newest frame.
            self._submit_frame_for_inference(mode, frame)
            next_capture_at += interval

    def _invalidate_camera_detections(self, mode):
        """Clear one camera's stale detections once it is confirmed offline."""
        with self._result_lock:
            if self._latest_dets[mode] is None and self._latest_names[mode] is None:
                return
            self._latest_dets[mode] = None
            self._latest_names[mode] = None
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
        dirty = False
        with self._task_lock:
            if self._last_frame is not None:
                self._last_frame = None
                dirty = True
            if self._last_task_update_at is not None:
                self._last_task_update_at = None
                dirty = True
            if self._consecutive_failures != 0:
                self._consecutive_failures = 0
                dirty = True
            if self.serial is not None and self.serial.current_frame is not None:
                self.serial.current_frame = None
                dirty = True
        return dirty

    def _on_inference_gate_closed(self):
        """Reset inference/display/serial state for a gate-off (idle) period.

        Called from the capture-worker submit path whenever the inference
        gate is observed closed. Idempotent under duplicate/concurrent
        invocation (both camera workers hit it each capture tick). Lock order
        matches _on_inference_done: _task_lock -> _result_lock ->
        _infer_pending_cv.
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
                self._zone_results = {"bean": {}, "digit": {}}
                dirty = True
        with self._infer_pending_cv:
            if (
                self._infer_pending["bean"] is not None
                or self._infer_pending["digit"] is not None
            ):
                self._infer_pending = {"bean": None, "digit": None}
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

    def _camera_state(self, mode):
        """Return (state, frame) for one camera: ok / grace / waiting / offline."""
        with self._capture_lock:
            ready = self._capture_ready.get(mode, False)
            failures = self._capture_failures.get(mode, 0)
            frame = self._captured_frames.get(mode)
        if failures >= CAPTURE_OFFLINE_AFTER_FAILURES:
            return ("offline", None)
        if ready and frame is not None:
            return ("ok", frame)
        if frame is not None:
            return ("grace", frame)
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

    def _build_controls(self):
        """Extend the control bar with a DEBUG toggle on the right side."""
        super()._build_controls()
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
                    self._submit_frame_for_inference(mode, frame)

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
        if self.task_gen is None:
            return

        # Model workers may finish at the same time. Task state and serial
        # output must still be updated as one ordered transaction.
        debug_forced = self._debug_forced()
        with self._task_lock:
            now = time.monotonic()
            if (
                self._last_task_update_at is not None
                and now - self._last_task_update_at < self._task_update_interval
            ):
                return
            self._last_task_update_at = now
            self._update_task_from_latest_detections(debug_forced=debug_forced)

    def _update_task_from_latest_detections(self, debug_forced=False):
        """Generate and publish one task update while holding _task_lock.

        debug_forced=True keeps the result display-only: the UI overlay is
        updated but the serial cache and the competition failure counters are
        untouched, so toggling DEBUG can never poison the real output.
        """

        with self._result_lock:
            bean_dets = self._latest_dets.get("bean")
            digit_dets = self._latest_dets.get("digit")

        if bean_dets is None or digit_dets is None:
            return

        # A camera confirmed offline must not contribute its stale detections
        # to a new result frame; keep sending the last valid frame (sticky).
        if not (
            self._camera_capture_healthy("bean")
            and self._camera_capture_healthy("digit")
        ):
            return

        # Use logical completion to tolerate missing/duplicate zones
        logic_cfg = self.cfg.get("logic", {})
        frame, bean_results, digit_results, reason = \
            self.task_gen.generate_frame_with_completion(
                bean_dets, digit_dets,
                confidence_threshold=logic_cfg.get("confidence_threshold", 0.5),
                max_inferred_beans=logic_cfg.get("max_inferred_beans", 1),
                max_inferred_digits=logic_cfg.get("max_inferred_digits", 2),
            )

        if frame is not None:
            with self._result_lock:
                self._zone_results["bean"] = bean_results
                self._zone_results["digit"] = digit_results
            if debug_forced:
                # Display-only: keep the UI overlay fresh without touching the
                # serial cache or the competition failure counters.
                logging.debug("DEBUG inference produced a result (display-only)")
                return
            self._consecutive_failures = 0
            self._last_result_failure_log_at = None
            if frame != self._last_frame:
                if self.serial is not None:
                    self.serial.current_frame = frame
                self._last_frame = frame
                inferred = sum(
                    1 for r in (bean_results | digit_results).values()
                    if r[1] == "inferred"
                )
                self._log_result_update(frame, inferred)
        else:
            if debug_forced:
                logging.warning("DEBUG inference produced no result: %s", reason)
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= 10:
                if self.serial is not None:
                    self.serial.current_frame = None
                self._log_result_failure(reason)
            else:
                logging.warning(
                    "generate_frame failed (%d/%d): %s",
                    self._consecutive_failures, 10, reason,
                )

    def _log_result_failure(self, reason):
        now = time.monotonic()
        if (
            self._last_result_failure_log_at is not None
            and now - self._last_result_failure_log_at < RESULT_FAILURE_LOG_INTERVAL_SEC
        ):
            return
        logging.error(
            "generate_frame failed %d times, clearing result: %s",
            self._consecutive_failures,
            reason,
        )
        self._last_result_failure_log_at = now

    def _log_result_update(self, frame, inferred):
        """Keep normal logs useful without writing once per inference frame."""
        now = time.monotonic()
        if (
            self._last_result_update_log_at is not None
            and now - self._last_result_update_log_at < RESULT_UPDATE_LOG_INTERVAL_SEC
        ):
            return
        logging.info(
            "Result updated: C1-C3=%s, N4-N8=%s (%d inferred)",
            frame[:3], frame[3:], inferred,
        )
        self._last_result_update_log_at = now

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
