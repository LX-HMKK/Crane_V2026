import logging
import queue
import threading
import tkinter as tk

import cv2
import yaml
from PIL import Image, ImageTk

from models.camera_viewer import CameraViewer, LINE_COLORS

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
    ):
        super().__init__(None, cfg_path, cam_bean, cam_digit, root=root)
        if not self._window_ok:
            return

        self.bean_model = bean_model
        self.digit_model = digit_model

        self._infer_queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest_dets = {"bean": None, "digit": None}
        self._latest_names = {"bean": None, "digit": None}
        self._infer_thread = None
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
        """Start a background thread for model inference if models are loaded."""
        if not self.inference_overlay:
            return
        if self.bean_model is None and self.digit_model is None:
            return
        self._infer_thread = threading.Thread(
            target=self._inference_worker, daemon=True
        )
        self._infer_thread.start()

    def _stop_inference_worker(self):
        """Signal the inference worker to stop and wait for it to finish."""
        if self._infer_thread is None or not self._infer_thread.is_alive():
            return
        # Wake the worker if it is blocked waiting for a frame.
        try:
            self._infer_queue.put_nowait(None)
        except queue.Full:
            pass
        self._infer_thread.join(timeout=1.0)
        self._infer_thread = None

    def _inference_worker(self):
        """Consume frames from the queue and run inference in the background."""
        while self._running:
            try:
                item = self._infer_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            mode, frame = item
            if not self._running:
                break
            model = self.bean_model if mode == "bean" else self.digit_model
            if model is None:
                continue
            try:
                dets = model.predict(frame)
                with self._result_lock:
                    self._latest_dets[mode] = dets
                    self._latest_names[mode] = model.names
                self._on_inference_done(mode, dets)
            except Exception as exc:
                logging.error("Inference worker error (%s): %s", mode, exc)

    def _on_inference_done(self, mode, dets):
        """Hook called after each inference completes (in worker thread).

        Override in subclasses to trigger task generation, serial updates, etc.
        """

    def _submit_frame_for_inference(self, mode, frame):
        """Hand the latest frame to the inference worker without blocking the UI."""
        if not self.inference_overlay:
            return
        if self.bean_model is None and self.digit_model is None:
            return
        try:
            self._infer_queue.put_nowait((mode, frame.copy()))
        except queue.Full:
            # The worker is still busy; replace the stale pending frame.
            try:
                self._infer_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._infer_queue.put_nowait((mode, frame.copy()))
            except queue.Full:
                pass

    def _build_controls(self):
        """Add control panel below the video area created by CameraViewer."""
        control_frame = tk.Frame(self._video_container.master)
        control_frame.grid(row=1, column=0, sticky="ew")
        self._control_frame = control_frame

        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 1))

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
            var = tk.IntVar(master=self.root, value=self.cfg["lines"][group][key])
            tk.Scale(
                row,
                from_=0,
                to=max_val,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda value, g=group, k=key: self._set_line(g, k, int(value)),
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
            camera_cfg = self.cfg["cameras"][camera_name]
            row = tk.Frame(self.sliders_frame)
            row.pack(fill=tk.X, pady=1)

            tk.Label(row, text=title, width=14, anchor=tk.W, font=font).pack(
                side=tk.LEFT, padx=(0, 2)
            )
            auto_var = tk.BooleanVar(
                master=self.root,
                value=bool(camera_cfg.get("auto_exposure", True)),
            )
            self._auto_exposure_vars[camera_name] = auto_var
            tk.Checkbutton(
                row,
                text="AUTO",
                variable=auto_var,
                command=lambda name=camera_name: self._set_camera_auto_exposure(
                    name, self._auto_exposure_vars[name].get()
                ),
                font=font,
            ).pack(side=tk.LEFT, padx=(0, 8))

            tk.Label(row, text="EXPOSURE", font=font).pack(side=tk.LEFT, padx=(0, 2))
            exposure_var = tk.IntVar(
                master=self.root,
                value=int(camera_cfg.get("exposure", 1)),
            )
            scale = tk.Scale(
                row,
                from_=1,
                to=10000,
                resolution=1,
                orient=tk.HORIZONTAL,
                variable=exposure_var,
                font=font,
                command=lambda current, name=camera_name: self._set_camera_exposure(
                    name, current
                ),
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
            value = float(self.cfg["models"][model_name].get("conf_thres", 0.7))
            var = tk.DoubleVar(master=self.root, value=value)
            tk.Scale(
                row,
                from_=0.05,
                to=0.95,
                resolution=0.05,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda current, name=model_name: self._set_model_confidence(
                    name, current
                ),
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
            logic_cfg = self.cfg.get("logic", {})
            value = int(logic_cfg.get(logic_key, 1))
            var = tk.IntVar(master=self.root, value=value)
            tk.Scale(
                row,
                from_=0,
                to=max_val,
                resolution=1,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda current, k=logic_key: self._set_logic_limit(k, current),
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

    def _set_camera_auto_exposure(self, camera_name, enabled):
        enabled = bool(enabled)
        self.cfg["cameras"][camera_name]["auto_exposure"] = enabled
        setter = getattr(self._camera_for_name(camera_name), "set_auto_exposure", None)
        if setter is not None:
            setter(enabled)
        self._refresh_exposure_control_states()

    def _set_camera_exposure(self, camera_name, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            logging.warning("Invalid %s camera exposure value: %r", camera_name, value)
            return

        self.cfg["cameras"][camera_name]["exposure"] = value
        if not self.cfg["cameras"][camera_name].get("auto_exposure", True):
            setter = getattr(self._camera_for_name(camera_name), "set_exposure", None)
            if setter is not None:
                setter(value)

    def _set_model_confidence(self, model_name, value):
        value = round(float(value), 2)
        self.cfg["models"][model_name]["conf_thres"] = value
        model = self.bean_model if model_name == "bean" else self.digit_model
        if model is not None:
            model.conf_thres = value

    def _set_logic_limit(self, key, value):
        """Update logic completion threshold in config."""
        self.cfg.setdefault("logic", {})[key] = int(value)

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

    def _set_line(self, group, key, value):
        # Update in-memory config only — write to disk happens on explicit SAVE.
        self.cfg["lines"][group][key] = int(value)

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

        if self.mode == "bean":
            ok, frame = self.cam_bean.capture()
            model = self.bean_model
        else:
            ok, frame = self.cam_digit.capture()
            model = self.digit_model

        if ok and frame is not None:
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

            # All calibration logic runs on the native-resolution frame;
            # only the displayed copy is scaled to fit the available container.
            try:
                avail_w = int(self._video_container.winfo_width())
                avail_h = int(self._video_container.winfo_height())
            except Exception:
                avail_w, avail_h = 0, 0
            if avail_w <= 1 or avail_h <= 1:
                avail_w, avail_h = self.window_width, self.video_max_height

            display_frame = self._resize_for_display(
                frame, max_w=avail_w, max_h=avail_h
            )

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

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
            self.cam_bean.release()
            self.cam_digit.release()


class CompetitionUI(CalibrationUI):
    """Competition-mode UI: calibration viewer + serial + task generation.

    Extends CalibrationUI with a SerialThread and a post-inference hook that
    runs TaskGenerator.generate_frame() and updates the serial thread.

    Unlike the calibration UI, both cameras are captured on every frame cycle
    so that generate_frame always has fresh detections from both views.
    """

    def __init__(self, cfg_path, cam_bean, cam_digit, bean_model, digit_model,
                 task_gen=None, serial_thread=None):
        super().__init__(cfg_path, cam_bean, cam_digit, bean_model, digit_model)
        if not self._window_ok:
            return
        self.task_gen = task_gen
        self.serial = serial_thread
        self._last_frame = None
        self._consecutive_failures = 0
        self._zone_results = {"bean": {}, "digit": {}}
        if self.serial is not None:
            self.serial.start()

    # ── dual-camera capture ────────────────────────────────────────────

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

    def _update_frame(self):
        """Capture both cameras every cycle so the competition always has fresh
        detections from both views for generate_frame()."""
        if not self._running:
            return

        ok_b, frame_b = self.cam_bean.capture()
        ok_d, frame_d = self.cam_digit.capture()

        if self.mode == "settings":
            # Keep detection and serial output live while the settings page
            # intentionally hides both camera previews.
            if ok_b and frame_b is not None and self.bean_model is not None:
                self._submit_frame_for_inference("bean", frame_b)
            if ok_d and frame_d is not None and self.digit_model is not None:
                self._submit_frame_for_inference("digit", frame_d)
            if self._running:
                self.root.after(30, self._update_frame)
            return

        with self._result_lock:
            zone_results_b = dict(self._zone_results.get("bean", {}))
            dets_b = self._latest_dets.get("bean")
            names_b = self._latest_names.get("bean")
            zone_results_d = dict(self._zone_results.get("digit", {}))
            dets_d = self._latest_dets.get("digit")
            names_d = self._latest_names.get("digit")

        if ok_b and frame_b is not None:
            self._draw_bean_lines(frame_b)
            if self.bean_model is not None:
                self._submit_frame_for_inference("bean", frame_b)
                if dets_b is not None:
                    self._draw_detections(
                        frame_b, dets_b, names_b,
                        zone_results=zone_results_b, mode="bean",
                    )
            self._draw_inferred_zone_labels(frame_b, zone_results_b, "bean")

        if ok_d and frame_d is not None:
            self._draw_digit_lines(frame_d)
            if self.digit_model is not None:
                self._submit_frame_for_inference("digit", frame_d)
                if dets_d is not None:
                    self._draw_detections(
                        frame_d, dets_d, names_d,
                        zone_results=zone_results_d, mode="digit",
                    )
            self._draw_inferred_zone_labels(frame_d, zone_results_d, "digit")

        # Show the active-mode camera, or bean camera by default.
        if self.mode == "digit" and ok_d and frame_d is not None:
            display_frame = frame_d
        elif ok_b and frame_b is not None:
            display_frame = frame_b
        elif ok_d and frame_d is not None:
            display_frame = frame_d
        else:
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

        display_frame = self._resize_for_display(
            display_frame, max_w=avail_w, max_h=avail_h
        )
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        if self._running:
            self.root.after(30, self._update_frame)

    # ── task generation hook ───────────────────────────────────────────

    def _on_inference_done(self, mode, dets):
        """After each inference, try to generate a task frame from both cameras."""
        if self.task_gen is None:
            return

        with self._result_lock:
            bean_dets = self._latest_dets.get("bean")
            digit_dets = self._latest_dets.get("digit")

        if bean_dets is None or digit_dets is None:
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
            self._consecutive_failures = 0
            with self._result_lock:
                self._zone_results["bean"] = bean_results
                self._zone_results["digit"] = digit_results
            if frame != self._last_frame:
                if self.serial is not None:
                    self.serial.current_frame = frame
                self._last_frame = frame
                inferred = sum(
                    1 for r in (bean_results | digit_results).values()
                    if r[1] == "inferred"
                )
                logging.info(
                    "Result updated: C1-C3=%s, N4-N8=%s (%d inferred)",
                    frame[:3], frame[3:], inferred,
                )
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 10:
                if self.serial is not None:
                    self.serial.current_frame = None
                logging.error(
                    "generate_frame failed %d times, clearing result: %s",
                    self._consecutive_failures, reason,
                )
            else:
                logging.warning(
                    "generate_frame failed (%d/%d): %s",
                    self._consecutive_failures, 10, reason,
                )

    # ── lifecycle ──────────────────────────────────────────────────────

    def _quit(self):
        self._running = False
        self._stop_inference_worker()
        if self.serial is not None:
            self.serial.stop()
        self.root.destroy()
