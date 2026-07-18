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
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f, sort_keys=False, allow_unicode=True)

    def _save_with_feedback(self):
        """Write config to disk and briefly show confirmation on the save button."""
        self._save_cfg()
        if hasattr(self, "_save_button") and self._save_button.winfo_exists():
            self._save_button.config(text="SAVED ✓", bg="#006600")
            self.root.after(1500, self._restore_save_button_text)

    def _restore_save_button_text(self):
        if hasattr(self, "_save_button") and self._save_button.winfo_exists():
            self._save_button.config(text="SAVE", bg="#0066cc")

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
            except Exception as exc:
                logging.error("Inference worker error (%s): %s", mode, exc)

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
            height=1,
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
            height=1,
        )
        self._digit_button.pack(side=tk.LEFT, padx=1)

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
            height=1,
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
            height=1,
        )
        self._save_button.pack(side=tk.RIGHT, padx=1)

        self.sliders_frame = tk.Frame(control_frame)
        self.sliders_frame.pack(fill=tk.X)

        self._create_sliders(self.mode)

        # Update the info bar to mention the touch-friendly buttons.
        self._info_label.config(text="1=BEAN  2=DIGIT  f=fullscreen  [SAVE] [EXIT]")

        self.root.update_idletasks()

    def _create_sliders(self, mode):
        """Recreate sliders for the active view, sized for the touchscreen."""
        for widget in self.sliders_frame.winfo_children():
            widget.destroy()

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

    def _set_mode(self, mode):
        if mode == self.mode:
            return
        super()._set_mode(mode)
        # Clear stale detection results so the new view doesn't briefly show
        # overlay from a previous calibration session.
        with self._result_lock:
            self._latest_dets[mode] = None
            self._latest_names[mode] = None
        self._update_button_appearance()
        self._create_sliders(mode)

    def _update_button_appearance(self):
        if self.mode == "bean":
            self._bean_button.config(bg="#00aa00", relief=tk.SUNKEN)
            self._digit_button.config(bg="#555555", relief=tk.RAISED)
        else:
            self._bean_button.config(bg="#555555", relief=tk.RAISED)
            self._digit_button.config(bg="#0066cc", relief=tk.SUNKEN)

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

    def _draw_detections(self, frame, dets, names):
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            name = self._class_name(names, cls_id)
            label = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(
                frame, label, (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
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
