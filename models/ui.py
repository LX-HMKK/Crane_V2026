import logging
import tkinter as tk

import cv2
import yaml
from PIL import Image, ImageTk

LINE_COLORS = {
    "bean_left_x": (0, 255, 0),
    "bean_right_x": (0, 255, 0),
    "digit_x1": (255, 0, 0),
    "digit_x2": (255, 0, 0),
    "digit_x3": (255, 0, 0),
    "digit_x4": (255, 0, 0),
}

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
        "slider_width": 15,
        "slider_height": 20,
        "button_height": 30,
        "padding": 5,
        "font_scale": 1.0,
    },
}


class CalibrationUI:
    """Tkinter-based calibration UI.

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
        self.cfg_path = cfg_path
        self.cam_bean = cam_bean
        self.cam_digit = cam_digit
        self.bean_model = bean_model
        self.digit_model = digit_model
        self.mode = "bean"
        self._window_ok = True
        self._running = False

        self._load_cfg()
        self._apply_ui_cfg()
        self._resolve_trackbar_limits()

        if root is None:
            try:
                self.root = tk.Tk()
            except tk.TclError as exc:
                self._window_ok = False
                logging.warning("Tkinter GUI unavailable: %s", exc)
                return
        else:
            self.root = root

        self._build_ui()

    def _load_cfg(self):
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def _apply_ui_cfg(self):
        """Merge UI config with defaults and pre-compute derived values."""
        import copy

        ui_cfg = copy.deepcopy(DEFAULT_UI_CFG)
        if self.cfg and isinstance(self.cfg.get("ui"), dict):
            ui_cfg["window"].update(self.cfg["ui"].get("window", {}))
            ui_cfg["video"].update(self.cfg["ui"].get("video", {}))
            ui_cfg["controls"].update(self.cfg["ui"].get("controls", {}))
        self.ui_cfg = ui_cfg

        win = ui_cfg["window"]
        self.window_width = int(win["width"])
        self.window_height = int(win["height"])
        self.fullscreen = bool(win.get("fullscreen", False))

        vid = ui_cfg["video"]
        self.video_max_height = int(vid["max_height"])
        self.video_fit_mode = str(vid.get("fit_mode", "fit_width"))

        ctl = ui_cfg["controls"]
        self.touch_mode = bool(ctl.get("touch_mode", False))
        self.slider_width = int(ctl["slider_width"])
        self.slider_height = int(ctl["slider_height"])
        self.button_height = int(ctl["button_height"])
        self.padding = int(ctl["padding"])
        self.font_scale = float(ctl["font_scale"])

        self.display_w = self.window_width
        self.display_h = self.video_max_height
        self._configure_pending = None

    def _compute_display_size(self, native_w, native_h, max_w=None, max_h=None):
        """Return (display_w, display_h) that fits inside the given bounds.

        The returned size always keeps the native aspect ratio and never
        exceeds the available width or height.  When no bounds are supplied,
        the configured window width and video max height are used.
        """
        if native_w <= 0 or native_h <= 0:
            return max_w or self.window_width, max_h or self.video_max_height

        max_w = max_w or self.window_width
        max_h = max_h or self.video_max_height
        aspect = native_h / native_w

        # Try to fill the available width, then clamp by height if needed.
        display_w = max_w
        display_h = int(round(display_w * aspect))
        if display_h > max_h:
            display_h = max_h
            display_w = int(round(display_h / aspect))

        display_w = max(1, min(display_w, max_w))
        display_h = max(1, min(display_h, max_h))
        return display_w, display_h

    def _resize_for_display(self, frame, max_w=None, max_h=None):
        """Scale a native-resolution frame to the computed display size."""
        display_w, display_h = self._compute_display_size(
            frame.shape[1], frame.shape[0], max_w=max_w, max_h=max_h
        )
        if frame.shape[1] == display_w and frame.shape[0] == display_h:
            return frame
        return cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

    def _on_configure(self, _event=None):
        """Debounced handler for window resize events."""
        if not self._window_ok or not self._running:
            return
        if self._configure_pending is not None:
            self.root.after_cancel(self._configure_pending)
        self._configure_pending = self.root.after(200, self._apply_geometry)

    def _apply_geometry(self):
        """Recompute display size and recreate sliders to fit current window."""
        if not self._window_ok:
            return
        self.window_width = max(1, self.root.winfo_width())
        self.window_height = max(1, self.root.winfo_height())
        self._create_sliders(self.mode)

    def _toggle_fullscreen(self, _event=None):
        """Toggle fullscreen state, bound to the 'f' key for field debugging."""
        if not self._window_ok:
            return
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _save_cfg(self):
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f, sort_keys=False, allow_unicode=True)

    @staticmethod
    def _camera_resolution(cam, default_w=640, default_h=480):
        w = getattr(cam, "width", None)
        h = getattr(cam, "height", None)
        w = int(w) if w is not None and int(w) > 0 else default_w
        h = int(h) if h is not None and int(h) > 0 else default_h
        return w, h

    def _resolve_trackbar_limits(self):
        bean_w, bean_h = self._camera_resolution(self.cam_bean)
        digit_w, digit_h = self._camera_resolution(self.cam_digit)
        self.max_x = max(bean_w, digit_w)
        self.max_y = max(bean_h, digit_h)

    def _build_ui(self):
        # Opt-in to per-monitor DPI awareness on Windows so that window and
        # widget dimensions map 1:1 to physical pixels on high-DPI screens.
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        except Exception:
            pass

        self.root.title("Crane Vision Calibration")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.bind("<KeyPress-1>", lambda _event: self._set_mode("bean"))
        self.root.bind("<KeyPress-2>", lambda _event: self._set_mode("digit"))
        self.root.bind("<Escape>", lambda _event: self._quit())
        self.root.bind("<KeyPress-f>", self._toggle_fullscreen)
        self.root.bind("<Configure>", self._on_configure)

        if self.fullscreen:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry(f"{self.window_width}x{self.window_height}")
            self.root.minsize(640, 480)

        # Use a grid so the video area expands to fill leftover space while the
        # control panel keeps its natural height and is always visible.
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)  # video expands
        main_frame.grid_rowconfigure(1, weight=0)  # controls fixed
        main_frame.grid_columnconfigure(0, weight=1)

        # The video container fills the top cell.  The label inside it keeps the
        # image at its natural size and centers it; empty space shows as black.
        video_container = tk.Frame(main_frame, bg="black")
        video_container.grid(row=0, column=0, sticky="nsew")
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_container, bg="black")
        self.video_label.grid(row=0, column=0)
        self._video_container = video_container

        control_frame = tk.Frame(main_frame)
        control_frame.grid(row=1, column=0, sticky="ew")
        self._control_frame = control_frame

        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(0, self.padding))

        font = ("Helvetica", int(12 * self.font_scale))

        self._bean_button = tk.Button(
            mode_frame,
            text="BEAN (1)",
            command=lambda: self._set_mode("bean"),
            bg="#00aa00",
            fg="white",
            font=font,
            height=self.button_height // 20,
        )
        self._bean_button.pack(side=tk.LEFT, padx=self.padding)

        self._digit_button = tk.Button(
            mode_frame,
            text="DIGIT (2)",
            command=lambda: self._set_mode("digit"),
            bg="#555555",
            fg="white",
            font=font,
            height=self.button_height // 20,
        )
        self._digit_button.pack(side=tk.LEFT, padx=self.padding)

        self.sliders_frame = tk.Frame(control_frame)
        self.sliders_frame.pack(fill=tk.X)

        self._create_sliders(self.mode)
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
            row.pack(fill=tk.X, pady=self.padding // 2)

            tk.Label(
                row,
                text=label_text,
                width=label_width,
                anchor=tk.W,
                font=font,
            ).pack(side=tk.LEFT, padx=(0, self.padding))

            max_val = getattr(self, limit_attr)
            var = tk.IntVar(master=self.root, value=self.cfg["lines"][group][key])
            tk.Scale(
                row,
                from_=0,
                to=max_val,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda value, g=group, k=key: self._set_line(g, k, int(value)),
                showvalue=True,
                length=slider_length,
                width=self.slider_width,
                sliderlength=self.slider_height,
                font=font,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _set_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
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
        self.cfg["lines"][group][key] = int(value)
        self._save_cfg()

    def _draw_bean_lines(self, frame):
        h, w = frame.shape[:2]
        left = self.cfg["lines"]["bean"]["left_x"]
        right = self.cfg["lines"]["bean"]["right_x"]
        cv2.line(frame, (left, 0), (left, h), LINE_COLORS["bean_left_x"], 2)
        cv2.line(frame, (right, 0), (right, h), LINE_COLORS["bean_right_x"], 2)

    def _draw_digit_lines(self, frame):
        h = frame.shape[0]
        for key in ("x1", "x2", "x3", "x4"):
            x = self.cfg["lines"]["digit"][key]
            cv2.line(frame, (x, 0), (x, h), LINE_COLORS[f"digit_{key}"], 2)

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
            self._draw_model_overlay(frame, model, error_y=30)

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

    def _quit(self):
        self._running = False
        self.root.destroy()

    def run(self):
        if not self._window_ok:
            raise RuntimeError(
                "GUI is not available; cannot run calibration UI on this machine"
            )

        print("Calibration UI started. Keys: 1=bean camera, 2=digit camera, ESC=exit")
        self._running = True
        self._update_frame()
        try:
            self.root.mainloop()
        finally:
            self._running = False
            self.cam_bean.release()
            self.cam_digit.release()
