import logging
import tkinter as tk

import cv2
import yaml
from PIL import Image, ImageTk


DEFAULT_VIEWER_CFG = {
    "window": {"fullscreen": False, "width": 1024, "height": 600},
    "video": {"max_height": 420, "fit_mode": "contain"},
    "controls": {
        "touch_mode": False,
        "font_scale": 0.8,
    },
}

LINE_COLORS = {
    "bean_left_x": (0, 255, 0),
    "bean_right_x": (0, 255, 0),
    "digit_x1": (255, 0, 0),
    "digit_x2": (255, 0, 0),
    "digit_x3": (255, 0, 0),
    "digit_x4": (255, 0, 0),
}


class CameraViewer:
    """Minimal dual-camera viewer with guide lines.

    No sliders, no config saving, no inference. Designed for standalone
    camera placement verification.
    """

    def __init__(self, cfg, cfg_path, cam_bean, cam_digit, root=None):
        self.cfg_path = cfg_path
        self.cfg = cfg
        self.cam_bean = cam_bean
        self.cam_digit = cam_digit
        self.mode = "bean"
        self._window_ok = True
        self._running = False
        self._configure_pending = None

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
        if self.cfg is not None:
            return
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def _apply_ui_cfg(self):
        import copy

        ui_cfg = copy.deepcopy(DEFAULT_VIEWER_CFG)
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
        self.font_scale = float(ctl.get("font_scale", 0.8))

        self.display_w = self.window_width
        self.display_h = self.video_max_height

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

    def _compute_display_size(self, native_w, native_h, max_w=None, max_h=None):
        if native_w <= 0 or native_h <= 0:
            return max_w or self.window_width, max_h or self.video_max_height

        max_w = max_w or self.window_width
        max_h = max_h or self.video_max_height
        aspect = native_h / native_w

        display_w = max_w
        display_h = int(round(display_w * aspect))
        if display_h > max_h:
            display_h = max_h
            display_w = int(round(display_h / aspect))

        display_w = max(1, min(display_w, max_w))
        display_h = max(1, min(display_h, max_h))
        return display_w, display_h

    def _resize_for_display(self, frame, max_w=None, max_h=None):
        display_w, display_h = self._compute_display_size(
            frame.shape[1], frame.shape[0], max_w=max_w, max_h=max_h
        )
        if frame.shape[1] == display_w and frame.shape[0] == display_h:
            return frame
        return cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

    def _on_configure(self, _event=None):
        if not self._window_ok or not self._running:
            return
        if self._configure_pending is not None:
            self.root.after_cancel(self._configure_pending)
        self._configure_pending = self.root.after(200, self._apply_geometry)

    def _apply_geometry(self):
        if not self._window_ok:
            return
        self.window_width = max(1, self.root.winfo_width())
        self.window_height = max(1, self.root.winfo_height())

    def _toggle_fullscreen(self, _event=None):
        if not self._window_ok:
            return
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _build_ui(self):
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        except Exception:
            pass

        self.root.title("Crane Vision Camera Viewer")
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

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        video_container = tk.Frame(main_frame, bg="black")
        video_container.grid(row=0, column=0, sticky="nsew")
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_container, bg="black")
        self.video_label.grid(row=0, column=0)
        self._video_container = video_container

        info_frame = tk.Frame(main_frame)
        info_frame.grid(row=1, column=0, sticky="ew")

        font = ("Helvetica", int(12 * self.font_scale))
        self._info_label = tk.Label(
            info_frame,
            text="1=BEAN  2=DIGIT  ESC=exit  f=fullscreen",
            font=font,
        )
        self._info_label.pack(side=tk.LEFT, padx=5, pady=2)

    def _set_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode

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

    def _update_frame(self):
        if not self._running:
            return

        if self.mode == "bean":
            ok, frame = self.cam_bean.capture()
        else:
            ok, frame = self.cam_digit.capture()

        if ok and frame is not None:
            if self.mode == "bean":
                self._draw_bean_lines(frame)
            else:
                self._draw_digit_lines(frame)

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
                "GUI is not available; cannot run camera viewer on this machine"
            )

        print("Camera viewer started. Keys: 1=bean camera, 2=digit camera, ESC=exit")
        self._running = True
        self._update_frame()
        try:
            self.root.mainloop()
        finally:
            self._running = False
            self.cam_bean.release()
            self.cam_digit.release()
