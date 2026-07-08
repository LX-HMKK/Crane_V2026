import logging
import tkinter as tk

import cv2
import yaml
from PIL import Image, ImageTk

LINE_COLORS = {
    "bean_left_x": (0, 255, 0),
    "bean_right_x": (0, 255, 0),
    "digit_horizontal_y": (255, 0, 0),
    "digit_left_x": (255, 0, 0),
    "digit_right_x": (255, 0, 0),
}

SLIDER_DEFS = {
    "bean": [
        ("bean_left_x", "Bean left x", "bean", "left_x", "max_x"),
        ("bean_right_x", "Bean right x", "bean", "right_x", "max_x"),
    ],
    "digit": [
        ("digit_h_y", "Digit horizontal y", "digit", "horizontal_y", "max_y"),
        ("digit_left_x", "Digit left x", "digit", "left_x", "max_x"),
        ("digit_right_x", "Digit right x", "digit", "right_x", "max_x"),
    ],
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
        self.root.title("Crane Vision Calibration")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.bind("<KeyPress-1>", lambda _event: self._set_mode("bean"))
        self.root.bind("<KeyPress-2>", lambda _event: self._set_mode("digit"))
        self.root.bind("<Escape>", lambda _event: self._quit())

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(main_frame)
        self.video_label.pack()

        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        mode_frame = tk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=(0, 5))

        self._bean_button = tk.Button(
            mode_frame,
            text="BEAN (1)",
            command=lambda: self._set_mode("bean"),
            bg="#00aa00",
            fg="white",
        )
        self._bean_button.pack(side=tk.LEFT, padx=5)

        self._digit_button = tk.Button(
            mode_frame,
            text="DIGIT (2)",
            command=lambda: self._set_mode("digit"),
            bg="#555555",
            fg="white",
        )
        self._digit_button.pack(side=tk.LEFT, padx=5)

        self.sliders_frame = tk.Frame(control_frame)
        self.sliders_frame.pack(fill=tk.X)

        self._create_sliders(self.mode)

    def _create_sliders(self, mode):
        """Recreate sliders for the active view."""
        for widget in self.sliders_frame.winfo_children():
            widget.destroy()

        for _slider_name, label_text, group, key, limit_attr in SLIDER_DEFS[mode]:
            row = tk.Frame(self.sliders_frame)
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=label_text, width=20, anchor=tk.W).pack(side=tk.LEFT)

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
                length=400,
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
        h, w = frame.shape[:2]
        hy = self.cfg["lines"]["digit"]["horizontal_y"]
        left = self.cfg["lines"]["digit"]["left_x"]
        right = self.cfg["lines"]["digit"]["right_x"]
        cv2.line(frame, (0, hy), (w, hy), LINE_COLORS["digit_horizontal_y"], 2)
        cv2.line(frame, (left, 0), (left, h), LINE_COLORS["digit_left_x"], 2)
        cv2.line(frame, (right, 0), (right, h), LINE_COLORS["digit_right_x"], 2)

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
