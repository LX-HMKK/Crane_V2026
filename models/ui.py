import cv2
import yaml
import numpy as np

LINE_COLORS = {
    "bean_left_x": (0, 255, 0),
    "bean_right_x": (0, 255, 0),
    "digit_horizontal_y": (255, 0, 0),
    "digit_left_x": (255, 0, 0),
    "digit_right_x": (255, 0, 0),
}


class CalibrationUI:
    """OpenCV-based UI to adjust guide lines and inspect live inference."""

    def __init__(self, cfg_path, cam_bean, cam_digit, bean_model=None, digit_model=None):
        self.cfg_path = cfg_path
        self.cam_bean = cam_bean
        self.cam_digit = cam_digit
        self.bean_model = bean_model
        self.digit_model = digit_model
        self.mode = "bean"
        self._window_ok = False
        self._load_cfg()
        self._resolve_trackbar_limits()
        self._init_window()

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
        return (
            int(w) if w is not None else default_w,
            int(h) if h is not None else default_h,
        )

    def _resolve_trackbar_limits(self):
        bean_w, bean_h = self._camera_resolution(self.cam_bean)
        digit_w, digit_h = self._camera_resolution(self.cam_digit)
        self.max_x = max(bean_w, digit_w)
        self.max_y = max(bean_h, digit_h)

    def _init_window(self):
        try:
            cv2.namedWindow("calibration")
            cv2.createTrackbar(
                "bean_left_x", "calibration",
                self.cfg["lines"]["bean"]["left_x"], self.max_x,
                lambda v: self._set_line("bean", "left_x", v),
            )
            cv2.createTrackbar(
                "bean_right_x", "calibration",
                self.cfg["lines"]["bean"]["right_x"], self.max_x,
                lambda v: self._set_line("bean", "right_x", v),
            )
            cv2.createTrackbar(
                "digit_h_y", "calibration",
                self.cfg["lines"]["digit"]["horizontal_y"], self.max_y,
                lambda v: self._set_line("digit", "horizontal_y", v),
            )
            cv2.createTrackbar(
                "digit_left_x", "calibration",
                self.cfg["lines"]["digit"]["left_x"], self.max_x,
                lambda v: self._set_line("digit", "left_x", v),
            )
            cv2.createTrackbar(
                "digit_right_x", "calibration",
                self.cfg["lines"]["digit"]["right_x"], self.max_x,
                lambda v: self._set_line("digit", "right_x", v),
            )
            self._window_ok = True
        except cv2.error:
            # Headless/OpenCV built without GUI support; UI can still be unit-tested.
            self._window_ok = False

    def _set_line(self, group, key, value):
        self.cfg["lines"][group][key] = value
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
        cv2.line(frame, (left, hy), (left, h), LINE_COLORS["digit_left_x"], 2)
        cv2.line(frame, (right, hy), (right, h), LINE_COLORS["digit_right_x"], 2)

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
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _draw_model_overlay(self, frame, model, error_y=30):
        if model is None:
            return
        try:
            dets = model.predict(frame)
            self._draw_detections(frame, dets, model.names)
        except Exception as e:
            cv2.putText(
                frame, f"infer error: {e}", (10, error_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

    def run(self):
        if not self._window_ok:
            raise RuntimeError(
                "OpenCV GUI is not available; cannot run calibration UI on this machine"
            )

        print("Calibration UI started. Keys: 1=bean camera, 2=digit camera, ESC=exit")
        try:
            while True:
                if self.mode == "bean":
                    ok, frame = self.cam_bean.capture()
                    if not ok:
                        continue
                    self._draw_bean_lines(frame)
                    self._draw_model_overlay(frame, self.bean_model, error_y=30)
                    label = "BEAN (1/2 switch, ESC exit)"
                else:
                    ok, frame = self.cam_digit.capture()
                    if not ok:
                        continue
                    self._draw_digit_lines(frame)
                    self._draw_model_overlay(frame, self.digit_model, error_y=50)
                    label = "DIGIT (1/2 switch, ESC exit)"

                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("calibration", frame)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord("1"):
                    self.mode = "bean"
                elif key == ord("2"):
                    self.mode = "digit"
        finally:
            self.cam_bean.release()
            self.cam_digit.release()
            cv2.destroyAllWindows()
