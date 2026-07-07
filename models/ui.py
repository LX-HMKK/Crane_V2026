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
        self._load_cfg()
        self._init_window()

    def _load_cfg(self):
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def _save_cfg(self):
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f, sort_keys=False, allow_unicode=True)

    def _init_window(self):
        try:
            cv2.namedWindow("calibration")
            cv2.createTrackbar("bean_left_x", "calibration", self.cfg["lines"]["bean"]["left_x"], 640, lambda v: self._set_line("bean", "left_x", v))
            cv2.createTrackbar("bean_right_x", "calibration", self.cfg["lines"]["bean"]["right_x"], 640, lambda v: self._set_line("bean", "right_x", v))
            cv2.createTrackbar("digit_h_y", "calibration", self.cfg["lines"]["digit"]["horizontal_y"], 480, lambda v: self._set_line("digit", "horizontal_y", v))
            cv2.createTrackbar("digit_left_x", "calibration", self.cfg["lines"]["digit"]["left_x"], 640, lambda v: self._set_line("digit", "left_x", v))
            cv2.createTrackbar("digit_right_x", "calibration", self.cfg["lines"]["digit"]["right_x"], 640, lambda v: self._set_line("digit", "right_x", v))
        except cv2.error:
            # Headless/OpenCV built without GUI support; UI can still be unit-tested.
            pass

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

    def _draw_detections(self, frame, dets, names):
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            name = names.get(cls_id, str(cls_id))
            label = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def run(self):
        print("Calibration UI started. Keys: 1=bean camera, 2=digit camera, ESC=exit")
        while True:
            if self.mode == "bean":
                ok, frame = self.cam_bean.capture()
                if not ok:
                    continue
                self._draw_bean_lines(frame)
                if self.bean_model is not None:
                    try:
                        dets = self.bean_model.predict(frame)
                        self._draw_detections(frame, dets, self.bean_model.names)
                    except Exception as e:
                        cv2.putText(frame, f"infer error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                label = "BEAN (1/2 switch, ESC exit)"
            else:
                ok, frame = self.cam_digit.capture()
                if not ok:
                    continue
                self._draw_digit_lines(frame)
                if self.digit_model is not None:
                    try:
                        dets = self.digit_model.predict(frame)
                        self._draw_detections(frame, dets, self.digit_model.names)
                    except Exception as e:
                        cv2.putText(frame, f"infer error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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

        cv2.destroyAllWindows()
