import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pytest
import yaml
from unittest.mock import MagicMock

from models.ui import CalibrationUI


def _write_cfg(tmp_path, values=None):
    cfg_path = tmp_path / "vision.yaml"
    cfg = {
        "lines": {
            "bean": {"left_x": 100, "right_x": 300},
            "digit": {"horizontal_y": 200, "left_x": 100, "right_x": 300},
        },
        "inference": {"imgsz": 640},
    }
    if values:
        cfg["lines"].update(values)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return cfg_path


def test_ui_saves_lines_on_change(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    ui._set_line("bean", "left_x", 150)
    ui._set_line("digit", "horizontal_y", 250)

    with open(cfg_path, "r") as f:
        saved = yaml.safe_load(f)
    assert saved["lines"]["bean"]["left_x"] == 150
    assert saved["lines"]["digit"]["horizontal_y"] == 250


def test_draw_bean_lines_smoke(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ui._draw_bean_lines(frame)


def test_draw_digit_lines_smoke(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ui._draw_digit_lines(frame)


@pytest.mark.parametrize("names", [{0: "bean", 1: "green"}, ["bean", "green"]])
def test_draw_detections_accepts_dict_and_list_names(tmp_path, names):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = np.array([[50, 50, 100, 100, 0.9, 1]])
    ui._draw_detections(frame, dets, names)


def test_draw_model_overlay_runs_infer_and_draws(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    model = MagicMock()
    model.predict.return_value = [[10, 10, 40, 40, 0.8, 0]]
    model.names = {0: "bean"}
    ui._draw_model_overlay(frame, model)
    model.predict.assert_called_once_with(frame)


def test_draw_model_overlay_shows_error_on_failure(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    model = MagicMock()
    model.predict.side_effect = RuntimeError("boom")
    ui._draw_model_overlay(frame, model)


def test_init_window_headless_fallback(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)

    def raise_cv2_error(*args, **kwargs):
        raise cv2.error("no GUI support")

    monkeypatch.setattr(cv2, "namedWindow", raise_cv2_error)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    assert ui._window_ok is False


def test_run_raises_when_window_unavailable(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    ui._window_ok = False
    with pytest.raises(RuntimeError, match="OpenCV GUI"):
        ui.run()


def test_run_releases_cameras_on_exit(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cam_bean.capture.return_value = (True, frame)
    cam_digit.capture.return_value = (True, frame)

    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    ui._window_ok = True

    monkeypatch.setattr(cv2, "imshow", lambda win, img: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 27)

    ui.run()
    cam_bean.release.assert_called_once()
    cam_digit.release.assert_called_once()


def test_trackbar_limits_derived_from_camera_resolution(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_bean.width = 800
    cam_bean.height = 600
    cam_digit = MagicMock()
    cam_digit.width = 640
    cam_digit.height = 480

    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    assert ui.max_x == 800
    assert ui.max_y == 600
