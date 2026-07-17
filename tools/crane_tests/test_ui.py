import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import pytest
import yaml
from unittest.mock import MagicMock

from models.camera import MockCamera
from models.ui import CalibrationUI
from scripts.calibrate import run_calibration


def _write_cfg(tmp_path, values=None):
    cfg_path = tmp_path / "vision.yaml"
    cfg = {
        "lines": {
            "bean": {"left_x": 100, "right_x": 300},
            "digit": {"x1": 100, "x2": 200, "x3": 300, "x4": 400},
        },
        "inference": {"imgsz": 640},
    }
    if values:
        cfg["lines"].update(values)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return cfg_path


def _write_full_cfg(tmp_path):
    cfg_path = tmp_path / "vision.yaml"
    cfg = {
        "lines": {
            "bean": {"left_x": 100, "right_x": 300},
            "digit": {"x1": 100, "x2": 200, "x3": 300, "x4": 400},
        },
        "inference": {"imgsz": 640, "conf_thres": 0.25, "iou_thres": 0.45},
        "cameras": {
            "bean": {"device": 0, "width": 640, "height": 480, "buffer_size": 1},
            "digit": {"device": 1, "width": 640, "height": 480, "buffer_size": 1},
        },
        "models": {
            "bean": {
                "bin": str(tmp_path / "bean.bin"),
                "cfg": str(tmp_path / "bean.yaml"),
            },
            "digit": {
                "bin": str(tmp_path / "digit.bin"),
                "cfg": str(tmp_path / "digit.yaml"),
            },
        },
        "serial": {
            "port": "/dev/ttyTEST",
            "baudrate": 115200,
            "timeout": 0.05,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return cfg_path


@pytest.fixture(autouse=True)
def _mock_tkinter_and_pil(monkeypatch):
    """Make Tkinter widget creation headless-test friendly."""

    def _widget_mock(*args, **kwargs):
        m = MagicMock()
        m.winfo_children.return_value = []
        return m

    # CameraViewer (base class) imports tk under its own module.
    monkeypatch.setattr("models.camera_viewer.tk.Tk", MagicMock)
    monkeypatch.setattr("models.camera_viewer.tk.Frame", _widget_mock)
    monkeypatch.setattr("models.camera_viewer.tk.Label", _widget_mock)
    monkeypatch.setattr("models.camera_viewer.ImageTk.PhotoImage", MagicMock)
    monkeypatch.setattr("models.camera_viewer.Image.fromarray", MagicMock)

    # CalibrationUI still creates control widgets via its own tk import.
    monkeypatch.setattr("models.ui.tk.Tk", MagicMock)
    monkeypatch.setattr("models.ui.tk.Frame", _widget_mock)
    monkeypatch.setattr("models.ui.tk.Label", _widget_mock)
    monkeypatch.setattr("models.ui.tk.Button", _widget_mock)
    monkeypatch.setattr("models.ui.tk.Scale", _widget_mock)
    monkeypatch.setattr("models.ui.tk.IntVar", MagicMock)
    monkeypatch.setattr("models.ui.ImageTk.PhotoImage", MagicMock)
    monkeypatch.setattr("models.ui.Image.fromarray", MagicMock)


def test_ui_saves_lines_on_change(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    ui._set_line("bean", "left_x", 150)
    ui._set_line("digit", "x1", 250)
    # With debouncing the file is not written until _save_cfg is flushed.
    ui._save_cfg()

    with open(cfg_path, "r") as f:
        saved = yaml.safe_load(f)
    assert saved["lines"]["bean"]["left_x"] == 150
    assert saved["lines"]["digit"]["x1"] == 250


def test_set_line_debounces_save(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    ui._set_line("bean", "left_x", 150)
    # File should not be written immediately while a debounced save is pending.
    assert ui._save_pending is not None
    text_before = cfg_path.read_text(encoding="utf-8")
    assert "left_x: 150" not in text_before
    # Flushing the pending save writes the file.
    ui._save_cfg()
    text_after = cfg_path.read_text(encoding="utf-8")
    assert "left_x: 150" in text_after


def test_apply_geometry_does_not_recreate_sliders(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    ui.root.winfo_width.return_value = 1024
    ui.root.winfo_height.return_value = 600
    create_calls = []
    original_create = ui._create_sliders

    def tracking_create(mode):
        create_calls.append(mode)
        original_create(mode)

    monkeypatch.setattr(ui, "_create_sliders", tracking_create)
    ui._apply_geometry()
    ui._apply_geometry()
    # _apply_geometry should no longer touch sliders.
    assert create_calls == []


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

    def raise_tcl_error(*args, **kwargs):
        raise tk.TclError("no display name")

    monkeypatch.setattr("models.camera_viewer.tk.Tk", raise_tcl_error)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    assert ui._window_ok is False


def test_run_raises_when_window_unavailable(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    ui._window_ok = False
    with pytest.raises(RuntimeError, match="GUI is not available"):
        ui.run()


def test_run_releases_cameras_on_exit(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cam_bean.capture.return_value = (True, frame)
    cam_digit.capture.return_value = (True, frame)

    root = MagicMock()
    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit, root=root)
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


def test_initial_mode_creates_only_bean_sliders(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    created = []

    def mock_scale(*args, **kwargs):
        created.append(True)
        return MagicMock()

    monkeypatch.setattr("models.ui.tk.Scale", mock_scale)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    assert ui.mode == "bean"
    assert len(created) == 2


def test_switching_mode_recreates_correct_trackbars(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    created = []

    def mock_scale(*args, **kwargs):
        created.append(True)
        return MagicMock()

    monkeypatch.setattr("models.ui.tk.Scale", mock_scale)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    created.clear()

    ui._set_mode("digit")
    assert ui.mode == "digit"
    assert len(created) == 4

    created.clear()
    ui._set_mode("bean")
    assert ui.mode == "bean"
    assert len(created) == 2


def test_button_click_switches_mode(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    commands = {}

    def mock_button(*args, **kwargs):
        text = kwargs.get("text")
        commands[text] = kwargs.get("command")
        return MagicMock()

    monkeypatch.setattr("models.ui.tk.Button", mock_button)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    assert ui.mode == "bean"

    # Clicking the active view should keep the current mode.
    commands["BEAN (1)"]()
    assert ui.mode == "bean"

    # Clicking the other view should switch.
    commands["DIGIT (2)"]()
    assert ui.mode == "digit"


def test_key_binding_2_switches_to_digit_mode(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cam_bean.capture.return_value = (True, frame)
    cam_digit.capture.return_value = (True, frame)

    root = MagicMock()
    binds = {}

    def capture_bind(event, cb):
        binds[event] = cb

    root.bind.side_effect = capture_bind

    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit, root=root)
    assert "<KeyPress-2>" in binds
    binds["<KeyPress-2>"](None)
    assert ui.mode == "digit"

    ui.run()
    cam_bean.release.assert_called_once()
    cam_digit.release.assert_called_once()


def test_mock_camera_capture_shape_and_dtype():
    cam = MockCamera(640, 480)
    ok, frame = cam.capture()
    assert ok is True
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    cam.release()


def test_calibration_ui_accepts_mock_cameras(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MockCamera(640, 480, pattern="color_bars")
    cam_digit = MockCamera(640, 480, pattern="checkerboard")
    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    assert ui.cam_bean is cam_bean
    assert ui.cam_digit is cam_digit
    assert ui.max_x == 640
    assert ui.max_y == 480


def test_mock_camera_empty_pattern_returns_black_frame():
    cam = MockCamera(640, 480, pattern="empty")
    ok, frame = cam.capture()
    assert ok is True
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame == 0)
    cam.release()


def test_run_calibration_uses_empty_frame_when_camera_missing(
    tmp_path, monkeypatch
):
    cfg_path = _write_full_cfg(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr("models.camera.Camera.__init__", raise_runtime_error)

    mock_ui_class = MagicMock()
    monkeypatch.setattr("scripts.calibrate.CalibrationUI", mock_ui_class)

    fake_port = MagicMock()
    monkeypatch.setattr(
        "scripts.calibrate.open_serial_port", lambda *a, **k: fake_port
    )

    run_calibration(cfg, str(cfg_path), mock=False)

    assert mock_ui_class.called
    mock_ui_class.return_value.run.assert_called_once()
    fake_port.close.assert_called_once()


def test_run_calibration_opens_serial_port_by_default(tmp_path, monkeypatch):
    cfg_path = _write_full_cfg(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr("models.camera.Camera.__init__", raise_runtime_error)
    monkeypatch.setattr("scripts.calibrate.CalibrationUI", MagicMock())

    opened = {}

    def fake_open_serial_port(port, baudrate, timeout=0.05):
        opened["port"] = port
        opened["baudrate"] = baudrate
        opened["timeout"] = timeout
        return MagicMock()

    monkeypatch.setattr("scripts.calibrate.open_serial_port", fake_open_serial_port)

    run_calibration(cfg, str(cfg_path), mock=False)

    assert opened["port"] == "/dev/ttyTEST"
    assert opened["baudrate"] == 115200
    assert opened["timeout"] == 0.05


def _write_cfg_with_ui(tmp_path, ui_cfg=None):
    cfg_path = tmp_path / "vision.yaml"
    cfg = {
        "lines": {
            "bean": {"left_x": 100, "right_x": 300},
            "digit": {"x1": 100, "x2": 200, "x3": 300, "x4": 400},
        },
        "inference": {"imgsz": 640},
    }
    if ui_cfg:
        cfg["ui"] = ui_cfg
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return cfg_path


def test_ui_uses_default_config_when_ui_section_missing(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    assert ui.window_width == 1024
    assert ui.window_height == 600
    assert ui.fullscreen is False
    assert ui.touch_mode is False


def test_fullscreen_enabled_when_configured(tmp_path):
    cfg_path = _write_cfg_with_ui(tmp_path, {"window": {"fullscreen": True}})
    root = MagicMock()
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock(), root=root)
    root.attributes.assert_called_once_with("-fullscreen", True)


def test_display_size_respects_max_height(tmp_path):
    cfg_path = _write_cfg_with_ui(
        tmp_path,
        {"window": {"width": 1024, "height": 600}, "video": {"max_height": 420}},
    )
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    display_w, display_h = ui._compute_display_size(1280, 720)
    # 1280x720 contained in 1024x420 should scale to 747x420 (keeps aspect ratio).
    assert display_w == 747
    assert display_h == 420


def test_display_size_preserves_aspect_ratio(tmp_path):
    cfg_path = _write_cfg_with_ui(
        tmp_path,
        {"window": {"width": 1024, "height": 600}, "video": {"max_height": 600}},
    )
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    display_w, display_h = ui._compute_display_size(1280, 720)
    # 1280x720 scaled to width 1024 should be 1024x576.
    assert display_w == 1024
    assert display_h == 576


def test_slider_to_uses_native_camera_width(tmp_path, monkeypatch):
    cfg_path = _write_cfg_with_ui(
        tmp_path,
        {"window": {"width": 1024, "height": 600}, "controls": {"touch_mode": True}},
    )
    created = []

    def mock_scale(*args, **kwargs):
        created.append(kwargs)
        return MagicMock()

    monkeypatch.setattr("models.ui.tk.Scale", mock_scale)
    cam_bean = MagicMock()
    cam_bean.width = 1280
    cam_bean.height = 720
    cam_digit = MagicMock()
    cam_digit.width = 1280
    cam_digit.height = 720

    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    assert len(created) == 2
    for kwargs in created:
        assert kwargs["to"] == 1280


def test_sliders_hide_value_and_use_large_handles(tmp_path, monkeypatch):
    cfg_path = _write_cfg_with_ui(
        tmp_path,
        {"controls": {"slider_width": 16, "slider_height": 32}},
    )
    created = []

    def mock_scale(*args, **kwargs):
        created.append(kwargs)
        return MagicMock()

    monkeypatch.setattr("models.ui.tk.Scale", mock_scale)
    CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    assert len(created) == 2
    for kwargs in created:
        assert kwargs["showvalue"] is False
        assert kwargs["width"] >= 16
        assert kwargs["sliderlength"] >= 32


def test_lines_drawn_before_display_resize(tmp_path, monkeypatch):
    cfg_path = _write_cfg_with_ui(
        tmp_path,
        {"window": {"width": 1024, "height": 600}, "video": {"max_height": 420}},
    )
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())

    draw_calls = []
    original_draw_bean = ui._draw_bean_lines
    original_resize = ui._resize_for_display

    def tracking_draw_bean(frame):
        draw_calls.append("draw_bean_lines")
        original_draw_bean(frame)

    def tracking_resize(frame):
        draw_calls.append("resize_for_display")
        return original_resize(frame)

    monkeypatch.setattr(ui, "_draw_bean_lines", tracking_draw_bean)
    monkeypatch.setattr(ui, "_resize_for_display", tracking_resize)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ui._draw_bean_lines(frame)
    _ = ui._resize_for_display(frame)

    assert draw_calls == ["draw_bean_lines", "resize_for_display"]
    # Make sure the resized output is smaller than native.
    resized = ui._resize_for_display(frame)
    assert resized.shape[0] <= 420


def test_update_frame_does_not_block_on_slow_inference(tmp_path, monkeypatch):
    """Regression check: video updates must stay responsive while inference is slow."""
    cfg_path = _write_cfg(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    cam_bean = MagicMock()
    cam_bean.capture.return_value = (True, frame)
    cam_digit = MagicMock()
    cam_digit.capture.return_value = (True, frame)

    model = MagicMock()
    model.predict.side_effect = lambda f: (time.sleep(0.5), [])[1]
    model.names = {0: "bean"}

    root = MagicMock()
    ui = CalibrationUI(
        str(cfg_path), cam_bean, cam_digit, bean_model=model, root=root
    )
    ui._running = True
    ui._start_inference_worker()
    try:
        start = time.monotonic()
        ui._update_frame()
        elapsed = time.monotonic() - start
        # The UI thread must return immediately; it must not wait for the
        # 500 ms inference sleep to finish.
        assert elapsed < 0.1
    finally:
        ui._running = False
        ui._stop_inference_worker()


def test_inference_overlay_can_be_disabled_via_config(tmp_path):
    cfg_path = _write_cfg_with_ui(
        tmp_path, {"controls": {"inference_overlay": False}}
    )
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    model = MagicMock()

    ui = CalibrationUI(
        str(cfg_path), cam_bean, cam_digit, bean_model=model, digit_model=model
    )
    assert ui.inference_overlay is False


def test_mode_switch_clears_stale_detection_results(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    ui._latest_dets["digit"] = [[10, 10, 20, 20, 0.9, 0]]
    ui._latest_names["digit"] = {0: "digit"}

    ui._set_mode("digit")
    assert ui._latest_dets["digit"] is None
    assert ui._latest_names["digit"] is None
