import sys
import threading
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests"))

import cv2
import numpy as np
import pytest
import yaml
from unittest.mock import MagicMock

from models.ui import (
    CalibrationUI,
    CompetitionUI,
    CAPTURE_OFFLINE_AFTER_FAILURES,
    _InferenceRateMeter,
)


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


@pytest.fixture(autouse=True)
def _mock_tkinter_and_pil(monkeypatch):
    """Make Tkinter widget creation headless-test friendly."""

    def _widget_mock(*args, **kwargs):
        m = MagicMock()
        m.winfo_children.return_value = []
        return m

    # CameraViewer (base class) imports tk under its own module.
    monkeypatch.setattr("models.ui.viewer.tk.Tk", MagicMock)
    monkeypatch.setattr("models.ui.viewer.tk.Frame", _widget_mock)
    monkeypatch.setattr("models.ui.viewer.tk.Label", _widget_mock)
    monkeypatch.setattr("models.ui.viewer.ImageTk.PhotoImage", MagicMock)
    monkeypatch.setattr("models.ui.viewer.Image.fromarray", MagicMock)

    # CalibrationUI still creates control widgets via its own tk import.
    monkeypatch.setattr("models.ui.calibration.tk.Tk", MagicMock)
    monkeypatch.setattr("models.ui.calibration.tk.Frame", _widget_mock)
    monkeypatch.setattr("models.ui.calibration.tk.Label", _widget_mock)
    monkeypatch.setattr("models.ui.calibration.tk.Button", _widget_mock)
    monkeypatch.setattr("models.ui.calibration.tk.Scale", _widget_mock)
    monkeypatch.setattr("models.ui.calibration.tk.IntVar", MagicMock)
    monkeypatch.setattr("models.ui.calibration.ImageTk.PhotoImage", MagicMock)
    monkeypatch.setattr("models.ui.calibration.Image.fromarray", MagicMock)
    monkeypatch.setattr("models.ui.competition.tk.Button", _widget_mock)
    monkeypatch.setattr("models.ui.competition.tk.Frame", _widget_mock)
    monkeypatch.setattr("models.ui.competition.tk.Label", _widget_mock)


@pytest.mark.parametrize("names", [{0: "bean", 1: "green"}, ["bean", "green"]])
def test_draw_detections_accepts_dict_and_list_names(tmp_path, names):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = np.array([[50, 50, 100, 100, 0.9, 1]])
    ui._draw_detections(frame, dets, names)


def test_init_window_headless_fallback(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)

    def raise_tcl_error(*args, **kwargs):
        raise tk.TclError("no display name")

    monkeypatch.setattr("models.ui.viewer.tk.Tk", raise_tcl_error)
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


def test_run_keeps_borrowed_cameras_open_on_exit(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cam_bean.capture.return_value = (True, frame)
    cam_digit.capture.return_value = (True, frame)
    root = MagicMock()

    ui = CalibrationUI(
        str(cfg_path), cam_bean, cam_digit, root=root, owns_cameras=False
    )
    ui.run()

    cam_bean.release.assert_not_called()
    cam_digit.release.assert_not_called()


def test_competition_ui_keeps_borrowed_serial_open(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    cam_bean = MagicMock()
    cam_digit = MagicMock()
    serial = MagicMock()

    ui = CompetitionUI(
        str(cfg_path),
        cam_bean,
        cam_digit,
        MagicMock(),
        MagicMock(),
        serial_thread=serial,
        root=MagicMock(),
        owns_cameras=False,
        owns_serial=False,
        start_serial=False,
    )
    ui._quit()

    serial.start.assert_not_called()
    serial.stop.assert_not_called()
    cam_bean.release.assert_not_called()
    cam_digit.release.assert_not_called()


def test_competition_starts_one_capture_worker_per_camera(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    started = []

    class RecordingThread:
        def __init__(self, *, target, args, name, daemon):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append((self.name, self.args))

        def join(self, timeout=None):
            del timeout

        def is_alive(self):
            return False

    monkeypatch.setattr("models.ui.competition.threading.Thread", RecordingThread)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    ui._start_capture_workers()
    ui._stop_capture_workers()

    assert [name for name, _args in started] == [
        "crane-camera-bean", "crane-camera-digit",
    ]
    assert [args[0] for _name, args in started] == ["bean", "digit"]


def test_calibration_starts_one_inference_worker_per_loaded_model(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    started = []

    class RecordingThread:
        def __init__(self, *, target, args, name, daemon):
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append((self.name, self.args))

        def join(self, timeout=None):
            del timeout

        def is_alive(self):
            return False

    monkeypatch.setattr("models.ui.calibration.threading.Thread", RecordingThread)
    ui = CalibrationUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )
    ui._start_inference_worker()
    ui._stop_inference_worker()

    assert started == [
        ("crane-inference-bean", ("bean",)),
        ("crane-inference-digit", ("digit",)),
    ]


def test_competition_capture_interval_prefers_configured_target_fps():
    camera = type("Camera", (), {"target_fps": 15.0, "fps": 25.0})()

    assert CompetitionUI._capture_interval(camera) == 1.0 / 15.0


def test_competition_capture_worker_submits_latest_frame_for_inference(
    tmp_path, monkeypatch
):
    cfg_path = _write_cfg(tmp_path)
    raw_bean = np.zeros((480, 640, 3), dtype=np.uint8)
    cam_bean = MagicMock()
    cam_bean.capture.return_value = (True, raw_bean)
    cam_digit = MagicMock()
    ui = CompetitionUI(
        str(cfg_path), cam_bean, cam_digit, MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    submitted = []
    original_submit = ui._submit_frame_for_inference

    def submit_and_stop(mode, frame, **metadata):
        submitted.append((mode, frame, metadata))
        original_submit(mode, frame, **metadata)
        ui._capture_stop.set()

    monkeypatch.setattr(ui, "_submit_frame_for_inference", submit_and_stop)
    ui._capture_worker("bean", cam_bean)

    cam_bean.capture.assert_called_once()
    assert [mode for mode, _frame, _metadata in submitted] == ["bean"]
    assert submitted[0][1] is raw_bean
    assert submitted[0][2]["capture_sequence"] == 1
    assert submitted[0][2]["captured_at"] is not None
    assert ui._capture_sequences["bean"] == 1
    # The capture slot and inference slot must never share mutable frame data.
    assert not np.shares_memory(ui._infer_pending["bean"], raw_bean)


def test_competition_renders_only_the_active_camera(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    ui._capture_ready["bean"] = True
    ui._captured_frames["bean"] = frame

    queried = []
    original_camera_state = ui._camera_state

    def tracking_camera_state(mode):
        queried.append((mode, original_camera_state(mode)))
        return queried[-1][1]

    shown = []
    original_show_frame = ui._show_frame

    def tracking_show_frame(display_frame):
        shown.append(display_frame)
        original_show_frame(display_frame)

    monkeypatch.setattr(ui, "_camera_state", tracking_camera_state)
    monkeypatch.setattr(ui, "_show_frame", tracking_show_frame)
    ui._running = True
    try:
        ui._update_frame()
    finally:
        ui._running = False

    # Only the active camera is queried; the other camera is never consulted
    # as a fallback.
    assert len(queried) == 1
    assert queried[0][0] == "bean"
    assert queried[0][1][0] == "ok"
    assert not np.shares_memory(queried[0][1][1], frame)
    assert np.all(frame == 0)
    # A real frame is rendered, not an offline placeholder.
    assert len(shown) == 1


def test_competition_failure_log_is_rate_limited(tmp_path, caplog):
    import logging

    cfg_path = _write_cfg(tmp_path)
    task_gen = MagicMock()
    task_gen.generate_frame_with_completion.return_value = (None, {}, {}, "no frame")
    ui = CompetitionUI(
        str(cfg_path),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        task_gen=task_gen,
    )
    ui._result_controller.update_interval = 0.0
    ui._latest_dets = {"bean": [], "digit": []}

    with caplog.at_level(logging.ERROR):
        for _ in range(12):
            ui._on_inference_done("bean", [])

    errors = [
        record for record in caplog.records
        if "generate_frame failed" in record.message
    ]
    assert len(errors) == 1


def test_competition_result_update_log_is_rate_limited(tmp_path, caplog):
    import logging

    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        start_serial=False,
    )
    with caplog.at_level(logging.INFO):
        controller = ui._result_controller
        controller._log_result_update((1, 2, 3, 1, 2, 3, 4, 5), 0, 10.0)
        controller._log_result_update((1, 2, 3, 1, 2, 3, 4, 5), 0, 10.5)
        controller._log_result_update((1, 2, 3, 1, 2, 3, 4, 5), 0, 11.1)

    updates = [
        record for record in caplog.records if "Result updated" in record.message
    ]
    assert len(updates) == 2


def test_competition_limits_task_updates_to_serial_interval(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    task_gen = MagicMock()
    task_gen.generate_frame_with_completion.return_value = (None, {}, {}, "no frame")
    ui = CompetitionUI(
        str(cfg_path),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        task_gen=task_gen,
        start_serial=False,
    )
    ui._latest_dets = {"bean": [], "digit": []}
    timestamps = iter((10.0, 10.02, 10.06))
    monkeypatch.setattr("models.ui.competition.time.monotonic", lambda: next(timestamps))

    ui._on_inference_done("bean", [])
    ui._on_inference_done("digit", [])
    ui._on_inference_done("bean", [])

    assert task_gen.generate_frame_with_completion.call_count == 2


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


def test_inference_workers_run_models_in_parallel(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    release_models = threading.Event()
    bean_started = threading.Event()
    digit_started = threading.Event()

    def blocking_predict(started):
        def predict(_frame):
            started.set()
            assert release_models.wait(timeout=2.0)
            return []

        return predict

    bean_model = MagicMock()
    bean_model.names = {0: "bean"}
    bean_model.predict.side_effect = blocking_predict(bean_started)
    digit_model = MagicMock()
    digit_model.names = {0: "digit"}
    digit_model.predict.side_effect = blocking_predict(digit_started)

    ui = CalibrationUI(
        str(cfg_path),
        MagicMock(),
        MagicMock(),
        bean_model=bean_model,
        digit_model=digit_model,
    )
    ui._running = True
    ui._start_inference_worker()
    try:
        ui._submit_frame_for_inference("bean", frame)
        ui._submit_frame_for_inference("digit", frame)

        assert bean_started.wait(timeout=1.0)
        assert digit_started.wait(timeout=1.0)
    finally:
        release_models.set()
        ui._running = False
        ui._stop_inference_worker()


def test_stop_inference_worker_stops_all_model_workers(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )
    ui._running = True
    ui._start_inference_worker()

    assert set(ui._infer_threads) == {"bean", "digit"}

    ui._running = False
    ui._stop_inference_worker()

    assert ui._infer_threads == {}


def test_competition_serializes_parallel_task_updates(tmp_path):
    cfg_path = _write_cfg(tmp_path)

    class BlockingTaskGenerator:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.calls = 0
            self.lock = threading.Lock()
            self.first_call_started = threading.Event()
            self.second_call_started = threading.Event()
            self.release = threading.Event()

        def generate_frame_with_completion(self, *_args, **_kwargs):
            with self.lock:
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if self.calls == 1:
                    self.first_call_started.set()
                else:
                    self.second_call_started.set()
            assert self.release.wait(timeout=2.0)
            with self.lock:
                self.active -= 1
            return None, {}, {}, "no frame"

    task_gen = BlockingTaskGenerator()
    ui = CompetitionUI(
        str(cfg_path),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        task_gen=task_gen,
        start_serial=False,
    )
    ui._task_update_interval = 0.0
    ui._latest_dets = {"bean": [], "digit": []}
    first = threading.Thread(target=ui._on_inference_done, args=("bean", []))
    second = threading.Thread(target=ui._on_inference_done, args=("digit", []))

    first.start()
    assert task_gen.first_call_started.wait(timeout=1.0)
    second.start()
    try:
        assert not task_gen.second_call_started.wait(timeout=0.1)
    finally:
        task_gen.release.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert task_gen.max_active == 1


def test_inference_rate_meter_reports_per_camera_and_total_rates():
    meter = _InferenceRateMeter(interval=2.0, started_at=10.0)

    assert meter.record("bean", now=10.5) is None
    assert meter.record("digit", now=11.0) is None
    assert meter.record("bean", now=12.0) == {
        "bean_fps": 1.0,
        "digit_fps": 0.5,
        "total_fps": 1.5,
        "window_sec": 2.0,
    }

    assert meter.record("digit", now=13.0) is None
    assert meter.record("digit", now=14.0) == {
        "bean_fps": 0.0,
        "digit_fps": 1.0,
        "total_fps": 1.0,
        "window_sec": 2.0,
    }


def test_mode_switch_clears_stale_detection_results(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CalibrationUI(str(cfg_path), MagicMock(), MagicMock())
    ui._latest_dets["digit"] = [[10, 10, 20, 20, 0.9, 0]]
    ui._latest_names["digit"] = {0: "digit"}

    ui._set_mode("digit")
    assert ui._latest_dets["digit"] is None
    assert ui._latest_names["digit"] is None


def test_competition_update_frame_offline_shows_placeholder_not_other_camera(
    tmp_path, monkeypatch
):
    cfg_path = _write_cfg(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    digit_frame = np.full((480, 640, 3), 99, dtype=np.uint8)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    ui.mode = "bean"
    ui._capture_ready = {"bean": False, "digit": True}
    ui._capture_failures = {
        "bean": CAPTURE_OFFLINE_AFTER_FAILURES, "digit": 0,
    }
    ui._captured_frames = {"bean": frame, "digit": digit_frame}

    placeholder = np.zeros((10, 10, 3), dtype=np.uint8)
    shown = []
    monkeypatch.setattr(ui, "_render_placeholder", lambda *a, **k: placeholder)
    monkeypatch.setattr(
        ui, "_show_frame", lambda display_frame: shown.append(display_frame)
    )
    ui._running = True
    try:
        ui._update_frame()
    finally:
        ui._running = False

    # Offline shows the placeholder; the other camera's frame is never rendered.
    assert shown == [placeholder]


def test_competition_update_frame_shows_live_frame_when_ok(tmp_path, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    ui.mode = "bean"
    ui._capture_ready = {"bean": True, "digit": False}
    ui._capture_failures = {"bean": 0, "digit": 0}
    ui._captured_frames = {"bean": frame, "digit": None}

    shown = []
    monkeypatch.setattr(ui, "_render_placeholder", lambda *a, **k: object())
    monkeypatch.setattr(ui, "_resize_for_display", lambda f, **k: f)
    monkeypatch.setattr(
        ui, "_show_frame", lambda display_frame: shown.append(display_frame)
    )
    ui._running = True
    try:
        ui._update_frame()
    finally:
        ui._running = False

    assert len(shown) == 1
    assert not np.shares_memory(shown[0], frame)
    assert np.all(frame == 0)


def test_competition_update_frame_waiting_uses_neutral_text_not_offline(
    tmp_path, monkeypatch
):
    """Startup (no frame, no failures yet) must not show the offline alarm."""
    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    ui.mode = "bean"
    # Startup state: no frame delivered yet, and no confirmed failures.
    ui._capture_ready = {"bean": False, "digit": False}
    ui._capture_failures = {"bean": 0, "digit": 0}
    ui._captured_frames = {"bean": None, "digit": None}

    placeholder_args = []

    def recording_placeholder(*args, **kwargs):
        placeholder_args.append((args, kwargs))
        return np.zeros((480, 640, 3), dtype=np.uint8)

    monkeypatch.setattr(ui, "_render_placeholder", recording_placeholder)
    monkeypatch.setattr(ui, "_show_frame", lambda _frame: None)
    ui._running = True
    try:
        ui._update_frame()
    finally:
        ui._running = False

    assert placeholder_args, "placeholder should be rendered while waiting"
    text = placeholder_args[0][0][0]
    assert "WAITING" in text
    assert "OFFLINE" not in text


def test_competition_update_status_maps_offline_states(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    serial = MagicMock()
    serial.connected = False
    serial.is_start_active = False
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=serial, root=MagicMock(), start_serial=False,
    )
    ui._capture_failures = {
        "bean": CAPTURE_OFFLINE_AFTER_FAILURES, "digit": 0,
    }

    ui._update_status()

    assert ui._status_states["bean"] is False
    assert ui._status_states["digit"] is True
    assert ui._status_states["serial"] is False
    assert ui._status_states["start"] is False
    ui._status_labels["bean"].config.assert_called_with(
        text="BEAN: OFFLINE", bg="#aa0000"
    )
    ui._status_labels["serial"].config.assert_called_with(
        text="SERIAL: OFFLINE", bg="#aa0000"
    )


def test_competition_capture_worker_invalidates_detections_after_offline(
    tmp_path, monkeypatch
):
    cfg_path = _write_cfg(tmp_path)
    failing_camera = MagicMock()
    failing_camera.capture.return_value = (False, None)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    ui._capture_failures["bean"] = CAPTURE_OFFLINE_AFTER_FAILURES - 1
    ui._latest_dets["bean"] = [[1, 2, 3, 4, 0.9, 0]]
    ui._latest_names["bean"] = {0: "bean"}

    waits = {"n": 0}

    def counting_wait(timeout=None):
        del timeout
        waits["n"] += 1
        if waits["n"] >= 2:
            ui._capture_stop.set()
        return False

    monkeypatch.setattr(ui._capture_stop, "wait", counting_wait)
    ui._capture_worker("bean", failing_camera)

    assert ui._latest_dets["bean"] is None
    assert ui._latest_names["bean"] is None


# ── START-gated inference + DEBUG mode (CompetitionUI) ────────────────


def _serial_with_start(is_start_active):
    serial = MagicMock()
    serial.is_start_active = is_start_active
    return serial


def test_competition_should_infer_true_without_serial(tmp_path):
    """x86 (no serial) must keep inferring; the gate is a no-op there."""
    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    assert ui._should_infer() is True


def test_competition_should_infer_on_start_active(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=_serial_with_start(True), root=MagicMock(), start_serial=False,
    )
    assert ui._should_infer() is True


def test_competition_should_not_infer_when_idle(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=_serial_with_start(False), root=MagicMock(), start_serial=False,
    )
    assert ui._should_infer() is False


def test_competition_should_infer_in_debug_mode(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=_serial_with_start(False), root=MagicMock(), start_serial=False,
    )
    ui._debug_force_inference = True
    assert ui._should_infer() is True


def test_competition_accepts_bool_and_callable_is_start_active(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    serial = _serial_with_start(False)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=serial, root=MagicMock(), start_serial=False,
    )
    assert ui._should_infer() is False

    serial.is_start_active = lambda: True
    assert ui._should_infer() is True


def test_competition_start_gated_false_disables_gate(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui_cfg = {"inference": {"start_gated": False}}
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.update(ui_cfg)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=_serial_with_start(False), root=MagicMock(), start_serial=False,
    )
    assert ui._should_infer() is True


def test_submit_frame_gated_off_clears_state_and_does_not_queue(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    serial = _serial_with_start(False)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=serial, root=MagicMock(), start_serial=False,
    )
    ui._latest_dets = {"bean": [[1, 2, 3, 4, 0.9, 0]], "digit": None}
    ui._latest_names = {"bean": {0: "bean"}, "digit": None}
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    ui._submit_frame_for_inference("bean", frame)

    assert ui._infer_pending["bean"] is None
    assert ui._latest_dets["bean"] is None
    assert serial.current_frame is None


def test_submit_frame_gated_on_queues(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    ui._submit_frame_for_inference("bean", frame)

    assert ui._infer_pending["bean"] is not None
    assert not np.shares_memory(ui._infer_pending["bean"], frame)


def test_competition_gate_off_resets_last_frame_for_sticky_resume(tmp_path):
    """gate-off must reset _last_frame so a re-generated identical result can
    be re-published to serial after START resumes."""
    cfg_path = _write_cfg(tmp_path)
    serial = _serial_with_start(False)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=serial, root=MagicMock(), start_serial=False,
    )
    controller = ui._result_controller
    controller._last_frame = (1, 2, 3, 4, 5, 1, 2, 3)
    controller._last_updated_at = 123.0
    controller._consecutive_failures = 7

    ui._on_inference_gate_closed()

    assert controller._last_frame is None
    assert controller._last_updated_at is None
    assert controller.consecutive_failures == 0
    assert serial.current_frame is None


def test_competition_debug_forced_does_not_write_serial(tmp_path):
    """DEBUG results update the UI overlay but never the serial cache/counters."""
    cfg_path = _write_cfg(tmp_path)
    serial = _serial_with_start(False)
    task_gen = MagicMock()
    task_gen.generate_frame_with_completion.return_value = (
        (1, 2, 3, 4, 5, 1, 2, 3),
        {1: (1, "detected", 0.9)},
        {},
        "ok",
    )
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        task_gen=task_gen, serial_thread=serial, root=MagicMock(), start_serial=False,
    )
    ui._latest_dets = {"bean": [["bean"]], "digit": [["digit"]]}
    ui._result_controller._consecutive_failures = 3
    serial.current_frame = None

    ui._update_task_from_latest_detections(debug_forced=True)

    assert serial.current_frame is None
    assert ui._result_controller.consecutive_failures == 3
    assert ui._zone_results["bean"] == {1: (1, "detected", 0.9)}


def test_competition_debug_toggle_flips_gate_and_button(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=_serial_with_start(False), root=MagicMock(), start_serial=False,
    )
    assert ui._debug_force_inference is False
    assert ui._should_infer() is False

    ui._toggle_debug()

    assert ui._debug_force_inference is True
    assert ui._should_infer() is True
    assert ui._debug_button.config.call_args.kwargs["relief"] == tk.SUNKEN

    ui._toggle_debug()
    assert ui._debug_force_inference is False
    assert ui._should_infer() is False


def test_competition_debug_button_disabled_without_serial(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        root=MagicMock(), start_serial=False,
    )
    assert ui._debug_button_usable() is False
    assert ui._debug_button.config.call_args.kwargs["state"] == tk.DISABLED


def test_competition_idle_update_frame_suppresses_overlay(tmp_path, monkeypatch):
    """Idle (no START, no DEBUG) renders the live feed with no overlays."""
    cfg_path = _write_cfg(tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=_serial_with_start(False), root=MagicMock(), start_serial=False,
    )
    ui.mode = "bean"
    ui._capture_ready = {"bean": True, "digit": False}
    ui._capture_failures = {"bean": 0, "digit": 0}
    ui._captured_frames = {"bean": frame, "digit": None}
    ui._latest_dets = {"bean": [[1, 2, 3, 4, 0.9, 0]], "digit": None}
    ui._latest_names = {"bean": {0: "bean"}, "digit": None}
    ui._zone_results = {"bean": {1: (1, "detected", 0.9)}, "digit": {}}

    draw_calls = []
    monkeypatch.setattr(
        ui, "_draw_detections", lambda *a, **k: draw_calls.append((a, k))
    )
    shown = []
    monkeypatch.setattr(ui, "_show_frame", lambda f: shown.append(f))
    monkeypatch.setattr(ui, "_resize_for_display", lambda f, **k: f)

    ui._running = True
    try:
        ui._update_frame()
    finally:
        ui._running = False

    assert draw_calls == []
    assert len(shown) == 1
    assert not np.shares_memory(shown[0], frame)
    assert np.all(frame == 0)


def test_competition_debug_on_drops_serial_cache(tmp_path):
    """DEBUG on with START inactive must keep the serial cache empty (F2),
    so a pre-gap frame is never transmitted when START resumes."""
    cfg_path = _write_cfg(tmp_path)
    serial = _serial_with_start(False)
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=serial, root=MagicMock(), start_serial=False,
    )
    serial.current_frame = (1, 2, 3, 4, 5, 1, 2, 3)
    ui._debug_force_inference = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    ui._submit_frame_for_inference("bean", frame)

    assert serial.current_frame is None


def test_competition_debug_forced_is_start_active_callable(tmp_path):
    """_debug_forced() must handle a callable is_start_active like
    _should_infer() does (F3); otherwise DEBUG results would be treated as
    normal and written to the serial cache."""
    cfg_path = _write_cfg(tmp_path)
    serial = MagicMock()
    serial.is_start_active = lambda: False
    ui = CompetitionUI(
        str(cfg_path), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        serial_thread=serial, root=MagicMock(), start_serial=False,
    )
    ui._debug_force_inference = True

    assert ui._should_infer() is True
    assert ui._debug_forced() is True
