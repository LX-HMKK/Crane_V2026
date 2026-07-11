import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Block heavy optional dependencies so that importing the entry points does not
# pull in Ultralytics/torch (and possibly initialize a CUDA context) during
# pytest collection. The tests themselves mock model loading.
sys.modules["ultralytics"] = None
sys.modules["torch"] = None

import pytest
import numpy as np
import serial
import time
import threading

import models.runtime as runtime
import main_amd64
import main_arm64
from models.camera import MockCamera
from models.serial_port import START_COMMAND, SerialPort


def test_main_loads_vision_config(tmp_path):
    cfg_path = tmp_path / "vision.yaml"
    cfg_path.write_text("lines:\n  bean:\n    left_x: 100\n", encoding="utf-8")
    cfg = runtime.load_vision_cfg(str(cfg_path))
    assert cfg["lines"]["bean"]["left_x"] == 100


def test_open_camera_mock_returns_MockCamera():
    cfg = {"device": 0, "width": 640, "height": 480, "buffer_size": 1}
    cam = runtime.open_camera(cfg, mock=True)
    assert isinstance(cam, MockCamera)
    assert cam.width == 640
    assert cam.height == 480
    cam.release()


def test_open_camera_fallback_empty_on_runtime_error():
    cfg = {"device": 99, "width": 640, "height": 480, "buffer_size": 1}
    with patch("models.runtime.Camera") as MockCameraCls:
        MockCameraCls.side_effect = RuntimeError("no camera")
        cam = runtime.open_camera(cfg, fallback_empty=True)
    assert isinstance(cam, MockCamera)
    assert cam.pattern == "empty"
    cam.release()


def test_main_amd64_does_not_depend_on_serial():
    # The module must be importable without bringing in serial/SerialPort.
    assert "serial" not in main_amd64.__dict__
    assert "SerialPort" not in main_amd64.__dict__


def test_load_pt_models_raises_when_ultralytics_unavailable(tmp_path):
    cfg_path = tmp_path / "vision.yaml"
    cfg_path.write_text(
        "models:\n"
        "  bean:\n"
        "    pt: assets/beans_v4.pt\n"
        "  digit:\n"
        "    pt: assets/num.pt\n"
        "inference:\n"
        "  imgsz: 640\n",
        encoding="utf-8",
    )
    cfg = runtime.load_vision_cfg(str(cfg_path))
    with patch("models.pt_model.HAS_ULTRALYTICS", False):
        with pytest.raises(EnvironmentError, match="ultralytics is not available"):
            runtime.load_pt_models(cfg)


def test_main_amd64_once_mock_cameras(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_amd64.py", "--once", "--mock-cameras"])
    monkeypatch.setattr(main_amd64.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(main_amd64.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(main_amd64.cv2, "waitKey", lambda *a, **k: 27)
    monkeypatch.setattr(main_amd64.cv2, "destroyAllWindows", lambda *a, **k: None)
    # Force camera-only mode so the test is deterministic regardless of whether
    # ultralytics is installed in the current environment.
    monkeypatch.setattr("models.pt_model.HAS_ULTRALYTICS", False)

    main_amd64.main()

    captured = capsys.readouterr()
    assert "SCAN FAILED: task generator unavailable" in captured.out


def test_serial_frame_round_trip():
    ser = serial.serial_for_url("loop://", timeout=0.01)

    def writer():
        time.sleep(0.1)
        ser.write(START_COMMAND)

    t = threading.Thread(target=writer)
    t.start()

    port = SerialPort(ser)
    ok = port.wait_start(timeout=2.0)
    t.join()
    assert ok
    port.send_result(1, 2, 3, 3, 1, 5, 2, 4)
    time.sleep(0.05)
    assert ser.read(20) == b"\xA5\x5A\x01\x02\x03\x03\x01\x05\x02\x04\x0D\x0A"
    port.close()


def test_send_result_accepts_numpy_integers():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    port.send_result(np.int64(1), np.uint8(2), 3, np.int32(3), 1, 5, 2, 4)
    time.sleep(0.05)
    assert ser.read(20) == b"\xA5\x5A\x01\x02\x03\x03\x01\x05\x02\x04\x0D\x0A"
    port.close()


def _minimal_vision_cfg():
    return {
        "serial": {"port": "loop://", "baudrate": 921600, "timeout": 0.01},
        "protocol": {
            "start_timeout_sec": 0.3,
            "send_interval_sec": 0.1,
            "poll_interval_sec": 0.01,
            "scan_retry_interval_sec": 0.5,
        },
        "cameras": {
            "bean": {"device": 0, "width": 640, "height": 480, "buffer_size": 1},
            "digit": {"device": 1, "width": 640, "height": 480, "buffer_size": 1},
        },
        "models": {
            "bean": {"bin": "assets/beans_v4.bin", "cfg": "assets/beans_v4_rdkx5_config.yaml"},
            "digit": {"bin": "assets/num.bin", "cfg": "assets/num_rdkx5_config.yaml"},
        },
        "inference": {"imgsz": 640, "conf_thres": 0.25, "iou_thres": 0.45, "pad": 127},
        "scan": {"frames": 1, "max_retries": 0, "timeout_sec": 5},
        "lines": {
            "bean": {"left_x": 213, "right_x": 426},
            "digit": {"x1": 150, "x2": 250, "x3": 350, "x4": 450},
        },
        "mapping": {"bean_to_digit": {"yellow": 1, "green": 2, "white": 3}},
    }


def test_main_amd64_once_success(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_amd64.py", "--once", "--mock-cameras"])
    monkeypatch.setattr(main_amd64.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(main_amd64.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(main_amd64.cv2, "waitKey", lambda *a, **k: 27)
    monkeypatch.setattr(main_amd64.cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(main_amd64, "load_vision_cfg", lambda _path: _minimal_vision_cfg())

    def fake_load_pt_models(cfg):
        bean = MagicMock()
        bean.names = {0: "white", 1: "green", 2: "yellow"}
        # zone1 green -> 2, zone2 yellow -> 1, zone3 white -> 3
        bean.predict.return_value = [
            [95.0, 0.0, 105.0, 10.0, 0.9, 1],   # zone 1, green
            [295.0, 0.0, 305.0, 10.0, 0.9, 2],  # zone 2, yellow
            [495.0, 0.0, 505.0, 10.0, 0.9, 0],  # zone 3, white
        ]
        digit = MagicMock()
        digit.names = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
        digit.predict.return_value = [
            [95.0, 95.0, 105.0, 105.0, 0.9, 1],   # zone 4, value 2
            [195.0, 95.0, 205.0, 105.0, 0.9, 0],  # zone 5, value 1
            [300.0, 95.0, 310.0, 105.0, 0.9, 2],  # zone 6, value 3
        ]
        return bean, digit

    monkeypatch.setattr(main_amd64, "load_pt_models", fake_load_pt_models)

    main_amd64.main()

    captured = capsys.readouterr()
    assert "SCAN RESULT: 4 5 6" in captured.out


def test_main_arm64_competition_sends_result(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_arm64.py", "--no-ui"])
    monkeypatch.setattr(main_arm64, "load_vision_cfg", lambda _path: _minimal_vision_cfg())
    monkeypatch.setattr(main_arm64, "init_logging", lambda: None)
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    monkeypatch.setattr(
        main_arm64, "open_camera",
        lambda _cfg, **kwargs: MockCamera(640, 480),
    )
    monkeypatch.setattr(main_arm64, "warmup_models", lambda *a, **k: None)

    def fake_load_models(cfg):
        bean = MagicMock()
        bean.names = {0: "white", 1: "green", 2: "yellow"}
        bean.predict.return_value = [
            [95.0, 0.0, 105.0, 10.0, 0.9, 1],   # zone 1, green -> color 2
            [295.0, 0.0, 305.0, 10.0, 0.9, 2],  # zone 2, yellow -> color 1
            [495.0, 0.0, 505.0, 10.0, 0.9, 0],  # zone 3, white -> color 3
        ]
        digit = MagicMock()
        digit.names = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
        digit.predict.return_value = [
            [95.0, 95.0, 105.0, 105.0, 0.9, 1],   # zone 4, value 2
            [195.0, 95.0, 205.0, 105.0, 0.9, 0],  # zone 5, value 1
            [300.0, 95.0, 310.0, 105.0, 0.9, 2],  # zone 6, value 3
            [400.0, 95.0, 410.0, 105.0, 0.9, 3],  # zone 7, value 4
            [500.0, 95.0, 510.0, 105.0, 0.9, 4],  # zone 8, value 5
        ]
        return bean, digit

    monkeypatch.setattr(main_arm64, "load_models", fake_load_models)

    class FakeSerialPort:
        def __init__(self, ser):
            self.ser = ser
            self.calls = []
            self._active = False

        def poll_start(self):
            self._active = True

        def is_start_active(self, timeout=None):
            return self._active

        def send_result(self, *frame):
            self.calls.append(tuple(frame))
            raise KeyboardInterrupt()

        def close(self):
            pass

    fake_port = FakeSerialPort(MagicMock())
    # main_arm64 now uses the encapsulated open_serial_port helper.
    monkeypatch.setattr(main_arm64, "open_serial_port", lambda *a, **k: fake_port)

    main_arm64.main()

    assert fake_port.calls, "send_result was never called"
    # zone1 color=2, zone2 color=1, zone3 color=3
    # zone4 digit=2, zone5 digit=1, zone6 digit=3, zone7 digit=4, zone8 digit=5
    assert fake_port.calls[0] == (2, 1, 3, 2, 1, 3, 4, 5)
