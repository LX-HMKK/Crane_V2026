import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test"))

sys.modules["ultralytics"] = None
sys.modules["torch"] = None

import pytest

import send_test_frame


def test_send_default_frame_reads_config_and_sends_once(tmp_path, monkeypatch):
    cfg_path = tmp_path / "vision.yaml"
    cfg_path.write_text(
        "serial:\n  port: /dev/ttyTEST\n  baudrate: 115200\n  timeout: 0.05\n",
        encoding="utf-8",
    )

    fake_port = MagicMock()
    opened = {}

    def fake_open_serial_port(port, baudrate, timeout=0.05):
        opened["port"] = port
        opened["baudrate"] = baudrate
        opened["timeout"] = timeout
        return fake_port

    monkeypatch.setattr(
        send_test_frame, "open_serial_port", fake_open_serial_port
    )

    with patch.object(
        sys, "argv", ["send_test_frame.py", "--config", str(cfg_path), "--count", "1"]
    ):
        send_test_frame.main()

    assert opened["port"] == "/dev/ttyTEST"
    assert opened["baudrate"] == 115200
    assert opened["timeout"] == 0.05
    fake_port.send_result.assert_called_once_with(*send_test_frame.DEFAULT_FRAME)
    fake_port.close.assert_called_once()


def test_send_custom_frame_and_count(tmp_path, monkeypatch):
    cfg_path = tmp_path / "vision.yaml"
    cfg_path.write_text(
        "serial:\n  port: /dev/ttyTEST\n  baudrate: 115200\n",
        encoding="utf-8",
    )

    fake_port = MagicMock()
    monkeypatch.setattr(send_test_frame, "open_serial_port", lambda *a, **k: fake_port)
    monkeypatch.setattr(send_test_frame.time, "sleep", lambda _s: None)

    with patch.object(
        sys,
        "argv",
        [
            "send_test_frame.py",
            "--config",
            str(cfg_path),
            "--frame",
            "1",
            "2",
            "3",
            "5",
            "4",
            "3",
            "2",
            "1",
            "--count",
            "3",
        ],
    ):
        send_test_frame.main()

    assert fake_port.send_result.call_count == 3
    fake_port.send_result.assert_called_with(1, 2, 3, 5, 4, 3, 2, 1)
    fake_port.close.assert_called_once()


def test_missing_serial_section_raises(tmp_path):
    cfg_path = tmp_path / "vision.yaml"
    cfg_path.write_text("lines:\n  bean:\n    left_x: 100\n", encoding="utf-8")

    with patch.object(sys, "argv", ["send_test_frame.py", "--config", str(cfg_path)]):
        with pytest.raises(RuntimeError, match="No 'serial' section"):
            send_test_frame.main()
