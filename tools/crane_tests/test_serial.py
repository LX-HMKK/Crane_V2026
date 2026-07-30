import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Keep this module independent of the heavy inference dependencies.
sys.modules["ultralytics"] = None
sys.modules["torch"] = None

import pytest
import serial
import time
import threading
from models.serial_port import SerialPort, SerialThread, START_COMMAND, open_serial_port


def test_wait_start_detects_start_command():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)

    def writer():
        time.sleep(0.05)
        ser.write(b"\x00\x01" + START_COMMAND + b"\x00")

    t = threading.Thread(target=writer)
    t.start()
    ok = port.wait_start(timeout=2.0)
    t.join()
    assert ok
    port.close()


def test_wait_start_can_be_called_again_after_match():
    """After a successful match the internal buffer must be cleared."""
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)

    def writer():
        time.sleep(0.05)
        ser.write(START_COMMAND)
        time.sleep(0.05)
        ser.write(START_COMMAND)

    t = threading.Thread(target=writer)
    t.start()
    assert port.wait_start(timeout=2.0)
    assert port.wait_start(timeout=2.0)
    t.join()
    port.close()


def test_wait_start_split_command():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)

    def writer():
        time.sleep(0.05)
        ser.write(START_COMMAND[:2])
        time.sleep(0.05)
        ser.write(START_COMMAND[2:])

    t = threading.Thread(target=writer)
    t.start()
    ok = port.wait_start(timeout=2.0)
    t.join()
    assert ok
    port.close()


def test_wait_start_timeout_returns_false():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    start = time.monotonic()
    ok = port.wait_start(timeout=0.2)
    elapsed = time.monotonic() - start
    assert not ok
    assert elapsed >= 0.15
    port.close()


def test_poll_start_detects_start_command():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    ser.write(b"\x00" + START_COMMAND + b"\x00")
    time.sleep(0.05)
    assert port.poll_start()
    assert port.is_start_active()
    port.close()


def test_poll_start_split_command():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    ser.write(START_COMMAND[:2])
    time.sleep(0.05)
    assert not port.poll_start()
    ser.write(START_COMMAND[2:])
    time.sleep(0.05)
    assert port.poll_start()
    port.close()


def test_poll_start_discards_heartbeat_burst_and_keeps_split_prefix():
    class BufferedSerial:
        def __init__(self, data=b""):
            self._data = bytearray(data)

        @property
        def in_waiting(self):
            return len(self._data)

        def read(self, size):
            data = bytes(self._data[:size])
            del self._data[:size]
            return data

        def write(self, data):
            self._data.extend(data)

    ser = BufferedSerial(START_COMMAND * 500 + START_COMMAND[:2])
    port = SerialPort(ser)

    assert port.poll_start()
    assert port._buf == bytearray(START_COMMAND[:2])

    ser.write(START_COMMAND[2:])
    assert port.poll_start()


def test_is_start_active_times_out():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    ser.write(START_COMMAND)
    time.sleep(0.05)
    assert port.poll_start()
    assert port.is_start_active(timeout=0.3)
    time.sleep(0.35)
    assert not port.is_start_active(timeout=0.3)
    port.close()


def test_send_result_writes_correct_frame():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    port.send_result(1, 2, 3, 3, 1, 5, 2, 4)
    time.sleep(0.05)
    data = ser.read(20)
    assert data == b"\xA5\x5A\x01\x02\x03\x03\x01\x05\x02\x04\x0D\x0A"
    port.close()


def test_send_result_raises_on_partial_write():
    ser = mock.Mock()
    ser.write.return_value = 11
    port = SerialPort(ser)

    with pytest.raises(TimeoutError, match="11/12"):
        port.send_result(1, 2, 3, 3, 1, 5, 2, 4)

    ser.flush.assert_not_called()


def test_serial_thread_does_not_log_each_start_heartbeat(monkeypatch, caplog):
    import logging
    import models.serial_port as serial_port

    class HeartbeatPort:
        def poll_start(self):
            return True

        def is_start_active(self, _timeout):
            return False

    thread = SerialThread.__new__(SerialThread)
    thread._port = HeartbeatPort()
    thread._start_timeout = 0.3
    thread._send_interval = 0.1
    thread._running = True
    thread._reported_missing_frame = False
    thread._reported_transmission = False

    def stop_after_iteration(_seconds):
        thread._running = False

    monkeypatch.setattr(serial_port.time, "sleep", stop_after_iteration)
    with caplog.at_level(logging.INFO):
        thread._run()

    assert not [record for record in caplog.records if "START received" in record.message]


def test_send_result_rejects_invalid_payload():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    with pytest.raises(ValueError):
        port.send_result(1, 2, 3, 3, 1, 5, 2, 300)
    port.close()


def test_send_result_rejects_duplicate_colors():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    with pytest.raises(ValueError):
        port.send_result(1, 1, 3, 3, 1, 5, 2, 4)
    port.close()


def test_send_result_rejects_duplicate_digits():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    with pytest.raises(ValueError):
        port.send_result(1, 2, 3, 3, 1, 5, 3, 4)
    port.close()


def test_send_result_accepts_numpy_integers():
    import numpy as np

    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    port.send_result(
        np.int64(1), np.uint8(2), 3, np.int32(3), 1, 5, 2, 4
    )
    time.sleep(0.05)
    assert ser.read(20) == b"\xA5\x5A\x01\x02\x03\x03\x01\x05\x02\x04\x0D\x0A"
    port.close()


def test_send_result_rejects_float_values():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    with pytest.raises(ValueError):
        port.send_result(1.5, 2, 3, 3, 1, 5, 2, 4)
    port.close()


def test_open_serial_port_uses_eight_n_one():
    fake_ser = mock.Mock(spec=serial.Serial)
    with mock.patch("serial.Serial", return_value=fake_ser) as mock_serial:
        port = open_serial_port("/dev/ttyS1", 921600, timeout=0.05)

    mock_serial.assert_called_once_with(
        "/dev/ttyS1",
        921600,
        bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
            write_timeout=0.05,
        )
    assert isinstance(port, SerialPort)
    assert port.ser is fake_ser
