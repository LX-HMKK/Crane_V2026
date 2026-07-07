import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import serial
import time
import threading

import main as main_module
from models.serial_port import START_FRAME, SerialPort


def test_main_loads_vision_config(tmp_path):
    cfg_path = tmp_path / "vision.yaml"
    cfg_path.write_text("lines:\n  bean:\n    left_x: 100\n", encoding="utf-8")
    cfg = main_module.load_vision_cfg(str(cfg_path))
    assert cfg["lines"]["bean"]["left_x"] == 100


def test_serial_frame_round_trip():
    ser = serial.serial_for_url("loop://", timeout=0.01)

    def writer():
        time.sleep(0.1)
        ser.write(START_FRAME)

    t = threading.Thread(target=writer)
    t.start()

    port = SerialPort(ser)
    ok = port.wait_start(timeout=2.0)
    t.join()
    assert ok
    port.send_task(4, 5, 6)
    time.sleep(0.05)
    assert ser.read(10) == b"\xBB\x66\x04\x05\x06\x1B\x1A"
    port.close()
