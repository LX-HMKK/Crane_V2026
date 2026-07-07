import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import serial
import time
import threading
from models.serial_port import SerialPort, START_FRAME


def test_wait_start_detects_start_frame():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)

    def writer():
        time.sleep(0.05)
        ser.write(b"\x00\x01" + START_FRAME + b"\x00")

    t = threading.Thread(target=writer)
    t.start()
    ok = port.wait_start(timeout=2.0)
    t.join()
    assert ok
    port.close()


def test_send_task_writes_correct_frame():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    port.send_task(5, 6, 4)
    time.sleep(0.05)
    data = ser.read(20)
    assert data == b"\xBB\x66\x05\x06\x04\x1B\x1A"
    port.close()
