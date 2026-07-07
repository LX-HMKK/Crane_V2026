import struct
import time


START_FRAME = bytes([0xAA, 0x55, 0x01, 0x0D, 0x0A])
TASK_FRAME_SIZE = 7
TASK_HEAD = bytes([0xBB, 0x66])
TASK_TAIL = bytes([0x1B, 0x1A])


class SerialPort:
    """Handle START frame reception and TASK frame transmission."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = bytearray()

    def wait_start(self, timeout=None):
        """Sliding-window match of AA 55 01 0D 0A."""
        deadline = None
        if timeout is not None:
            deadline = time.time() + timeout

        while True:
            if deadline and time.time() > deadline:
                return False
            byte = self.ser.read(1)
            if not byte:
                continue
            self.buf.extend(byte)
            if len(self.buf) > 16:
                self.buf.pop(0)
            if len(self.buf) >= 5 and self.buf[-5:] == START_FRAME:
                return True

    def send_task(self, p1, p2, p3):
        frame = struct.pack(
            ">2B3B2B",
            TASK_HEAD[0],
            TASK_HEAD[1],
            p1,
            p2,
            p3,
            TASK_TAIL[0],
            TASK_TAIL[1],
        )
        self.ser.write(frame)

    def close(self):
        self.ser.close()
