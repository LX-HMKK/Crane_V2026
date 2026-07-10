import numbers
import struct
import time


START_FRAME = bytes([0xAA, 0x55, 0x01, 0x0D, 0x0A])
TASK_FRAME_SIZE = 7
TASK_HEAD = bytes([0xBB, 0x66])
TASK_TAIL = bytes([0x1B, 0x1A])


def open_serial_port(port, baudrate, timeout=0.05, **kwargs):
    """Open a pyserial handle and wrap it in SerialPort.

    Defaults to 8/N/1 with no software/hardware flow control.
    Extra keyword arguments are forwarded to serial.Serial.
    """
    import serial

    return SerialPort(
        serial.Serial(
            port,
            baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            **kwargs,
        )
    )


class SerialPort:
    """Handle START frame reception and TASK frame transmission."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = bytearray()

    def wait_start(self, timeout=None):
        """Sliding-window match of AA 55 01 0D 0A."""
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        while True:
            if deadline is not None and time.monotonic() > deadline:
                return False
            byte = self.ser.read(1)
            if not byte:
                continue
            self.buf.extend(byte)
            if len(self.buf) > 16:
                self.buf.pop(0)
            if len(self.buf) >= 5 and self.buf[-5:] == START_FRAME:
                # Clear buffer so a subsequent call starts fresh.
                self.buf.clear()
                return True

    def send_task(self, p1, p2, p3):
        """Send a fixed 7-byte TASK frame: BB 66 p1 p2 p3 1B 1A."""
        for p in (p1, p2, p3):
            if not isinstance(p, numbers.Integral) or not (0 <= p <= 255):
                raise ValueError(
                    f"TASK payload values must be 0-255 ints, got {p!r}"
                )
        frame = TASK_HEAD + bytes([p1, p2, p3]) + TASK_TAIL
        self.ser.write(frame)
        self.ser.flush()

    def close(self):
        self.ser.close()
