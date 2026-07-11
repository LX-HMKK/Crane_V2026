import numbers
import time


START_COMMAND = bytes([0xA5, 0x5A, 0x01])
RESULT_HEAD = bytes([0xA5, 0x5A])
RESULT_TAIL = bytes([0x0D, 0x0A])
RESULT_FRAME_SIZE = 12


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
    """Handle START command reception and 12-byte result frame transmission."""

    def __init__(self, ser):
        self.ser = ser
        self._buf = bytearray()
        self.last_start_time = 0.0

    def poll_start(self):
        """Read available RX bytes and update last_start_time if START_COMMAND is seen.

        Returns True if a new START_COMMAND was detected during this call.
        The internal buffer is trimmed to at most len(START_COMMAND) - 1 bytes
        so split commands are not lost.
        """
        data = self.ser.read(self.ser.in_waiting or 0)
        if data:
            self._buf.extend(data)

        idx = self._buf.find(START_COMMAND)
        if idx >= 0:
            self.last_start_time = time.monotonic()
            self._buf = self._buf[idx + len(START_COMMAND) :]
            return True

        # Keep only the trailing bytes that could form the start of a command.
        max_keep = len(START_COMMAND) - 1
        if len(self._buf) > max_keep:
            self._buf = self._buf[-max_keep:]
        return False

    def is_start_active(self, timeout=0.3):
        """Return True if a START_COMMAND was received within `timeout` seconds."""
        return time.monotonic() - self.last_start_time < timeout

    def wait_start(self, timeout=None):
        """Blocking helper: keep polling until a START_COMMAND arrives or timeout.

        Provided for tests and backward compatibility; the competition loop
        should use poll_start() / is_start_active() directly.
        """
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        while True:
            if deadline is not None and time.monotonic() > deadline:
                return False
            self.poll_start()
            if self.is_start_active():
                return True

    def send_result(self, c1, c2, c3, n4, n5, n6, n7, n8):
        """Send the fixed 12-byte result frame: A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A.

        C1-C3 must be 1,2,3 (bean colors) each used exactly once.
        N4-N8 must be 1..5 (placement digits) each used exactly once.
        """
        raw_values = (c1, c2, c3, n4, n5, n6, n7, n8)
        for value in raw_values:
            if not isinstance(value, numbers.Integral) or not (0 <= value <= 255):
                raise ValueError(
                    f"Result payload values must be 0-255 ints, got {value!r}"
                )

        colors = [int(c1), int(c2), int(c3)]
        digits = [int(n4), int(n5), int(n6), int(n7), int(n8)]

        if sorted(colors) != [1, 2, 3]:
            raise ValueError(
                f"Pickup colors C1-C3 must be 1,2,3 exactly once, got {colors}"
            )
        if sorted(digits) != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Placement digits N4-N8 must be 1..5 exactly once, got {digits}"
            )

        frame = RESULT_HEAD + bytes(colors) + bytes(digits) + RESULT_TAIL
        self.ser.write(frame)
        self.ser.flush()

    def close(self):
        self.ser.close()
