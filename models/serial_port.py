import logging
import numbers
import threading
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

    # A result frame is only 12 bytes; bound writes to the same short timeout
    # as reads so a stalled UART cannot indefinitely block the serial worker.
    kwargs.setdefault("write_timeout", timeout)
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
        All complete commands from a heartbeat burst are consumed at once. The
        internal buffer keeps at most len(START_COMMAND) - 1 trailing bytes so
        split commands are not lost or allowed to accumulate.
        """
        data = self.ser.read(self.ser.in_waiting or 0)
        if data:
            self._buf.extend(data)

        last_idx = self._buf.rfind(START_COMMAND)
        if last_idx >= 0:
            self.last_start_time = time.monotonic()
            self._buf = self._buf[last_idx + len(START_COMMAND) :]
            max_keep = len(START_COMMAND) - 1
            if len(self._buf) > max_keep:
                self._buf = self._buf[-max_keep:]
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
        written = self.ser.write(frame)
        if written != len(frame):
            raise TimeoutError(
                f"Serial write timed out after {written}/{len(frame)} result bytes"
            )
        # write_timeout bounds the kernel enqueue. Avoid flush(), whose driver
        # drain can block indefinitely despite a finite write timeout.

    def close(self):
        self.ser.close()


class SerialThread:
    """Background thread: continuous serial RX polling + timed result TX.

    Owns a SerialPort exclusively — all serial I/O happens on this thread.
    The main thread sets ``current_frame`` when vision produces a new result;
    the serial thread sends it at *send_interval* Hz whenever START is active.
    """

    def __init__(self, device, baudrate, timeout=0.05,
                 start_timeout=0.3, send_interval=0.1,
                 reconnect_enabled=True, reconnect_after=5,
                 reconnect_interval=1.0):
        self._device = device
        self._baudrate = baudrate
        self._timeout = timeout
        self._port = open_serial_port(device, baudrate, timeout=timeout)
        self._start_timeout = start_timeout
        self._send_interval = send_interval
        self._reconnect_enabled = reconnect_enabled
        self._reconnect_after = max(reconnect_after, 1)
        self._reconnect_interval = max(reconnect_interval, 0.05)
        self._consecutive_failures = 0
        self._last_reopen_at = 0.0
        self._current_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._reported_missing_frame = False
        self._reported_transmission = False

    # ── thread-safe properties (main thread ↔ serial thread) ──────────

    @property
    def current_frame(self):
        with self._lock:
            return self._current_frame

    @current_frame.setter
    def current_frame(self, frame):
        with self._lock:
            self._current_frame = frame

    @property
    def connected(self):
        """True while a serial port handle is held (even mid-reconnect)."""
        return self._port is not None

    @property
    def is_start_active(self):
        if self._port is None:
            return False
        return self._port.is_start_active(self._start_timeout)

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self):
        """Launch the background serial thread (daemon)."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("SerialThread is already running")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the thread to exit, join, and close the serial port."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._port is not None:
            self._port.close()

    # ── internal ───────────────────────────────────────────────────────

    def _run(self):
        last_send = 0.0
        while self._running:
            try:
                if self._port is None:
                    # Previous reopen failed; retry on the reconnect interval.
                    self._maybe_reopen()
                    time.sleep(0.1)
                    continue
                self._port.poll_start()
                if self._port.is_start_active(self._start_timeout):
                    frame = self.current_frame
                    if frame is not None:
                        self._reported_missing_frame = False
                        now = time.monotonic()
                        if now - last_send >= self._send_interval:
                            self._port.send_result(*frame)
                            if not self._reported_transmission:
                                logging.info("Serial result transmission active")
                                self._reported_transmission = True
                            last_send = now
                    elif not self._reported_missing_frame:
                        logging.warning(
                            "Serial START is active but no valid vision result is available"
                        )
                        self._reported_missing_frame = True
                else:
                    self._reported_missing_frame = False
                    self._reported_transmission = False
                self._consecutive_failures = 0
            except Exception:
                self._consecutive_failures += 1
                self._maybe_reopen()
                logging.exception("SerialThread send failed")
                time.sleep(1.0)  # backoff on persistent fault
            time.sleep(0.005)  # 200 Hz poll rate

    def _maybe_reopen(self):
        """Attempt a reopen after persistent failures, rate-limited."""
        if not self._reconnect_enabled:
            return
        if self._consecutive_failures < self._reconnect_after:
            return
        if time.monotonic() - self._last_reopen_at < self._reconnect_interval:
            return
        self._last_reopen_at = time.monotonic()
        self._reopen()

    def _reopen(self):
        """Close the dead port and open a fresh handle on the same device."""
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                logging.warning("Failed to close serial port before reopen")
        self._port = None
        try:
            self._port = open_serial_port(
                self._device, self._baudrate, timeout=self._timeout
            )
            logging.info("Serial port %s reconnected", self._device)
            self._consecutive_failures = 0
        except Exception:
            logging.warning("Serial port %s reconnect failed", self._device)
            self._port = None
