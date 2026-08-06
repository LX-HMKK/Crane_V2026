"""Unit tests for the RDK entry point's headless result path."""

from scripts.main_arm64 import HeadlessCompetitionLoop, build_completed_result


class _TaskGenerator:
    def __init__(self):
        self.calls = []

    def generate_frame_with_completion(self, bean_dets, digit_dets, **kwargs):
        self.calls.append((bean_dets, digit_dets, kwargs))
        return (1, 2, 3, 4, 5, 1, 2, 3), {}, {}, "ok"


def test_headless_result_uses_ui_logical_completion_settings():
    task_gen = _TaskGenerator()
    cfg = {
        "logic": {
            "confidence_threshold": 0.65,
            "max_inferred_beans": 2,
            "max_inferred_digits": 3,
        }
    }

    frame, reason = build_completed_result(task_gen, [["bean"]], [["digit"]], cfg)

    assert frame == (1, 2, 3, 4, 5, 1, 2, 3)
    assert reason == "ok"
    assert task_gen.calls == [
        (
            [["bean"]],
            [["digit"]],
            {
                "confidence_threshold": 0.65,
                "max_inferred_beans": 2,
                "max_inferred_digits": 3,
            },
        )
    ]


def test_headless_result_uses_existing_logic_defaults():
    task_gen = _TaskGenerator()

    _frame, _reason = build_completed_result(task_gen, [], [], {})

    assert task_gen.calls[0][2] == {
        "confidence_threshold": 0.5,
        "max_inferred_beans": 1,
        "max_inferred_digits": 2,
    }


def test_headless_loop_hands_off_only_after_a_complete_scan(monkeypatch):
    import scripts.main_arm64 as main_arm64

    monkeypatch.setattr(
        main_arm64,
        "scan_once_dets",
        lambda *_args, **_kwargs: (["bean"], ["digit"], "ok"),
    )
    task_gen = _TaskGenerator()
    serial = type("Serial", (), {"current_frame": None})()
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), task_gen, serial, 0.0
    )

    loop.run(handoff_ready=lambda: True)

    assert serial.current_frame == (1, 2, 3, 4, 5, 1, 2, 3)
    assert len(task_gen.calls) == 1


def test_headless_camera_failures_back_off_and_rate_limit_logs(monkeypatch, caplog):
    import logging
    import scripts.main_arm64 as main_arm64

    scan_kwargs = []

    def failed_scan(*_args, **kwargs):
        scan_kwargs.append(kwargs)
        kwargs["on_capture_failure"](False, False)
        return [], [], "too many camera capture failures"

    monkeypatch.setattr(main_arm64, "scan_once_dets", failed_scan)
    sleep_calls = []
    monkeypatch.setattr(main_arm64.time, "sleep", sleep_calls.append)

    serial = type("Serial", (), {"current_frame": (1, 2, 3, 4, 5, 1, 2, 3)})()
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), object(), serial, 0.0
    )
    handoff_checks = iter((False, True))

    with caplog.at_level(logging.WARNING):
        loop.run(handoff_ready=lambda: next(handoff_checks))

    assert sleep_calls == [main_arm64.CAMERA_FAILURE_BACKOFF_SEC]
    assert len(scan_kwargs) == 2
    assert all(call["on_capture_failure"].__self__ is loop for call in scan_kwargs)
    capture_logs = [
        record for record in caplog.records
        if "Camera capture failure" in record.message
    ]
    assert len(capture_logs) == 1


def test_create_serial_thread_passes_reconnect_config(monkeypatch):
    """serial.reconnect in vision.yaml must reach the SerialThread constructor."""
    import scripts.main_arm64 as main_arm64

    captured = {}

    def fake_serial_thread(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(main_arm64, "SerialThread", fake_serial_thread)
    cfg = {
        "serial": {
            "port": "/dev/ttyS1",
            "baudrate": 921600,
            "timeout": 0.05,
            "reconnect": {"enabled": False, "after_failures": 9, "interval_sec": 2.0},
        },
        "protocol": {"start_timeout_sec": 0.3, "send_interval_sec": 0.05},
    }
    main_arm64._create_serial_thread(cfg)

    assert captured["args"] == ("/dev/ttyS1", 921600)
    assert captured["kwargs"]["timeout"] == 0.05
    assert captured["kwargs"]["start_timeout"] == 0.3
    assert captured["kwargs"]["send_interval"] == 0.05
    assert captured["kwargs"]["reconnect_enabled"] is False
    assert captured["kwargs"]["reconnect_after"] == 9
    assert captured["kwargs"]["reconnect_interval"] == 2.0


def test_headless_camera_outage_keeps_last_frame_sticky(monkeypatch):
    """Camera outage must NOT clear the last valid serial frame (sticky)."""
    import scripts.main_arm64 as main_arm64

    def failed_scan(*_args, **kwargs):
        kwargs["on_capture_failure"](False, False)
        return [], [], "too many camera capture failures"

    monkeypatch.setattr(main_arm64, "scan_once_dets", failed_scan)
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    serial = type("Serial", (), {"current_frame": (1, 2, 3, 4, 5, 1, 2, 3)})()
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), object(), serial, 0.0
    )
    scans = {"n": 0}

    def handoff_ready():
        scans["n"] += 1
        return scans["n"] >= 12  # run 12 failed scans, past the 10-clear threshold

    loop.run(handoff_ready=handoff_ready)

    assert serial.current_frame == (1, 2, 3, 4, 5, 1, 2, 3)


def test_headless_logic_failure_clears_after_outage_recovery(monkeypatch):
    """After a camera outage, a later logic-failure streak must still clear.

    Regression: outage scans used to inflate consecutive_failures past 10, so
    the clear-on-10 could never fire again after the camera came back.
    """
    import scripts.main_arm64 as main_arm64

    class FailingTaskGen:
        def generate_frame_with_completion(self, *_args, **_kwargs):
            return None, {}, {}, "no arrangement"

    calls = {"n": 0}

    def scanning(*_args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 12:
            kwargs["on_capture_failure"](False, False)
            return [], [], "too many camera capture failures"
        return [], [], "ok"

    monkeypatch.setattr(main_arm64, "scan_once_dets", scanning)
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    serial = type("Serial", (), {"current_frame": (1, 2, 3, 4, 5, 1, 2, 3)})()
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), FailingTaskGen(), serial, 0.0
    )
    scans = {"n": 0}

    def handoff_ready():
        scans["n"] += 1
        return scans["n"] >= 24  # 12 outage scans + 12 logic-failure scans

    loop.run(handoff_ready=handoff_ready)

    # After 10 consecutive logic failures post-recovery, the sticky frame clears.
    assert serial.current_frame is None


# ── START-gated inference (headless) ───────────────────────────────────


def _serial_stub(current_frame=None, is_start_active=True):
    """SerialThread stand-in with an explicit is_start_active attribute."""
    return type(
        "Serial",
        (),
        {"current_frame": current_frame, "is_start_active": is_start_active},
    )()


def test_headless_idles_without_start(monkeypatch):
    """With START inactive the headless loop must not scan, and must clear
    any stale serial frame so it is not transmitted on START resume."""
    import scripts.main_arm64 as main_arm64

    scan_calls = []

    def counting_scan(*_args, **_kwargs):
        scan_calls.append(1)
        return [], [], "ok"

    monkeypatch.setattr(main_arm64, "scan_once_dets", counting_scan)
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    serial = _serial_stub((1, 2, 3, 4, 5, 1, 2, 3), is_start_active=False)
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), object(), serial, 0.0
    )
    checks = {"n": 0}

    def handoff_ready():
        checks["n"] += 1
        return checks["n"] >= 3  # a few idle ticks, then hand off

    loop.run(handoff_ready=handoff_ready)

    assert scan_calls == []
    assert serial.current_frame is None
    assert loop.consecutive_failures == 0


def test_headless_scans_when_start_active(monkeypatch):
    """With START active the headless loop scans and publishes a result."""
    import scripts.main_arm64 as main_arm64

    monkeypatch.setattr(
        main_arm64,
        "scan_once_dets",
        lambda *_a, **_k: (["bean"], ["digit"], "ok"),
    )
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    serial = _serial_stub(None, is_start_active=True)
    task_gen = _TaskGenerator()
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), task_gen, serial, 0.0
    )
    loop.run(handoff_ready=lambda: True)

    assert serial.current_frame == (1, 2, 3, 4, 5, 1, 2, 3)
    assert len(task_gen.calls) == 1


def test_headless_idle_does_not_count_as_failures(monkeypatch):
    """Idle ticks must not run scans, and must reset stale failure state."""
    import scripts.main_arm64 as main_arm64

    scan_calls = []

    def counting_scan(*_args, **_kwargs):
        scan_calls.append(1)
        return [], [], "ok"

    monkeypatch.setattr(main_arm64, "scan_once_dets", counting_scan)
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    serial = _serial_stub((1, 2, 3, 4, 5, 1, 2, 3), is_start_active=False)
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), object(), serial, 0.0
    )
    loop.consecutive_failures = 5
    checks = {"n": 0}

    def handoff_ready():
        checks["n"] += 1
        return checks["n"] >= 12  # run past the clear-on-10 threshold in idle

    loop.run(handoff_ready=handoff_ready)

    assert loop.consecutive_failures == 0
    assert serial.current_frame is None
    assert scan_calls == []


def test_headless_gate_off_clears_stale_frame_before_resume(monkeypatch):
    """A frame published before START dropped must be cleared while idle, so
    it is never transmitted when START resumes (F1)."""
    import scripts.main_arm64 as main_arm64

    monkeypatch.setattr(
        main_arm64,
        "scan_once_dets",
        lambda *_a, **_k: (["bean"], ["digit"], "ok"),
    )
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    class FlippingSerial:
        current_frame = (1, 2, 3, 4, 5, 1, 2, 3)

        def __init__(self):
            self.active = True

        @property
        def is_start_active(self):
            return self.active

    serial = FlippingSerial()
    task_gen = _TaskGenerator()
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), task_gen, serial, 0.0
    )

    # START active: one scan publishes the frame to the serial cache.
    loop._scan_once()
    assert serial.current_frame == (1, 2, 3, 4, 5, 1, 2, 3)

    # START drops: the idle branch must clear the stale cache before resume.
    serial.active = False
    checks = iter((False, True))
    loop.run(handoff_ready=lambda: next(checks))

    assert serial.current_frame is None


def test_headless_handoff_during_idle_returns_promptly(monkeypatch):
    """The UI handoff must still be honored while inference is gated off."""
    import scripts.main_arm64 as main_arm64

    scan_calls = []

    def counting_scan(*_args, **_kwargs):
        scan_calls.append(1)
        return [], [], "ok"

    monkeypatch.setattr(main_arm64, "scan_once_dets", counting_scan)
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    serial = _serial_stub(None, is_start_active=False)
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), object(), serial, 0.0
    )
    loop.run(handoff_ready=lambda: True)

    assert scan_calls == []


def test_headless_debug_force_overrides_inactive_start(monkeypatch):
    """debug_force_inference bypasses an inactive START and runs inference,
    but stays display-only: it never writes the serial cache (F4)."""
    import scripts.main_arm64 as main_arm64

    monkeypatch.setattr(
        main_arm64,
        "scan_once_dets",
        lambda *_a, **_k: (["bean"], ["digit"], "ok"),
    )
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    serial = _serial_stub(None, is_start_active=False)
    task_gen = _TaskGenerator()
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), task_gen, serial, 0.0
    )
    loop.debug_force_inference = True
    loop.run(handoff_ready=lambda: True)

    # Inference ran (task_gen called) but the serial cache stayed empty.
    assert len(task_gen.calls) == 1
    assert serial.current_frame is None


def test_headless_start_gated_false_scans_without_start(monkeypatch):
    """inference.start_gated=False disables the gate entirely."""
    import scripts.main_arm64 as main_arm64

    monkeypatch.setattr(
        main_arm64,
        "scan_once_dets",
        lambda *_a, **_k: (["bean"], ["digit"], "ok"),
    )
    monkeypatch.setattr(main_arm64.time, "sleep", lambda _s: None)

    serial = _serial_stub(None, is_start_active=False)
    task_gen = _TaskGenerator()
    loop = HeadlessCompetitionLoop(
        {"inference": {"start_gated": False}},
        object(), object(), object(), object(), task_gen, serial, 0.0,
    )
    loop.run(handoff_ready=lambda: True)

    assert serial.current_frame == (1, 2, 3, 4, 5, 1, 2, 3)


def test_headless_idle_uses_poll_interval(monkeypatch):
    """The idle branch must sleep, never busy-loop the CPU."""
    import scripts.main_arm64 as main_arm64

    monkeypatch.setattr(
        main_arm64, "scan_once_dets", lambda *_a, **_k: ([], [], "ok")
    )
    sleeps = []
    monkeypatch.setattr(main_arm64.time, "sleep", sleeps.append)

    serial = _serial_stub(None, is_start_active=False)
    loop = HeadlessCompetitionLoop(
        {}, object(), object(), object(), object(), object(), serial, 0.0
    )
    checks = iter((False, True))  # one idle tick, then hand off
    loop.run(handoff_ready=lambda: next(checks))

    assert sleeps == [main_arm64.IDLE_POLL_INTERVAL_SEC]
