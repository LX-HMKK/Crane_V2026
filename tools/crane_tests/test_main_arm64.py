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
