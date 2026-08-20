import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.application.competition_controller import CompetitionResultController


class _Serial:
    def __init__(self):
        self.current_frame = None


class _TaskGenerator:
    def __init__(self):
        self.calls = 0

    def generate_frame_with_completion(self, *_args, **_kwargs):
        self.calls += 1
        return (
            (1, 2, 3, 1, 2, 3, 4, 5),
            {1: (1, "detected", 0.9)},
            {4: (1, "detected", 0.9)},
            "ok",
        )


def _controller(task_generator, serial):
    return CompetitionResultController(
        task_generator,
        serial,
        {},
        update_interval=0.0,
        pair_max_skew_sec=0.15,
    )


def test_controller_rejects_stale_cross_camera_pair():
    task_generator = _TaskGenerator()
    controller = _controller(task_generator, _Serial())

    result = controller.update(
        [],
        [],
        bean_meta={"sequence": 3, "captured_at": 10.0},
        digit_meta={"sequence": 7, "captured_at": 10.2},
    )

    assert result is None
    assert task_generator.calls == 0


def test_controller_publishes_time_aligned_pair():
    task_generator = _TaskGenerator()
    serial = _Serial()
    controller = _controller(task_generator, serial)

    result = controller.update(
        [],
        [],
        bean_meta={"sequence": 3, "captured_at": 10.0},
        digit_meta={"sequence": 7, "captured_at": 10.1},
    )

    assert result.bean_zones == {1: (1, "detected", 0.9)}
    assert serial.current_frame == (1, 2, 3, 1, 2, 3, 4, 5)


def test_controller_keeps_debug_result_out_of_serial_cache():
    task_generator = _TaskGenerator()
    serial = _Serial()
    controller = _controller(task_generator, serial)

    result = controller.update([], [], debug_forced=True)

    assert result is not None
    assert serial.current_frame is None
