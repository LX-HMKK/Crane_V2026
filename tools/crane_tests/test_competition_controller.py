import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Keep this module independent of the heavy inference dependencies.
sys.modules["ultralytics"] = None
sys.modules["torch"] = None

from models.application.competition_controller import CompetitionResultController
from models.domain.mapping import DEFAULT_BEAN_TO_DIGIT
from models.domain.task_generator import TaskGenerator


LINES = {
    "bean": {"left_x": 500, "right_x": 800},
    "digit": {"x1": 200, "x2": 400, "x3": 600, "x4": 800},
}
BEAN_NAMES = {0: "yellow", 1: "green", 2: "white"}
DIGIT_NAMES = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}


def _bean_det(cls_id, cx, cy=360, conf=0.9):
    return [cx - 30, cy - 30, cx + 30, cy + 30, float(conf), cls_id]


def _digit_det(cls_id, cx, cy=360, conf=0.9):
    return _bean_det(cls_id, cx, cy, conf)


def _controller(serial):
    return CompetitionResultController(
        TaskGenerator(DEFAULT_BEAN_TO_DIGIT, LINES, BEAN_NAMES, DIGIT_NAMES),
        serial,
        {},
        update_interval=0.0,
        pair_max_skew_sec=0.15,
    )


def test_controller_rejects_stale_cross_camera_pair():
    serial = type("Serial", (), {"current_frame": None})()
    controller = _controller(serial)

    result = controller.update(
        [],
        [],
        bean_meta={"sequence": 3, "captured_at": 10.0},
        digit_meta={"sequence": 7, "captured_at": 10.2},
    )

    # Skew 0.2s > 0.15s; the pair is rejected and nothing is published.
    assert result is None
    assert serial.current_frame is None


def test_controller_debug_result_stays_out_of_serial_cache():
    serial = type("Serial", (), {"current_frame": None})()
    controller = _controller(serial)

    bean = [_bean_det(0, 300), _bean_det(1, 600), _bean_det(2, 900)]
    digit = [
        _digit_det(0, 100), _digit_det(1, 300), _digit_det(2, 500),
        _digit_det(3, 700), _digit_det(4, 900),
    ]

    result = controller.update(bean, digit, debug_forced=True)

    # DEBUG produces overlay data but does not publish to the serial cache.
    assert result is not None
    assert serial.current_frame is None
