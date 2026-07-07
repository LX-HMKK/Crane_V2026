import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.task_generator import TaskGenerator


def digit_det(zone, digit):
    """Helper: [x1, y1, x2, y2, conf, cls_id]; names map cls_id -> digit value."""
    centers = {
        4: (100, 100),  # top-left
        5: (100, 300),  # bottom-left
        6: (300, 300),  # bottom-middle
        7: (500, 300),  # bottom-right
        8: (500, 100),  # top-right
    }
    cx, cy = centers[zone]
    return [cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0, 0.9, digit - 1]


def bean_det(zone, cls_id):
    centers = {1: 100, 2: 300, 3: 500}
    cx = centers[zone]
    return [cx - 5.0, 0.0, cx + 5.0, 10.0, 0.9, cls_id]


def test_generate_task_success():
    # 豆子：zone1=yellow(0), zone2=green(1), zone3=white(2)
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    # 数字：4/5/6/7/8 分别为 1/2/3/4/5
    digit_names = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    digit_dets = [
        digit_det(4, 1), digit_det(5, 2), digit_det(6, 3),
        digit_det(7, 4), digit_det(8, 5),
    ]
    lines = {
        "bean": {"left_x": 213, "right_x": 426},
        "digit": {"horizontal_y": 240, "left_x": 213, "right_x": 426},
    }
    mapping = {"yellow": 1, "green": 2, "white": 3}
    gen = TaskGenerator(mapping, lines, digit_names)
    result, reason = gen.generate(bean_dets, digit_dets)
    # zone1(yellow->1) -> placement 4
    # zone2(green->2)  -> placement 5
    # zone3(white->3)  -> placement 6
    assert result == (4, 5, 6)
    assert reason == "ok"


def test_generate_task_missing_digit():
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    digit_names = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    digit_dets = [digit_det(4, 1)]  # missing 2 and 3
    lines = {
        "bean": {"left_x": 213, "right_x": 426},
        "digit": {"horizontal_y": 240, "left_x": 213, "right_x": 426},
    }
    mapping = {"yellow": 1, "green": 2, "white": 3}
    gen = TaskGenerator(mapping, lines, digit_names)
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result is None
    assert "missing" in reason.lower()
