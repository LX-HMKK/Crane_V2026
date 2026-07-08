import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mapping import DEFAULT_BEAN_TO_DIGIT, bean_name_to_digit
from models.task_generator import TaskGenerator


def digit_det(zone, digit, conf=0.9):
    """Helper: [x1, y1, x2, y2, conf, cls_id]; names map cls_id -> digit value."""
    centers = {
        4: (100, 100),  # top-left
        5: (100, 300),  # bottom-left
        6: (300, 300),  # bottom-middle
        7: (500, 300),  # bottom-right
        8: (500, 100),  # top-right
    }
    cx, cy = centers[zone]
    return [cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0, conf, digit - 1]


def bean_det(zone, cls_id, conf=0.9):
    centers = {1: 100, 2: 300, 3: 500}
    cx = centers[zone]
    return [cx - 5.0, 0.0, cx + 5.0, 10.0, conf, cls_id]


def make_lines():
    return {
        "bean": {"left_x": 213, "right_x": 426},
        "digit": {"horizontal_y": 240, "left_x": 213, "right_x": 426},
    }


def make_names():
    return (
        {"yellow": 1, "green": 2, "white": 3},  # mapping
        {0: "white", 1: "green", 2: "yellow"},  # bean_names
        {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"},  # digit_names
    )


def make_gen(mapping=None):
    mapping_default, bean_names, digit_names = make_names()
    lines = make_lines()
    return TaskGenerator(mapping or mapping_default, lines, bean_names, digit_names)


def test_generate_task_success():
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    digit_dets = [
        digit_det(4, 1), digit_det(5, 2), digit_det(6, 3),
        digit_det(7, 4), digit_det(8, 5),
    ]
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result == (4, 5, 6)
    assert reason == "ok"


def test_generate_task_missing_digit():
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    digit_dets = [digit_det(4, 1)]  # missing 2 and 3
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result is None
    assert "missing" in reason.lower()


def test_generate_task_unknown_bean_class():
    bean_dets = [bean_det(1, 99), bean_det(2, 1), bean_det(3, 0)]
    digit_dets = [digit_det(4, 1), digit_det(5, 2), digit_det(6, 3)]
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result is None
    assert "unknown bean class 99" in reason


def test_generate_task_missing_bean_in_zone():
    bean_dets = [bean_det(1, 2), bean_det(3, 0)]  # no bean in zone 2
    digit_dets = [digit_det(4, 1), digit_det(5, 2), digit_det(6, 3)]
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result is None
    assert "missing bean in pickup zone 2" in reason


def test_generate_task_bean_zone_tiebreak():
    # Zone 1: low-conf yellow (digit 1, no placement) vs high-conf green (digit 2).
    # If yellow wins, target digit 1 is missing and generation fails.
    bean_dets = [
        bean_det(1, 2, conf=0.5),  # yellow
        bean_det(1, 1, conf=0.95),  # green
        bean_det(2, 1, conf=0.9),
        bean_det(3, 0, conf=0.9),
    ]
    digit_dets = [digit_det(5, 2), digit_det(6, 3)]  # value 1 is not placed
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result == (5, 5, 6)
    assert reason == "ok"


def test_generate_task_digit_zone_tiebreak():
    # Zone 4: low-conf value 1 vs high-conf value 2.
    # Winner value 2 means target digit 1 must be found in zone 5.
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    digit_dets = [
        [95.0, 95.0, 105.0, 105.0, 0.5, 0],   # zone 4, value 1
        [95.0, 95.0, 105.0, 105.0, 0.99, 1],  # zone 4, value 2
        digit_det(5, 1),  # zone 5, value 1
        digit_det(6, 3),  # zone 6, value 3
    ]
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result == (5, 4, 6)
    assert reason == "ok"


def test_generate_task_duplicate_digit_values():
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    # Values 1 appear in both zone 4 and zone 5.
    digit_dets = [
        digit_det(4, 1), digit_det(5, 1),
        digit_det(6, 3),
    ]
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result is None
    assert "duplicate digit 1" in reason


def test_generate_task_boundary_lines():
    # Beans on boundary lines; digits on boundary lines.
    # Bean cx == left_x (213) falls into zone 2; cx == right_x (426) into zone 3.
    bean_dets = [
        [100.0, 0.0, 106.0, 10.0, 0.9, 2],   # zone 1, yellow -> 1
        [210.0, 0.0, 216.0, 10.0, 0.9, 1],   # cx center 213, zone 2, green -> 2
        [423.0, 0.0, 429.0, 10.0, 0.9, 0],   # cx center 426, zone 3, white -> 3
    ]
    # Digit cy == horizontal_y (240) falls to bottom row;
    # cx == left_x (213) in bottom row falls into zone 6.
    digit_dets = [
        [95.0, 95.0, 105.0, 105.0, 0.9, 0],     # zone 4, value 1
        [95.0, 240.0, 105.0, 250.0, 0.9, 1],    # zone 5, value 2
        [210.0, 240.0, 216.0, 250.0, 0.9, 2],   # zone 6, value 3
    ]
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result == (4, 5, 6)
    assert reason == "ok"


def test_generate_task_malformed_lines():
    mapping, bean_names, digit_names = make_names()
    gen = TaskGenerator(mapping, {"bean": {"left_x": 213}}, bean_names, digit_names)
    result, reason = gen.generate([], [])
    assert result is None
    assert "invalid bean line config" in reason


def test_generate_task_reversed_bean_lines():
    mapping, bean_names, digit_names = make_names()
    lines = make_lines()
    lines["bean"]["left_x"] = 500
    lines["bean"]["right_x"] = 100
    gen = TaskGenerator(mapping, lines, bean_names, digit_names)
    result, reason = gen.generate([], [])
    assert result is None
    assert "left_x must be less than right_x" in reason


def test_generate_task_negative_line_coordinate():
    mapping, bean_names, digit_names = make_names()
    lines = make_lines()
    lines["digit"]["left_x"] = -10
    gen = TaskGenerator(mapping, lines, bean_names, digit_names)
    result, reason = gen.generate([], [])
    assert result is None
    assert "non-negative" in reason


def test_generate_task_malformed_digit_lines():
    mapping, bean_names, digit_names = make_names()
    gen = TaskGenerator(
        mapping,
        {"bean": {"left_x": 213, "right_x": 426}, "digit": {"left_x": 213}},
        bean_names,
        digit_names,
    )
    result, reason = gen.generate([], [])
    assert result is None
    assert "invalid digit line config" in reason


def test_generate_task_malformed_detection():
    gen = make_gen()
    result, reason = gen.generate([[0, 1, 2, 3]], [])
    assert result is None
    assert "malformed detection" in reason


def test_generate_task_unknown_digit_class():
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    digit_dets = [
        digit_det(4, 1),
        # Unknown class placed alone in zone 5 so it is selected and reported.
        [95.0, 300.0, 105.0, 310.0, 0.9, 99],
        digit_det(6, 3),
    ]
    gen = make_gen()
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result is None
    assert "unknown digit class 99" in reason


def test_generate_task_non_numeric_digit_name():
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    digit_dets = [digit_det(4, 1), digit_det(5, 2), digit_det(6, 3)]
    mapping, bean_names, _ = make_names()
    digit_names = {0: "1", 1: "2", 2: "not_a_number", 3: "4", 4: "5"}
    gen = TaskGenerator(mapping, make_lines(), bean_names, digit_names)
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result is None
    assert "non-numeric digit name 'not_a_number'" in reason


def test_default_mapping_and_helper():
    # Exercise models.mapping from the task generator side.
    assert bean_name_to_digit("yellow") == 1
    assert bean_name_to_digit("green") == 2
    assert bean_name_to_digit("white") == 3
    assert bean_name_to_digit("unknown") is None

    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    digit_dets = [digit_det(4, 1), digit_det(5, 2), digit_det(6, 3)]
    mapping, bean_names, digit_names = make_names()
    # Use the shared default mapping via the helper.
    gen = TaskGenerator(DEFAULT_BEAN_TO_DIGIT, make_lines(), bean_names, digit_names)
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result == (4, 5, 6)
    assert reason == "ok"
