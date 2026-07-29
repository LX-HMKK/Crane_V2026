import numpy as np

from models.ui import CompetitionUI


def test_inferred_labels_use_readable_values():
    assert CompetitionUI._inferred_label("bean", 2, 2) == "INFER Z2: GREEN"
    assert CompetitionUI._inferred_label("digit", 6, 3) == "INFER Z6: 3"


def test_inferred_zone_labels_draw_only_for_inferred_zones():
    viewer = object.__new__(CompetitionUI)
    viewer.cfg = {
        "lines": {
            "bean": {"left_x": 400, "right_x": 800},
            "digit": {"x1": 250, "x2": 500, "x3": 750, "x4": 1000},
        }
    }
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    viewer._draw_inferred_zone_labels(
        frame,
        {1: (1, "detected", 0.9), 2: (2, "inferred", None)},
        "bean",
    )

    assert frame[280:440].any()
    assert not frame[:180].any()
