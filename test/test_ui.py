import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from unittest.mock import MagicMock
from models.ui import CalibrationUI


def test_ui_saves_lines_on_change(tmp_path):
    cfg_path = tmp_path / "vision.yaml"
    cfg = {
        "lines": {
            "bean": {"left_x": 100, "right_x": 300},
            "digit": {"horizontal_y": 200, "left_x": 100, "right_x": 300},
        },
        "inference": {"imgsz": 640},
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    cam_bean = MagicMock()
    cam_digit = MagicMock()
    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    ui._set_line("bean", "left_x", 150)
    ui._set_line("digit", "horizontal_y", 250)

    with open(cfg_path, "r") as f:
        saved = yaml.safe_load(f)
    assert saved["lines"]["bean"]["left_x"] == 150
    assert saved["lines"]["digit"]["horizontal_y"] == 250
