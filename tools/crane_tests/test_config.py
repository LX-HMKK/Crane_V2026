import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml


def test_vision_yaml_exists_and_parses():
    cfg_path = Path(__file__).parent.parent.parent / "configs" / "vision.yaml"
    assert cfg_path.exists(), "configs/vision.yaml must exist"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["serial"]["baudrate"] == 921600
    assert cfg["cameras"]["bean"]["device"] == "/dev/crane_cam_beans"
    assert cfg["mapping"]["bean_to_digit"]["yellow"] == 1
    assert cfg["models"]["bean"]["pt"] == "assets/beans_v4.pt"
    assert cfg["models"]["digit"]["pt"] == "assets/num.pt"


def test_models_package_exists():
    init_path = Path(__file__).parent.parent.parent / "models" / "__init__.py"
    assert init_path.exists(), "models/__init__.py must exist"
