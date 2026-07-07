import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def test_vision_yaml_exists_and_parses():
    cfg_path = Path(__file__).parent.parent / "configs" / "vision.yaml"
    assert cfg_path.exists(), "configs/vision.yaml must exist"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["serial"]["baudrate"] == 115200
    assert cfg["cameras"]["bean"]["device"] == 0
    assert cfg["mapping"]["bean_to_digit"]["yellow"] == 1


def test_models_package_exists():
    init_path = Path(__file__).parent.parent / "models" / "__init__.py"
    assert init_path.exists(), "models/__init__.py must exist"
