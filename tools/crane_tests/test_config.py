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
    assert cfg["models"]["bean"]["pt"] == "assets/beans_v7.pt"
    assert cfg["models"]["digit"]["pt"] == "assets/num_v7.pt"
    assert cfg["inference"]["start_gated"] is True
    assert cfg["debug"]["force_inference"] is False


def test_models_namespace_package_imports():
    import models

    models_path = (Path(__file__).parent.parent.parent / "models").resolve()
    assert models_path in {Path(path).resolve() for path in models.__path__}
