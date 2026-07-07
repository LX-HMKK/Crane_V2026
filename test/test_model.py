from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

from models.bean_model import BeanModel
from models.bpu_loader import BpuModel
from models.digit_model import DigitModel


def test_bpu_model_raises_on_x86():
    with pytest.raises(EnvironmentError):
        BpuModel("assets/num.bin", "assets/num_rdkx5_config.yaml")


def _write_cfg(tmp_path, **kwargs):
    cfg = {
        "imgsz": 640,
        "nc": 10,
        "names": [str(i) for i in range(10)],
        "reg_max": 7,
        "strides": [8, 16, 32],
    }
    cfg.update(kwargs)
    cfg_file = tmp_path / "num_rdkx5_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg, cfg_file


def test_bpu_model_happy_path_with_mock(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    cfg, cfg_file = _write_cfg(tmp_path)
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    mock_model = mock.Mock()
    mock_model.forward.return_value = ["output"]
    mock_load = mock.Mock(return_value=[mock_model])
    monkeypatch.setattr("models.bpu_loader.dnn", mock.Mock(load=mock_load))

    model = BpuModel(str(bin_file), str(cfg_file))
    assert model.cfg == cfg

    tensor = np.zeros((1, 3, 640, 640), dtype=np.uint8)
    result = model.infer(tensor)
    assert result == ["output"]
    mock_load.assert_called_once_with(str(bin_file))
    mock_model.forward.assert_called_once()


def test_bpu_model_load_returns_non_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    _, cfg_file = _write_cfg(tmp_path)
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    monkeypatch.setattr(
        "models.bpu_loader.dnn", mock.Mock(load=mock.Mock(return_value=object()))
    )

    with pytest.raises(TypeError, match="sequence"):
        BpuModel(str(bin_file), str(cfg_file))


def test_bpu_model_missing_bin(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    _, cfg_file = _write_cfg(tmp_path)

    with pytest.raises(FileNotFoundError, match="bin not found"):
        BpuModel(str(tmp_path / "missing.bin"), str(cfg_file))


def test_bpu_model_missing_config_key(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    cfg_file = tmp_path / "num_rdkx5_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"imgsz": 640}, f)
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    monkeypatch.setattr(
        "models.bpu_loader.dnn", mock.Mock(load=mock.Mock(return_value=[mock.Mock()]))
    )

    with pytest.raises(ValueError, match="missing required key"):
        BpuModel(str(bin_file), str(cfg_file))


def test_bpu_model_infer_validation(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    _, cfg_file = _write_cfg(tmp_path)
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    mock_model = mock.Mock()
    monkeypatch.setattr(
        "models.bpu_loader.dnn", mock.Mock(load=mock.Mock(return_value=[mock_model]))
    )

    model = BpuModel(str(bin_file), str(cfg_file))

    with pytest.raises(TypeError, match="np.ndarray"):
        model.infer("not an array")

    with pytest.raises(TypeError, match="uint8"):
        model.infer(np.zeros((1, 3, 640, 640), dtype=np.float32))

    with pytest.raises(ValueError, match="\\(1, 3, H, W\\)"):
        model.infer(np.zeros((1, 640, 640), dtype=np.uint8))

    with pytest.raises(ValueError, match="\\(1, 3, H, W\\)"):
        model.infer(np.zeros((1, 1, 640, 640), dtype=np.uint8))

    # Non-contiguous input should be accepted after conversion.
    tensor = np.zeros((1, 3, 640, 640), dtype=np.uint8)[..., ::-1]
    assert not tensor.flags["C_CONTIGUOUS"]
    model.infer(tensor)
    mock_model.forward.assert_called_once()


def test_bpu_model_bin_is_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    _, cfg_file = _write_cfg(tmp_path)
    bin_dir = tmp_path / "num.bin"
    bin_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="bin not found"):
        BpuModel(str(bin_dir), str(cfg_file))


def test_bpu_model_cfg_is_directory(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    cfg_dir = tmp_path / "num_rdkx5_config.yaml"
    cfg_dir.mkdir()
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    with pytest.raises(FileNotFoundError, match="config not found"):
        BpuModel(str(bin_file), str(cfg_dir))


def test_bpu_model_empty_config(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    cfg_file = tmp_path / "num_rdkx5_config.yaml"
    cfg_file.write_text("\n\n", encoding="utf-8")
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    monkeypatch.setattr(
        "models.bpu_loader.dnn", mock.Mock(load=mock.Mock(return_value=[mock.Mock()]))
    )

    with pytest.raises(ValueError, match="config file is empty"):
        BpuModel(str(bin_file), str(cfg_file))


def test_bpu_model_load_returns_empty_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    _, cfg_file = _write_cfg(tmp_path)
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    monkeypatch.setattr(
        "models.bpu_loader.dnn", mock.Mock(load=mock.Mock(return_value=[]))
    )

    with pytest.raises(TypeError, match="sequence"):
        BpuModel(str(bin_file), str(cfg_file))


def create_dummy_cfg(path, names):
    cfg = {
        "imgsz": 640,
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
        "reg_max": 16,
        "strides": [8, 16, 32],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg


_FAKE_OUTPUTS = [
    np.zeros((1, 80, 80, 64), dtype=np.float32),
    np.zeros((1, 40, 40, 64), dtype=np.float32),
    np.zeros((1, 20, 20, 64), dtype=np.float32),
    np.zeros((1, 80, 80, 3), dtype=np.float32),
    np.zeros((1, 40, 40, 3), dtype=np.float32),
    np.zeros((1, 20, 20, 3), dtype=np.float32),
]


@pytest.mark.parametrize("model_class", [BeanModel, DigitModel])
def test_model_predict_calls_bpu_and_postprocess(model_class, tmp_path):
    cfg_path = tmp_path / f"{model_class.__name__.lower()}.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.infer.return_value = _FAKE_OUTPUTS
        MockBpu.return_value = bpu

        model = model_class(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = model.predict(img)
        assert dets.shape[0] == 0
        bpu.infer.assert_called_once()
        tensor = bpu.infer.call_args[0][0]
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.uint8


def test_model_predict_asserts_tensor_rgb_order(tmp_path):
    cfg_path = tmp_path / "rgb.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.infer.return_value = _FAKE_OUTPUTS
        MockBpu.return_value = bpu

        model = BeanModel(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:, :, 0] = 1  # B
        img[:, :, 1] = 2  # G
        img[:, :, 2] = 3  # R
        model.predict(img)
        tensor = bpu.infer.call_args[0][0]
        # Channel 0 is R, channel 1 is G, channel 2 is B after BGR→RGB conversion.
        assert tensor[0, 0, :, :].mean() == 3.0
        assert tensor[0, 1, :, :].mean() == 2.0
        assert tensor[0, 2, :, :].mean() == 1.0


@pytest.mark.parametrize("model_class", [BeanModel, DigitModel])
def test_model_predict_rejects_bad_input(model_class, tmp_path):
    cfg_path = tmp_path / f"{model_class.__name__.lower()}.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        MockBpu.return_value = bpu
        model = model_class(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )

        with pytest.raises(TypeError, match="np.ndarray"):
            model.predict("not an array")

        with pytest.raises(ValueError, match="3 dimensions"):
            model.predict(np.zeros((480, 640), dtype=np.uint8))

        with pytest.raises(ValueError, match="3 channels"):
            model.predict(np.zeros((480, 640, 1), dtype=np.uint8))
