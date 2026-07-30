from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import yaml

from models.bean_model import BeanModel
from models.bpu_loader import BpuModel, _nv12_from_bgr, _nv12_from_rgb_nchw
from models.digit_model import DigitModel


def test_bpu_model_raises_on_x86():
    with pytest.raises(EnvironmentError):
        BpuModel("assets/num.bin", "assets/num_rdkx5_config.yaml")


def test_direct_bgr_nv12_matches_legacy_rgb_nchw_conversion():
    bgr = np.random.default_rng(7).integers(
        0, 256, (640, 640, 3), dtype=np.uint8
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    nchw = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.uint8)

    assert np.array_equal(_nv12_from_bgr(bgr), _nv12_from_rgb_nchw(nchw))


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


def _fake_outputs_for_cfg(cfg):
    """Generate zero-filled 6-head outputs with the correct shapes for cfg."""
    imgsz = cfg["imgsz"]
    if isinstance(imgsz, int):
        imgsz_h = imgsz_w = imgsz
    else:
        imgsz_h, imgsz_w = imgsz
    reg_max = cfg["reg_max"]
    nc = cfg["nc"]
    strides = cfg["strides"]
    outputs = []
    for stride in strides:
        h = imgsz_h // stride
        w = imgsz_w // stride
        outputs.append(np.zeros((1, h, w, 4 * reg_max), dtype=np.float32))
    for stride in strides:
        h = imgsz_h // stride
        w = imgsz_w // stride
        outputs.append(np.zeros((1, h, w, nc), dtype=np.float32))
    return outputs


@pytest.mark.parametrize("model_class", [BeanModel, DigitModel])
def test_model_predict_calls_bpu_and_postprocess(model_class, tmp_path):
    cfg_path = tmp_path / f"{model_class.__name__.lower()}.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.infer.return_value = _fake_outputs_for_cfg(cfg)
        MockBpu.return_value = bpu

        model = model_class(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = model.predict(img)
        assert len(dets) == 0
        bpu.infer.assert_called_once()
        tensor = bpu.infer.call_args[0][0]
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.uint8


def test_model_predict_uses_direct_nv12_input_when_supported(tmp_path):
    cfg_path = tmp_path / "nv12.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])
    image = np.random.default_rng(9).integers(
        0, 256, (640, 640, 3), dtype=np.uint8
    )

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.input_type = "NV12"
        bpu.infer_nv12.return_value = _fake_outputs_for_cfg(cfg)
        MockBpu.return_value = bpu

        model = BeanModel(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )
        assert model.predict(image) == []

    bpu.infer.assert_not_called()
    np.testing.assert_array_equal(
        bpu.infer_nv12.call_args[0][0], _nv12_from_bgr(image)
    )


def test_model_predict_nv12_path_preserves_legacy_uint8_cast(tmp_path):
    cfg_path = tmp_path / "nv12_float.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])
    image = np.full((640, 640, 3), 17.9, dtype=np.float32)

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.input_type = "NV12"
        bpu.infer_nv12.return_value = _fake_outputs_for_cfg(cfg)
        MockBpu.return_value = bpu

        model = BeanModel(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )
        assert model.predict(image) == []

    np.testing.assert_array_equal(
        bpu.infer_nv12.call_args[0][0], _nv12_from_bgr(image.astype(np.uint8))
    )


def test_model_predict_uses_per_model_thresholds(tmp_path):
    cfg_path = tmp_path / "override.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.infer.return_value = _fake_outputs_for_cfg(cfg)
        MockBpu.return_value = bpu

        model = BeanModel(
            "assets/fake.bin",
            str(cfg_path),
            {"conf_thres": 0.25, "iou_thres": 0.45},
            {"conf_thres": 0.7, "iou_thres": 0.3},
        )
        assert model.conf_thres == 0.7
        assert model.iou_thres == 0.3
    cfg_path = tmp_path / "rgb.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.infer.return_value = _fake_outputs_for_cfg(cfg)
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


def test_model_predict_uses_runtime_threshold_updates(tmp_path):
    cfg_path = tmp_path / "runtime_thresholds.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])
    outputs = _fake_outputs_for_cfg(cfg)

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu, mock.patch(
        "models.yolo11_model.postprocess",
        return_value=np.zeros((0, 6), dtype=np.float32),
    ) as postprocess_mock:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.infer.return_value = outputs
        MockBpu.return_value = bpu
        model = BeanModel(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )
        model.conf_thres = 0.72
        model.iou_thres = 0.31
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8))

    post_cfg = postprocess_mock.call_args.args[1]
    assert post_cfg["conf_thres"] == 0.72
    assert post_cfg["iou_thres"] == 0.31


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


def test_model_normalize_names_int_keys(tmp_path):
    cfg_path = tmp_path / "names.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])
    # YAML may load integer-like keys as strings; _normalize_names must coerce them.
    cfg["names"] = {"0": "white", "1": "green", "2": "yellow"}

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        MockBpu.return_value = bpu
        model = BeanModel(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )
        assert model.names == {0: "white", 1: "green", 2: "yellow"}


def test_model_normalize_names_rejects_non_int_keys(tmp_path):
    cfg_path = tmp_path / "names.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])
    cfg["names"] = {"a": "white", "1": "green", "2": "yellow"}

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        MockBpu.return_value = bpu
        with pytest.raises(ValueError, match="int-convertible"):
            BeanModel("assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45})


def test_model_predict_rectangular_imgsz(tmp_path):
    cfg_path = tmp_path / "rect.yaml"
    cfg = create_dummy_cfg(cfg_path, ["white", "green", "yellow"])
    cfg["imgsz"] = [384, 640]

    with mock.patch("models.yolo11_model.BpuModel") as MockBpu:
        bpu = mock.MagicMock()
        bpu.cfg = cfg
        bpu.infer.return_value = _fake_outputs_for_cfg(cfg)
        MockBpu.return_value = bpu

        model = BeanModel(
            "assets/fake.bin", str(cfg_path), {"conf_thres": 0.6, "iou_thres": 0.45}
        )
        img = np.zeros((384, 640, 3), dtype=np.uint8)
        dets = model.predict(img)
        assert isinstance(dets, list)
        tensor = bpu.infer.call_args[0][0]
        assert tensor.shape == (1, 3, 384, 640)


@pytest.mark.parametrize(
    "bad_cfg,error_match",
    [
        ({"imgsz": -640}, "positive"),
        ({"imgsz": [640, -640]}, "positive ints"),
        ({"nc": "3"}, "positive integer"),
        ({"names": {0: "white"}}, "length"),  # names length != nc
        ({"names": ["white"]}, "length"),
        ({"names": {"a": "white", 1: "green", 2: "yellow"}}, "int-convertible"),
        ({"reg_max": 0}, "positive integer"),
        ({"strides": []}, "non-empty"),
        ({"strides": [8, -16, 32]}, "positive integer"),
    ],
)
def test_bpu_model_config_validation(tmp_path, monkeypatch, bad_cfg, error_match):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    base = {"imgsz": 640, "nc": 3, "names": {0: "white", 1: "green", 2: "yellow"}, "reg_max": 16, "strides": [8, 16, 32]}
    base.update(bad_cfg)
    cfg_file = tmp_path / "bad.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f)
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    monkeypatch.setattr(
        "models.bpu_loader.dnn", mock.Mock(load=mock.Mock(return_value=[mock.Mock()]))
    )

    with pytest.raises(ValueError, match=error_match):
        BpuModel(str(bin_file), str(cfg_file))


def test_bpu_model_infer_rejects_wrong_spatial_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    cfg, cfg_file = _write_cfg(tmp_path, imgsz=640)
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    mock_model = mock.Mock()
    monkeypatch.setattr(
        "models.bpu_loader.dnn", mock.Mock(load=mock.Mock(return_value=[mock_model]))
    )

    model = BpuModel(str(bin_file), str(cfg_file))
    with pytest.raises(ValueError, match="spatial shape"):
        model.infer(np.zeros((1, 3, 480, 640), dtype=np.uint8))


def test_bpu_model_infer_accepts_list_imgsz(tmp_path, monkeypatch):
    monkeypatch.setattr("models.bpu_loader.HAS_HOBOT", True)

    cfg, cfg_file = _write_cfg(tmp_path, imgsz=[384, 640])
    bin_file = tmp_path / "num.bin"
    bin_file.write_text("fake bin")

    mock_model = mock.Mock()
    monkeypatch.setattr(
        "models.bpu_loader.dnn", mock.Mock(load=mock.Mock(return_value=[mock_model]))
    )

    model = BpuModel(str(bin_file), str(cfg_file))
    tensor = np.zeros((1, 3, 384, 640), dtype=np.uint8)
    model.infer(tensor)
    mock_model.forward.assert_called_once()


_FAKE_OUTPUTS = _fake_outputs_for_cfg(
    {"imgsz": 640, "nc": 3, "reg_max": 16, "strides": [8, 16, 32]}
)
