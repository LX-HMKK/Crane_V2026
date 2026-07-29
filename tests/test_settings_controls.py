from types import SimpleNamespace

import cv2

from models.camera import Camera
from models.ui import CalibrationUI


class FakeCapture:
    def __init__(self):
        self.calls = []

    def set(self, prop, value):
        self.calls.append((prop, value))
        return True


class FakeCamera:
    def __init__(self):
        self.auto_exposure = None
        self.exposure = None

    def set_auto_exposure(self, enabled):
        self.auto_exposure = bool(enabled)
        return True

    def set_exposure(self, value):
        self.exposure = float(value)
        return True


def _settings_ui():
    ui = object.__new__(CalibrationUI)
    ui.cfg = {
        "cameras": {
            "bean": {"auto_exposure": True, "exposure": -6.0},
            "digit": {"auto_exposure": True, "exposure": -6.0},
        },
        "models": {
            "bean": {"conf_thres": 0.7},
            "digit": {"conf_thres": 0.7},
        },
        "logic": {
            "max_inferred_beans": 1,
            "max_inferred_digits": 2,
        },
    }
    ui.cam_bean = FakeCamera()
    ui.cam_digit = FakeCamera()
    ui.bean_model = SimpleNamespace(conf_thres=0.7)
    ui.digit_model = SimpleNamespace(conf_thres=0.7)
    ui._auto_exposure_vars = {}
    ui._exposure_scales = {}
    return ui


def test_camera_runtime_setters_use_v4l2_values():
    camera = object.__new__(Camera)
    camera.cap = FakeCapture()
    camera.device_index = "fake"
    camera.auto_exposure = None
    camera.exposure = None

    assert camera.set_auto_exposure(False)
    assert camera.cap.calls[-1] == (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    assert camera.set_auto_exposure(True)
    assert camera.cap.calls[-1] == (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    assert camera.set_exposure(-5.5)
    assert camera.cap.calls[-1] == (cv2.CAP_PROP_EXPOSURE, -5.5)


def test_settings_update_runtime_and_persistable_config():
    ui = _settings_ui()

    ui._set_camera_auto_exposure("bean", False)
    ui._set_camera_exposure("bean", -4.5)
    ui._set_camera_exposure("digit", -3.0)
    ui._set_model_confidence("bean", 0.65)
    ui._set_model_confidence("digit", 0.8)

    assert ui.cfg["cameras"]["bean"] == {
        "auto_exposure": False,
        "exposure": -4.5,
    }
    assert ui.cam_bean.auto_exposure is False
    assert ui.cam_bean.exposure == -4.5
    assert ui.cfg["cameras"]["digit"]["exposure"] == -3.0
    assert ui.cam_digit.exposure is None  # Auto exposure leaves the driver alone.
    assert ui.cfg["models"]["bean"]["conf_thres"] == 0.65
    assert ui.bean_model.conf_thres == 0.65
    assert ui.cfg["models"]["digit"]["conf_thres"] == 0.8
    assert ui.digit_model.conf_thres == 0.8


def test_logic_limit_sliders_update_cfg():
    """Setter writes max_inferred_beans / max_inferred_digits into cfg["logic"]."""
    ui = _settings_ui()

    ui._set_logic_limit("max_inferred_beans", 2)
    ui._set_logic_limit("max_inferred_digits", 4)

    assert ui.cfg["logic"]["max_inferred_beans"] == 2
    assert ui.cfg["logic"]["max_inferred_digits"] == 4
    assert isinstance(ui.cfg["logic"]["max_inferred_beans"], int)
    assert isinstance(ui.cfg["logic"]["max_inferred_digits"], int)


def test_logic_limit_slider_string_to_int_conversion():
    """Scale callbacks pass strings; ensure int coercion."""
    ui = _settings_ui()

    ui._set_logic_limit("max_inferred_beans", "2")
    ui._set_logic_limit("max_inferred_digits", "3")

    assert ui.cfg["logic"]["max_inferred_beans"] == 2
    assert ui.cfg["logic"]["max_inferred_digits"] == 3
    assert isinstance(ui.cfg["logic"]["max_inferred_beans"], int)
