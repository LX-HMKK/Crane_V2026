import threading
from types import SimpleNamespace

import cv2

from models.devices.camera import Camera
from models.config.config_binding import ConfigBinding
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


# ── Camera V4L2 setters (unchanged) ─────────────────────────────────

def test_camera_runtime_setters_use_v4l2_values():
    camera = object.__new__(Camera)
    camera._io_lock = threading.RLock()  # normally set in Camera.__init__
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


# ── ConfigBinding read / write ──────────────────────────────────────

def test_config_binding_read_write_roundtrip():
    ui = _settings_ui()
    binding = ConfigBinding(ui.cfg, "logic", "max_inferred_beans", default=1, type_=int)
    assert binding.read() == 1
    binding.write(2)
    assert ui.cfg["logic"]["max_inferred_beans"] == 2
    assert isinstance(ui.cfg["logic"]["max_inferred_beans"], int)


def test_config_binding_coerces_string_to_type():
    ui = _settings_ui()
    binding = ConfigBinding(ui.cfg, "logic", "max_inferred_digits", default=2, type_=int)
    binding.write("3")
    assert ui.cfg["logic"]["max_inferred_digits"] == 3
    assert isinstance(ui.cfg["logic"]["max_inferred_digits"], int)


def test_config_binding_creates_intermediate_keys():
    ui = _settings_ui()
    ui.cfg.pop("logic", None)
    binding = ConfigBinding(ui.cfg, "logic", "new_key", default=5, type_=int)
    assert binding.read() == 5  # default when path is missing
    binding.write(7)
    assert ui.cfg["logic"]["new_key"] == 7


def test_config_binding_default_fallback():
    cfg = {}
    binding = ConfigBinding(cfg, "a", "b", "c", default=42, type_=int)
    assert binding.read() == 42


def test_config_binding_on_change_callback():
    cfg = {}
    side_effects = []
    binding = ConfigBinding(cfg, "x", type_=int).on_change(lambda v: side_effects.append(v))
    binding.write(10)
    assert side_effects == [10]


def test_config_binding_float_coercion():
    cfg = {}
    binding = ConfigBinding(cfg, "conf", default=0.5, type_=float)
    binding.write("0.65")
    assert cfg["conf"] == 0.65
    assert isinstance(cfg["conf"], float)


def test_config_binding_bool_coercion():
    cfg = {}
    binding = ConfigBinding(cfg, "flag", default=True, type_=bool)
    binding.write(0)
    assert cfg["flag"] is False
    binding.write(1)
    assert cfg["flag"] is True


# ── Side-effect helpers (replacing old _set_* methods) ──────────────

def test_apply_auto_exposure_pushes_to_camera():
    ui = _settings_ui()
    ui._apply_auto_exposure("bean", False)
    assert ui.cam_bean.auto_exposure is False
    ui._apply_auto_exposure("bean", True)
    assert ui.cam_bean.auto_exposure is True


def test_apply_exposure_skips_when_auto_is_on():
    ui = _settings_ui()
    ui._apply_exposure("digit", -3.0)
    # auto_exposure still True → set_exposure must NOT be called
    assert ui.cam_digit.exposure is None


def test_apply_exposure_pushes_when_auto_is_off():
    ui = _settings_ui()
    ui.cfg["cameras"]["bean"]["auto_exposure"] = False
    ui._apply_exposure("bean", -4.5)
    assert ui.cam_bean.exposure == -4.5


def test_apply_confidence_pushes_to_model():
    ui = _settings_ui()
    ui._apply_confidence("bean", 0.65)
    assert ui.bean_model.conf_thres == 0.65
    # rounding
    ui._apply_confidence("digit", 0.876)
    assert ui.digit_model.conf_thres == 0.88


# ── Integration: ConfigBinding → cfg + side-effect ──────────────────

def test_binding_with_on_change_writes_cfg_and_fires_side_effect():
    ui = _settings_ui()
    binding = ConfigBinding(
        ui.cfg, "cameras", "bean", "auto_exposure", type_=bool,
    ).on_change(lambda v: ui._apply_auto_exposure("bean", v))
    binding.write(False)

    assert ui.cfg["cameras"]["bean"]["auto_exposure"] is False
    assert ui.cam_bean.auto_exposure is False
