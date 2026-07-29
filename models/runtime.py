"""Shared runtime helpers for crane vision entry points."""

import logging
import sys
import time
from pathlib import Path

import yaml

# Add repo root to path so scripts/ can import models/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.bean_model import BeanModel
from models.camera import Camera, MockCamera
from models.digit_model import DigitModel
from models.mapping import DEFAULT_BEAN_TO_DIGIT
from models.pt_model import PtBeanModel, PtDigitModel
from models.task_generator import TaskGenerator


def load_vision_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_logging(debug=False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def list_cameras():
    import cv2

    print("Available cameras:")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  /dev/video{i}: {w}x{h}")
            cap.release()


def open_camera(camera_cfg, mock=False, fallback_empty=False):
    """Open a camera from config.

    Args:
        camera_cfg: Dict with device, width, height, optional buffer_size and
            optional fps.
        mock: If True, return a MockCamera instead of a real one.
        fallback_empty: If True and real camera fails, return an empty-pattern
            MockCamera instead of raising.
    """
    if mock:
        return MockCamera(camera_cfg["width"], camera_cfg["height"])
    try:
        return Camera(
            camera_cfg["device"],
            camera_cfg["width"],
            camera_cfg["height"],
            camera_cfg.get("buffer_size", 1),
            camera_cfg.get("fps"),
            auto_exposure=camera_cfg.get("auto_exposure", True),
            grayscale=camera_cfg.get("grayscale", False),
        )
    except RuntimeError:
        if fallback_empty:
            logging.warning(
                "Cannot open camera device %s; falling back to empty mock frame",
                camera_cfg["device"],
            )
            return MockCamera(
                camera_cfg["width"], camera_cfg["height"], pattern="empty"
            )
        raise


def load_models(cfg):
    """Load BPU models; returns (bean_model, digit_model)."""
    bean_model = BeanModel(
        cfg["models"]["bean"]["bin"],
        cfg["models"]["bean"]["cfg"],
        cfg["inference"],
        cfg["models"]["bean"],
    )
    digit_model = DigitModel(
        cfg["models"]["digit"]["bin"],
        cfg["models"]["digit"]["cfg"],
        cfg["inference"],
        cfg["models"]["digit"],
    )
    return bean_model, digit_model


def load_pt_models(cfg):
    """Load PyTorch `.pt` models; returns (bean_model, digit_model).

    Raises EnvironmentError if ultralytics is not installed.
    """
    bean_model = PtBeanModel(
        cfg["models"]["bean"]["pt"],
        cfg["inference"],
        cfg["models"]["bean"],
    )
    digit_model = PtDigitModel(
        cfg["models"]["digit"]["pt"],
        cfg["inference"],
        cfg["models"]["digit"],
    )
    return bean_model, digit_model


def create_task_generator(cfg, bean_model, digit_model):
    """Build a TaskGenerator from config and loaded models.

    Returns None if either model is None (e.g. BPU unavailable on x86).
    """
    if bean_model is None or digit_model is None:
        return None

    mapping = cfg.get("mapping", {}).get("bean_to_digit", DEFAULT_BEAN_TO_DIGIT)
    return TaskGenerator(
        mapping,
        cfg["lines"],
        bean_model.names,
        digit_model.names,
    )


def warmup_models(cam_bean, cam_digit, bean_model, digit_model):
    """Run a couple of inference cycles to warm up models."""
    logging.info("Warming up models...")
    for _ in range(2):
        ok, frame = cam_bean.capture()
        if ok and bean_model is not None:
            bean_model.predict(frame)
        ok, frame = cam_digit.capture()
        if ok and digit_model is not None:
            digit_model.predict(frame)


def scan_once_dets(bean_cam, digit_cam, bean_model, digit_model, cfg):
    """Capture frames, run inference, and return raw detections.

    Returns (all_bean_dets, all_digit_dets, reason). On camera failure
    reason is set and the detection lists may be empty.
    """
    frames = cfg["scan"]["frames"]
    all_bean_dets = []
    all_digit_dets = []
    failures = 0

    for _ in range(frames):
        ok_b, frame_b = bean_cam.capture()
        ok_d, frame_d = digit_cam.capture()
        if not ok_b or not ok_d:
            failures += 1
            logging.warning(
                "Camera capture failure (bean_ok=%s, digit_ok=%s)", ok_b, ok_d
            )
            continue
        if bean_model is not None:
            all_bean_dets.extend(bean_model.predict(frame_b))
        if digit_model is not None:
            all_digit_dets.extend(digit_model.predict(frame_d))

    if failures >= frames:
        return all_bean_dets, all_digit_dets, "too many camera capture failures"

    return all_bean_dets, all_digit_dets, "ok"


def scan_once(bean_cam, digit_cam, bean_model, digit_model, task_gen, cfg):
    """Capture frames, run inference, and generate a task triplet."""
    if task_gen is None:
        return None, "task generator unavailable"

    bean_dets, digit_dets, reason = scan_once_dets(
        bean_cam, digit_cam, bean_model, digit_model, cfg
    )
    if reason != "ok":
        return None, reason

    return task_gen.generate(bean_dets, digit_dets)


def release_cameras(*cameras):
    """Release all non-None cameras."""
    for cam in cameras:
        if cam is not None:
            cam.release()
