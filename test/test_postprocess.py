import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.yolo11_postprocess import letterbox, postprocess, scale_coords


def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, scale, pad = letterbox(img, new_shape=(640, 640), color=127)
    assert out.shape == (640, 640, 3)
    assert scale == 1.0
    assert pad == (0, 80)


def test_postprocess_returns_empty_for_zero_tensors():
    cfg = {
        "imgsz": 640,
        "nc": 5,
        "reg_max": 16,
        "strides": [8, 16, 32],
        "conf_thres": 0.6,
        "iou_thres": 0.45,
    }
    outputs = [
        np.zeros((1, 80, 80, 64), dtype=np.float32),
        np.zeros((1, 40, 40, 64), dtype=np.float32),
        np.zeros((1, 20, 20, 64), dtype=np.float32),
        np.zeros((1, 80, 80, 5), dtype=np.float32),
        np.zeros((1, 40, 40, 5), dtype=np.float32),
        np.zeros((1, 20, 20, 5), dtype=np.float32),
    ]
    dets = postprocess(outputs, cfg)
    assert dets.shape[0] == 0


def test_scale_coords_reverses_letterbox():
    dets = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
    scaled = scale_coords(dets, scale=1.0, pad=(10, 20))
    assert scaled[0, 0] == 90
    assert scaled[0, 1] == 80
    assert scaled[0, 2] == 190
    assert scaled[0, 3] == 180
