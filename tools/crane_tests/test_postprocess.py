import numpy as np
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.inference import yolo11_postprocess
from models.inference.yolo11_postprocess import _make_anchors, letterbox, postprocess, scale_coords


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


def test_postprocess_decodes_only_confident_candidates():
    cfg = {
        "imgsz": 640,
        "nc": 3,
        "reg_max": 16,
        "strides": [8, 16, 32],
        "conf_thres": 0.6,
        "iou_thres": 0.45,
    }
    empty_outputs = [
        np.zeros((1, 80, 80, 64), dtype=np.float32),
        np.zeros((1, 40, 40, 64), dtype=np.float32),
        np.zeros((1, 20, 20, 64), dtype=np.float32),
        np.zeros((1, 80, 80, 3), dtype=np.float32),
        np.zeros((1, 40, 40, 3), dtype=np.float32),
        np.zeros((1, 20, 20, 3), dtype=np.float32),
    ]

    with mock.patch.object(
        yolo11_postprocess,
        "_dfl_decode",
        wraps=yolo11_postprocess._dfl_decode,
    ) as decode:
        assert postprocess(empty_outputs, cfg).shape == (0, 6)
        decode.assert_not_called()

    outputs = make_yolo11_outputs(
        [[300.0, 300.0, 320.0, 320.0, 0.9, 1]], cfg
    )
    with mock.patch.object(
        yolo11_postprocess,
        "_dfl_decode",
        wraps=yolo11_postprocess._dfl_decode,
    ) as decode:
        result = postprocess(outputs, cfg)

    assert result.shape[0] == 1
    assert decode.call_count == 1
    assert decode.call_args.args[0].shape == (1, 4 * cfg["reg_max"])


def test_postprocess_accepts_cached_metadata():
    cfg = {
        "imgsz": 640,
        "nc": 3,
        "reg_max": 16,
        "strides": [8, 16, 32],
        "conf_thres": 0.6,
        "iou_thres": 0.45,
    }
    outputs = make_yolo11_outputs(
        [[300.0, 300.0, 320.0, 320.0, 0.9, 1]], cfg
    )

    expected = postprocess(outputs, cfg)
    cached = postprocess(
        outputs,
        cfg,
        anchors=_make_anchors(cfg["imgsz"], cfg["strides"]),
        dfl_proj=np.arange(cfg["reg_max"], dtype=np.float32),
    )

    np.testing.assert_allclose(cached, expected)


def test_scale_coords_reverses_letterbox():
    dets = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
    scaled = scale_coords(dets, scale=1.0, pad=(10, 20), img_shape=(480, 640))
    assert scaled[0, 0] == 90
    assert scaled[0, 1] == 80
    assert scaled[0, 2] == 190
    assert scaled[0, 3] == 180


def test_scale_coords_clips_to_image_bounds():
    dets = np.array([[-10, -20, 650, 660, 0.9, 0]], dtype=np.float32)
    scaled = scale_coords(dets, scale=1.0, pad=(0, 0), img_shape=(480, 640))
    assert scaled[0, 0] == 0
    assert scaled[0, 1] == 0
    assert scaled[0, 2] == 640
    assert scaled[0, 3] == 480


def _encode_dfl_distance(dist, reg_max, peak=10.0):
    """Create a sharp DFL logit vector whose expected value is approximately dist."""
    bin_idx = int(round(np.clip(dist, 0, reg_max - 1)))
    vec = np.zeros(reg_max, dtype=np.float32)
    vec[bin_idx] = peak
    return vec


def make_yolo11_outputs(dets, cfg):
    """Build deterministic 6-head YOLO11 raw outputs that decode to ``dets``.

    ``dets`` must be in letterboxed image coordinates, i.e. after letterbox
    resizing/padding.  Each det is ``[x1, y1, x2, y2, conf, cls_id]``.
    """
    imgsz = cfg["imgsz"]
    if isinstance(imgsz, int):
        imgsz_h = imgsz_w = imgsz
    else:
        imgsz_h, imgsz_w = imgsz
    nc = cfg["nc"]
    reg_max = cfg["reg_max"]
    strides = cfg["strides"]

    level_info = []
    bbox_outputs = []
    cls_outputs = []
    for stride in strides:
        h = imgsz_h // stride
        w = imgsz_w // stride
        xv, yv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xv = (xv + 0.5) * stride
        yv = (yv + 0.5) * stride
        anchors = np.stack([xv.ravel(), yv.ravel()], axis=-1)
        level_info.append({"stride": stride, "h": h, "w": w, "anchors": anchors})
        bbox_outputs.append(np.zeros((1, h, w, 4 * reg_max), dtype=np.float32))
        # Use a negative baseline so only the explicitly-set anchor activates.
        cls_outputs.append(np.full((1, h, w, nc), -10.0, dtype=np.float32))

    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        chosen = None
        for lvl in level_info:
            stride, h, w, anchors = lvl["stride"], lvl["h"], lvl["w"], lvl["anchors"]
            gx = int(cx // stride)
            gy = int(cy // stride)
            if 0 <= gx < w and 0 <= gy < h:
                idx = gy * w + gx
                chosen = (lvl, idx)
                break
        if chosen is None:
            raise ValueError(f"detection {det} does not fit any stride grid")

        lvl, idx = chosen
        stride, h, w, anchors = lvl["stride"], lvl["h"], lvl["w"], lvl["anchors"]
        ax, ay = anchors[idx]
        # The model predicts DFL distances in feature-map cells; postprocess
        # converts them back to input pixels with this head's stride.
        l = (ax - x1) / stride
        t = (ay - y1) / stride
        r = (x2 - ax) / stride
        b = (y2 - ay) / stride

        dfl = np.concatenate(
            [_encode_dfl_distance(d, reg_max) for d in (l, t, r, b)]
        )
        out_h = idx // w
        out_w = idx % w
        level_idx = level_info.index(lvl)
        bbox_outputs[level_idx][0, out_h, out_w, :] = dfl
        cls_outputs[level_idx][0, out_h, out_w, int(cls_id)] = 10.0

    outputs = bbox_outputs + cls_outputs
    return outputs


def test_postprocess_decodes_single_detection():
    cfg = {
        "imgsz": 640,
        "nc": 3,
        "reg_max": 16,
        "strides": [8, 16, 32],
        "conf_thres": 0.25,
        "iou_thres": 0.45,
    }
    # A small box whose edge distances fit inside reg_max=16 bins.
    dets = [[300.0, 300.0, 320.0, 320.0, 0.9, 1]]
    outputs = make_yolo11_outputs(dets, cfg)
    result = postprocess(outputs, cfg)
    assert result.shape[0] == 1
    assert int(result[0, 5]) == 1
    assert result[0, 4] > 0.9
    # Synthetic logits use one whole DFL bin; with stride=8 this quantizes an
    # edge by up to 4 px, plus a small softmax tail from the remaining bins.
    np.testing.assert_allclose(result[0, :4], [300, 300, 320, 320], atol=5.0)


def test_postprocess_class_aware_nms_keeps_overlapping_different_classes():
    cfg = {
        "imgsz": 640,
        "nc": 3,
        "reg_max": 16,
        "strides": [8, 16, 32],
        "conf_thres": 0.25,
        "iou_thres": 0.45,
    }
    # Two nearly identical boxes of different classes; class-agnostic NMS would drop one.
    dets = [
        [286.0, 286.0, 366.0, 366.0, 0.9, 0],
        [288.0, 288.0, 368.0, 368.0, 0.85, 1],
    ]
    outputs = make_yolo11_outputs(dets, cfg)
    result = postprocess(outputs, cfg)
    assert result.shape[0] == 2
    classes = set(int(r[5]) for r in result)
    assert classes == {0, 1}


def test_postprocess_class_aware_nms_suppresses_same_class_duplicates():
    cfg = {
        "imgsz": 640,
        "nc": 3,
        "reg_max": 16,
        "strides": [8, 16, 32],
        "conf_thres": 0.25,
        "iou_thres": 0.45,
    }
    # Two heavily overlapping boxes of the same class; only the higher-conf one survives.
    dets = [
        [286.0, 286.0, 366.0, 366.0, 0.9, 2],
        [288.0, 288.0, 368.0, 368.0, 0.85, 2],
    ]
    outputs = make_yolo11_outputs(dets, cfg)
    result = postprocess(outputs, cfg)
    assert result.shape[0] == 1
    assert int(result[0, 5]) == 2
    assert result[0, 4] > 0.89


def test_postprocess_rectangular_imgsz():
    cfg = {
        "imgsz": [384, 640],
        "nc": 3,
        "reg_max": 16,
        "strides": [8, 16, 32],
        "conf_thres": 0.25,
        "iou_thres": 0.45,
    }
    # Small box in 640x384 space with edge distances inside reg_max bins.
    dets = [[300.0, 150.0, 320.0, 170.0, 0.9, 0]]
    outputs = make_yolo11_outputs(dets, cfg)
    result = postprocess(outputs, cfg)
    assert result.shape[0] == 1
    np.testing.assert_allclose(result[0, :4], [300, 150, 320, 170], atol=5.0)
