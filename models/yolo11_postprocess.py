import cv2
import numpy as np


def _softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def letterbox(im, new_shape=(640, 640), color=127, stride=32):
    """Resize and pad image to new_shape while preserving aspect ratio."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def _make_anchors(imgsz, strides):
    """Generate anchor centers for each stride."""
    if isinstance(imgsz, int):
        imgsz_h = imgsz_w = imgsz
    else:
        imgsz_h, imgsz_w = imgsz
    anchors = []
    for stride in strides:
        h = imgsz_h // stride
        w = imgsz_w // stride
        xv, yv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xv = (xv + 0.5) * stride
        yv = (yv + 0.5) * stride
        anchors.append(np.stack([xv.ravel(), yv.ravel()], axis=-1))
    return np.concatenate(anchors, axis=0)


def _dfl_decode(pred_dfl, reg_max):
    """Decode Distribution Focal Loss output to l,t,r,b distances."""
    b = pred_dfl.shape[0]
    pred = pred_dfl.reshape(b, 4, reg_max)
    pred = _softmax(pred, axis=2)
    proj = np.arange(reg_max, dtype=np.float32)
    return (pred * proj).sum(axis=2)  # (b, 4)


def postprocess(outputs, cfg):
    """
    Convert raw 6-head outputs to detections.
    outputs order: [s_bbox, m_bbox, l_bbox, s_cls, m_cls, l_cls]
    Returns: ndarray of shape (N, 6) -> [x1, y1, x2, y2, conf, cls_id]
    """
    imgsz = cfg["imgsz"]
    if isinstance(imgsz, int):
        imgsz_h = imgsz_w = imgsz
    else:
        imgsz_h, imgsz_w = imgsz
    nc = cfg["nc"]
    reg_max = cfg["reg_max"]
    strides = cfg["strides"]
    conf_thres = cfg.get("conf_thres", 0.25)
    iou_thres = cfg.get("iou_thres", 0.45)

    anchors = _make_anchors((imgsz_h, imgsz_w), strides)

    dfl_parts = []
    cls_parts = []
    for i, stride in enumerate(strides):
        h = imgsz_h // stride
        w = imgsz_w // stride
        dfl_parts.append(outputs[i].reshape(-1, 4 * reg_max))
        cls_parts.append(outputs[i + 3].reshape(-1, nc))

    dfl_preds = np.concatenate(dfl_parts, axis=0)
    cls_preds = np.concatenate(cls_parts, axis=0)

    bbox = _dfl_decode(dfl_preds, reg_max)

    xyxy = np.empty_like(bbox)
    xyxy[:, 0] = anchors[:, 0] - bbox[:, 0]
    xyxy[:, 1] = anchors[:, 1] - bbox[:, 1]
    xyxy[:, 2] = anchors[:, 0] + bbox[:, 2]
    xyxy[:, 3] = anchors[:, 1] + bbox[:, 3]

    cls_sigmoid = 1.0 / (1.0 + np.exp(-cls_preds))
    conf = cls_sigmoid.max(axis=1)
    cls_id = cls_sigmoid.argmax(axis=1)

    mask = conf >= conf_thres
    xyxy = xyxy[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]

    if len(xyxy) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Class-aware NMS: do not suppress boxes of different classes.
    keep = []
    for cls in np.unique(cls_id):
        mask = cls_id == cls
        idxs = np.where(mask)[0]
        boxes = [xyxy[i].tolist() for i in idxs]
        scores = conf[mask].tolist()
        selected = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
        if len(selected) == 0:
            continue
        if isinstance(selected, tuple):
            selected = selected[0]
        keep.extend(idxs[selected.flatten()])

    if len(keep) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    keep = np.array(keep, dtype=int)
    order = np.argsort(-conf[keep])
    keep = keep[order]

    dets = np.concatenate(
        [xyxy[keep], conf[keep][:, None], cls_id[keep][:, None]],
        axis=1,
    )
    return dets


def scale_coords(dets, scale, pad, img_shape):
    """Map detection coordinates from letterbox space back to original image."""
    dets = dets.copy()
    left, top = pad
    dets[:, [0, 2]] -= left
    dets[:, [1, 3]] -= top
    dets[:, :4] /= scale
    height, width = img_shape[:2]
    dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, width)
    dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, height)
    return dets
