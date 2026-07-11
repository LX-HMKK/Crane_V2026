"""PyTorch YOLO inference wrapper for x86 development/testing.

Loads `.pt` models via the Ultralytics YOLO API. On environments without
`ultralytics` installed, importing this module succeeds but instantiation
raises `EnvironmentError`, mirroring the behaviour of `models.bpu_loader`.
"""

import logging

try:
    from ultralytics import YOLO
    import torch

    HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None
    torch = None
    HAS_ULTRALYTICS = False


def _default_device():
    """Return the best available torch device for Ultralytics inference."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PtModel:
    """Ultralytics YOLO wrapper with the same interface as Yolo11Model."""

    def __init__(self, pt_path, inference_cfg, model_cfg=None):
        if not HAS_ULTRALYTICS:
            raise EnvironmentError(
                "ultralytics is not available; install it to run .pt models on x86"
            )
        self.device = inference_cfg.get("device", _default_device())
        self.model = YOLO(pt_path)
        self.model.to(self.device)
        self.imgsz = inference_cfg.get("imgsz", 640)
        model_cfg = model_cfg or {}
        self.conf_thres = model_cfg.get("conf_thres", inference_cfg.get("conf_thres", 0.25))
        self.iou_thres = model_cfg.get("iou_thres", inference_cfg.get("iou_thres", 0.45))
        self.names = {int(i): str(name) for i, name in self.model.names.items()}
        logging.info("Loaded %s on device %s", pt_path, self.device)

    def predict(self, bgr_image):
        """Run inference on a BGR image and return detections.

        Each detection is a list: [x1, y1, x2, y2, conf, cls_id].
        """
        results = self.model(
            bgr_image,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )[0]

        dets = []
        if results.boxes is None:
            return dets

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            dets.append([x1, y1, x2, y2, conf, cls_id])
        return dets


class PtBeanModel(PtModel):
    """Bean maturity detector backed by a Ultralytics YOLO `.pt` model."""


class PtDigitModel(PtModel):
    """Digit detector backed by a Ultralytics YOLO `.pt` model."""
