import cv2
import numpy as np

from models.bpu_loader import BpuModel
from models.yolo11_postprocess import letterbox, postprocess, scale_coords


class Yolo11Model:
    """Shared inference pipeline for YOLO11-style BPU models."""

    def __init__(self, bin_path, cfg_path, inference_cfg):
        self.bpu = BpuModel(bin_path, cfg_path)
        cfg = self.bpu.cfg
        self.imgsz = cfg["imgsz"]
        self.nc = cfg["nc"]
        self.names = self._normalize_names(cfg["names"])
        self.reg_max = cfg["reg_max"]
        self.strides = cfg["strides"]
        self.conf_thres = inference_cfg["conf_thres"]
        self.iou_thres = inference_cfg["iou_thres"]
        self.pad = inference_cfg.get("pad", 127)

    @staticmethod
    def _normalize_names(names):
        if isinstance(names, dict):
            normalized = {}
            for key, value in names.items():
                try:
                    cls_id = int(key)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"names key must be int-convertible, got {key!r}"
                    ) from exc
                normalized[cls_id] = str(value)
            return normalized
        if isinstance(names, (list, tuple)):
            return {i: str(name) for i, name in enumerate(names)}
        raise TypeError(
            f"names must be a dict or list, got {type(names).__name__}"
        )

    def predict(self, bgr_image):
        if not isinstance(bgr_image, np.ndarray):
            raise TypeError(
                f"input image must be np.ndarray, got {type(bgr_image).__name__}"
            )
        if bgr_image.ndim != 3:
            raise ValueError(
                f"input image must have 3 dimensions, got {bgr_image.ndim}"
            )
        if bgr_image.shape[2] != 3:
            raise ValueError(
                f"input image must have 3 channels (BGR), got {bgr_image.shape[2]}"
            )

        if isinstance(self.imgsz, int):
            new_shape = (self.imgsz, self.imgsz)
        else:
            new_shape = tuple(self.imgsz)
        letterboxed, scale, pad = letterbox(
            bgr_image, new_shape=new_shape, color=self.pad
        )
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.uint8)
        outputs = self.bpu.infer(tensor)
        post_cfg = {
            "imgsz": self.imgsz,
            "nc": self.nc,
            "reg_max": self.reg_max,
            "strides": self.strides,
            "conf_thres": self.conf_thres,
            "iou_thres": self.iou_thres,
        }
        dets = postprocess(outputs, post_cfg)
        dets = scale_coords(dets, scale, pad, bgr_image.shape)
        return dets.tolist()
