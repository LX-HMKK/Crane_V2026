import cv2
import numpy as np

from models.bpu_loader import BpuModel, _nv12_from_bgr
from models.yolo11_postprocess import (
    _make_anchors,
    letterbox,
    postprocess,
    scale_coords,
)


class Yolo11Model:
    """Shared inference pipeline for YOLO11-style BPU models."""

    def __init__(self, bin_path, cfg_path, inference_cfg, model_cfg=None):
        self.bpu = BpuModel(bin_path, cfg_path)
        cfg = self.bpu.cfg
        self.imgsz = cfg["imgsz"]
        self.nc = cfg["nc"]
        self.names = self._normalize_names(cfg["names"])
        self.reg_max = cfg["reg_max"]
        self.strides = cfg["strides"]
        model_cfg = model_cfg or {}
        self.conf_thres = model_cfg.get("conf_thres", inference_cfg["conf_thres"])
        self.iou_thres = model_cfg.get("iou_thres", inference_cfg["iou_thres"])
        self.pad = inference_cfg.get("pad", 127)
        self._post_cfg = {
            "imgsz": self.imgsz,
            "nc": self.nc,
            "reg_max": self.reg_max,
            "strides": self.strides,
            "conf_thres": self.conf_thres,
            "iou_thres": self.iou_thres,
        }
        self._anchors = _make_anchors(self.imgsz, self.strides)
        self._dfl_proj = np.arange(self.reg_max, dtype=np.float32)

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
        if getattr(self.bpu, "input_type", None) == "NV12":
            # Match the legacy RGB/NCHW path, which cast input pixels before
            # handing them to the uint8-only NV12 color converter.
            if letterboxed.dtype != np.uint8:
                letterboxed = letterboxed.astype(np.uint8)
            outputs = self.bpu.infer_nv12(_nv12_from_bgr(letterboxed))
        else:
            rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
            tensor = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)[None, ...]
            outputs = self.bpu.infer(tensor)
        # The settings UI updates these attributes at runtime. Keep the reusable
        # postprocess config in sync without rebuilding immutable model metadata.
        self._post_cfg["conf_thres"] = self.conf_thres
        self._post_cfg["iou_thres"] = self.iou_thres
        dets = postprocess(
            outputs,
            self._post_cfg,
            anchors=self._anchors,
            dfl_proj=self._dfl_proj,
        )
        dets = scale_coords(dets, scale, pad, bgr_image.shape)
        return dets.tolist()
