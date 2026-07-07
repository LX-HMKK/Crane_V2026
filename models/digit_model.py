import cv2
import numpy as np
import yaml

from models.bpu_loader import BpuModel
from models.yolo11_postprocess import letterbox, postprocess, scale_coords


class DigitModel:
    def __init__(self, bin_path, cfg_path, inference_cfg):
        self.bpu = BpuModel(bin_path, cfg_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.imgsz = self.cfg["imgsz"]
        self.nc = self.cfg["nc"]
        self.names = self.cfg["names"]
        self.reg_max = self.cfg["reg_max"]
        self.strides = self.cfg["strides"]
        self.conf_thres = inference_cfg["conf_thres"]
        self.iou_thres = inference_cfg["iou_thres"]
        self.pad = inference_cfg.get("pad", 127)

    def predict(self, bgr_image):
        letterboxed, scale, pad = letterbox(
            bgr_image, new_shape=(self.imgsz, self.imgsz), color=self.pad
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
        dets = scale_coords(dets, scale, pad)
        return dets
