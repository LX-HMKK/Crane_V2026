from pathlib import Path

try:
    import hobot_dnn as dnn
    HAS_HOBOT = True
except ImportError:
    dnn = None
    HAS_HOBOT = False


class BpuModel:
    """Thin wrapper around hobot_dnn for RDKx5 BPU inference."""

    def __init__(self, bin_path, cfg_path):
        if not HAS_HOBOT:
            raise EnvironmentError(
                "hobot_dnn is not available; BpuModel only runs on RDKx5"
            )
        bin_path = Path(bin_path)
        if not bin_path.is_file():
            raise FileNotFoundError(f"bin not found: {bin_path}")
        self.bin_path = str(bin_path)
        self.cfg = self._load_cfg(cfg_path)
        loaded = dnn.load(self.bin_path)
        # hobot_dnn.load returns a sequence of models; take the first one.
        try:
            self._bpu_model = loaded[0]
        except (TypeError, IndexError, KeyError) as exc:
            raise TypeError(
                f"hobot_dnn.load returned {type(loaded).__name__}, expected a sequence"
            ) from exc

    @staticmethod
    def _load_cfg(cfg_path):
        if cfg_path is None:
            return None
        cfg_path = Path(cfg_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config not found: {cfg_path}")
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError(f"config file is empty: {cfg_path}")
        for key in ("imgsz", "nc", "names", "reg_max", "strides"):
            if key not in cfg:
                raise ValueError(f"config missing required key: {key}")
        return cfg

    def infer(self, tensor):
        """tensor: np.ndarray uint8, shape (1, 3, H, W) NCHW"""
        import numpy as np
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"tensor must be np.ndarray, got {type(tensor).__name__}")
        if tensor.dtype != np.uint8:
            raise TypeError(f"tensor dtype must be uint8, got {tensor.dtype}")
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError(
                f"tensor shape must be (1, 3, H, W), got {tensor.shape}"
            )
        if not tensor.flags["C_CONTIGUOUS"]:
            tensor = np.ascontiguousarray(tensor)
        return self._bpu_model.forward(tensor)
