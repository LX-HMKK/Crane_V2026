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
        BpuModel._validate_cfg(cfg)
        return cfg

    @staticmethod
    def _validate_cfg(cfg):
        """Validate type and consistency of the BPU config."""
        for key in ("imgsz", "nc", "names", "reg_max", "strides"):
            if key not in cfg:
                raise ValueError(f"config missing required key: {key}")

        imgsz = cfg["imgsz"]
        if isinstance(imgsz, int):
            if imgsz <= 0:
                raise ValueError(f"imgsz must be positive, got {imgsz}")
        elif isinstance(imgsz, (list, tuple)):
            if len(imgsz) != 2 or not all(isinstance(v, int) and v > 0 for v in imgsz):
                raise ValueError(
                    f"imgsz as list/tuple must be two positive ints, got {imgsz}"
                )
        else:
            raise ValueError(
                f"imgsz must be int or [h, w], got {type(imgsz).__name__}"
            )

        nc = cfg["nc"]
        if not isinstance(nc, int) or nc <= 0:
            raise ValueError(f"nc must be a positive integer, got {nc!r}")

        names = cfg["names"]
        if isinstance(names, dict):
            if len(names) != nc:
                raise ValueError(
                    f"names dict length ({len(names)}) does not match nc ({nc})"
                )
            for key in names:
                try:
                    int(key)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"names key must be int-convertible, got {key!r}"
                    ) from exc
        elif isinstance(names, (list, tuple)):
            if len(names) != nc:
                raise ValueError(
                    f"names list length ({len(names)}) does not match nc ({nc})"
                )
        else:
            raise ValueError(
                f"names must be dict or list/tuple, got {type(names).__name__}"
            )

        reg_max = cfg["reg_max"]
        if not isinstance(reg_max, int) or reg_max <= 0:
            raise ValueError(f"reg_max must be a positive integer, got {reg_max!r}")

        strides = cfg["strides"]
        if not isinstance(strides, (list, tuple)) or len(strides) == 0:
            raise ValueError(f"strides must be a non-empty list/tuple, got {strides!r}")
        for s in strides:
            if not isinstance(s, int) or s <= 0:
                raise ValueError(f"each stride must be a positive integer, got {s!r}")

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
        imgsz = self.cfg["imgsz"]
        if isinstance(imgsz, int):
            expected = (imgsz, imgsz)
        else:
            expected = tuple(imgsz)
        actual = tuple(tensor.shape[2:])
        if actual != expected:
            raise ValueError(
                f"tensor spatial shape {actual} does not match model imgsz {expected}"
            )
        if not tensor.flags["C_CONTIGUOUS"]:
            tensor = np.ascontiguousarray(tensor)
        return self._bpu_model.forward(tensor)
