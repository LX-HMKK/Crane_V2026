try:
    import hobot_dnn as dnn
    HAS_HOBOT = True
except Exception:
    HAS_HOBOT = False


class BpuModel:
    """Thin wrapper around hobot_dnn for RDKx5 BPU inference."""

    def __init__(self, bin_path, cfg_path):
        if not HAS_HOBOT:
            raise EnvironmentError(
                "hobot_dnn is not available; BpuModel only runs on RDKx5"
            )
        self.bin_path = bin_path
        self.cfg_path = cfg_path
        self.model = dnn.load(bin_path)
        self.m = self.model[0]

    def infer(self, tensor):
        """tensor: np.ndarray uint8, shape (1, 3, H, W) NCHW"""
        return self.m.forward(tensor)
