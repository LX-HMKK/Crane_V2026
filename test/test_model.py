import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.bpu_loader import BpuModel


def test_bpu_model_raises_on_x86():
    with pytest.raises(EnvironmentError):
        BpuModel("assets/num.bin", "assets/num_rdkx5_config.yaml")
