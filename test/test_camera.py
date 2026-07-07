import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from models.camera import Camera


def test_camera_capture_returns_bgr_frame():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("models.camera.cv2.VideoCapture") as MockCap:
        instance = MagicMock()
        instance.read.return_value = (True, fake_frame)
        instance.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        MockCap.return_value = instance

        cam = Camera(0, 640, 480)
        ok, frame = cam.capture()
        assert ok
        assert frame.shape == (480, 640, 3)
        cam.release()
