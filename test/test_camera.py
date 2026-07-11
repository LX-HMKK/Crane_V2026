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


def test_camera_enables_auto_exposure_by_default():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("models.camera.cv2.VideoCapture") as MockCap:
        instance = MagicMock()
        instance.read.return_value = (True, fake_frame)
        instance.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        MockCap.return_value = instance

        cam = Camera(0, 640, 480)
        calls = [call for call in instance.set.call_args_list
                 if call[0][0] == cv2.CAP_PROP_AUTO_EXPOSURE]
        assert len(calls) == 1
        assert calls[0][0][1] == 0.75
        cam.release()


def test_camera_can_disable_auto_exposure():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("models.camera.cv2.VideoCapture") as MockCap:
        instance = MagicMock()
        instance.read.return_value = (True, fake_frame)
        instance.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        MockCap.return_value = instance

        cam = Camera(0, 640, 480, auto_exposure=False)
        calls = [call for call in instance.set.call_args_list
                 if call[0][0] == cv2.CAP_PROP_AUTO_EXPOSURE]
        assert len(calls) == 1
        assert calls[0][0][1] == 0.25
        cam.release()


def test_camera_converts_grayscale_input_to_bgr():
    gray_frame = np.zeros((480, 640), dtype=np.uint8)
    with patch("models.camera.cv2.VideoCapture") as MockCap:
        instance = MagicMock()
        instance.read.return_value = (True, gray_frame)
        instance.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        MockCap.return_value = instance

        cam = Camera(0, 640, 480)
        ok, frame = cam.capture()
        assert ok
        assert frame.shape == (480, 640, 3)
        cam.release()


def test_camera_grayscale_option_outputs_three_channel_gray():
    color_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    color_frame[:, :] = (10, 20, 30)  # BGR
    with patch("models.camera.cv2.VideoCapture") as MockCap:
        instance = MagicMock()
        instance.read.return_value = (True, color_frame)
        instance.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        MockCap.return_value = instance

        cam = Camera(0, 640, 480, grayscale=True)
        ok, frame = cam.capture()
        assert ok
        assert frame.shape == (480, 640, 3)
        # All channels should be equal after BGR -> Gray -> BGR.
        assert np.all(frame[:, :, 0] == frame[:, :, 1])
        assert np.all(frame[:, :, 1] == frame[:, :, 2])
        cam.release()
