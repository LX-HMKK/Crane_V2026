import sys
import threading
from subprocess import TimeoutExpired
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from models.devices.camera import Camera


def test_camera_capture_returns_bgr_frame():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("models.devices.camera.cv2.VideoCapture") as MockCap:
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
    with patch("models.devices.camera.cv2.VideoCapture") as MockCap:
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
    with patch("models.devices.camera.cv2.VideoCapture") as MockCap:
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
    with patch("models.devices.camera.cv2.VideoCapture") as MockCap:
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
    with patch("models.devices.camera.cv2.VideoCapture") as MockCap:
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


def test_camera_v4l2_timeout_falls_back_without_blocking():
    cam = Camera.__new__(Camera)
    cam.device_index = "/dev/video0"

    with patch("models.devices.camera.shutil.which", return_value="/usr/bin/v4l2-ctl"), \
         patch(
             "models.devices.camera.subprocess.run",
             side_effect=TimeoutExpired("v4l2-ctl", 2.0),
         ):
        assert cam._set_v4l2_control("auto_exposure", 3) is False


def test_camera_reopens_after_sustained_read_failures():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("models.devices.camera.cv2.VideoCapture") as factory:
        bad = MagicMock()
        bad.read.return_value = (False, None)
        bad.isOpened.return_value = True
        bad.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480

        good = MagicMock()
        good.read.return_value = (True, fake_frame)
        good.isOpened.return_value = True
        good.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        factory.side_effect = [bad, good]

        cam = Camera(0, 640, 480, reconnect_after=1, reconnect_interval=0.0)
        ok, _frame = cam.capture()
        assert ok is False
        ok, frame = cam.capture()
        assert ok is True
        assert frame is fake_frame
        # The first failure triggered a reopen, so the factory was called twice.
        assert factory.call_count == 2
        cam.release()


def test_camera_transient_single_failure_does_not_reopen():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("models.devices.camera.cv2.VideoCapture") as MockCap:
        instance = MagicMock()
        instance.read.side_effect = [(False, None), (True, fake_frame)]
        instance.isOpened.return_value = True
        instance.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        MockCap.return_value = instance

        cam = Camera(0, 640, 480, reconnect_after=3, reconnect_interval=0.0)
        ok, _frame = cam.capture()
        assert ok is False
        ok, frame = cam.capture()
        assert ok is True
        # One transient failure is below reconnect_after; never reopened.
        assert MockCap.call_count == 1
        cam.release()


def test_camera_no_reopen_when_reconnect_disabled():
    with patch("models.devices.camera.cv2.VideoCapture") as MockCap:
        instance = MagicMock()
        instance.read.return_value = (False, None)
        instance.isOpened.return_value = True
        instance.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        MockCap.return_value = instance

        cam = Camera(0, 640, 480, reconnect_enabled=False)
        for _ in range(5):
            ok, _frame = cam.capture()
            assert ok is False
        assert MockCap.call_count == 1
        cam.release()


def test_camera_reopen_reapplies_capture_parameters():
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("models.devices.camera.cv2.VideoCapture") as factory:
        bad = MagicMock()
        bad.read.return_value = (False, None)
        bad.isOpened.return_value = True
        bad.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480

        good = MagicMock()
        good.read.return_value = (True, fake_frame)
        good.isOpened.return_value = True
        good.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        factory.side_effect = [bad, good]

        cam = Camera(
            0, 640, 480, auto_exposure=False, exposure=123,
            reconnect_after=1, reconnect_interval=0.0,
        )
        cam.capture()
        cam.capture()

        assert factory.call_count == 2
        # _configure_cap() replays FourCC / size / exposure on the new handle.
        good.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
        good.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fourcc_calls = [
            call for call in good.set.call_args_list
            if call[0][0] == cv2.CAP_PROP_FOURCC
        ]
        assert len(fourcc_calls) == 1
        good.set.assert_any_call(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        good.set.assert_any_call(cv2.CAP_PROP_EXPOSURE, 123.0)
        cam.release()


def test_camera_release_and_setters_safe_when_cap_none():
    cam = Camera.__new__(Camera)
    cam.device_index = 0
    cam._io_lock = threading.RLock()
    cam.cap = None

    # No handle held: none of these may raise.
    cam.release()
    assert cam.set_auto_exposure(True) is False
    assert cam.set_exposure(123) is False


def test_camera_reopen_replays_runtime_exposure_change():
    """Exposure changed via the settings UI must survive a reconnect."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("models.devices.camera.cv2.VideoCapture") as factory:
        bad = MagicMock()
        bad.read.return_value = (False, None)
        bad.isOpened.return_value = True
        bad.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480

        good = MagicMock()
        good.read.return_value = (True, fake_frame)
        good.isOpened.return_value = True
        good.get.side_effect = lambda p: 640 if p == cv2.CAP_PROP_FRAME_WIDTH else 480
        factory.side_effect = [bad, good]

        cam = Camera(
            0, 640, 480, auto_exposure=False, exposure=100,
            reconnect_after=1, reconnect_interval=0.0,
        )
        # Operator adjusts exposure at runtime, then the device drops.
        cam.set_exposure(800)
        cam.capture()  # fails -> reopen on the good handle
        cam.capture()  # succeeds on the new handle

        assert factory.call_count == 2
        # The runtime value (800) is replayed on the new handle, not boot 100.
        good.set.assert_any_call(cv2.CAP_PROP_EXPOSURE, 800.0)
        cam.release()
