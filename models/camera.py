import cv2
import numpy as np


class Camera:
    """Simple wrapper around cv2.VideoCapture."""

    def __init__(self, device_index, width, height, buffer_size=1, fps=None):
        self.device_index = device_index
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def capture(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class MockCamera:
    """Synthetic camera for UI testing without real USB cameras."""

    def __init__(self, width=640, height=480, pattern="color_bars"):
        self.width = int(width)
        self.height = int(height)
        self.pattern = pattern
        self._counter = 0

    def capture(self):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.pattern == "color_bars":
            bar_width = max(1, self.width // 8)
            colors = [
                (0, 0, 255),    # red
                (0, 165, 255),  # orange
                (0, 255, 255),  # yellow
                (0, 255, 0),    # green
                (255, 0, 0),    # blue
                (255, 0, 255),  # magenta
                (255, 255, 0),  # cyan
                (255, 255, 255),  # white
            ]
            for i, color in enumerate(colors):
                x1 = i * bar_width
                x2 = min(self.width, x1 + bar_width)
                frame[:, x1:x2] = color
        elif self.pattern == "empty":
            pass
        else:
            # checkerboard
            square = max(20, min(self.width, self.height) // 8)
            for y in range(0, self.height, square):
                for x in range(0, self.width, square):
                    if ((x // square) + (y // square)) % 2 == 0:
                        color = (255, 255, 255)
                    else:
                        color = (0, 0, 0)
                    y2 = min(self.height, y + square)
                    x2 = min(self.width, x + square)
                    frame[y:y2, x:x2] = color

        if self.pattern != "empty":
            label = f"MOCK CAMERA #{self._counter}"
            cv2.putText(
                frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
        self._counter += 1
        return True, frame

    def release(self):
        pass


class StaticImageCamera:
    """Camera that repeatedly returns a fixed image file.

    Useful for x86 offline tests with captured frames. The image is resized to
    the configured width/height on first capture.
    """

    def __init__(self, image_path, width, height):
        self.image_path = image_path
        self.width = int(width)
        self.height = int(height)
        self._frame = None

    def capture(self):
        if self._frame is None:
            frame = cv2.imread(str(self.image_path))
            if frame is None:
                raise RuntimeError(f"Cannot load image {self.image_path}")
            if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height))
            self._frame = frame
        return True, self._frame.copy()

    def release(self):
        self._frame = None
