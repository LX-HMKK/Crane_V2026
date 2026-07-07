import cv2


class Camera:
    """Simple wrapper around cv2.VideoCapture."""

    def __init__(self, device_index, width, height, buffer_size=1):
        self.device_index = device_index
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def capture(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
