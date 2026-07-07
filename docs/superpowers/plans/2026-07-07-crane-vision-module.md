# Crane 视觉模块实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 RDKx5 上实现双相机、双 BPU 模型、串口通信的竞赛视觉模块，并附带 x86 可测的单元测试、OpenCV 校准 UI 和看门狗脚本。

**Architecture:** 所有板端可复用模块放在 `models/` 下（后处理、BPU 加载、相机、串口、业务逻辑、UI）；`scripts/` 只保留 `main.py` 竞赛入口和 `watchdog.sh`；`configs/vision.yaml` 集中保存运行配置与辅助线位置；`test/` 存放针对串口、相机、模型、后处理、任务生成的单元测试。

**Tech Stack:** Python 3, OpenCV, NumPy, PyYAML, pyserial, hobot_dnn (仅 RDKx5)

---

## 文件结构总览

```text
crane/
├── configs/vision.yaml              # 运行配置（将被 UI 自动修改）
├── models/
│   ├── __init__.py
│   ├── yolo11_postprocess.py        # letterbox / DFL / NMS / 坐标回映
│   ├── bpu_loader.py                # hobot_dnn 封装，x86 上抛 EnvironmentError
│   ├── camera.py                    # 单相机取流封装
│   ├── serial_port.py               # START / TASK 帧
│   ├── mapping.py                   # 豆子类别 → 目标数字
│   ├── task_generator.py            # 辅助线分区 + 反向查找
│   ├── bean_model.py                # 豆子 BPU 模型 + 后处理封装
│   ├── digit_model.py               # 数字 BPU 模型 + 后处理封装
│   └── ui.py                        # OpenCV 校准 UI
├── scripts/
│   ├── main.py                      # 状态机 + 竞赛入口（--ui 调试模式）
│   └── watchdog.sh                  # 开机自启动 + 崩溃重启
├── test/
│   ├── test_postprocess.py
│   ├── test_camera.py
│   ├── test_serial.py
│   ├── test_model.py
│   ├── test_task_generator.py
│   ├── test_ui.py
│   └── test_main.py
└── docs/
    ├── RDKx5_Bin_Inference.md       # 修正 YOLOv5 遗留描述
    ├── vision_runtime.md            # 新增
    ├── vision_mapping.md            # 新增
    └── README.md                    # 更新
```

---

## Task 1: 创建运行配置 `configs/vision.yaml`

**Files:**
- Create: `configs/vision.yaml`

**说明：** 这是运行时的唯一真相源。`models/ui.py` 调节辅助线后会重写此文件；`scripts/main.py` 启动时读取。

- [ ] **Step 1: 创建配置文件**

```yaml
# Crane 视觉模块运行时配置
# 注意：辅助线位置会在 UI 校准后自动更新，请通过 python scripts/main.py --ui 调节

serial:
  port: /dev/ttyUSB0
  baudrate: 115200
  timeout: 0.05

cameras:
  bean:
    device: 0
    width: 640
    height: 480
    buffer_size: 1
  digit:
    device: 2
    width: 640
    height: 480
    buffer_size: 1

models:
  bean:
    bin: assets/beans_v4.bin
    cfg: assets/beans_v4_rdkx5_config.yaml
  digit:
    bin: assets/num.bin
    cfg: assets/num_rdkx5_config.yaml

inference:
  imgsz: 640
  conf_thres: 0.25
  iou_thres: 0.45
  pad: 127

scan:
  frames: 5
  max_retries: 2
  timeout_sec: 30

lines:
  bean:
    left_x: 213
    right_x: 426
  digit:
    horizontal_y: 240
    left_x: 213
    right_x: 426

mapping:
  bean_to_digit:
    yellow: 1
    green: 2
    white: 3
```

- [ ] **Step 2: 提交**

```bash
git add configs/vision.yaml models/__init__.py
git commit -m "feat(config): 添加视觉模块运行时配置与 models 包标识"
```

---

## Task 2: 共享后处理库 `models/yolo11_postprocess.py`

**Files:**
- Create: `models/yolo11_postprocess.py`
- Create: `test/test_postprocess.py`

### Step 1: 写第一个失败测试

- [ ] 创建 `test/test_postprocess.py`：

```python
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.yolo11_postprocess import letterbox, postprocess, scale_coords


def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, scale, pad = letterbox(img, new_shape=(640, 640), color=127)
    assert out.shape == (640, 640, 3)
    assert scale == 1.0
    assert pad == (0, 0)


def test_postprocess_returns_empty_for_zero_tensors():
    cfg = {
        "imgsz": 640,
        "nc": 5,
        "reg_max": 16,
        "strides": [8, 16, 32],
        "conf_thres": 0.25,
        "iou_thres": 0.45,
    }
    outputs = [
        np.zeros((1, 80, 80, 64), dtype=np.float32),
        np.zeros((1, 40, 40, 64), dtype=np.float32),
        np.zeros((1, 20, 20, 64), dtype=np.float32),
        np.zeros((1, 80, 80, 5), dtype=np.float32),
        np.zeros((1, 40, 40, 5), dtype=np.float32),
        np.zeros((1, 20, 20, 5), dtype=np.float32),
    ]
    dets = postprocess(outputs, cfg)
    assert dets.shape[0] == 0


def test_scale_coords_reverses_letterbox():
    dets = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)
    scaled = scale_coords(dets, scale=1.0, pad=(10, 20))
    assert scaled[0, 0] == 90
    assert scaled[0, 1] == 80
    assert scaled[0, 2] == 190
    assert scaled[0, 3] == 180
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest test/test_postprocess.py -v
```

Expected: 失败，提示 `models/yolo11_postprocess.py` 不存在或函数未定义。

### Step 3: 实现后处理库

- [ ] 创建 `models/yolo11_postprocess.py`：

```python
import cv2
import numpy as np


def _softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def letterbox(im, new_shape=(640, 640), color=127, stride=32):
    """Resize and pad image to new_shape while preserving aspect ratio."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def _make_anchors(imgsz, strides):
    """Generate anchor centers for each stride."""
    anchors = []
    for stride in strides:
        h = w = imgsz // stride
        xv, yv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        xv = (xv + 0.5) * stride
        yv = (yv + 0.5) * stride
        anchors.append(np.stack([xv.ravel(), yv.ravel()], axis=-1))
    return np.concatenate(anchors, axis=0)


def _dfl_decode(pred_dfl, reg_max):
    """Decode Distribution Focal Loss output to l,t,r,b distances."""
    b = pred_dfl.shape[0]
    pred = pred_dfl.reshape(b, 4, reg_max)
    pred = _softmax(pred, axis=2)
    proj = np.arange(reg_max, dtype=np.float32)
    return (pred * proj).sum(axis=2)  # (b, 4)


def postprocess(outputs, cfg):
    """
    Convert raw 6-head outputs to detections.
    outputs order: [s_bbox, m_bbox, l_bbox, s_cls, m_cls, l_cls]
    Returns: ndarray of shape (N, 6) -> [x1, y1, x2, y2, conf, cls_id]
    """
    imgsz = cfg["imgsz"]
    nc = cfg["nc"]
    reg_max = cfg["reg_max"]
    strides = cfg["strides"]
    conf_thres = cfg.get("conf_thres", 0.25)
    iou_thres = cfg.get("iou_thres", 0.45)

    anchors = _make_anchors(imgsz, strides)

    dfl_parts = []
    cls_parts = []
    for i, stride in enumerate(strides):
        h = w = imgsz // stride
        dfl_parts.append(outputs[i].reshape(-1, 4 * reg_max))
        cls_parts.append(outputs[i + 3].reshape(-1, nc))

    dfl_preds = np.concatenate(dfl_parts, axis=0)
    cls_preds = np.concatenate(cls_parts, axis=0)

    bbox = _dfl_decode(dfl_preds, reg_max)

    xyxy = np.empty_like(bbox)
    xyxy[:, 0] = anchors[:, 0] - bbox[:, 0]
    xyxy[:, 1] = anchors[:, 1] - bbox[:, 1]
    xyxy[:, 2] = anchors[:, 0] + bbox[:, 2]
    xyxy[:, 3] = anchors[:, 1] + bbox[:, 3]

    cls_sigmoid = 1.0 / (1.0 + np.exp(-cls_preds))
    conf = cls_sigmoid.max(axis=1)
    cls_id = cls_sigmoid.argmax(axis=1)

    mask = conf >= conf_thres
    xyxy = xyxy[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]

    if len(xyxy) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    indices = cv2.dnn.NMSBoxes(
        [xyxy[i].tolist() for i in range(len(xyxy))],
        conf.tolist(),
        conf_thres,
        iou_thres,
    )

    if len(indices) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    if isinstance(indices, tuple):
        indices = indices[0]
    indices = indices.flatten()

    dets = np.concatenate(
        [xyxy[indices], conf[indices][:, None], cls_id[indices][:, None]],
        axis=1,
    )
    return dets


def scale_coords(dets, scale, pad):
    """Map detection coordinates from letterbox space back to original image."""
    dets = dets.copy()
    left, top = pad
    dets[:, [0, 2]] -= left
    dets[:, [1, 3]] -= top
    dets[:, :4] /= scale
    return dets
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest test/test_postprocess.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: 提交**

```bash
git add models/yolo11_postprocess.py test/test_postprocess.py
git commit -m "feat(postprocess): 添加 YOLO11 6-head 后处理库及测试"
```

---

## Task 3: BPU 加载器 `models/bpu_loader.py`

**Files:**
- Create: `models/bpu_loader.py`
- Create: `test/test_model.py`（仅含对 BpuModel 的 mock 测试）

### Step 1: 写失败测试

- [ ] 创建 `test/test_model.py`：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.bpu_loader import BpuModel


def test_bpu_model_raises_on_x86():
    with pytest.raises(EnvironmentError):
        BpuModel("assets/num.bin", "assets/num_rdkx5_config.yaml")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest test/test_model.py -v
```

Expected: 失败，提示 `models/bpu_loader.py` 不存在。

### Step 3: 实现 BpuModel

- [ ] 创建 `models/bpu_loader.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest test/test_model.py -v
```

Expected: PASS（x86 上导入 hobot_dnn 失败，构造 BpuModel 抛出 EnvironmentError）。

- [ ] **Step 5: 提交**

```bash
git add models/bpu_loader.py test/test_model.py
git commit -m "feat(bpu): 添加 hobot_dnn 封装并在 x86 上优雅失败"
```

---

## Task 4: 相机封装 `models/camera.py`

**Files:**
- Create: `models/camera.py`
- Create: `test/test_camera.py`

### Step 1: 写失败测试

- [ ] 创建 `test/test_camera.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest test/test_camera.py -v
```

Expected: 失败，提示 `models/camera.py` 不存在。

### Step 3: 实现 Camera

- [ ] 创建 `models/camera.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest test/test_camera.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add models/camera.py test/test_camera.py
git commit -m "feat(camera): 添加 OpenCV 相机封装及 mock 测试"
```

---

## Task 5: 串口封装 `models/serial_port.py`

**Files:**
- Create: `models/serial_port.py`
- Create: `test/test_serial.py`

### Step 1: 写失败测试

- [ ] 创建 `test/test_serial.py`：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import serial
import time
import threading
from models.serial_port import SerialPort, START_FRAME


def test_wait_start_detects_start_frame():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)

    def writer():
        time.sleep(0.05)
        ser.write(b"\x00\x01" + START_FRAME + b"\x00")

    t = threading.Thread(target=writer)
    t.start()
    ok = port.wait_start(timeout=2.0)
    t.join()
    assert ok
    port.close()


def test_send_task_writes_correct_frame():
    ser = serial.serial_for_url("loop://", timeout=0.01)
    port = SerialPort(ser)
    port.send_task(5, 6, 4)
    time.sleep(0.05)
    data = ser.read(20)
    assert data == b"\xBB\x66\x05\x06\x04\x1B\x1A"
    port.close()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest test/test_serial.py -v
```

Expected: 失败，提示 `models/serial_port.py` 不存在。

### Step 3: 实现 SerialPort

- [ ] 创建 `models/serial_port.py`：

```python
import struct
import time


START_FRAME = bytes([0xAA, 0x55, 0x01, 0x0D, 0x0A])
TASK_FRAME_SIZE = 7
TASK_HEAD = bytes([0xBB, 0x66])
TASK_TAIL = bytes([0x1B, 0x1A])


class SerialPort:
    """Handle START frame reception and TASK frame transmission."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = bytearray()

    def wait_start(self, timeout=None):
        """Sliding-window match of AA 55 01 0D 0A."""
        deadline = None
        if timeout is not None:
            deadline = time.time() + timeout

        while True:
            if deadline and time.time() > deadline:
                return False
            byte = self.ser.read(1)
            if not byte:
                continue
            self.buf.extend(byte)
            if len(self.buf) > 16:
                self.buf.pop(0)
            if len(self.buf) >= 5 and self.buf[-5:] == START_FRAME:
                return True

    def send_task(self, p1, p2, p3):
        frame = struct.pack(
            ">2B3B2B",
            TASK_HEAD[0],
            TASK_HEAD[1],
            p1,
            p2,
            p3,
            TASK_TAIL[0],
            TASK_TAIL[1],
        )
        self.ser.write(frame)

    def close(self):
        self.ser.close()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest test/test_serial.py -v
```

Expected: 2 tests PASS。

- [ ] **Step 5: 提交**

```bash
git add models/serial_port.py test/test_serial.py
git commit -m "feat(serial): 添加串口 START/TASK 帧封装及测试"
```

---

## Task 6: 豆子到数字映射 `models/mapping.py`

**Files:**
- Create: `models/mapping.py`

- [ ] **Step 1: 创建映射模块**

```python
DEFAULT_BEAN_TO_DIGIT = {
    "yellow": 1,  # 黄豆
    "green": 2,   # 绿豆
    "white": 3,   # 白芸豆
}


def bean_name_to_digit(name, mapping=None):
    if mapping is None:
        mapping = DEFAULT_BEAN_TO_DIGIT
    return mapping.get(name)
```

- [ ] **Step 2: 提交**

```bash
git add models/mapping.py
git commit -m "feat(mapping): 添加豆子类别到目标数字映射"
```

---

## Task 7: 任务生成器 `models/task_generator.py`

**Files:**
- Create: `models/task_generator.py`
- Create: `test/test_task_generator.py`

### Step 1: 写失败测试

- [ ] 创建 `test/test_task_generator.py`：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.task_generator import TaskGenerator


def digit_det(zone, digit):
    """Helper: [x1, y1, x2, y2, conf, cls_id]; names map cls_id -> digit value."""
    return [0.0, 0.0, 10.0, 10.0, 0.9, digit - 1]


def bean_det(zone, cls_id):
    return [0.0, 0.0, 10.0, 10.0, 0.9, cls_id]


def test_generate_task_success():
    # 豆子：zone1=yellow(0), zone2=green(1), zone3=white(2)
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    # 数字：4/5/6/7/8 分别为 1/2/3/4/5
    digit_names = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    digit_dets = [
        digit_det(4, 1), digit_det(5, 2), digit_det(6, 3),
        digit_det(7, 4), digit_det(8, 5),
    ]
    lines = {
        "bean": {"left_x": 213, "right_x": 426},
        "digit": {"horizontal_y": 240, "left_x": 213, "right_x": 426},
    }
    mapping = {"yellow": 1, "green": 2, "white": 3}
    gen = TaskGenerator(mapping, lines, digit_names)
    result = gen.generate(bean_dets, digit_dets)
    # zone1(yellow->1) -> placement 4
    # zone2(green->2)  -> placement 5
    # zone3(white->3)  -> placement 6
    assert result == (4, 5, 6)


def test_generate_task_missing_digit():
    bean_dets = [bean_det(1, 2), bean_det(2, 1), bean_det(3, 0)]
    digit_names = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
    digit_dets = [digit_det(4, 1)]  # missing 2 and 3
    lines = {
        "bean": {"left_x": 213, "right_x": 426},
        "digit": {"horizontal_y": 240, "left_x": 213, "right_x": 426},
    }
    mapping = {"yellow": 1, "green": 2, "white": 3}
    gen = TaskGenerator(mapping, lines, digit_names)
    result, reason = gen.generate(bean_dets, digit_dets)
    assert result is None
    assert "missing" in reason.lower()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest test/test_task_generator.py -v
```

Expected: 失败。

### Step 3: 实现 TaskGenerator

- [ ] 创建 `models/task_generator.py`：

```python
class TaskGenerator:
    """Convert bean/digit detections into three placement-zone numbers."""

    def __init__(self, mapping, lines, digit_names):
        self.mapping = mapping
        self.lines = lines
        self.digit_names = digit_names

    def _bean_zone(self, cx):
        left = self.lines["bean"]["left_x"]
        right = self.lines["bean"]["right_x"]
        if cx < left:
            return 1
        if cx < right:
            return 2
        return 3

    def _digit_zone(self, cx, cy):
        d = self.lines["digit"]
        horizontal_y = d["horizontal_y"]
        left_x = d["left_x"]
        right_x = d["right_x"]
        midpoint = (left_x + right_x) / 2

        if cy < horizontal_y:
            return 4 if cx < midpoint else 8
        if cx < left_x:
            return 5
        if cx < right_x:
            return 6
        return 7

    def _center(self, det):
        x1, y1, x2, y2 = det[:4]
        return (x1 + x2) / 2, (y1 + y2) / 2

    def generate(self, bean_dets, digit_dets):
        # Aggregate best bean per pickup zone
        bean_best = {1: None, 2: None, 3: None}
        for det in bean_dets:
            cx, cy = self._center(det)
            zone = self._bean_zone(cx)
            if bean_best[zone] is None or det[4] > bean_best[zone][4]:
                bean_best[zone] = det

        # Map beans to target digits
        target_digits = {}
        for zone in (1, 2, 3):
            det = bean_best[zone]
            if det is None:
                return None, f"missing bean in pickup zone {zone}"
            cls_id = int(det[5])
            # bean config names: 0:white, 1:green, 2:yellow
            name_map = {0: "white", 1: "green", 2: "yellow"}
            name = name_map.get(cls_id)
            digit = self.mapping.get(name)
            if digit is None:
                return None, f"unknown bean class {cls_id} in zone {zone}"
            target_digits[zone] = digit

        # Aggregate best digit per placement zone
        digit_best = {4: None, 5: None, 6: None, 7: None, 8: None}
        for det in digit_dets:
            cx, cy = self._center(det)
            zone = self._digit_zone(cx, cy)
            if digit_best[zone] is None or det[4] > digit_best[zone][4]:
                digit_best[zone] = det

        # Reverse lookup: target digit -> placement zone
        placement_for_digit = {}
        for zone, det in digit_best.items():
            if det is None:
                continue
            cls_id = int(det[5])
            name = self.digit_names.get(cls_id)
            if name is None:
                continue
            try:
                value = int(name)
            except ValueError:
                continue
            placement_for_digit[value] = zone

        result = []
        for zone in (1, 2, 3):
            target = target_digits[zone]
            place = placement_for_digit.get(target)
            if place is None:
                return None, f"missing digit {target} in placement zones"
            result.append(place)

        return tuple(result), "ok"
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest test/test_task_generator.py -v
```

Expected: 2 tests PASS。

- [ ] **Step 5: 提交**

```bash
git add models/task_generator.py test/test_task_generator.py
git commit -m "feat(task): 添加任务生成器与辅助线分区逻辑"
```

---

## Task 8: 模型封装 `models/bean_model.py` 与 `models/digit_model.py`

**Files:**
- Create: `models/bean_model.py`
- Create: `models/digit_model.py`
- Modify: `test/test_model.py`

### Step 1: 写失败测试

- [ ] 追加到 `test/test_model.py`：

```python
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.bean_model import BeanModel


def create_dummy_cfg(path, names):
    cfg = {
        "imgsz": 640,
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
        "reg_max": 16,
        "strides": [8, 16, 32],
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return cfg


def test_bean_model_predict_calls_bpu_and_postprocess(tmp_path):
    cfg_path = tmp_path / "bean.yaml"
    create_dummy_cfg(cfg_path, ["white", "green", "yellow"])

    fake_outputs = [
        np.zeros((1, 80, 80, 64), dtype=np.float32),
        np.zeros((1, 40, 40, 64), dtype=np.float32),
        np.zeros((1, 20, 20, 64), dtype=np.float32),
        np.zeros((1, 80, 80, 3), dtype=np.float32),
        np.zeros((1, 40, 40, 3), dtype=np.float32),
        np.zeros((1, 20, 20, 3), dtype=np.float32),
    ]

    with patch("models.bean_model.BpuModel") as MockBpu:
        bpu = MagicMock()
        bpu.infer.return_value = fake_outputs
        MockBpu.return_value = bpu

        model = BeanModel("assets/beans_v4.bin", str(cfg_path), {"conf_thres": 0.25, "iou_thres": 0.45})
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = model.predict(img)
        assert dets.shape[0] == 0
        bpu.infer.assert_called_once()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest test/test_model.py -v
```

Expected: 失败，提示 `models/bean_model.py` 不存在。

### Step 3: 实现 BeanModel 与 DigitModel

- [ ] 创建 `models/bean_model.py`：

```python
import cv2
import numpy as np
import yaml

from models.bpu_loader import BpuModel
from models.yolo11_postprocess import letterbox, postprocess, scale_coords


class BeanModel:
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
```

- [ ] 创建 `models/digit_model.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest test/test_model.py -v
```

Expected: 2 tests PASS（BpuModel x86 失败测试 + BeanModel mock 预测测试）。

- [ ] **Step 5: 提交**

```bash
git add models/bean_model.py models/digit_model.py test/test_model.py
git commit -m "feat(model): 添加豆子/数字模型推理封装"
```

---

## Task 9: 校准 UI `models/ui.py`

**Files:**
- Create: `models/ui.py`
- Create: `test/test_ui.py`（仅测试配置读写逻辑，避免依赖真实相机）

### Step 1: 写失败测试

- [ ] 创建 `test/test_ui.py`：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from unittest.mock import MagicMock
from models.ui import CalibrationUI


def test_ui_saves_lines_on_change(tmp_path):
    cfg_path = tmp_path / "vision.yaml"
    cfg = {
        "lines": {
            "bean": {"left_x": 100, "right_x": 300},
            "digit": {"horizontal_y": 200, "left_x": 100, "right_x": 300},
        },
        "inference": {"imgsz": 640},
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    cam_bean = MagicMock()
    cam_digit = MagicMock()
    ui = CalibrationUI(str(cfg_path), cam_bean, cam_digit)
    ui._set_line("bean", "left_x", 150)
    ui._set_line("digit", "horizontal_y", 250)

    with open(cfg_path, "r") as f:
        saved = yaml.safe_load(f)
    assert saved["lines"]["bean"]["left_x"] == 150
    assert saved["lines"]["digit"]["horizontal_y"] == 250
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest test/test_ui.py -v
```

Expected: 失败。

### Step 3: 实现 CalibrationUI

- [ ] 创建 `models/ui.py`：

```python
import cv2
import yaml
import numpy as np

LINE_COLORS = {
    "bean_left_x": (0, 255, 0),
    "bean_right_x": (0, 255, 0),
    "digit_horizontal_y": (255, 0, 0),
    "digit_left_x": (255, 0, 0),
    "digit_right_x": (255, 0, 0),
}


class CalibrationUI:
    """OpenCV-based UI to adjust guide lines and inspect live inference."""

    def __init__(self, cfg_path, cam_bean, cam_digit, bean_model=None, digit_model=None):
        self.cfg_path = cfg_path
        self.cam_bean = cam_bean
        self.cam_digit = cam_digit
        self.bean_model = bean_model
        self.digit_model = digit_model
        self.mode = "bean"
        self._load_cfg()
        self._init_window()

    def _load_cfg(self):
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def _save_cfg(self):
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f, sort_keys=False, allow_unicode=True)

    def _init_window(self):
        cv2.namedWindow("calibration")
        cv2.createTrackbar("bean_left_x", "calibration", self.cfg["lines"]["bean"]["left_x"], 640, lambda v: self._set_line("bean", "left_x", v))
        cv2.createTrackbar("bean_right_x", "calibration", self.cfg["lines"]["bean"]["right_x"], 640, lambda v: self._set_line("bean", "right_x", v))
        cv2.createTrackbar("digit_h_y", "calibration", self.cfg["lines"]["digit"]["horizontal_y"], 480, lambda v: self._set_line("digit", "horizontal_y", v))
        cv2.createTrackbar("digit_left_x", "calibration", self.cfg["lines"]["digit"]["left_x"], 640, lambda v: self._set_line("digit", "left_x", v))
        cv2.createTrackbar("digit_right_x", "calibration", self.cfg["lines"]["digit"]["right_x"], 640, lambda v: self._set_line("digit", "right_x", v))

    def _set_line(self, group, key, value):
        self.cfg["lines"][group][key] = value
        self._save_cfg()

    def _draw_bean_lines(self, frame):
        h, w = frame.shape[:2]
        left = self.cfg["lines"]["bean"]["left_x"]
        right = self.cfg["lines"]["bean"]["right_x"]
        cv2.line(frame, (left, 0), (left, h), LINE_COLORS["bean_left_x"], 2)
        cv2.line(frame, (right, 0), (right, h), LINE_COLORS["bean_right_x"], 2)

    def _draw_digit_lines(self, frame):
        h, w = frame.shape[:2]
        hy = self.cfg["lines"]["digit"]["horizontal_y"]
        left = self.cfg["lines"]["digit"]["left_x"]
        right = self.cfg["lines"]["digit"]["right_x"]
        cv2.line(frame, (0, hy), (w, hy), LINE_COLORS["digit_horizontal_y"], 2)
        cv2.line(frame, (left, hy), (left, h), LINE_COLORS["digit_left_x"], 2)
        cv2.line(frame, (right, hy), (right, h), LINE_COLORS["digit_right_x"], 2)

    def _draw_detections(self, frame, dets, names):
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            name = names.get(cls_id, str(cls_id))
            label = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def run(self):
        print("Calibration UI started. Keys: 1=bean camera, 2=digit camera, ESC=exit")
        while True:
            if self.mode == "bean":
                ok, frame = self.cam_bean.capture()
                if not ok:
                    continue
                self._draw_bean_lines(frame)
                if self.bean_model is not None:
                    try:
                        dets = self.bean_model.predict(frame)
                        self._draw_detections(frame, dets, self.bean_model.names)
                    except Exception as e:
                        cv2.putText(frame, f"infer error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                label = "BEAN (1/2 switch, ESC exit)"
            else:
                ok, frame = self.cam_digit.capture()
                if not ok:
                    continue
                self._draw_digit_lines(frame)
                if self.digit_model is not None:
                    try:
                        dets = self.digit_model.predict(frame)
                        self._draw_detections(frame, dets, self.digit_model.names)
                    except Exception as e:
                        cv2.putText(frame, f"infer error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                label = "DIGIT (1/2 switch, ESC exit)"

            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("calibration", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("1"):
                self.mode = "bean"
            elif key == ord("2"):
                self.mode = "digit"

        cv2.destroyAllWindows()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest test/test_ui.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add models/ui.py test/test_ui.py
git commit -m "feat(ui): 添加 OpenCV 辅助线校准 UI"
```

---

## Task 10: 主程序 `scripts/main.py`

**Files:**
- Create: `scripts/main.py`
- Modify: `models/__init__.py`

### Step 1: 写失败测试

- [ ] 创建 `test/test_main.py`：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import serial
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import main as main_module


def test_main_loads_vision_config(tmp_path):
    cfg_path = tmp_path / "vision.yaml"
    cfg_path.write_text("lines:\n  bean:\n    left_x: 100\n", encoding="utf-8")
    cfg = main_module.load_vision_cfg(str(cfg_path))
    assert cfg["lines"]["bean"]["left_x"] == 100


def test_serial_frame_round_trip():
    # Smoke test for START/TASK frame plumbing.
    from models.serial_port import START_FRAME, SerialPort
    ser = serial.serial_for_url("loop://", timeout=0.01)

    def writer():
        time.sleep(0.1)
        ser.write(START_FRAME)

    t = threading.Thread(target=writer)
    t.start()

    port = SerialPort(ser)
    ok = port.wait_start(timeout=2.0)
    t.join()
    assert ok
    port.send_task(4, 5, 6)
    time.sleep(0.05)
    assert ser.read(10) == b"\xBB\x66\x04\x05\x06\x1B\x1A"
    port.close()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest test/test_main.py -v
```

Expected: 失败，提示 `scripts/main.py` 不存在或无法导入。

### Step 3: 实现 main.py

- [ ] 创建 `scripts/main.py`：

```python
#!/usr/bin/env python3
"""Crane vision runtime entry point."""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Add repo root to path so scripts/ can import models/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.bean_model import BeanModel
from models.camera import Camera
from models.digit_model import DigitModel
from models.mapping import DEFAULT_BEAN_TO_DIGIT
from models.serial_port import SerialPort
from models.task_generator import TaskGenerator
from models.ui import CalibrationUI


def load_vision_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def list_cameras():
    import cv2
    print("Available cameras:")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"  /dev/video{i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            cap.release()


def run_calibration(cfg):
    bean_cfg = cfg["cameras"]["bean"]
    digit_cfg = cfg["cameras"]["digit"]
    cam_bean = Camera(bean_cfg["device"], bean_cfg["width"], bean_cfg["height"], bean_cfg.get("buffer_size", 1))
    cam_digit = Camera(digit_cfg["device"], digit_cfg["width"], digit_cfg["height"], digit_cfg.get("buffer_size", 1))

    bean_model = None
    digit_model = None
    try:
        bean_model = BeanModel(cfg["models"]["bean"]["bin"], cfg["models"]["bean"]["cfg"], cfg["inference"])
        digit_model = DigitModel(cfg["models"]["digit"]["bin"], cfg["models"]["digit"]["cfg"], cfg["inference"])
    except EnvironmentError:
        logging.warning("BPU not available; running UI in camera-only mode")

    ui = CalibrationUI(str(ROOT / "configs" / "vision.yaml"), cam_bean, cam_digit, bean_model, digit_model)
    try:
        ui.run()
    finally:
        cam_bean.release()
        cam_digit.release()


def scan_once(bean_cam, digit_cam, bean_model, digit_model, task_gen, cfg):
    frames = cfg["scan"]["frames"]
    all_bean_dets = []
    all_digit_dets = []

    for _ in range(frames):
        ok_b, frame_b = bean_cam.capture()
        ok_d, frame_d = digit_cam.capture()
        if ok_b and bean_model is not None:
            all_bean_dets.extend(bean_model.predict(frame_b))
        if ok_d and digit_model is not None:
            all_digit_dets.extend(digit_model.predict(frame_d))

    return task_gen.generate(all_bean_dets, all_digit_dets)


def run_competition(cfg):
    import serial
    init_logging()
    logging.info("Starting crane vision in competition mode")

    port = SerialPort(serial.Serial(cfg["serial"]["port"], cfg["serial"]["baudrate"], timeout=cfg["serial"]["timeout"]))

    bean_cfg = cfg["cameras"]["bean"]
    digit_cfg = cfg["cameras"]["digit"]
    cam_bean = Camera(bean_cfg["device"], bean_cfg["width"], bean_cfg["height"], bean_cfg.get("buffer_size", 1))
    cam_digit = Camera(digit_cfg["device"], digit_cfg["width"], digit_cfg["height"], digit_cfg.get("buffer_size", 1))

    bean_model = BeanModel(cfg["models"]["bean"]["bin"], cfg["models"]["bean"]["cfg"], cfg["inference"])
    digit_model = DigitModel(cfg["models"]["digit"]["bin"], cfg["models"]["digit"]["cfg"], cfg["inference"])

    with open(cfg["models"]["digit"]["cfg"], "r", encoding="utf-8") as f:
        digit_names = yaml.safe_load(f)["names"]

    task_gen = TaskGenerator(DEFAULT_BEAN_TO_DIGIT, cfg["lines"], digit_names)

    # Warm-up
    logging.info("Warming up models...")
    for _ in range(2):
        ok, frame = cam_bean.capture()
        if ok:
            bean_model.predict(frame)
        ok, frame = cam_digit.capture()
        if ok:
            digit_model.predict(frame)

    logging.info("Ready. Waiting for START frame...")

    try:
        while True:
            if not port.wait_start():
                continue
            logging.info("START received, scanning...")
            result = None
            reason = ""
            for attempt in range(cfg["scan"]["max_retries"] + 1):
                result, reason = scan_once(cam_bean, cam_digit, bean_model, digit_model, task_gen, cfg)
                if result is not None:
                    break
                logging.warning(f"Scan attempt {attempt + 1} failed: {reason}; retrying...")
                time.sleep(0.1)

            if result is None:
                logging.error(f"Failed to generate task: {reason}")
                continue

            p1, p2, p3 = result
            logging.info(f"Sending TASK: {p1} {p2} {p3}")
            port.send_task(p1, p2, p3)
            logging.info("Done. Returning to WAIT_START.")
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        port.close()
        cam_bean.release()
        cam_digit.release()


def main():
    parser = argparse.ArgumentParser(description="Crane vision runtime")
    parser.add_argument("--config", default=str(ROOT / "configs" / "vision.yaml"), help="Path to vision.yaml")
    parser.add_argument("--ui", action="store_true", help="Run calibration UI")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    args = parser.parse_args()

    init_logging()

    if args.list_cameras:
        list_cameras()
        return

    cfg = load_vision_cfg(args.config)

    if args.ui:
        run_calibration(cfg)
    else:
        run_competition(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest test/test_main.py -v
python scripts/main.py --help
```

Expected: pytest PASS；`--help` 显示正常。

- [ ] **Step 5: 提交**

```bash
git add scripts/main.py test/test_main.py
git commit -m "feat(main): 添加竞赛状态机主程序"
```

---

## Task 11: 看门狗脚本 `scripts/watchdog.sh`

**Files:**
- Create: `scripts/watchdog.sh`

- [ ] **Step 1: 创建脚本**

```bash
#!/usr/bin/env bash
# Auto-restart crane vision main.py on RDKx5
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

while true; do
    echo "[watchdog] Starting scripts/main.py at $(date)"
    python3 scripts/main.py || true
    echo "[watchdog] main.py exited ($?), restarting in 2s..."
    sleep 2
done
```

- [ ] **Step 2: 添加可执行权限**

```bash
chmod +x scripts/watchdog.sh
```

- [ ] **Step 3: 在 x86 上仅做语法检查**

```bash
bash -n scripts/watchdog.sh
echo $?
```

Expected: 输出 `0`。

- [ ] **Step 4: 提交**

```bash
git add scripts/watchdog.sh
git commit -m "feat(watchdog): 添加 main.py 自启动/崩溃重启脚本"
```

---

## Task 12: 文档更新

**Files:**
- Modify: `docs/RDKx5_Bin_Inference.md`
- Create: `docs/vision_runtime.md`
- Create: `docs/vision_mapping.md`
- Modify: `README.md`

### Step 1: 修正 `docs/RDKx5_Bin_Inference.md`

- [ ] 把文中“由 `export_rdkx5_onnx.py` 导出 ONNX”等明显指向 YOLOv5 遗留脚本的说法，改为：

```markdown
本文档假设你已经拥有 YOLO11 6-head ONNX 模型，并已完成 `hb_mapper makertbin` 编译。YOLOv5 导出桥 `tools/yolov5/export_rdkx5_onnx.py` 与本项目当前资产无关，请勿使用。
```

同时把文中对 `scripts/deployment/infer_rdkx5_yolo11.py` 的引用改为“板端测试请使用 `scripts/main.py --ui` 中的单图预览功能”。本项目 `scripts/` 下只保留 `main.py` 与 `watchdog.sh`，不再单独维护单图推理脚本。

### Step 2: 创建 `docs/vision_runtime.md`

- [ ] 写入：

```markdown
# Crane 视觉模块运行时说明

## 启动方式

```bash
# 竞赛模式（看门狗会自动调用）
python3 scripts/main.py

# 校准 UI
python3 scripts/main.py --ui

# 查看可用相机
python3 scripts/main.py --list-cameras
```

## 配置字段

见 `configs/vision.yaml`。

## 状态机

INIT → WAIT_START → SCANNING → READY → SEND_TASK → DONE

收到 `START` 帧（`AA 55 01 0D 0A`）前，程序不发送任何数据。

## 看门狗

```bash
./scripts/watchdog.sh
```

`main.py` 退出后会自动重启。
```

### Step 3: 创建 `docs/vision_mapping.md`

- [ ] 写入辅助线规则、豆子→数字→放置区映射、缺失处理策略（直接引用设计文档第 6、9 节）。

### Step 4: 更新 `README.md`

- [ ] 写入项目简介、目录说明、快速开始命令、部署清单。

### Step 5: 提交

```bash
git add docs/RDKx5_Bin_Inference.md docs/vision_runtime.md docs/vision_mapping.md README.md
git commit -m "docs: 更新板端推理、运行时与映射文档"
```

---

## Task 13: 集成与板上验证

**Files:**
- N/A（仅在 RDKx5 执行）

- [ ] **Step 1: x86 全量测试**

```bash
python -m pytest test/ -v
```

Expected: 所有测试 PASS。

- [ ] **Step 2: 部署到 RDKx5**

```bash
scp -r models/ scripts/ configs/ test/ assets/*.bin assets/*_rdkx5_config.yaml \
    rdkx5:/home/root/crane/
```

- [ ] **Step 3: 板端校准**

```bash
ssh rdkx5
cd /home/root/crane
python3 scripts/main.py --ui
```

调节辅助线后保存，确认 `configs/vision.yaml` 已更新。

- [ ] **Step 4: 板端竞赛模式测试**

```bash
python3 scripts/main.py
```

从下位机发送 `START` 帧，验证返回 `TASK` 帧是否正确。

- [ ] **Step 5: 看门狗测试**

```bash
./scripts/watchdog.sh &
kill %1  # 模拟 main.py 退出
```

Expected: watchdog 在 2s 后自动重新启动 `main.py`。

- [ ] **Step 6: 提交测试结果记录（可选）**

若测试通过，可在 `docs/vision_runtime.md` 末尾追加测试记录片段并提交。

---

## Self-Review Checklist

- [ ] **Spec coverage**: 每个设计文档章节都有对应任务。
- [ ] **No placeholders**: 所有任务包含具体代码、命令与提交信息。
- [ ] **Type consistency**: `BpuModel`、`TaskGenerator`、`SerialPort`、`Camera` 接口在所有文件中一致。
- [ ] **x86 可测**: 所有板端依赖均可在 x86 上通过 mock 或异常处理测试。
- [ ] **路径一致**: 默认配置文件路径为 `configs/vision.yaml`，所有脚本从该文件读取。

---

**下一步**: 将本计划保存到 `docs/superpowers/plans/2026-07-07-crane-vision-module.md` 后，选择执行方式：

1. **Subagent-Driven (推荐)**：每个 Task 派一个独立子代理完成，我负责审查。
2. **Inline Execution**：在当前会话中按 Task 顺序直接执行。
