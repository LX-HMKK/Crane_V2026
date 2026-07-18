# Crane Vision 测试与入口解耦实现计划

> **已实现。** 2026-07-18 最终调整：`calibrate.py` 与 `send_test_frame.py` 从 `scripts/` 移入 `tests/`，与 `test_camera_view.py` 等统一作为独立功能测试入口。生产入口 `scripts/` 仅保留 `main_arm64.py`、`main_amd64.py`、`watchdog.sh`。
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `main_arm64.py` 精简为纯竞赛入口，把校准、串口测试、相机测试、推理测试拆分为独立脚本，并将细碎单元测试迁移到 `tools/crane_tests/`。

**架构：** 新增 `models/camera_viewer.py` 作为极简双相机查看器基类；`models/ui.py` 中的 `CalibrationUI` 继承该基类并增加滑块、保存、推理叠加；`tests/calibrate.py` 承接校准入口；`tests/` 下新增三个独立功能测试脚本；现有 `tests/test_*.py` 迁移到 `tools/crane_tests/`。

**Tech Stack:** Python 3, OpenCV, Tkinter, PIL/Pillow, PyYAML, pyserial, pytest, 现有 `models/` 运行时模块。

---

## 文件变更总览

### 新增文件

- `tests/calibrate.py` — 校准 UI 入口（从 `main_arm64.py` 迁出）
- `models/camera_viewer.py` — 极简双相机查看器基类
- `tests/test_camera_view.py` — 双相机画面检查
- `tests/test_inference_ui.py` — 双相机推理检查
- `tests/test_serial_loopback.py` — 串口完整收发测试
- `tools/crane_tests/README.md` — 单元测试目录说明

### 迁移文件（从 `tests/` 到 `tools/crane_tests/`）

- `tests/test_postprocess.py`
- `tests/test_task_generator.py`
- `tests/test_serial.py`
- `tests/test_config.py`
- `tests/test_camera.py`
- `tests/test_model.py`
- `tests/test_ui.py`
- `tests/test_send_test_frame.py`

### 修改文件

- `scripts/main_arm64.py` — 删除校准逻辑，仅保留竞赛模式
- `models/ui.py` — `CalibrationUI` 继承/组合 `CameraViewer`
- `docs/runbooks/commands.md` — 更新命令清单
- `CLAUDE.md` — 更新快速入口与说明

### 删除文件

- `tests/test_main.py` — 原 `main_arm64.py` 混合逻辑不再适合单元测试

---

## Task 1: 迁移现有单元测试到 `tools/crane_tests/`

**目标：** 清空 `tests/` 根目录的细碎模块测试，为独立功能测试脚本腾出位置。

**Files:**
- Create: `tools/crane_tests/README.md`
- Move: `tests/test_postprocess.py` → `tools/crane_tests/test_postprocess.py`
- Move: `tests/test_task_generator.py` → `tools/crane_tests/test_task_generator.py`
- Move: `tests/test_serial.py` → `tools/crane_tests/test_serial.py`
- Move: `tests/test_config.py` → `tools/crane_tests/test_config.py`
- Move: `tests/test_camera.py` → `tools/crane_tests/test_camera.py`
- Move: `tests/test_model.py` → `tools/crane_tests/test_model.py`
- Move: `tests/test_ui.py` → `tools/crane_tests/test_ui.py`
- Move: `tests/test_send_test_frame.py` → `tools/crane_tests/test_send_test_frame.py`
- Delete: `tests/test_main.py`

- [ ] **Step 1: 创建 `tools/crane_tests/` 目录**

```bash
mkdir -p tools/crane_tests
```

- [ ] **Step 2: 移动所有模块测试文件**

```bash
git mv tests/test_postprocess.py tools/crane_tests/
git mv tests/test_task_generator.py tools/crane_tests/
git mv tests/test_serial.py tools/crane_tests/
git mv tests/test_config.py tools/crane_tests/
git mv tests/test_camera.py tools/crane_tests/
git mv tests/test_model.py tools/crane_tests/
git mv tests/test_ui.py tools/crane_tests/
git mv tests/test_send_test_frame.py tools/crane_tests/
```

- [ ] **Step 3: 删除 `tests/test_main.py`**

```bash
git rm tests/test_main.py
```

- [ ] **Step 4: 创建 `tools/crane_tests/README.md`**

```markdown
# crane_tests

本目录存放 Crane Vision 的**模块级单元测试**，由 pytest 驱动。

运行方式：

```bash
pytest tools/crane_tests/ -v
```

这些测试覆盖 `models/` 中各模块的独立行为，不依赖真实相机、BPU 模型或串口硬件。

独立功能测试入口请见仓库根目录 `tests/` 下的脚本：

- `tests/test_camera_view.py` — 双相机画面检查
- `tests/test_inference_ui.py` — 双相机推理检查
- `tests/test_serial_loopback.py` — 串口完整收发测试
```

- [ ] **Step 5: 验证单元测试仍可通过**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m pytest tools/crane_tests/ -v
```

Expected: 所有测试通过（当前应全部通过）。

- [ ] **Step 6: 提交**

```bash
git add tools/crane_tests/ tests/test_main.py
cat > .git-msg << 'EOF'
refactor(test): 将模块单元测试迁移到 tools/crane_tests/

- 移动现有 tests/test_*.py 到 tools/crane_tests/
- 删除 tests/test_main.py
- 新增 README.md 说明目录用途
EOF
git commit -F .git-msg
rm .git-msg
```

---

## Task 2: 从 `CalibrationUI` 提取 `CameraViewer` 基类

**目标：** 创建一个只显示双相机画面和辅助线的极简 UI，供 `test_camera_view.py` 使用；`CalibrationUI` 在此基础上扩展滑块、保存和推理叠加。

**Files:**
- Create: `models/camera_viewer.py`
- Modify: `models/ui.py`

- [ ] **Step 1: 创建 `models/camera_viewer.py`**

完整代码：

```python
import logging
import tkinter as tk

import cv2
import yaml
from PIL import Image, ImageTk


DEFAULT_VIEWER_CFG = {
    "window": {"fullscreen": False, "width": 1024, "height": 600},
    "video": {"max_height": 420, "fit_mode": "contain"},
    "controls": {
        "touch_mode": False,
        "font_scale": 0.8,
    },
}

LINE_COLORS = {
    "bean_left_x": (0, 255, 0),
    "bean_right_x": (0, 255, 0),
    "digit_x1": (255, 0, 0),
    "digit_x2": (255, 0, 0),
    "digit_x3": (255, 0, 0),
    "digit_x4": (255, 0, 0),
}


class CameraViewer:
    """Minimal dual-camera viewer with guide lines.

    No sliders, no config saving, no inference. Designed for standalone
    camera placement verification.
    """

    def __init__(self, cfg, cfg_path, cam_bean, cam_digit, root=None):
        self.cfg_path = cfg_path
        self.cfg = cfg
        self.cam_bean = cam_bean
        self.cam_digit = cam_digit
        self.mode = "bean"
        self._window_ok = True
        self._running = False
        self._configure_pending = None

        self._load_cfg()
        self._apply_ui_cfg()
        self._resolve_trackbar_limits()

        if root is None:
            try:
                self.root = tk.Tk()
            except tk.TclError as exc:
                self._window_ok = False
                logging.warning("Tkinter GUI unavailable: %s", exc)
                return
        else:
            self.root = root

        self._build_ui()

    def _load_cfg(self):
        if self.cfg is not None:
            return
        with open(self.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def _apply_ui_cfg(self):
        import copy

        ui_cfg = copy.deepcopy(DEFAULT_VIEWER_CFG)
        if self.cfg and isinstance(self.cfg.get("ui"), dict):
            ui_cfg["window"].update(self.cfg["ui"].get("window", {}))
            ui_cfg["video"].update(self.cfg["ui"].get("video", {}))
            ui_cfg["controls"].update(self.cfg["ui"].get("controls", {}))
        self.ui_cfg = ui_cfg

        win = ui_cfg["window"]
        self.window_width = int(win["width"])
        self.window_height = int(win["height"])
        self.fullscreen = bool(win.get("fullscreen", False))

        vid = ui_cfg["video"]
        self.video_max_height = int(vid["max_height"])
        self.video_fit_mode = str(vid.get("fit_mode", "fit_width"))

        ctl = ui_cfg["controls"]
        self.touch_mode = bool(ctl.get("touch_mode", False))
        self.font_scale = float(ctl.get("font_scale", 0.8))

        self.display_w = self.window_width
        self.display_h = self.video_max_height

    @staticmethod
    def _camera_resolution(cam, default_w=640, default_h=480):
        w = getattr(cam, "width", None)
        h = getattr(cam, "height", None)
        w = int(w) if w is not None and int(w) > 0 else default_w
        h = int(h) if h is not None and int(h) > 0 else default_h
        return w, h

    def _resolve_trackbar_limits(self):
        bean_w, bean_h = self._camera_resolution(self.cam_bean)
        digit_w, digit_h = self._camera_resolution(self.cam_digit)
        self.max_x = max(bean_w, digit_w)
        self.max_y = max(bean_h, digit_h)

    def _compute_display_size(self, native_w, native_h, max_w=None, max_h=None):
        if native_w <= 0 or native_h <= 0:
            return max_w or self.window_width, max_h or self.video_max_height

        max_w = max_w or self.window_width
        max_h = max_h or self.video_max_height
        aspect = native_h / native_w

        display_w = max_w
        display_h = int(round(display_w * aspect))
        if display_h > max_h:
            display_h = max_h
            display_w = int(round(display_h / aspect))

        display_w = max(1, min(display_w, max_w))
        display_h = max(1, min(display_h, max_h))
        return display_w, display_h

    def _resize_for_display(self, frame, max_w=None, max_h=None):
        display_w, display_h = self._compute_display_size(
            frame.shape[1], frame.shape[0], max_w=max_w, max_h=max_h
        )
        if frame.shape[1] == display_w and frame.shape[0] == display_h:
            return frame
        return cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

    def _on_configure(self, _event=None):
        if not self._window_ok or not self._running:
            return
        if self._configure_pending is not None:
            self.root.after_cancel(self._configure_pending)
        self._configure_pending = self.root.after(200, self._apply_geometry)

    def _apply_geometry(self):
        if not self._window_ok:
            return
        self.window_width = max(1, self.root.winfo_width())
        self.window_height = max(1, self.root.winfo_height())

    def _toggle_fullscreen(self, _event=None):
        if not self._window_ok:
            return
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _build_ui(self):
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
        except Exception:
            pass

        self.root.title("Crane Vision Camera Viewer")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.bind("<KeyPress-1>", lambda _event: self._set_mode("bean"))
        self.root.bind("<KeyPress-2>", lambda _event: self._set_mode("digit"))
        self.root.bind("<Escape>", lambda _event: self._quit())
        self.root.bind("<KeyPress-f>", self._toggle_fullscreen)
        self.root.bind("<Configure>", self._on_configure)

        if self.fullscreen:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry(f"{self.window_width}x{self.window_height}")
            self.root.minsize(640, 480)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        video_container = tk.Frame(main_frame, bg="black")
        video_container.grid(row=0, column=0, sticky="nsew")
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_container, bg="black")
        self.video_label.grid(row=0, column=0)
        self._video_container = video_container

        info_frame = tk.Frame(main_frame)
        info_frame.grid(row=1, column=0, sticky="ew")

        font = ("Helvetica", int(12 * self.font_scale))
        self._info_label = tk.Label(
            info_frame,
            text="1=BEAN  2=DIGIT  ESC=exit  f=fullscreen",
            font=font,
        )
        self._info_label.pack(side=tk.LEFT, padx=5, pady=2)

    def _set_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode

    def _draw_bean_lines(self, frame):
        h, w = frame.shape[:2]
        left = self.cfg["lines"]["bean"]["left_x"]
        right = self.cfg["lines"]["bean"]["right_x"]
        cv2.line(frame, (left, 0), (left, h), LINE_COLORS["bean_left_x"], 2)
        cv2.line(frame, (right, 0), (right, h), LINE_COLORS["bean_right_x"], 2)

    def _draw_digit_lines(self, frame):
        h = frame.shape[0]
        for key in ("x1", "x2", "x3", "x4"):
            x = self.cfg["lines"]["digit"][key]
            cv2.line(frame, (x, 0), (x, h), LINE_COLORS[f"digit_{key}"], 2)

    def _update_frame(self):
        if not self._running:
            return

        if self.mode == "bean":
            ok, frame = self.cam_bean.capture()
        else:
            ok, frame = self.cam_digit.capture()

        if ok and frame is not None:
            if self.mode == "bean":
                self._draw_bean_lines(frame)
            else:
                self._draw_digit_lines(frame)

            try:
                avail_w = int(self._video_container.winfo_width())
                avail_h = int(self._video_container.winfo_height())
            except Exception:
                avail_w, avail_h = 0, 0
            if avail_w <= 1 or avail_h <= 1:
                avail_w, avail_h = self.window_width, self.video_max_height

            display_frame = self._resize_for_display(
                frame, max_w=avail_w, max_h=avail_h
            )

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        if self._running:
            self.root.after(30, self._update_frame)

    def _quit(self):
        self._running = False
        self.root.destroy()

    def run(self):
        if not self._window_ok:
            raise RuntimeError(
                "GUI is not available; cannot run camera viewer on this machine"
            )

        print("Camera viewer started. Keys: 1=bean camera, 2=digit camera, ESC=exit")
        self._running = True
        self._update_frame()
        try:
            self.root.mainloop()
        finally:
            self._running = False
            self.cam_bean.release()
            self.cam_digit.release()
```

- [ ] **Step 2: 修改 `models/ui.py` 让 `CalibrationUI` 继承 `CameraViewer`**

保留 `CalibrationUI` 的滑块、保存、推理叠加逻辑，但把窗口创建、画面缩放、辅助线绘制、相机切换等通用逻辑委托给父类。

关键修改点（不是完整文件，只给出需要重构的部分）：

1. 删除 `CalibrationUI` 中已迁移到 `CameraViewer` 的以下方法/属性：
   - `DEFAULT_UI_CFG` 可保留（因为包含 `inference_overlay` 等额外字段）。
   - `_load_cfg`, `_apply_ui_cfg`, `_resolve_trackbar_limits`, `_compute_display_size`, `_resize_for_display`, `_on_configure`, `_apply_geometry`, `_toggle_fullscreen`, `_set_mode`, `_draw_bean_lines`, `_draw_digit_lines`, `_update_frame`, `_camera_resolution`, `_quit`, `run()` 中的通用释放逻辑。

2. 在文件顶部导入：

```python
from models.camera_viewer import CameraViewer, LINE_COLORS
```

3. 修改类定义：

```python
class CalibrationUI(CameraViewer):
```

4. `__init__` 中先调用父类 `__init__`，再创建控制面板：

```python
def __init__(self, cfg_path, cam_bean, cam_digit, bean_model=None, digit_model=None, root=None):
    super().__init__(None, cfg_path, cam_bean, cam_digit, root=root)
    # 如果父类初始化失败（无窗口），直接返回
    if not self._window_ok:
        return

    self.bean_model = bean_model
    self.digit_model = digit_model

    self._infer_queue = queue.Queue(maxsize=1)
    self._result_lock = threading.Lock()
    self._latest_dets = {"bean": None, "digit": None}
    self._latest_names = {"bean": None, "digit": None}
    self._infer_thread = None
    self._save_pending = None

    # 父类已创建 main_frame/video_container/video_label；
    # 这里在 main_frame 底部添加控制面板。
    self._build_controls()
```

5. `_build_ui` 改名为 `_build_controls`，只负责控制面板和滑块：

```python
def _build_controls(self):
    """Add control panel below the video area created by CameraViewer."""
    control_frame = tk.Frame(self._video_container.master)
    control_frame.grid(row=1, column=0, sticky="ew")
    self._control_frame = control_frame

    mode_frame = tk.Frame(control_frame)
    mode_frame.pack(fill=tk.X, pady=(0, 1))

    font = ("Helvetica", int(12 * self.font_scale))

    self._bean_button = tk.Button(
        mode_frame,
        text="BEAN (1)",
        command=lambda: self._set_mode("bean"),
        bg="#00aa00",
        fg="white",
        font=font,
        height=self.button_height // 20,
    )
    self._bean_button.pack(side=tk.LEFT, padx=1)

    self._digit_button = tk.Button(
        mode_frame,
        text="DIGIT (2)",
        command=lambda: self._set_mode("digit"),
        bg="#555555",
        fg="white",
        font=font,
        height=self.button_height // 20,
    )
    self._digit_button.pack(side=tk.LEFT, padx=1)

    self.sliders_frame = tk.Frame(control_frame)
    self.sliders_frame.pack(fill=tk.X)

    self._create_sliders(self.mode)
    self.root.update_idletasks()
```

6. 重写 `_set_mode` 以保留清除 stale detection 和重建滑块的逻辑：

```python
def _set_mode(self, mode):
    if mode == self.mode:
        return
    super()._set_mode(mode)
    with self._result_lock:
        self._latest_dets[mode] = None
        self._latest_names[mode] = None
    self._update_button_appearance()
    self._create_sliders(mode)
```

7. 重写 `_update_frame` 以保留推理叠加：

```python
def _update_frame(self):
    if not self._running:
        return

    if self.mode == "bean":
        ok, frame = self.cam_bean.capture()
        model = self.bean_model
    else:
        ok, frame = self.cam_digit.capture()
        model = self.digit_model

    if ok and frame is not None:
        if self.mode == "bean":
            self._draw_bean_lines(frame)
        else:
            self._draw_digit_lines(frame)

        if model is not None:
            self._submit_frame_for_inference(self.mode, frame)
            with self._result_lock:
                latest_dets = self._latest_dets[self.mode]
                latest_names = self._latest_names[self.mode]
            if latest_dets is not None:
                self._draw_detections(frame, latest_dets, latest_names)

        try:
            avail_w = int(self._video_container.winfo_width())
            avail_h = int(self._video_container.winfo_height())
        except Exception:
            avail_w, avail_h = 0, 0
        if avail_w <= 1 or avail_h <= 1:
            avail_w, avail_h = self.window_width, self.video_max_height

        display_frame = self._resize_for_display(
            frame, max_w=avail_w, max_h=avail_h
        )

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    if self._running:
        self.root.after(30, self._update_frame)
```

8. 保留以下方法不变：
   - `_save_cfg`
   - `_start_inference_worker`, `_stop_inference_worker`, `_inference_worker`, `_submit_frame_for_inference`
   - `_class_name`, `_draw_detections`, `_draw_model_overlay`
   - `_create_sliders`, `_update_button_appearance`, `_set_line`
   - `run()` 中启动/停止推理 worker 的逻辑

- [ ] **Step 3: 运行单元测试确认 UI 行为未退化**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m pytest tools/crane_tests/test_ui.py -v
```

Expected: 全部通过。如果测试 mocks 了 `tk.Tk` 等但类继承改变导致失败，需相应调整 test_ui.py 中的 monkeypatch 目标（例如父类中的 `tk.Tk` 引用）。

- [ ] **Step 4: 提交**

```bash
git add models/camera_viewer.py models/ui.py tools/crane_tests/test_ui.py
cat > .git-msg << 'EOF'
refactor(ui): 提取 CameraViewer 基类并重构 CalibrationUI

- 新增 models/camera_viewer.py：只显示双相机画面 + 辅助线
- CalibrationUI 继承 CameraViewer，保留滑块/保存/推理叠加
- 为后续独立测试脚本提供可复用的 Viewer
EOF
git commit -F .git-msg
rm .git-msg
```

---

## Task 3: 创建 `tests/calibrate.py`

**目标：** 把校准入口从 `main_arm64.py` 迁出，成为独立脚本。

**Files:**
- Create: `tests/calibrate.py`
- Modify: `scripts/main_arm64.py`（在 Task 4 中彻底删除校准逻辑）

- [ ] **Step 1: 创建 `tests/calibrate.py`**

完整代码：

```python
#!/usr/bin/env python3
"""Crane vision calibration UI entry point."""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.camera import StaticImageCamera
from models.runtime import (
    ROOT as RUNTIME_ROOT,
    init_logging,
    list_cameras,
    load_models,
    load_vision_cfg,
    open_camera,
    release_cameras,
)
from models.serial_port import open_serial_port
from models.ui import CalibrationUI


def _open_image_camera(image_path, camera_cfg):
    """Open a StaticImageCamera for offline file-based testing."""
    return StaticImageCamera(
        image_path, camera_cfg["width"], camera_cfg["height"]
    )


def run_calibration(cfg, cfg_path, mock=False, bean_image=None, digit_image=None):
    if bean_image is not None:
        cam_bean = _open_image_camera(bean_image, cfg["cameras"]["bean"])
    else:
        cam_bean = open_camera(
            cfg["cameras"]["bean"], mock=mock, fallback_empty=True
        )

    if digit_image is not None:
        cam_digit = _open_image_camera(digit_image, cfg["cameras"]["digit"])
    else:
        cam_digit = open_camera(
            cfg["cameras"]["digit"], mock=mock, fallback_empty=True
        )

    port = None
    try:
        bean_model = None
        digit_model = None
        try:
            bean_model, digit_model = load_models(cfg)
        except EnvironmentError:
            logging.warning("BPU not available; running UI in camera-only mode")

        serial_cfg = cfg.get("serial")
        if serial_cfg:
            port = open_serial_port(
                serial_cfg["port"],
                serial_cfg["baudrate"],
                timeout=serial_cfg.get("timeout", 0.05),
            )
            logging.info(
                "Serial port opened for calibration: %s @ %s",
                serial_cfg["port"],
                serial_cfg["baudrate"],
            )
        else:
            logging.warning(
                "No serial configuration found; running UI without serial port"
            )

        ui = CalibrationUI(
            cfg_path,
            cam_bean,
            cam_digit,
            bean_model,
            digit_model,
        )
        ui.run()
    finally:
        if port is not None:
            port.close()
        release_cameras(cam_bean, cam_digit)


def main():
    parser = argparse.ArgumentParser(description="Crane vision calibration UI")
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--mock-cameras",
        action="store_true",
        help="Use synthetic cameras for UI testing",
    )
    parser.add_argument(
        "--bean-image",
        help="Path to a static bean camera image (overrides real/mock camera)",
    )
    parser.add_argument(
        "--digit-image",
        help="Path to a static digit camera image (overrides real/mock camera)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras",
    )
    args = parser.parse_args()

    init_logging()

    if args.list_cameras:
        list_cameras()
        return

    cfg = load_vision_cfg(args.config)
    run_calibration(
        cfg,
        args.config,
        mock=args.mock_cameras,
        bean_image=args.bean_image,
        digit_image=args.digit_image,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 `calibrate.py` 语法正确**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m py_compile tests/calibrate.py
```

Expected: 无输出（编译成功）。

- [ ] **Step 3: 提交**

```bash
git add tests/calibrate.py
cat > .git-msg << 'EOF'
feat(scripts): 新增 calibrate.py 作为独立校准入口

- 从 main_arm64.py 迁出 run_calibration 逻辑
- 支持 --mock-cameras、--bean-image、--digit-image、--list-cameras
EOF
git commit -F .git-msg
rm .git-msg
```

---

## Task 4: 精简 `scripts/main_arm64.py` 为纯竞赛入口

**目标：** 删除校准逻辑和冗余参数，只保留竞赛模式。

**Files:**
- Modify: `scripts/main_arm64.py`

- [ ] **Step 1: 重写 `scripts/main_arm64.py`**

完整代码：

```python
#!/usr/bin/env python3
"""Crane vision competition runtime entry point for RDK X5 (ARM64)."""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.runtime import (
    ROOT as RUNTIME_ROOT,
    create_task_generator,
    init_logging,
    load_models,
    load_vision_cfg,
    open_camera,
    release_cameras,
    scan_once_dets,
    warmup_models,
)
from models.serial_port import open_serial_port


def _scan_for_frame(cam_bean, cam_digit, bean_model, digit_model, task_gen, cfg):
    """Run scan_once_dets with retries until a valid 8-byte result frame is produced.

    Returns (frame, reason) where frame is an 8-tuple on success or None.
    """
    deadline = time.monotonic() + cfg["scan"].get("timeout_sec", 30)
    max_attempts = cfg["scan"].get("max_retries", 0) + 1
    for attempt in range(max_attempts):
        if time.monotonic() >= deadline:
            return None, "scan timeout"
        bean_dets, digit_dets, reason = scan_once_dets(
            cam_bean, cam_digit, bean_model, digit_model, cfg
        )
        if reason != "ok":
            logging.warning(
                "Scan attempt %d failed: %s; retrying...",
                attempt + 1,
                reason,
            )
            time.sleep(0.1)
            continue
        frame, reason = task_gen.generate_frame(bean_dets, digit_dets)
        if frame is not None:
            return frame, "ok"
        logging.warning(
            "Scan attempt %d failed: %s; retrying...",
            attempt + 1,
            reason,
        )
        time.sleep(0.1)
    return None, reason


def run_competition(cfg):
    init_logging()
    logging.info("Starting crane vision in competition mode (ARM64)")

    cam_bean = open_camera(cfg["cameras"]["bean"])
    cam_digit = open_camera(cfg["cameras"]["digit"])
    try:
        bean_model, digit_model = load_models(cfg)
        task_gen = create_task_generator(cfg, bean_model, digit_model)
        if task_gen is None:
            raise RuntimeError(
                "Task generator unavailable; competition mode requires BPU models"
            )

        warmup_models(cam_bean, cam_digit, bean_model, digit_model)
        logging.info("Ready. Waiting for START command...")

        port = open_serial_port(
            cfg["serial"]["port"],
            cfg["serial"]["baudrate"],
            timeout=cfg["serial"].get("timeout", 0.05),
        )

        proto = cfg.get("protocol", {})
        start_timeout = proto.get("start_timeout_sec", 0.3)
        send_interval = proto.get("send_interval_sec", 0.1)
        poll_interval = proto.get("poll_interval_sec", 0.01)

        cached_frame = None
        last_send_time = 0.0
        last_scan_attempt = 0.0
        scan_retry_interval = proto.get("scan_retry_interval_sec", 0.5)

        try:
            while True:
                try:
                    port.poll_start()
                    active = port.is_start_active(start_timeout)

                    if not active:
                        cached_frame = None
                        time.sleep(poll_interval)
                        continue

                    if cached_frame is None:
                        now = time.monotonic()
                        if now - last_scan_attempt >= scan_retry_interval:
                            last_scan_attempt = now
                            frame, reason = _scan_for_frame(
                                cam_bean,
                                cam_digit,
                                bean_model,
                                digit_model,
                                task_gen,
                                cfg,
                            )
                            if frame is None:
                                logging.error("Failed to generate result: %s", reason)
                            else:
                                cached_frame = frame
                                last_send_time = 0.0
                                logging.info(
                                    "Result ready: C1-C3=%s, N4-N8=%s",
                                    cached_frame[:3],
                                    cached_frame[3:],
                                )
                        time.sleep(poll_interval)
                        continue

                    if time.monotonic() - last_send_time >= send_interval:
                        port.send_result(*cached_frame)
                        logging.info(
                            "Sent result frame: %s",
                            " ".join(f"{b:02X}" for b in cached_frame),
                        )
                        last_send_time = time.monotonic()

                    time.sleep(poll_interval)
                except Exception:
                    logging.exception("Error during competition cycle")
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            logging.info("Shutdown requested")
        finally:
            port.close()
    finally:
        release_cameras(cam_bean, cam_digit)


def main():
    parser = argparse.ArgumentParser(
        description="Crane vision competition runtime for RDK X5"
    )
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    init_logging(debug=args.verbose)
    cfg = load_vision_cfg(args.config)
    run_competition(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 `main_arm64.py` 语法正确**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m py_compile scripts/main_arm64.py
```

Expected: 无输出。

- [ ] **Step 3: 运行单元测试确认无意外破坏**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m pytest tools/crane_tests/ -v
```

Expected: 全部通过。

- [ ] **Step 4: 提交**

```bash
git add scripts/main_arm64.py
cat > .git-msg << 'EOF'
refactor(scripts): main_arm64.py 仅保留竞赛模式

- 删除校准 UI 逻辑与 --ui/--no-ui/--mock-cameras 等参数
- 删除 --list-cameras、--bean-image、--digit-image
- 仅保留 --config 与 --verbose
EOF
git commit -F .git-msg
rm .git-msg
```

---

## Task 5: 创建 `tests/test_camera_view.py`

**目标：** 独立的双相机画面检查脚本，无模型、无串口、无滑块。

**Files:**
- Create: `tests/test_camera_view.py`

- [ ] **Step 1: 创建 `tests/test_camera_view.py`**

完整代码：

```python
#!/usr/bin/env python3
"""Standalone dual-camera viewer for placement verification.

No model inference, no serial port, no sliders. Press 1/2 to switch cameras.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.camera import StaticImageCamera
from models.camera_viewer import CameraViewer
from models.runtime import (
    ROOT as RUNTIME_ROOT,
    init_logging,
    load_vision_cfg,
    open_camera,
)


def _open_image_camera(image_path, camera_cfg):
    return StaticImageCamera(
        image_path, camera_cfg["width"], camera_cfg["height"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="View bean and digit cameras for placement check"
    )
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--mock-cameras",
        action="store_true",
        help="Use synthetic cameras",
    )
    parser.add_argument(
        "--bean-image",
        help="Path to a static bean camera image",
    )
    parser.add_argument(
        "--digit-image",
        help="Path to a static digit camera image",
    )
    args = parser.parse_args()

    init_logging()
    cfg = load_vision_cfg(args.config)

    if args.bean_image is not None:
        cam_bean = _open_image_camera(args.bean_image, cfg["cameras"]["bean"])
    else:
        cam_bean = open_camera(
            cfg["cameras"]["bean"], mock=args.mock_cameras, fallback_empty=True
        )

    if args.digit_image is not None:
        cam_digit = _open_image_camera(args.digit_image, cfg["cameras"]["digit"])
    else:
        cam_digit = open_camera(
            cfg["cameras"]["digit"], mock=args.mock_cameras, fallback_empty=True
        )

    viewer = CameraViewer(cfg, args.config, cam_bean, cam_digit)
    try:
        viewer.run()
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        cam_bean.release()
        cam_digit.release()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m py_compile tests/test_camera_view.py
```

Expected: 无输出。

- [ ] **Step 3: 提交**

```bash
git add tests/test_camera_view.py
cat > .git-msg << 'EOF'
feat(test): 新增双相机画面检查脚本

- tests/test_camera_view.py：只显示画面和辅助线，无模型/串口/滑块
- 支持 --mock-cameras 和静态图片
EOF
git commit -F .git-msg
rm .git-msg
```

---

## Task 6: 创建 `tests/test_inference_ui.py`

**目标：** 独立的双相机推理检查脚本，加载 BPU 模型，不打开串口。

**Files:**
- Create: `tests/test_inference_ui.py`

- [ ] **Step 1: 创建 `tests/test_inference_ui.py`**

完整代码：

```python
#!/usr/bin/env python3
"""Standalone dual-camera inference viewer.

Loads BPU models and overlays detections / guide lines. No serial port.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.camera import StaticImageCamera
from models.runtime import (
    ROOT as RUNTIME_ROOT,
    init_logging,
    load_models,
    load_vision_cfg,
    open_camera,
)
from models.ui import CalibrationUI


def _open_image_camera(image_path, camera_cfg):
    return StaticImageCamera(
        image_path, camera_cfg["width"], camera_cfg["height"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on bean and digit cameras"
    )
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--mock-cameras",
        action="store_true",
        help="Use synthetic cameras",
    )
    parser.add_argument(
        "--bean-image",
        help="Path to a static bean camera image",
    )
    parser.add_argument(
        "--digit-image",
        help="Path to a static digit camera image",
    )
    args = parser.parse_args()

    init_logging()
    cfg = load_vision_cfg(args.config)

    if args.bean_image is not None:
        cam_bean = _open_image_camera(args.bean_image, cfg["cameras"]["bean"])
    else:
        cam_bean = open_camera(
            cfg["cameras"]["bean"], mock=args.mock_cameras, fallback_empty=True
        )

    if args.digit_image is not None:
        cam_digit = _open_image_camera(args.digit_image, cfg["cameras"]["digit"])
    else:
        cam_digit = open_camera(
            cfg["cameras"]["digit"], mock=args.mock_cameras, fallback_empty=True
        )

    try:
        bean_model, digit_model = load_models(cfg)
    except EnvironmentError:
        logging.warning("BPU not available; running inference UI in camera-only mode")
        bean_model = None
        digit_model = None

    ui = CalibrationUI(
        args.config,
        cam_bean,
        cam_digit,
        bean_model,
        digit_model,
    )

    try:
        ui.run()
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    finally:
        cam_bean.release()
        cam_digit.release()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m py_compile tests/test_inference_ui.py
```

Expected: 无输出。

- [ ] **Step 3: 提交**

```bash
git add tests/test_inference_ui.py
cat > .git-msg << 'EOF'
feat(test): 新增双相机推理检查脚本

- tests/test_inference_ui.py：加载 BPU 模型并叠加检测框，不打开串口
- 支持 --mock-cameras 和静态图片
EOF
git commit -F .git-msg
rm .git-msg
```

---

## Task 7: 创建 `tests/test_serial_loopback.py`

**目标：** 独立的串口完整收发测试，支持回环模式和协议模式。

**Files:**
- Create: `tests/test_serial_loopback.py`

- [ ] **Step 1: 创建 `tests/test_serial_loopback.py`**

完整代码：

```python
#!/usr/bin/env python3
"""Serial port loopback / protocol test for RDK X5.

Modes:
  loopback  -- TX and RX are shorted; send bytes and verify echo.
  protocol  -- Use SerialPort to send a result frame and wait for START.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import serial

from models.runtime import ROOT as RUNTIME_ROOT, init_logging, load_vision_cfg
from models.serial_port import START_COMMAND, SerialPort, open_serial_port


def _load_serial_cfg(config_path):
    cfg = load_vision_cfg(config_path)
    if "serial" not in cfg:
        raise RuntimeError(f"No 'serial' section in {config_path}")
    return cfg["serial"]


def _run_loopback(ser, count, interval):
    """Send random payloads and expect echo on a TX-RX shorted port."""
    import random

    errors = 0
    for i in range(count):
        payload = bytes(random.randint(0, 255) for _ in range(12))
        ser.write(payload)
        ser.flush()
        time.sleep(0.05)
        echo = ser.read(len(payload))
        if echo != payload:
            errors += 1
            logging.error(
                "Loopback mismatch at frame %d: sent=%s echo=%s",
                i + 1,
                payload.hex(),
                echo.hex(),
            )
        else:
            logging.info("Loopback frame %d OK", i + 1)
        if i + 1 < count:
            time.sleep(interval)
    return errors


def _run_protocol(port, count, interval):
    """Send a valid result frame and wait for START command in response."""
    errors = 0
    for i in range(count):
        try:
            port.send_result(1, 2, 3, 3, 1, 5, 2, 4)
            logging.info("Protocol frame %d sent", i + 1)
        except Exception as exc:
            errors += 1
            logging.error("Failed to send frame %d: %s", i + 1, exc)
            continue

        if port.wait_start(timeout=2.0):
            logging.info("Protocol frame %d: START received", i + 1)
        else:
            errors += 1
            logging.error("Protocol frame %d: START timeout", i + 1)

        if i + 1 < count:
            time.sleep(interval)
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Serial port loopback / protocol test"
    )
    parser.add_argument(
        "--config",
        default=str(RUNTIME_ROOT / "configs" / "vision.yaml"),
        help="Path to vision.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["loopback", "protocol"],
        required=True,
        help="loopback: TX-RX shorted; protocol: send result and wait START",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of frames to send",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between frames",
    )
    args = parser.parse_args()

    init_logging()
    serial_cfg = _load_serial_cfg(args.config)

    logging.info(
        "Opening serial port %s @ %s for %s test",
        serial_cfg["port"],
        serial_cfg["baudrate"],
        args.mode,
    )

    if args.mode == "loopback":
        ser = serial.Serial(
            serial_cfg["port"],
            serial_cfg["baudrate"],
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
        )
        try:
            # Drain any stale data.
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            errors = _run_loopback(ser, args.count, args.interval)
        finally:
            ser.close()
    else:
        port = open_serial_port(
            serial_cfg["port"],
            serial_cfg["baudrate"],
            timeout=serial_cfg.get("timeout", 0.05),
        )
        try:
            errors = _run_protocol(port, args.count, args.interval)
        finally:
            port.close()

    if errors:
        logging.error("Test finished with %d error(s)", errors)
        sys.exit(1)
    logging.info("Test finished successfully")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m py_compile tests/test_serial_loopback.py
```

Expected: 无输出。

- [ ] **Step 3: 提交**

```bash
git add tests/test_serial_loopback.py
cat > .git-msg << 'EOF'
feat(test): 新增串口完整收发测试脚本

- tests/test_serial_loopback.py：支持 loopback 回环测试与 protocol 协议测试
- 读取 configs/vision.yaml 的 serial 配置
EOF
git commit -F .git-msg
rm .git-msg
```

---

## Task 8: 更新文档

**目标：** 同步更新 `docs/runbooks/commands.md` 和 `CLAUDE.md` 中的命令和说明。

**Files:**
- Modify: `docs/runbooks/commands.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: 更新 `docs/runbooks/commands.md`**

需要修改的内容（具体行号以实际文件为准）：

1. 在“单元测试”小节更新命令：

```markdown
# 单元测试
conda run --prefix /d/conda_envs/crane-clean python -m pytest tools/crane_tests/ -v
```

2. 在“RDK X5 部署与测试”小节新增/更新：

```markdown
# 竞赛模式（RDK X5）
python3 scripts/main_arm64.py

# 校准辅助线
python3 tests/calibrate.py

# 独立功能测试
python3 tests/test_camera_view.py
python3 tests/test_inference_ui.py
python3 tests/test_serial_loopback.py --mode loopback --count 100
python3 tests/test_serial_loopback.py --mode protocol --count 10
```

3. 删除或更新文档中所有 `main_arm64.py --ui`、`--no-ui`、`--mock-cameras` 的引用。

- [ ] **Step 2: 更新 `CLAUDE.md`**

1. 在“快速入口”段落替换为：

```markdown
## 常用命令

# 单元测试
conda run --prefix /d/conda_envs/crane-clean python -m pytest tools/crane_tests/ -v

# 校准辅助线
conda run --prefix /d/conda_envs/crane-clean python tests/calibrate.py --mock-cameras

# 竞赛模式（仅限 RDK X5）
python3 scripts/main_arm64.py
```

2. 在“视觉运行时映射规则”段落，把 `python scripts/main_arm64.py` 改为 `python tests/calibrate.py`。

3. 在“重要注意事项”段落，更新：
   - `main_arm64.py` 仅用于竞赛模式；校准请用 `tests/calibrate.py`。
   - 测试脚本说明。

- [ ] **Step 3: 验证文档无语法错误**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m py_compile scripts/main_arm64.py tests/calibrate.py tests/test_camera_view.py tests/test_inference_ui.py tests/test_serial_loopback.py
```

Expected: 无输出。

- [ ] **Step 4: 提交**

```bash
git add docs/runbooks/commands.md CLAUDE.md
cat > .git-msg << 'EOF'
docs: 更新命令与入口说明

- 单元测试命令改为 pytest tools/crane_tests/
- 新增 calibrate.py 和三个独立测试脚本的命令
- 明确 main_arm64.py 仅用于竞赛模式
EOF
git commit -F .git-msg
rm .git-msg
```

---

## Task 9: 最终验证

**目标：** 确保所有改动协同工作，单元测试全部通过，新脚本语法正确。

- [ ] **Step 1: 运行全部单元测试**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m pytest tools/crane_tests/ -v
```

Expected: 全部通过。

- [ ] **Step 2: 验证所有入口脚本语法**

```bash
conda run --prefix /d/conda_envs/crane-clean python -m py_compile scripts/main_arm64.py tests/calibrate.py tests/test_camera_view.py tests/test_inference_ui.py tests/test_serial_loopback.py
```

Expected: 无输出。

- [ ] **Step 3: 检查 `tests/` 目录结构**

```bash
ls tests/
```

Expected: 只包含 `test_camera_view.py`、`test_inference_ui.py`、`test_serial_loopback.py`（以及可能的 `__init__.py`、`.gitkeep`）。

- [ ] **Step 4: 检查 `scripts/` 目录结构**

```bash
ls scripts/
```

Expected: 包含 `main_arm64.py`、`main_amd64.py`、`watchdog.sh`。

- [ ] **Step 5: 最终提交（如无问题则无需额外提交）**

如果所有验证通过，无需额外提交。若发现小问题，就地修复后单独提交。

---

## 计划自检

### Spec 覆盖检查

| Spec 章节 | 对应 Task |
|-----------|-----------|
| 2. 目录结构 | Task 1, 5, 6, 7 |
| 3. `main_arm64.py` 精简 | Task 4 |
| 4. `calibrate.py` | Task 3 |
| 5. `CameraViewer` | Task 2 |
| 6. `CalibrationUI` 改造 | Task 2 |
| 7. `tests/` 功能脚本 | Task 5, 6, 7 |
| 8. 单元测试迁移 | Task 1 |
| 9. 文档更新 | Task 8 |
| 10. 数据流与依赖 | Task 2, 3, 4, 5, 6, 7 |
| 11. 错误处理 | 各 Task 中已体现 |
| 12. 验收标准 | Task 9 |

### Placeholder 检查

- 无 TBD/TODO。
- 所有步骤包含具体代码或命令。
- 所有文件路径明确。

### 类型一致性检查

- `CameraViewer.__init__` 签名保持一致。
- `CalibrationUI` 继承后方法签名不变。
- 各测试脚本使用统一的 `load_vision_cfg`、`open_camera` helper。

---

## 执行方式选择

Plan complete and saved to `docs/superpowers/plans/2026-07-17-crane-vision-test-refactor-plan.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
