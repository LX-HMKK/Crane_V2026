# 相机/串口掉线重连 + UI 状态显示 设计文档

日期：2026-08-03
状态：已批准（用户确认全部决策）

## 背景与问题

RDK X5 竞赛运行时有双 USB 相机（bean/digit）+ 串口。现状：

1. **相机掉线无重连**：`models/camera.py` 的 `Camera.capture()` 在 `cap.read()` 失败后只返回 `(False, None)`，`CompetitionUI._capture_worker` 等调用方背退重试但**从不重建 `cv2.VideoCapture` 句柄**——底层设备指针已死，永远读不出帧。
2. **串口掉线无重连**：`models/serial_port.py` 的 `SerialThread._run` 在异常时 `except` 捕获 + 背退 1s，但**从不 close/reopen pyserial 句柄**。
3. **任务合成层"读旧有效帧"冗余（bug）**：`CompetitionUI._update_task_from_latest_detections` 只要求两路 `_latest_dets` 非 None。相机 A 掉线后，相机 B 的持续推理不断触发合成，把 A 的**最后一次有效检测**与 B 的新检测合并 → 持续生成含死相机旧数据的帧 → 发给下位机。且因 generate_frame 一直"成功"，`_consecutive_failures` 永远到不了 10，`serial.current_frame` 不会被清。
4. **UI 无掉线显示**：掉线相机无任何状态提示。

## 决策总览

| 决策 | 选择 |
|---|---|
| 相机重连位置 | **`Camera` 类内部自我修复**（所有调用方零改动受益） |
| 串口重连 | **一并实现**（`SerialThread` 内部） |
| 相机掉线帧显示 | **黑屏占位**「CAMERA OFFLINE」**不回退**另一台 |
| 相机掉线后串口 | **sticky：继续发最后有效帧，不清空** |
| 旧帧冗余修复 | **确认掉线后失效检测**（`_latest_dets[mode]` 置 None），串口 sticky 不受影响 |
| UI 状态显示 | 状态栏 BEAN/DIGIT/SERIAL(+START)；校准/查看器掉线占位 |

## 1. 相机重连 — `models/camera.py`

### 状态字段（`__init__` 新增）

- 重建参数：`_req_width`/`_req_height`、`_auto_exposure_requested`、`_exposure_requested`、`buffer_size`、`target_fps`、`grayscale`（后两者已有）。
- 重连参数（新关键字参数，默认值向后兼容）：
  - `reconnect_enabled=True`
  - `reconnect_after=3`（连续失败多少次后开始尝试重连）
  - `reconnect_interval=1.0`（重连尝试最小间隔秒，限速）
- 重连状态：`_consecutive_failures=0`、`_last_reopen_at=0.0`、`_opened=True`、`self.cap=None`（初值）。

### 方法

- **`_configure_cap()`**：从 `__init__` 抽出。对当前 `self.cap` 重放 FourCC=MJPG、`_req_width`/`_req_height`、FPS、BUFFERSIZE，并刷新 `self.width/height/fps`（驱动实际值）、`set_auto_exposure(_auto_exposure_requested)`、手动曝光时 `set_exposure(_exposure_requested)`。
- **`capture()`**：
  - 持 `_io_lock`：`self.cap` 为 None → 跳过读帧，失败计数 +1，走 `_maybe_reopen()`；否则 `cap.read()`。
  - 读到有效帧 → `_consecutive_failures=0`，`_opened=True`。
  - 失败 → `_consecutive_failures+=1` → `_maybe_reopen()`。
  - 返回后，仍失败 → 直接 `return (False, None)`；成功 → 原有灰度/BGR 后处理不变。
- **`_maybe_reopen()`**（在 `_io_lock` 内）：`reconnect_enabled=False` 或 `_consecutive_failures < reconnect_after` → return；距 `_last_reopen_at` 不足 `reconnect_interval` → return；否则 `_reopen()` 并更新 `_last_reopen_at`。
- **`_reopen()`**：release 旧句柄（容错）→ `cv2.VideoCapture(device_index)` → `isOpened()` 失败则记 WARNING、`self.cap=None`、`_opened=False`、return False；成功则 `_configure_cap()`、`_consecutive_failures=0`、`_opened=True`、记 INFO「camera reconnected」、return True。
- **防御**：`release()`/`set_auto_exposure()`/`set_exposure()` 在 `self.cap is None` 时不再崩溃（跳过/返回 False + WARNING）。

### 线程安全

重连全程持 `_io_lock`（RLock，`_configure_cap` 内 `set_*` 可重入）。读帧与重连串行，UI 设置滑块不会操作被替换的旧句柄。

### 影响

`MockCamera`/`StaticImageCamera` 永不失败、无 `cap`，不参与重连，现有测试不受影响。所有调用方（`CompetitionUI._capture_worker`、`HeadlessCompetitionLoop`、`CameraViewer/CalibrationUI._update_frame`、`warmup_models`）零改动获得重连。

## 2. 串口重连 — `models/serial_port.py`

### 现状（审查结论）

`SerialThread._run` 的 `except` 只 `logging.exception` + `time.sleep(1.0)`，pyserial 句柄损坏后永远复用死句柄。**无重连。**

### 改动

- `__init__` 保留 `_device`/`_baudrate`/`_timeout`；新增 `reconnect_enabled=True`、`reconnect_after=5`、`reconnect_interval=1.0`，状态 `_consecutive_failures=0`、`_last_reopen_at=0.0`。
- 新增 `connected` property：`return self._port is not None`（供 UI 状态栏）。
- `_run`：
  - 正常一轮迭代（try 内无异常）→ `_consecutive_failures=0`。
  - `except` 路径 → `_consecutive_failures+=1` → `_maybe_reopen()` → `logging.exception` → `time.sleep(1.0)`（保留现有背退）。
- `_maybe_reopen()`：`reconnect_enabled=False` 或 `< reconnect_after` → return；距 `_last_reopen_at` 不足 `reconnect_interval` → return；否则 `_reopen()`。
- `_reopen()`：close 旧 `SerialPort`（容错）→ `open_serial_port(_device, _baudrate, timeout=_timeout)` → 成功记 INFO「serial port reconnected」、`_consecutive_failures=0`；失败记 WARNING、`self._port=None`。
- 防御：`stop()` 对 `self._port is None` 跳过 close；`is_start_active` property 对 None 返回 False。

### 注释

`_run` 正常路径用 `getattr` 或类级默认值兜底 `_consecutive_failures` 等，避免破坏现有用 `SerialThread.__new__` 构造的测试（`test_serial_thread_does_not_log_each_start_heartbeat`）。

## 3. 旧帧冗余修复 + sticky — `models/ui.py`（CompetitionUI）

### 常量

- `CAPTURE_OFFLINE_AFTER_FAILURES = 5`（约 250ms @ 0.05s 背退；瞬时丢帧容忍阈值）。

### `__init__` 新增

- `self._capture_failures = {"bean": 0, "digit": 0}`。

### `_capture_worker` 失败分支

- 持 `_capture_lock`：`_capture_ready[mode]=False`、`_capture_failure_logged[mode]=True`、`_capture_failures[mode]+=1`，算 `invalidate = _capture_failures[mode] >= CAPTURE_OFFLINE_AFTER_FAILURES`。
- **退出 `_capture_lock` 后**，`if invalidate`：持 `_result_lock` 清 `_latest_dets[mode]=None`、`_latest_names[mode]=None`，记一次 WARNING「camera offline; detections invalidated」。
- 成功分支：`_capture_failures[mode]=0`。

### `_update_task_from_latest_detections`

无需改动——`_latest_dets[mode]` 被清为 None 后，`bean_dets is None or digit_dets is None` 提前 return，`_consecutive_failures` 不递增 → **`serial.current_frame` 不被清 → sticky 保持**。

### 锁安全

worker 在 `_capture_lock` 内只算 `invalidate` 布尔，退出后再取 `_result_lock`。两把锁不同时持有，与 `_update_task_from_latest_detections`（`_task_lock`→`_result_lock`）、推理 worker（`_result_lock`）无 AB-BA 死锁。

## 4. UI 状态显示 — `models/ui.py` + `models/camera_viewer.py`

### 基类 helper（`camera_viewer.py`，供三层 `_update_frame` 复用）

- `import numpy as np`。
- `_show_frame(frame)`：BGR→RGB→PhotoImage→label 显示（提取现有内联逻辑）。
- `_render_placeholder(text, width=None, height=None)`：黑底、居中红字 `cv2.putText`，默认 `window_width`×`video_max_height`。
- `CameraViewer._update_frame`：采集失败 → `_show_frame(self._render_placeholder(f"CAMERA OFFLINE: {self.mode.upper()}", avail_w, avail_h))`。

### `CalibrationUI._update_frame`

无 capture worker / 无 SerialThread。采集失败 → 同上占位（Tk 主线程同步判断，无防抖）。**不加状态栏。**

### `CompetitionUI`

- **状态栏**：`_build_status_bar()`（在 `_control_frame` 底部 pack 一行 Label：BEAN / DIGIT / SERIAL / START）+ `_update_status()`。
  - 相机状态：`_capture_failures[mode] >= CAPTURE_OFFLINE_AFTER_FAILURES` → OFFLINE（红），否则 OK（绿）。
  - 串口状态：`self.serial.connected`（`_port is not None`）→ OK/OFFLINE；`self.serial is None` → N/A（灰）。
  - START：`self.serial.is_start_active` → ON/OFF（灰绿）。
  - 仅状态变化时重配 Label（存 `_status_states`，防 30ms 无谓重绘）。
  - `_update_status()` 在 `_update_frame` 顶部调用，settings 模式也刷新。
- **`_camera_state(mode)` helper**：返回 `("ok", frame)` / `("grace", last_frame)` / `("offline", None)`：
  - ready 且有帧 → ok。
  - 未 ready 但 `_capture_failures[mode] < CAPTURE_OFFLINE_AFTER_FAILURES` 且有最后帧 → grace（显示最后帧，防抖）。
  - 其余 → offline。
- **`_update_frame` 重写**：
  - 顶部 `_update_status()`。
  - settings 模式：reschedule，return。
  - `state, frame = _camera_state(self.mode)`（mode 非 settings 时为 bean/digit）。
  - offline → `_show_frame(placeholder)`；ok/grace → 画辅助线 + 检测（grace 时检测仍有效）+ 推断标签，resize，`_show_frame`。
  - **移除原回退另一台相机的逻辑**（行为变更）。
- 状态栏在 settings 布局下仍可见（控制面板随 `_set_settings_layout` 移动）。

## 5. 配置变更

### `configs/vision.yaml`

```yaml
cameras:
  bean:
    ...
    reconnect:
      enabled: true
      after_failures: 3
      interval_sec: 1.0
  digit:
    ...
    reconnect:
      enabled: true
      after_failures: 3
      interval_sec: 1.0
serial:
  ...
  reconnect:
    enabled: true
    after_failures: 5
    interval_sec: 1.0
```

### `configs/vision_1024x600_test.yaml`

同步加相机 `reconnect:` 子块（数字设备号 `0/2`，`cv2.VideoCapture(n)` 重开同样有效）。

### 透传

- `models/runtime.py` `open_camera()`：读 `camera_cfg.get("reconnect", {})` → `reconnect_enabled/after/interval`。
- `scripts/main_arm64.py` `_create_serial_thread()`：读 `cfg["serial"].get("reconnect", {})` → 透传。

## 6. 错误处理与日志

- 相机：瞬时失败（< after）不动作；`_reopen` 成功 INFO / 失败 WARNING，失败被 `interval_sec` 限速。worker 掉线失效记一次 WARNING。
- 串口：`logging.exception` 保留（1s 背退）；`_reopen` 成功 INFO / 失败 WARNING（限速）。
- 掉线期间串口 sticky：结果不变继续发最后有效帧；**相机掉线不清空**（用户决策）。

## 7. 测试计划（`tools/crane_tests/`，全 mock）

- `test_camera.py`：
  - 连续失败 N 次后重开（`VideoCapture` 工厂第二次返回好帧 → 最终捕获成功）。
  - 瞬时失败（< after）不重开（工厂只调一次）。
  - `reconnect_enabled=False` 不重开。
  - 重开后 `_configure_cap` 的 `set()` 被重放（FourCC/尺寸/曝光）。
  - `cap is None` 时 `release()`/`set_*()` 不抛。
  - 现有 6 个测试保持通过。
- `test_serial.py`：
  - `open_serial_port` 连续抛错 N 次后再次被调用（重开）。
  - `reconnect_interval` 限速：间隔内不重开。
  - `reconnect_enabled=False` 不重开。
  - `connected` property：`_port` None 时 False。
  - 现有测试保持通过（`__new__` 构造需类级默认值兜底）。
- `test_ui.py`：
  - `_update_frame`：相机 offline → 占位（不再显示另一台）；ok → 正常帧；grace → 显示最后帧。
  - `_update_status`：相机/串口 OFFLINE 状态落到 Label。
  - `_capture_worker` 连续失败 N 次后 `_latest_dets[mode]` 被清。
  - 现有测试保持通过。
- `test_main_arm64.py`：`_create_serial_thread` 透传 reconnect 配置；现有测试保持通过。

## 8. 行为变更清单

1. 相机掉线：`capture()` 内部自动重连（新）。
2. 串口掉线：`SerialThread` 自动重连（新）。
3. 相机掉线帧显示：黑屏占位，**不再回退另一台相机**（变更）。
4. 相机确认掉线后：检测失效 → 不再合成含死相机旧数据的帧（修复冗余）；串口 sticky 保持（不变）。
5. UI 新增状态栏 BEAN/DIGIT/SERIAL/START（新）。
6. 校准/查看器：掉线显示占位（新）。

## 9. 开放问题 / 后续

- `CAPTURE_OFFLINE_AFTER_FAILURES` 与 reconnect 参数目前是模块常量/配置字段；如需 UI 可调可后续加。
- 校准 UI 串口仅用于协议测试（raw SerialPort，非 SerialThread），不显示串口状态；如需可在后续加。
