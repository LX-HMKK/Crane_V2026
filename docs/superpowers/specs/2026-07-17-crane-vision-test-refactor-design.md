# Crane Vision 测试与入口解耦设计

**日期：** 2026-07-17  
**状态：** 待实现  
**关联需求：** 用户反馈：`main_arm64.py` 功能堆砌，`test/` 被滥用为模块测试目录，需要独立的完整功能测试入口。

## 1. 背景与问题

当前代码存在以下问题：

- `scripts/main_arm64.py` 同时承担**竞赛模式**和**校准 UI** 两种职责，参数列表臃肿。
- `test/` 目录下存放大量细碎模块单元测试，导致真正的独立功能测试入口没有位置。
- 实际部署中需要单独验证：串口完整收发、双相机画面、双相机推理，但这些能力都淹没在 `main_arm64.py` 里，无法独立运行。

本设计目标：

1. `main_arm64.py` 成为**极简的完整竞赛入口**。
2. 校准、串口测试、相机测试、推理测试各自成为**独立可运行脚本**。
3. 细碎模块单元测试迁移到 `tools/crane_tests/`，`test/` 只保留功能级测试入口。

## 2. 目录结构

```text
scripts/
  main_arm64.py              # 只保留竞赛模式
  calibrate.py               # 校准 UI 入口（从 main_arm64.py 迁出）
  main_amd64.py              # x86 测试入口（保持现状，按需精简）
  send_test_frame.py         # 现有，不动

test/
  test_serial_loopback.py    # 串口完整收发测试（RDK X5 真实硬件/回环）
  test_camera_view.py        # 双相机画面检查 UI（无模型、无串口）
  test_inference_ui.py       # 双相机推理检查 UI（加载 BPU 模型、无串口）

tools/crane_tests/           # 迁移现有细碎模块测试
  test_postprocess.py
  test_task_generator.py
  test_serial.py             # 保留单元测试部分
  test_config.py
  test_camera.py
  test_model.py
  test_ui.py
  test_send_test_frame.py
  README.md

models/
  camera_viewer.py           # 新增：极简相机查看器基类
  ui.py                      # CalibrationUI 继承/组合 CameraViewer
  runtime.py                 # 共享加载、相机打开、模型加载等 helper
```

## 3. `scripts/main_arm64.py` 设计

### 3.1 职责

唯一职责：在 RDK X5 上运行完整竞赛循环。

### 3.2 参数

```bash
python scripts/main_arm64.py [--config PATH] [-v]
```

- 删除 `--ui`、`--no-ui`、`--mock-cameras`、`--bean-image`、`--digit-image`、`--list-cameras`。
- `--list-cameras` 可临时移到 `scripts/calibrate.py`，后续再决定是否单独拆出 `scripts/list_cameras.py`。

### 3.3 主流程

```python
def main():
    args = parse_args()
    init_logging(debug=args.verbose)
    cfg = load_vision_cfg(args.config)
    run_competition(cfg)
```

`run_competition()` 逻辑基本不变，但不再与校准代码共存于同一文件。

## 4. `scripts/calibrate.py` 设计

### 4.1 职责

承接原有校准 UI 入口。

### 4.2 参数

```bash
python scripts/calibrate.py [--config PATH] [--mock-cameras]
                            [--bean-image PATH] [--digit-image PATH]
                            [--list-cameras]
```

### 4.3 行为

- 读取 `configs/vision.yaml`。
- 打开真实相机或 mock/静态图片相机。
- 尝试加载 BPU 模型；在 x86 上加载失败则进入 camera-only 模式。
- 启动 `CalibrationUI`。
- 退出时释放相机、关闭串口。

## 5. `models/camera_viewer.py` 设计

### 5.1 职责

提供**只显示双相机画面 + 辅助线**的极简 UI 基类，无滑块、无保存、无推理。

### 5.2 关键能力

- 打开 Tkinter 窗口（可配置全屏/窗口）。
- 按 `1` 切换 bean 相机，按 `2` 切换 digit 相机。
- 绘制 `configs/vision.yaml` 中定义的辅助线。
- 按 `ESC` 退出并释放相机。

### 5.3 接口示例

```python
class CameraViewer:
    def __init__(self, cfg, cfg_path, cam_bean, cam_digit, root=None):
        ...

    def run(self):
        ...
```

## 6. `models/ui.py` 中 `CalibrationUI` 改造

### 6.1 方案

`CalibrationUI` 继承（或内部组合）`CameraViewer`，并额外提供：

- 滑块控制面板（按 `1/2` 切换时重新创建对应滑块）。
- 辅助线改动后自动保存回 `configs/vision.yaml`（debounce 300ms）。
- 可选的推理叠加（由 `ui.controls.inference_overlay` 控制）。

### 6.2 替代方案

不拆基类，直接在 `CalibrationUI` 上加开关参数（`enable_sliders`、`enable_save`、`enable_inference`）。该方案实现更快，但长期会导致 `CalibrationUI` 积累过多模式开关，再次臃肿。**本设计采用继承/组合方案。**

## 7. `test/` 功能测试脚本设计

### 7.1 `test/test_serial_loopback.py`

#### 用途

在 RDK X5 上验证串口硬件和协议层收发。

#### 参数

```bash
python test/test_serial_loopback.py [--config PATH] --mode {loopback,protocol}
                                    [--count N] [--interval SEC]
```

#### 行为

- 读取 `configs/vision.yaml` 的 `serial` 段。
- `--mode loopback`：用 `serial.Serial` 直接打开端口，发送随机 12 字节并读回校验；同时测试 `START_COMMAND` 的发送与接收。
- `--mode protocol`：通过 `SerialPort.send_result()` 发送结果帧，再用 `poll_start()` / `wait_start()` 等待下位机回传或触发 START，统计收发成功率。

### 7.2 `test/test_camera_view.py`

#### 用途

只测相机画面和摆放，不加载模型、不开串口。

#### 参数

```bash
python test/test_camera_view.py [--config PATH] [--mock-cameras]
```

#### 行为

- 打开两个相机（真实或 mock）。
- 使用 `CameraViewer` 显示画面，按 `1/2` 切换，叠加辅助线。
- 无滑块、不保存配置、不运行推理。

### 7.3 `test/test_inference_ui.py`

#### 用途

加载 BPU 模型，验证识别效果，与串口解耦。

#### 参数

```bash
python test/test_inference_ui.py [--config PATH] [--mock-cameras]
                                 [--bean-image PATH] [--digit-image PATH]
```

#### 行为

- 打开相机 + 加载 `models/*.bin`。
- 使用 `CalibrationUI`（或专用 `InferenceViewer`），按 `1/2` 切换相机，显示实时推理框、类别名、置信度、辅助线。
- 不打开串口，不按 START，不发送结果。
- 在 RDK X5 上跑真实模型；在 x86 上默认会报 BPU 不可用，可配合 `--mock-cameras` 和 mock 模型测试 UI 流程。

## 8. 单元测试迁移

### 8.1 迁移清单

| 原路径 | 新路径 |
|--------|--------|
| `test/test_postprocess.py` | `tools/crane_tests/test_postprocess.py` |
| `test/test_task_generator.py` | `tools/crane_tests/test_task_generator.py` |
| `test/test_serial.py` | `tools/crane_tests/test_serial.py` |
| `test/test_config.py` | `tools/crane_tests/test_config.py` |
| `test/test_camera.py` | `tools/crane_tests/test_camera.py` |
| `test/test_model.py` | `tools/crane_tests/test_model.py` |
| `test/test_ui.py` | `tools/crane_tests/test_ui.py` |
| `test/test_send_test_frame.py` | `tools/crane_tests/test_send_test_frame.py` |
| `test/test_main.py` | 删除或大幅精简 |

### 8.2 调整内容

- 保留顶部 `sys.modules["ultralytics"] = None` 等防止 PyTorch/CUDA 加载的语句。
- `pytest` 命令更新为 `pytest tools/crane_tests/ -v`。
- 在 `tools/crane_tests/README.md` 说明目录用途。

## 9. 文档更新

- `docs/runbooks/commands.md`：
  - 更新竞赛模式命令为 `python3 scripts/main_arm64.py`。
  - 新增校准命令 `python scripts/calibrate.py`。
  - 新增三个测试脚本的运行命令。
  - 更新单元测试命令为 `pytest tools/crane_tests/ -v`。
- `CLAUDE.md`：
  - 更新“快速入口”和“视觉运行时映射规则”段落。
  - 明确 `main_arm64.py` 仅用于竞赛。

## 10. 数据流与依赖

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ scripts/        │     │ test/           │     │ tools/          │
│ main_arm64.py   │     │ test_serial_    │     │ crane_tests/    │
│ calibrate.py    │     │   loopback.py   │     │ test_*.py       │
│                 │     │ test_camera_    │     │                 │
│                 │     │   view.py       │     │                 │
│                 │     │ test_inference_ │     │                 │
│                 │     │   ui.py         │     │                 │
└────────┬────────┘     └────────┬────────┘     └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │ models/runtime.py│
            │  load_vision_cfg │
            │  open_camera     │
            │  load_models     │
            │  run_competition │
            └────────┬────────┘
                     ▼
         ┌─────────────────────────┐
         │ models/ui.py            │
         │  CalibrationUI          │
         │ models/camera_viewer.py │
         │  CameraViewer           │
         │ models/serial_port.py   │
         │  SerialPort             │
         └─────────────────────────┘
```

## 11. 错误处理

- `main_arm64.py`：竞赛模式下，模型加载失败直接报错退出；串口打开失败直接报错退出。
- `calibrate.py`：模型加载失败进入 camera-only 模式；相机打开失败在 `fallback_empty=True` 时降级为 mock。
- `test/test_serial_loopback.py`：打开串口失败直接报错；收发失败统计错误并打印。
- `test/test_camera_view.py` / `test/test_inference_ui.py`：相机打开失败报错退出；模型加载失败根据脚本目的决定是否降级。

## 12. 验收标准

- [ ] `scripts/main_arm64.py` 参数不超过 `--config` 和 `--verbose`。
- [ ] `python scripts/calibrate.py` 能正常启动校准 UI。
- [ ] `test/test_serial_loopback.py` 能在 RDK X5 上完成回环/协议收发测试。
- [ ] `test/test_camera_view.py` 能显示两个相机画面和辅助线，无滑块、无推理。
- [ ] `test/test_inference_ui.py` 能加载模型并叠加检测框，不打开串口。
- [ ] `pytest tools/crane_tests/ -v` 全部通过。
- [ ] `docs/runbooks/commands.md` 和 `CLAUDE.md` 已同步更新。
