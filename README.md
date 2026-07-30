# Crane 视觉模块

2026 物流技术创意赛 RDK X5 视觉识别模块：双 USB 相机 + 两路 YOLO11 BPU 模型 + 串口通信，向下位机发送 12 字节结果帧。

## 功能

- **豆子检测**：识别取货区 ①–③ 的豆子种类（白芸豆 / 黄豆 / 绿豆），映射为协议颜色编码（黄豆→1、绿豆→2、白芸豆→3）。
- **数字检测**：识别放置区 ④–⑧ 货箱侧面的数字 1–5。
- **逻辑补全**：缺区/重复时通过全局排列评分推断最可能的值，推断区以橙色框显示；超过上限（豆子 ≤1、数字 ≤2）时拒绝发送。
- **串口输出**：收到下位机 START 帧（`A5 5A 01`）后，发送 12 字节结果帧（`A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`）。

## 目录结构

```text
assets/            # 训练好的模型产物（.pt、.onnx、.bin + *_rdkx5_config.yaml）
configs/           # 运行时配置
  vision.yaml                     # 主配置：串口、相机、模型、辅助线、映射
  vision_1024x600_test.yaml       # 1024×600 备选配置
data/              # 数据集（YOLO 格式，已加入 .gitignore）
docs/              # 部署与运行时文档
  guides/          #   RDKx5_Bin_Inference.md — BPU 板端工作流
                   #   vision_runtime.md        — 运行时入口、状态机与分区映射
  reference/       #   CRANE_VISION_PROTOCOL.md — 串口硬件与通信协议
                   #   conventions.md          — 数据集与资产命名规范
  runbooks/        #   commands.md             — 完整命令清单
models/            # 视觉运行时源码（15 个 .py 文件）
  camera_viewer.py              # CameraViewer 基类
  ui.py                         # CalibrationUI(CameraViewer) + CompetitionUI(CalibrationUI)
  yolo11_postprocess.py         # letterbox / DFL 解码 / NMS / 坐标回映
  yolo11_model.py / bpu_loader.py  # BPU 推理封装（x86 上回退）
  bean_model.py / digit_model.py   # 豆子/数字模型薄封装
  camera.py / pt_model.py          # 相机封装 / x86 PyTorch 推理
  serial_port.py                # SerialPort + SerialThread（后台串口）
  task_generator.py             # 分区聚合 + 逻辑补全
  logic_completion.py           # 置信度排列评分 + 缺区推断
  mapping.py / runtime.py       # 豆子→颜色映射 / 入口共享助手
scripts/           # 视觉运行时入口
  main_arm64.py                 # RDK X5 竞赛模式入口（CompetitionUI）
  main_amd64.py                 # x86 双相机 PyTorch 推理 + 逻辑补全（CompetitionUI，无串口）
  watchdog.sh                   # main_arm64.py 自动重启脚本
tests/             # 独立功能测试脚本（可能依赖真实相机/串口/BPU）
  calibrate.py                  # 校准 UI 入口
  test_camera_view.py           # 双相机画面检查
  test_inference_ui.py          # 推理 + 检测框叠加
  test_serial_loopback.py       # 串口回环/协议测试
  test_logic_completion.py      # 逻辑补全模块测试
  test_settings_controls.py     # 校准控件单元测试
  test_ui_completion.py         # 推断标签渲染测试
  send_test_frame.py            # 串口测试帧发送工具
tools/             # 训练/导出工具 + 模块级单元测试
  yolov5/                       # 内嵌 YOLOv5 仓库（train/detect/val/export）
  crane_tests/                  # 模块级单元测试（pytest 驱动，mock hobot_dnn）
  onnx/                         # 本地 ONNX 导出（已加入 .gitignore）
requirements.txt   # 运行时/测试依赖
CHANGELOG.md       # 变更日志
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
# 模块级单元测试（x86，mock hobot_dnn）
python -m pytest tools/crane_tests/ -v

# 独立功能测试（x86，直接运行）
python tests/test_logic_completion.py
```

### x86 完整流程测试（CompetitionUI + PyTorch）

`main_amd64.py` 启动 CompetitionUI，双相机同时 PyTorch 推理 + 逻辑补全，无串口依赖：

```bash
python scripts/main_amd64.py --mock-cameras
```

使用静态测试图片：

```bash
python scripts/main_amd64.py --bean-image data/image_for_test/beans_2.jpg --digit-image data/image_for_test/num_2.jpg
```

按 `1`=豆子画面，`2`=数字画面，`3`=settings（含推断上限滑块），`SAVE` 保存配置，`EXIT` 退出。

### 校准辅助线

```bash
python tests/calibrate.py --mock-cameras
```

在实时预览上通过滑块调整辅助线，点击 **SAVE** 手动写回 `configs/vision.yaml`。

### 板端部署

1. 将 `models/`、`scripts/`、`configs/`、`tests/` 与 `assets/*.bin`、`assets/*_rdkx5_config.yaml` 拷贝到 RDK X5。
2. 校准辅助线：
   ```bash
   python3 tests/calibrate.py
   ```
3. 启动竞赛模式（看门狗自动重启）：
   ```bash
   bash scripts/watchdog.sh
   ```
4. 下位机发送 START 帧 → 视觉返回 12 字节结果帧。

RDK 上电竞赛模式建议安装无界面 systemd 服务，避免等待 XFCE 桌面启动：

```bash
sudo bash scripts/install_crane_service.sh
```

服务先从 `main_arm64.py --headless --ui-when-ready` 启动结果通路，并在
XFCE 可用后自动显示竞赛 UI。校准时先执行 `sudo systemctl stop
crane-vision.service` 释放相机和串口。

## 配置

运行时唯一事实来源：[configs/vision.yaml](configs/vision.yaml)。关键配置段：

| 段 | 作用 |
| ---- | ------ |
| `serial` | 串口设备、波特率（默认 **921600** 8/N/1） |
| `cameras.bean` / `cameras.digit` | 设备节点、分辨率、帧率、曝光、灰度 |
| `models.bean` / `models.digit` | BPU `.bin`/`.cfg`（arm64）或 `.pt`（x86）权重，可选覆盖 `conf_thres`/`iou_thres` |
| `lines.bean` / `lines.digit` | 竖直辅助线，划分取货区 ①–③ 和放置区 ④–⑧ |
| `mapping.bean_to_digit` | 豆子类别 → 协议颜色编码（黄豆→1、绿豆→2、白芸豆→3） |
| `logic` | 置信度阈值 `confidence_threshold`，推断上限 `max_inferred_beans`/`max_inferred_digits` |
| `ui.controls` | 触摸屏模式、控件尺寸、推理 overlay 开关 |

## 文档

| 文档 | 内容 |
| ------ | ------ |
| [docs/guides/vision_runtime.md](docs/guides/vision_runtime.md) | 运行时入口、状态机与分区映射规则 |
| [docs/guides/RDKx5_Bin_Inference.md](docs/guides/RDKx5_Bin_Inference.md) | YOLO11 BPU 板端推理工作流 |
| [docs/reference/CRANE_VISION_PROTOCOL.md](docs/reference/CRANE_VISION_PROTOCOL.md) | 串口硬件接线与 12 字节协议细节 |
| [docs/reference/conventions.md](docs/reference/conventions.md) | 数据集与资产命名规范 |
| [docs/runbooks/commands.md](docs/runbooks/commands.md) | 完整命令清单（conda 环境、训练/导出、部署） |

## 约束

- `hobot_dnn` 仅在 RDK X5 上可用；x86 侧通过 mock 做单元测试。
- 默认输入尺寸 640×640，letterbox pad 值 127。
- 串口参数：**921600 8/N/1**（`/dev/ttyS1`，禁止使用 `/dev/ttyS0`）。
- 开发机不要运行 `main_arm64.py`（依赖 RDK X5 镜像中的 `hobot_dnn`）；x86 使用 `main_amd64.py` 或 `tests/` 下的独立脚本。
- 分支策略：默认分支 `main`；`master` 为历史分支。

## 许可证

随本项目仓库一起分发。
