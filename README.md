# 🏗️ Crane 视觉模块

> 2026 物流技术创意赛 RDK X5 视觉识别模块
>
> 双 USB 相机 + 两路 YOLO11 BPU 模型 + 串口通信，向下位机发送 12 字节结果帧。

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![PySerial](https://img.shields.io/badge/PySerial-3.x-007396?style=flat-square)](https://pypi.org/project/pyserial/)
[![YOLO11](https://img.shields.io/badge/YOLO11-BPU-00A0C4?style=flat-square)](https://docs.ultralytics.com/)
[![目标板](https://img.shields.io/badge/目标板-RDK%20X5-0aa0c4?style=flat-square)](https://developer.horizon.ai/)
[![串口](https://img.shields.io/badge/串口-921600%208N1-ff8c00?style=flat-square)](docs/reference/CRANE_VISION_PROTOCOL.md)

---

## 📋 目录

- [功能](#-功能) · [界面预览](#-界面预览) · [运行流程](#-运行流程) · [目录结构](#-目录结构)
- [快速开始](#-快速开始) · [配置](#-配置) · [文档](#-文档) · [约束](#-约束) · [许可证](#-许可证)

---

## 🎯 功能

- **豆子检测**：识别取货区 ①–③ 的豆子种类（白芸豆 / 黄豆 / 绿豆），映射为协议颜色编码（黄豆→1、绿豆→2、白芸豆→3）。
- **数字检测**：识别放置区 ④–⑧ 货箱侧面的数字 1–5。
- **逻辑补全**：缺区/重复时通过全局排列评分推断最可能的值，推断区以橙色框显示；超过上限（豆子 ≤1、数字 ≤2）时拒绝发送。
- **串口输出**：收到下位机 START 帧（`A5 5A 01`）后，发送 12 字节结果帧（`A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`）。
- **掉线重连**：相机/串口连接中断后自动重连；UI 状态栏显示 BEAN/DIGIT/SERIAL 连接状态，离线相机以占位画面提示。

## 🖼️ 界面预览

竞赛模式 CompetitionUI 截图（x86 静态图片 + PyTorch 推理）：

<p align="center">
  <img src="docs/images/ui_bean.jpg" width="80%" alt="豆子取货区检测：检测框 + 取货区辅助线 ①–③"/>
</p>
**豆子取货区检测**：豆子相机画面叠加检测框与取货区辅助线 ①–③。

<p align="center">
  <img src="docs/images/ui_digit.jpg" width="80%" alt="数字放置区检测：检测框 + 放置区辅助线 ④–⑧"/>
</p>
**数字放置区检测**：数字相机画面叠加检测框与放置区辅助线 ④–⑧。

<p align="center">
  <img src="docs/images/ui_settings.png" width="80%" alt="校准设置面板：辅助线 / 置信度 / 推断上限滑块"/>
</p>
**校准设置面板**：辅助线、置信度阈值与逻辑补全推断上限滑块，SAVE 写回配置。

## 🗺️ 运行流程

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 32, "rankSpacing": 42, "padding": 10}}}%%
flowchart TD
    subgraph CAP["采集"]
        C1(["豆子相机"]) --> T1["采集线程"]
        C2(["数字相机"]) --> T2["采集线程"]
    end
    subgraph INF["推理"]
        T1 --> M1["豆子模型"]
        T2 --> M2["数字模型"]
    end
    subgraph PRC["逻辑处理"]
        M1 --> M["结果组合"]
        M2 --> M
        M --> A["分区聚合 ①–③ / ④–⑧"]
        A --> Q{"逻辑补全?"}
        Q -->|"完整"| F["12 字节结果帧"]
        Q -->|"缺区/重复"| P["排列评分推断"]
        P -->|"未超上限"| F
        P -->|"超上限"| X["拒绝发送"]
    end
    subgraph OUT["串口输出"]
        F --> S["串口线程 20 Hz"]
        S --> L["下位机"]
        L -. "START 帧轮询" .-> S
    end
    M -. "并行显示" .-> U["主线程 UI"]
    U --> V["预览 + 检测框 + 状态栏"]

    classDef cam fill:#d4efdf,stroke:#1e8449,color:#145a32,stroke-width:2px
    classDef mdl fill:#d6eaf8,stroke:#2e86c1,color:#1b4f72,stroke-width:2px
    classDef lg  fill:#fdebd0,stroke:#d35400,color:#7e5109,stroke-width:2px
    classDef ser fill:#fadbd8,stroke:#cb4335,color:#7b241c,stroke-width:2px
    classDef ui  fill:#e8daef,stroke:#8e44ad,color:#5b2c6f,stroke-width:2px

    class C1,C2,T1,T2 cam
    class M1,M2 mdl
    class M,A,Q,P lg
    class F,S,L,X ser
    class U,V ui

    style CAP fill:#e9f7ef,stroke:#27ae60
    style INF fill:#ebf5fb,stroke:#3498db
    style PRC fill:#fef5e7,stroke:#f39c12
    style OUT fill:#fdedec,stroke:#e74c3c
```

## 📁 目录结构

**目录一览**：`assets` · `configs` · `data` · `docs` · `models` · `scripts` · `tests` · `tools`

<details>
<summary>📂 完整目录结构（点击展开）</summary>

```text
assets/            # 训练产物：.bin / .pt / .onnx + BPU 配置（beans.yaml / num.yaml）
configs/           # 运行时配置：vision.yaml（唯一事实来源）+ 1024×600 备选
data/              # 数据集（YOLO 格式，已 gitignore）
docs/              # 部署与运行时文档
  guides/          #   RDKx5_Bin_Inference.md — BPU 板端工作流
                   #   vision_runtime.md       — 运行时入口、状态机与分区映射
  reference/       #   CRANE_VISION_PROTOCOL.md — 串口硬件与通信协议
                   #   conventions.md         — 数据集与资产命名规范
  runbooks/        #   commands.md            — 完整命令清单
                   #   rdk_service.md         — crane-vision.service 系统服务运维
  images/          #   UI 界面预览截图
models/            # 视觉运行时源码（16 个 .py）
  camera_viewer.py              # CameraViewer 基类
  ui.py                         # CalibrationUI(CameraViewer) + CompetitionUI(CalibrationUI)
  config_binding.py             # ConfigBinding：类型化配置绑定
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
  install_crane_service.sh      # systemd 竞赛模式服务安装脚本
tests/             # 独立功能测试脚本
  calibrate.py                  # 校准 UI 入口
  test_camera_view.py           # 双相机画面检查
  test_inference_ui.py          # 推理 + 检测框叠加
  test_serial_loopback.py       # 串口回环/协议测试
  test_logic_completion.py      # 逻辑补全模块测试
  test_settings_controls.py     # 校准控件单元测试
  test_ui_completion.py         # 推断标签渲染测试
  send_test_frame.py            # 串口测试帧发送工具
tools/             # 训练导出 + 单元测试
  yolov5/                       # 内嵌 YOLOv5 仓库（train/detect/val/export）
  crane_tests/                  # 模块级单元测试（pytest 驱动，mock hobot_dnn）
  onnx/                         # 本地 ONNX 导出（已加入 .gitignore）
  99-crane-cameras.rules        # RDK X5 相机 udev 规则
  crane-vision.service          # systemd 竞赛模式服务
requirements.txt   # 运行时/测试依赖
CHANGELOG.md       # 变更日志
```
</details>

## 🚀 快速开始

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

1. 将 `models/`、`scripts/`、`configs/`、`tests/` 与 `assets/*.bin`、`assets/*.yaml`（BPU 部署配置）拷贝到 RDK X5。
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

## 🔧 配置

运行时唯一事实来源：[configs/vision.yaml](configs/vision.yaml)。关键配置段：

| 段 | 作用 |
| ---- | ------ |
| `serial` | 串口设备、波特率（默认 **921600** 8/N/1）、掉线重连参数 |
| `cameras.bean` / `cameras.digit` | 设备节点、分辨率、帧率、曝光、灰度、掉线重连参数 |
| `models.bean` / `models.digit` | BPU `.bin`/`.cfg`（arm64）或 `.pt`（x86）权重，可选覆盖 `conf_thres`/`iou_thres` |
| `lines.bean` / `lines.digit` | 竖直辅助线，划分取货区 ①–③ 和放置区 ④–⑧ |
| `mapping.bean_to_digit` | 豆子类别 → 协议颜色编码（黄豆→1、绿豆→2、白芸豆→3） |
| `logic` | 置信度阈值 `confidence_threshold`，推断上限 `max_inferred_beans`/`max_inferred_digits` |
| `ui.controls` | 触摸屏模式、控件尺寸、推理 overlay 开关 |

## 📚 文档

| 文档 | 内容 |
| ------ | ------ |
| [docs/guides/vision_runtime.md](docs/guides/vision_runtime.md) | 运行时入口、状态机与分区映射规则 |
| [docs/guides/RDKx5_Bin_Inference.md](docs/guides/RDKx5_Bin_Inference.md) | YOLO11 BPU 板端推理工作流 |
| [docs/reference/CRANE_VISION_PROTOCOL.md](docs/reference/CRANE_VISION_PROTOCOL.md) | 串口硬件接线与 12 字节协议细节 |
| [docs/reference/conventions.md](docs/reference/conventions.md) | 数据集与资产命名规范 |
| [docs/runbooks/commands.md](docs/runbooks/commands.md) | 完整命令清单（conda 环境、训练/导出、部署） |
| [docs/runbooks/rdk_service.md](docs/runbooks/rdk_service.md) | `crane-vision.service` systemd 服务安装与运维 |

## 🚧 约束

- `hobot_dnn` 仅在 RDK X5 上可用；x86 侧通过 mock 做单元测试。
- 默认输入尺寸 640×640，letterbox pad 值 127。
- 串口参数：**921600 8/N/1**（`/dev/ttyS1`，禁止使用 `/dev/ttyS0`）。
- 开发机不要运行 `main_arm64.py`（依赖 RDK X5 镜像中的 `hobot_dnn`）；x86 使用 `main_amd64.py` 或 `tests/` 下的独立脚本。
- 分支策略：默认分支 `main`；`master` 为历史分支。

## 📄 许可证

仓库根目录未附带独立 LICENSE 文件，随本项目仓库一起分发；内嵌 `tools/yolov5/` 遵循其自身 GPL 许可证。
