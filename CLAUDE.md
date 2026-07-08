# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本仓库是 2026 物流技术创意赛的计算机视觉目标检测工作空间。它包含位于 `tools/yolov5/` 的 Ultralytics YOLOv5 副本（用于训练与导出），以及位于 `models/`、`scripts/`、`configs/` 和 `test/` 的独立 RDK X5 视觉运行时。运行时使用两路 USB 相机、两个 YOLO11 BPU `.bin` 模型（`num.bin` 识别数字，`beans_v4.bin` 识别豆子）和串口通信，为下位机生成 TASK 帧。

> **注意：** `docs/RDKx5_Bin_Inference.md` 描述 YOLO11 风格的 6 头部署流程，但 `tools/yolov5/` 下的训练代码是标准 YOLOv5。板端推理以部署文档为准，训练/导出以 YOLOv5 树为准。

## 仓库目录结构

```text
assets/            # 训练好的模型产物（.pt、.onnx、.bin）
                   #   num.*      – 数字检测器
                   #   beans_v4.* – 豆子成熟度检测器
                   #   *_rdkx5_config.yaml – BPU 部署配置
configs/           # 运行时配置
                   #   vision.yaml – 串口、相机、模型、辅助线、映射
                   #   .gitkeep
data/              # 数据集（YOLO 格式：images/ + labels/ + dataset.yaml）
                   #   dataset3/       – 原始数字图片
                   #   output_gray/    – 处理后的数字数据集 + split_dataset/
                   #   beans/          – 豆子图片 + beans_split/
docs/              # 部署与运行时文档
                   #   RDKx5_Bin_Inference.md – YOLO11 BPU 板端工作流
                   #   vision_runtime.md      – 视觉运行时入口与状态机
                   #   vision_mapping.md      – 分区映射与辅助线规则
                   #   superpowers/           – 设计文档与实现计划
models/            # 视觉运行时源码
                   #   yolo11_postprocess.py, yolo11_model.py, bpu_loader.py
                   #   bean_model.py, digit_model.py, camera.py
                   #   serial_port.py, task_generator.py, mapping.py, ui.py
scripts/           # 视觉运行时入口
                   #   main_arm64.py – RDK X5 部署入口（竞赛 + 校准）
                   #   main_amd64.py – x86 本机完整流程测试入口
                   #   watchdog.sh – main_arm64.py 自动重启脚本
                   #   .gitkeep
test/              # x86 单元测试（mock hobot_dnn）
requirements.txt   # 运行时/测试依赖
README.md          # 项目概览（侧重视觉模块）
tools/yolov5/      # 内嵌 YOLOv5 仓库（train.py、detect.py、val.py、export.py 等）
tools/onnx/        # 本地 ONNX 导出；已加入 .gitignore
```

`data/*` 和 `tools/onnx/*` 已加入 `.gitignore`，仅保留 `data/.gitkeep`。大型资产和数据集只保存在本地工作区，不提交到仓库。

## 常用命令

以下命令均在仓库根目录执行。YOLOv5 训练/验证/导出命令引用 `tools/yolov5/` 中的脚本。

### 开发/测试虚拟环境

本项目使用独立的 Miniforge 虚拟环境，避免与系统或其他项目的 `opencv-python` / `opencv-python-headless` 等包冲突：

- 环境路径：`D:\conda_envs\crane-clean`
- Python：3.11
- 包源：USTC PyPI 镜像

所有 x86 开发/测试命令均通过该环境执行：

```bash
conda run --prefix /d/conda_envs/crane-clean python ...
```

#### 安装运行时/测试依赖

```bash
conda run --prefix /d/conda_envs/crane-clean pip install --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple -r requirements.txt
```

#### 运行单元测试

```bash
conda run --prefix /d/conda_envs/crane-clean python -m pytest test/ -v
```

### 视觉运行时命令

这些命令用于操作 RDK X5 竞赛运行时。`--ui` 校准视图和单元测试可在 x86 上运行，但完整 BPU 推理需要 RDK X5 板端的 `hobot_dnn`。

#### 使用实时预览校准辅助线

```bash
conda run --prefix /d/conda_envs/crane-clean python scripts/main_arm64.py
```

按 `1` / `2` 切换豆子/数字相机视图，通过控制面板滑块调整辅助线，按 `ESC` 保存到 `configs/vision.yaml`。

> `--ui` 仍可作为默认校准模式的向后兼容别名使用。在没有相机的 x86 开发机上，可配合 `--mock-cameras` 预览校准 UI。

#### 无相机测试校准 UI

```bash
conda run --prefix /d/conda_envs/crane-clean python scripts/main_arm64.py --mock-cameras
```

使用合成帧启动校准 UI，用于无相机环境下的 UI 开发与辅助线布局测试。按 `ESC` 保存的辅助线位置同样写回 `configs/vision.yaml`。

#### 列出可用相机

```bash
conda run --prefix /d/conda_envs/crane-clean python scripts/main_arm64.py --list-cameras
```

#### 运行竞赛模式（仅限 RDK X5）

```bash
python3 scripts/main_arm64.py --no-ui
```

#### 自动重启竞赛模式

```bash
./scripts/watchdog.sh
```

#### x86 完整流程测试（无串口）

`main_amd64.py` 使用 `assets/*.pt` 进行 PyTorch 推理（依赖 `ultralytics`）。当前 `crane-clean` 环境已安装 GPU 版 `torch 2.5.1+cu121` 和 `ultralytics`，可直接运行：

```bash
conda run --prefix /d/conda_envs/crane-clean python scripts/main_amd64.py --mock-cameras
```

按 `SPACE` 或 `s` 触发一次扫描，控制台打印 `SCAN RESULT: x y z` 或失败原因，按 `ESC` 退出。要单次扫描并退出，可加上 `--once`：

```bash
conda run --prefix /d/conda_envs/crane-clean python scripts/main_amd64.py --once --mock-cameras
```

如果当前环境没有 `ultralytics`，`main_amd64.py` 会自动退回到 camera-only 模式并打印 `SCAN FAILED: task generator unavailable`。

### 安装依赖（YOLOv5 训练）

```bash
pip install -r tools/yolov5/requirements.txt
```

### 训练检测器

```bash
python tools/yolov5/train.py \
  --data data/output_gray/images/dataset.yaml \
  --weights yolov5n.pt \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --project runs/train \
  --name exp
```

豆子数据集使用 `data/beans/beans_split/beans_split.yaml`。权重首次使用时会自动下载。

### 验证模型

```bash
python tools/yolov5/val.py \
  --data data/output_gray/images/dataset.yaml \
  --weights assets/num.pt \
  --img 640
```

### 运行推理

```bash
python tools/yolov5/detect.py \
  --weights assets/num.pt \
  --source data/dataset3/images/val/ \
  --img 640 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --project runs/detect \
  --name exp
```

### 导出 ONNX

```bash
python tools/yolov5/export.py \
  --weights assets/num.pt \
  --include onnx \
  --imgsz 640
```

### 运行 YOLOv5 测试

```bash
python -m pytest tools/yolov5/tests/
```

运行单个测试文件：

```bash
python -m pytest tools/yolov5/tests/test_invariant_export.py -v
```

### RDKx5 板端推理

完整流程见 `docs/RDKx5_Bin_Inference.md`。板端入口为 `scripts/main_arm64.py`；部署时需把 `models/`、`scripts/`、`configs/`、`assets/*.bin` 和 `assets/*_rdkx5_config.yaml` 一起拷贝到 RDKx5。使用 `python3 scripts/main_arm64.py` 进行校准，使用 `./scripts/watchdog.sh` 以看门狗方式运行竞赛模式。关键要点：

- 输入：RGB NCHW `uint8` 张量，形状 `1×3×640×640`；归一化（`/255`）已编译进模型。
- Letterbox 填充使用灰度值 `127`。
- 输出为 6 个张量：`s_bbox`、`m_bbox`、`l_bbox`、`s_cls`、`m_cls`、`l_cls`。
- 后处理流程：DFL 解码 → 还原 xyxy → sigmoid → NMS。
- 脚本依赖 `hobot_dnn`，仅在 RDKx5 镜像中可用。

## 数据集规范

每个数据集目录应包含：

- `images/train/` 和 `images/val/`
- `labels/train/` 和 `labels/val/`，文件为 YOLO 格式 `.txt`（`class x_center y_center width height`，已归一化）
- `dataset.yaml`（或 `<name>.yaml`），包含 `path`、`train`、`val`、`names`、`nc`

现有数据集 YAML 使用 `path: .` 和 `train: .`，因此需要在该 YAML 所在目录下调用。新建数据集配置时，建议使用绝对路径或仓库根目录相对路径，以避免路径歧义。

## 资产命名规范

- `.pt`  – PyTorch 权重（YOLOv5 / Ultralytics 格式）
- `.onnx` – x86 验证用 ONNX 导出
- `.bin`  – RDKx5 部署用 BPU 二进制

每个 `.bin` 基名都应配有一个部署配置 YAML（例如 `beans_v4_rdkx5_config.yaml`），包含 `imgsz`、`nc`、`names`、`reg_max`、`strides`。额外字段（如 `task`、`model`、`onnx`）不影响板端推理。

## 视觉运行时映射规则

`configs/vision.yaml` 驱动竞赛逻辑：

- `mapping.bean_to_digit`：将豆子类别映射为目标数字（默认黄豆→1、绿豆→2、白芸豆→3）。
- `lines.bean.left_x` / `lines.bean.right_x`：竖直辅助线，将豆子相机画面划分为取货区 ①–③。
- `lines.digit.horizontal_y` / `lines.digit.left_x` / `lines.digit.right_x`：辅助线，将数字相机画面划分为放置区 ④–⑧。

运行 `python scripts/main_arm64.py` 可在实时预览上通过滑块调整辅助线；位置会自动写回 `configs/vision.yaml`。`--ui` 是默认校准模式的向后兼容别名。具体分区规则见 `docs/vision_mapping.md`。

## 提交代码

提交信息按 Angular 规范写中文主题。在 PowerShell 中，避免把 `@` 解析成数组，使用 here-string 写入临时文件再提交：

```powershell
# 写入提交信息
@'
fix(rdkx5): 修复导出脚本路径问题

- 修复 ONNX 路径使用 POSIX 斜杠
- 校准数据 pad 值对齐 Ultralytics
'@ | Out-File -Encoding utf8 .git-msg

# 提交并删除临时文件
git commit -F .git-msg
Remove-Item .git-msg
```

### 协作者

**禁止在提交信息中添加 `Co-Authored-By` 尾注或设置任何协作者。** 所有提交只记录实际执行该提交的作者。

## 重要注意事项

- 请勿在开发机上以竞赛模式运行 `scripts/main_arm64.py`；完整 BPU 推理需要 RDKx5 镜像中的 `hobot_dnn`。`main_arm64.py` 校准视图、`main_amd64.py` 和单元测试可在 x86 上安全运行。
- `configs/vision.yaml` 是运行时唯一事实来源；在 UI 中调整的辅助线位置会自动保存回该文件。
- 新工作的默认分支是 `main`；`master` 是合并视觉运行时之前的历史分支。
- 提交新的训练/导出脚本或配置前，先确认它们应放在顶层 `scripts/`/`configs/`（视觉运行时）还是 `tools/yolov5/`（传统 YOLOv5 训练）。
