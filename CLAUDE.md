# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## 项目概述

2026 物流技术创意赛视觉目标检测工作空间：YOLOv5 训练/导出工具（`tools/yolov5/`）+ RDK X5 视觉运行时（`models/`、`scripts/`、`configs/`）。双 USB 相机 + 两路 YOLO11 BPU `.bin` 模型 + 串口，向下位机发送 12 字节结果帧。

> 板端推理以 `docs/guides/RDKx5_Bin_Inference.md` 为准；训练/导出以 `tools/yolov5/` 为准。

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
                   #   runbooks/
                   #     commands.md           – 完整命令清单
                   #     serial_port.md        – 串口硬件与协议
                   #   reference/
                   #     dataset_spec.md       – 数据集规范
                   #     asset_naming.md       – 资产命名规范
                   #     CRANE_VISION_PROTOCOL.md – 通信协议
                   #     requirements.md       – 原始需求（已归档）
                   #   guides/
                   #     RDKx5_Bin_Inference.md – YOLO11 BPU 板端工作流
                   #     vision_runtime.md      – 视觉运行时入口与状态机
                   #     vision_mapping.md      – 分区映射与辅助线规则
                   #   superpowers/           – 设计文档与实现计划
models/            # 视觉运行时源码
                   #   yolo11_postprocess.py, yolo11_model.py, bpu_loader.py
                   #   bean_model.py, digit_model.py, camera.py, pt_model.py
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

完整命令清单（conda 环境、YOLOv5 训练/导出、RDK X5 部署与测试）见 [docs/runbooks/commands.md](docs/runbooks/commands.md)。

快速入口：

```bash
# 单元测试
conda run --prefix /d/conda_envs/crane-clean python -m pytest test/ -v

# 校准辅助线
conda run --prefix /d/conda_envs/crane-clean python scripts/main_arm64.py --mock-cameras

# 竞赛模式（仅限 RDK X5）
python3 scripts/main_arm64.py --no-ui
```

## 数据集与资产规范

- 数据集目录与 YAML 规范：[docs/reference/dataset_spec.md](docs/reference/dataset_spec.md)
- 模型资产命名规范（.pt/.onnx/.bin）：[docs/reference/asset_naming.md](docs/reference/asset_naming.md)

## 视觉运行时映射规则

`configs/vision.yaml` 驱动竞赛逻辑：

- `mapping.bean_to_digit`：将豆子类别映射为协议颜色编码（默认黄豆→1、绿豆→2、白芸豆→3），该编码同时作为结果帧 `C1-C3` 的颜色值。
- `lines.bean.left_x` / `lines.bean.right_x`：竖直辅助线，将豆子相机画面划分为取货区 ①–③。
- `lines.digit.x1` / `x2` / `x3` / `x4`：竖直辅助线，将数字相机画面划分为放置区 ④–⑧。

结果帧格式为 `A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`：

- `C1-C3`：取货区 1–3 的豆子颜色（01=黄豆，02=绿豆，03=白芸豆）。
- `N4-N8`：放置区 4–8 上的数字（01–05）。

运行 `python scripts/main_arm64.py` 可在实时预览上通过滑块调整辅助线；位置会自动写回 `configs/vision.yaml`。`--ui` 是默认校准模式的向后兼容别名。具体分区规则见 [docs/guides/vision_mapping.md](docs/guides/vision_mapping.md)。

## 相机与模型配置字段

`configs/vision.yaml` 中除辅助线外，还经常需要调整以下字段：

- `cameras.bean` / `cameras.digit`：
  - `device`：V4L2 设备节点，可用 `scripts/main_arm64.py --list-cameras` 列出。
  - `width` / `height`：捕获分辨率。
  - `fps`：请求摄像头帧率（例如 `15`），实际帧率以设备返回为准。
  - `buffer_size`：OpenCV 内部缓冲，竞赛模式建议设为 `1` 降低延迟。
  - `auto_exposure`：`true` 启用自动曝光，`false` 改为手动曝光。
  - `grayscale`：`true` 时把输出强制转成单通道后再扩为 3 通道 BGR；数字相机若使用灰度数据集训练，可设为 `true`。

- `models.bean` / `models.digit`：
  - `bin` / `cfg`：RDK X5 BPU 模型与部署配置。
  - `pt`：`main_amd64.py` 在 x86 上使用的 Ultralytics 权重。
  - `conf_thres` / `iou_thres`：可选，覆盖全局 `inference.conf_thres` / `inference.iou_thres`，用于按模型调参。

- `ui.controls`：
  - `touch_mode`：触摸屏布局开关。
  - `slider_width` / `slider_height` / `button_height` / `padding` / `font_scale`：校准控件尺寸。
  - `inference_overlay`：`true` 时在校准画面上叠加推理框与分区编号，方便观察模型输出。

## 串口

RDK X5 与下位机通过 `/dev/ttyS1` 通信，默认配置 **921600 8/N/1**。参数来自 `configs/vision.yaml` 的 `serial` 段。**不要使用 `/dev/ttyS0`**（系统调试串口）。硬件引脚、接线与协议细节见 [docs/runbooks/serial_port.md](docs/runbooks/serial_port.md)。

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
- `main_arm64.py` 现已通过 `models.serial_port.open_serial_port()` 初始化串口，默认 921600 8/N/1；如需调整串口参数，应修改 `models/serial_port.py` 或 `configs/vision.yaml`。
- 在 x86 开发机上运行 `pytest test/` 时，测试模块会在顶部阻止 `ultralytics`/`torch` 导入，防止 pytest 收集阶段加载 PyTorch/CUDA 上下文导致内存暴涨。不要在测试文件顶部删除这些 `sys.modules[...] = None` 语句。

## 开发行为准则（八荣八耻）

| 序号 | 可耻行为（禁止） | 光荣行为（倡导） |
|------|------------------|------------------|
| 1 | 暗猜接口 | **认真查阅** — 文档、源码、现有实现 |
| 2 | 模糊执行 | **寻求确认** — 对不明确的需求先确认再动手 |
| 3 | 盲想业务 | **人类确认** — 关键业务逻辑与真人核对 |
| 4 | 创造接口 | **复用现有** — 优先使用已有函数/模块/约定 |
| 5 | 跳过验证 | **主动测试** — 改完必测，新增必验 |
| 6 | 破坏架构 | **遵循规范** — 按项目结构和约定行事 |
| 7 | 假装理解 | **诚实无知** — 不懂就说，不硬撑 |
| 8 | 盲目修改 | **谨慎重构** — 看清影响面再改 |

### 简化记忆版

> **查、问、核、复、测、规、诚、慎**
>
> 1. **查**接口  2. **问**清楚  3. **核**业务  4. **复**现有  
> 5. **测**改动  6. **规**架构  7. **诚**不知  8. **慎**重构

## 更多参考文档

- [docs/runbooks/commands.md](docs/runbooks/commands.md) — 完整命令清单
- [docs/runbooks/serial_port.md](docs/runbooks/serial_port.md) — 串口硬件与协议
- [docs/reference/dataset_spec.md](docs/reference/dataset_spec.md) — 数据集规范
- [docs/reference/asset_naming.md](docs/reference/asset_naming.md) — 资产命名规范
- [docs/reference/CRANE_VISION_PROTOCOL.md](docs/reference/CRANE_VISION_PROTOCOL.md) — 通信协议
- [docs/guides/RDKx5_Bin_Inference.md](docs/guides/RDKx5_Bin_Inference.md) — YOLO11 BPU 板端工作流
- [docs/guides/vision_runtime.md](docs/guides/vision_runtime.md) — 视觉运行时入口与状态机
- [docs/guides/vision_mapping.md](docs/guides/vision_mapping.md) — 分区映射与辅助线规则
