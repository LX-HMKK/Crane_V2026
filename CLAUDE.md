# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## 项目概述

2026 物流技术创意赛视觉目标检测工作空间：RDK X5 视觉运行时（`models/`、`scripts/`、`configs/`）。双 USB 相机 + 两路 YOLO11 BPU `.bin` 模型 + 串口，向下位机发送 12 字节结果帧。运行时模块已分层：`models/application/`（入口与结果协调）、`models/ui/`（viewer / 校准 / 竞赛 UI）、`models/devices/`（相机与串口）、`models/inference/`（BPU/PyTorch 模型）。

> 板端推理以 `docs/guides/RDKx5_Bin_Inference.md` 为准。目录结构、配置字段、提交规范等低频参考见[文档索引](#文档索引)。

## 竞赛模式架构

`main_arm64.py` 的竞赛 UI 使用独立的双路采集和双模型推理路径：

| 执行路径 | 类 | 职责 |
| --- | --- | --- |
| 主线程 (Tkinter) | `CompetitionUI` | 仅显示当前选中的相机预览、辅助线和检测框，不阻塞采集或推理 |
| 相机 worker (daemon, 每路一个) | `CompetitionUI._capture_worker` | 持续采集，覆盖各自模型的最新帧槽，不建立帧队列 |
| 推理 worker (daemon, 每模型一个) | `CalibrationUI._inference_worker`（位于 `models/ui/calibration.py`） | 各自执行 `model.predict()` 并发布最新检测结果 |
| 结果组合 | `CompetitionResultController`（`models/application/competition_controller.py`） | 按串口发送周期组合两路最新检测，更新 `serial.current_frame` |
| 串口线程 (daemon) | `SerialThread` | `poll_start()` @200Hz → `send_result()` @20Hz |

**逻辑补全**：`generate_frame_with_completion()` 在检测缺区/重复时，通过全局排列评分推断最可能的值。推断区在 UI 上以橙色框显示。推断超过上限（bean≤1, digit≤2）时拒绝发送。

**START 门控推理**：模型推理仅在收到下位机 START 时运行（空闲模型保持加载待命，由 `inference.start_gated` 控制）；START 停止后清空串口缓存，恢复后以新推理结果续发。`configs/vision.yaml` 的 `protocol.start_timeout_sec` 默认 3.0 秒，收到单帧 START 后也会保持 active 一段时间。`CompetitionUI` 右侧 **DEBUG** 按钮可强制推理并显示（仅调试显示，不改变串口门控）；`debug.force_inference` 供 headless 现场调试。

## 常用命令

完整命令清单见 [docs/runbooks/commands.md](docs/runbooks/commands.md)。快速入口：

```bash
# 单元测试
conda run --prefix /d/conda_envs/crane-clean python -m pytest tools/crane_tests/ -v
# 校准辅助线
conda run --prefix /d/conda_envs/crane-clean python tests/calibrate.py --mock-cameras
# 竞赛模式（仅限 RDK X5）
python3 scripts/main_arm64.py
# 上电竞赛模式（systemd 无界面服务）
systemctl start crane-vision.service
# x86 完整流程测试（CompetitionUI，PyTorch 推理 + 逻辑补全 + 无串口）
python scripts/main_amd64.py --mock-cameras
# 静态图片测试（x86 PyTorch）
python scripts/main_amd64.py --bean-image data/image_for_test/beans_2.jpg --digit-image data/image_for_test/num_2.jpg
```

## RDK X5 连接与同步

- 目标板：`root@192.168.128.10`，工作区 `/home/sunrise/Crane_WS`。
- 增量同步：只传自上次提交以来变更的文件，**绝不覆盖 `assets/*.bin`** 等大型资产。

```bash
git diff --name-only HEAD
scp configs/vision.yaml root@192.168.128.10:/home/sunrise/Crane_WS/configs/
```

完整同步规则、无串口测试命令和看门狗用法见 [docs/runbooks/commands.md](docs/runbooks/commands.md)。

## 串口

RDK X5 与下位机通过 `/dev/ttyS1` 通信，默认配置 **921600 8/N/1**。参数来自 `configs/vision.yaml` 的 `serial` 段。**不要使用 `/dev/ttyS0`**（系统调试串口）。硬件引脚、接线与协议细节见 [docs/reference/CRANE_VISION_PROTOCOL.md](docs/reference/CRANE_VISION_PROTOCOL.md)。

## 测试入口约定

`tools/crane_tests/`：模块级 mock 单元测试，`pytest tools/crane_tests/ -v` 驱动（x86 可跑）；`tests/`：独立功能脚本，`python tests/xxx.py`，可能依赖真实相机/串口/BPU。

## 重要注意事项

- `main_arm64.py` 默认使用 `CompetitionUI`；`crane-vision.service` 使用 `--headless --ui-when-ready`，先启动结果通路再显示 UI。校准前先停止该服务释放相机/串口。
- 请勿在开发机上以竞赛模式运行 `scripts/main_arm64.py`（需要 RDKx5 的 `hobot_dnn`）；`calibrate.py`、`test_camera_view.py`、`test_inference_ui.py`、`main_amd64.py`、`test_logic_completion.py` 和单元测试可在 x86 上安全运行。
- `configs/vision.yaml` 是运行时唯一事实来源；辅助线位置需点击 **SAVE** 按钮手动保存回该文件。
- 逻辑补全配置 `logic.confidence_threshold` / `max_inferred_beans` / `max_inferred_digits` 控制推断容忍度。推断区 UI 橙色框，检测区黄色框。
- 新工作的默认分支是 `main`；`master` 是合并视觉运行时之前的历史分支。
- 提交新的训练/导出脚本或配置前，先确认应放顶层 `scripts/`/`configs/`。
- 在 x86 开发机上运行 `pytest tools/crane_tests/` 时，测试模块顶部阻止 `ultralytics`/`torch` 导入防内存暴涨；**不要删除这些 `sys.modules[...] = None` 语句**。

## 文档索引

按需读取，不要预加载：

- [仓库目录结构](docs/reference/directory_structure.md) — 全仓库布局、各模块职责与数据集/资产命名
- [vision.yaml 配置参考](docs/reference/vision_config.md) — 相机/模型/UI 字段、映射规则、结果帧格式
- [提交规范](docs/reference/commit_guidelines.md) — Angular 中文提交 + PowerShell here-string + 禁止协作者
- [运行时入口与状态机](docs/guides/vision_runtime.md) — 启动方式、状态机、分区映射规则
- [串口协议](docs/reference/CRANE_VISION_PROTOCOL.md) — 硬件接线与 12 字节帧
- [完整命令清单](docs/runbooks/commands.md)
- [RDK X5 服务运维](docs/runbooks/rdk_service.md) — crane-vision.service 启停/日志/校准
- [RDKx5 BPU 板端工作流](docs/guides/RDKx5_Bin_Inference.md)
