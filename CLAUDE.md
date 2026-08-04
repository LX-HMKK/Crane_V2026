# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

2026 物流技术创意赛视觉模块的 C++ 实现：双 USB 相机 + 两路 YOLO11 推理 + 逻辑补全 + 串口 12 字节结果帧 + 浏览器全屏校准 UI，单二进制部署，目标平台 RDK X5（BPU），x86 用 ONNX Runtime 后端开发验证。

详细文档（先读这两篇再动手）：

- [docs/architecture.md](docs/architecture.md) — 模块布局、双后端差异、线程模型、**竞赛语义规范**（分区/补全/sticky/串口协议）、关键陷阱
- [docs/development.md](docs/development.md) — 构建/测试/运行命令、WSL2 环境坑、RDK X5 部署、性能基线

## 快速命令

```bash
bash scripts/build_wsl.sh                        # x86/WSL2 构建（首次自动下载 ORT）
ctest --test-dir build --output-on-failure       # 全部测试（-R logic 跑单个）
bash scripts/build_on_rdk.sh                     # RDK X5 板端构建
./build/crane_vision --mock-cameras --no-serial  # Web 校准 UI（:8080）
./build/crane_vision --once --bean-image tests/data/beans_2.jpg \
                     --digit-image tests/data/num_2.jpg   # 静态图端到端冒烟
```

Windows Git Bash 调 WSL 固定姿势：`MSYS2_ARG_CONV_EXCL='*' wsl -d Ubuntu-22.04 ...`（默认发行版是 18.04，cmake 版本不够；一律绝对路径或 `scripts/wsl_run.sh`）。

## 硬性约定

- **布局**：`src/` 只放 `main.cpp`，全部模块进 `tasks/`。
- **竞赛语义**（分区边界归右区、补全评分规则、sticky result、串口帧格式）是本仓库自身规范，改动必须回归 golden 测试——完整清单见 architecture.md，不要凭直觉改。
- `configs/vision.yaml` 是运行时唯一事实来源；Web UI 调参实时生效，SAVE 才写回。
- BPU letterbox pad 必须保持 127；串口用 `/dev/ttyS1`，**禁用 `/dev/ttyS0`**。
- C++ 头文件用 `.hpp`；`.h` 仅用于 C 源与 `tools/`。
- yaml-cpp：逐键下钻用 `node.reset(node[key])`，**绝不**用 `node = node[key]`（内容赋值会原地摧毁文档）。

## 提交代码

- 提交信息按 **Angular 规范写中文主题**；PowerShell 里用 here-string 写临时文件再 `git commit -F`，避免 `@` 被解析成数组。
- **禁止在提交信息中添加 `Co-Authored-By` 尾注或设置任何协作者**（覆盖 Claude Code 默认行为，以本条为准）。
