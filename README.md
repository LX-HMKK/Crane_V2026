# 🏗️ Crane Extreme

> **2026 物流技术创意赛 · 视觉识别模块（C++ 实现）**
> 双 USB 相机 + 两路 YOLO11 推理 + 逻辑补全 + 串口通信 + 浏览器校准 UI
> 单二进制部署，零 Python 依赖

![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![Platform](https://img.shields.io/badge/平台-RDK%20X5%20%7C%20x86%20Linux-lightgrey)
![Inference](https://img.shields.io/badge/推理-BPU%20%7C%20ONNX%20Runtime-orange)
![Build](https://img.shields.io/badge/构建-CMake%20%E2%89%A5%203.16-green)

## ✨ 设计亮点

| 维度 | 方案 |
|---|---|
| 推理后端 | **ONNX Runtime**（x86 验证，全局共享线程池）/ **libDNN C API 直调**（RDK X5 BPU，`hbDNNInfer`） |
| 线程模型 | `std::thread` 真并行：采集×2 → 推理×2 → 任务合并 → 串口，无 GIL |
| 启动开销 | 模型加载 ~0.2 s，即开即用 |
| 部署形态 | 单个二进制 + 模型文件 |
| 校准 UI | **浏览器全屏 Web UI**（内嵌 HTTP/MJPEG，零前端依赖） |

## 🚀 快速开始

```bash
# x86 Linux / WSL2：构建 + 测试
bash scripts/build_wsl.sh
ctest --test-dir build --output-on-failure

# 静态图端到端冒烟（无需相机/串口）
./build/crane_vision --once --bean-image tests/data/beans_2.jpg \
                     --digit-image tests/data/num_2.jpg
# 期望输出：FRAME A5 5A 01 03 02 01 04 05 03 02 0D 0A

# mock 相机 + Web 校准 UI → 浏览器打开 http://<host>:8080
./build/crane_vision --mock-cameras --no-serial --web-port 8080

# RDK X5 板端（竞赛模式）
bash scripts/build_on_rdk.sh
sudo ./build/crane_vision --backend bpu --web-port 8080
```

后端选择 `--backend auto|onnx|bpu`（auto：RDK X5 → bpu，其他 → onnx）。
完整 CLI 见 `./build/crane_vision --help`，完整命令手册见 [docs/development.md](docs/development.md)。

## 📁 目录结构

```text
src/main.cpp          # 唯一入口（CLI）
tasks/                # 全部运行时模块
  app.*                    # 竞赛编排：采集×2 / 推理×2 / 任务合并 / 串口 / Web
  common/                  # types / logging / config（vision.yaml 读写 + on_change 绑定）
  io/
    camera.*               # ICamera / Camera(V4L2) / MockCamera / StaticImageCamera
    serial_port.*          # termios UART 921600 8N1 + 200 Hz 轮询/20 Hz 发送 + 重连
  logic/
    mapping.*              # 类别→协议编码映射
    task_generator.*       # 分区聚合 + 严格/补全两种结果帧生成
    logic_completion.*     # 置信度排列评分推断（缺区/重复补全）
  inference/
    yolo11.*               # letterbox / DFL / sigmoid / 逐类 NMS / 坐标回映 / NV12
    model.*                # Yolo11Model：预处理→后端→后处理
    onnx_backend.*         # ONNX Runtime 后端（融合输出 [1,4+nc,8400]）
    bpu_backend.*          # RDK X5 BPU 后端（6 头原始输出 + 反量化）
    bpu_stub.c             # 非 RDK 主机的 libDNN 桩（保证全平台可编译链接）
  web/                     # 内嵌 HTTP server + MJPEG 流 + REST + 全屏校准页
tests/                # test_logic / test_serial / test_golden_onnx（ctest 驱动）
scripts/              # setup_onnxruntime / build_wsl / build_on_rdk / wsl_run
tools/                # 外部依赖：onnxruntime（预编译包）/ rdk_dnn（vendored 头文件）
configs/vision.yaml   # 运行时唯一事实来源
assets/               # 模型：.onnx（x86）/ .bin+.yaml（BPU）/ .pt（参考）
docs/                 # 架构规范 + 开发手册
```

## 🖥️ Web 校准 UI

- `BEAN(1)` / `DIGIT(2)` 视图：MJPEG 实时画面 + 检测框（🟡 检测 / 🟠 推断）+ 分区辅助线 + 状态条（BEAN/DIGIT/SERIAL/START + 推理 fps）
- `SETTINGS(3)`：双相机曝光/自动曝光、双模型 conf_thres、逻辑推断上限滑块
- 滑块实时生效（REST `POST /api/config`），`SAVE` 写回 `configs/vision.yaml`
- 首次点击页面时请求浏览器全屏（配合板载触摸屏）

## 📐 竞赛语义（摘要）

- **分区**：检测框中心 `cx < line` 归左区，边界线归右侧区（豆子 ①–③ / 数字 ④–⑧）
- **逻辑补全**：全局排列评分（严格 `>`，字典序首个最优），推断上限 bean≤1 / digit≤2，超限拒发
- **Sticky result**：相机掉线不清空；连续 10 次生成失败才清空串口帧
- **串口**：收到 `A5 5A 01` 后 20 Hz 发送 `A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`；300 ms 无 START 停发；掉线自动重连

> 以上为行为规范的摘要，**完整定义以 [docs/architecture.md](docs/architecture.md) 为准**，改动必须回归 golden 测试。

## 🧪 测试

```bash
ctest --test-dir build --output-on-failure
```

| 测试 | 内容 |
|---|---|
| `logic` | 逻辑补全 / 任务生成（含边界线、缺区、重复、推断上限用例） |
| `serial` | 帧编解码/校验、START 解析（分包/突发）、socketpair 回环 20 Hz 发送 |
| `golden_onnx` | 静态图 detections 对拍内置参考值 + 端到端 payload |

## ⚡ 性能（Ryzen 9 7945HX，YOLO11n @ 640，含前后处理）

| 路径 | 单帧耗时 |
|---|---|
| C++ ONNX Runtime（WSL2） | ~35 ms（letterbox 1.5 + 推理 33.6 + 后处理 0.1） |
| Python onnxruntime（WSL2） | ~36 ms |
| Python Ultralytics .pt（Windows 原生 CPU） | ~28 ms |
| Python onnxruntime（Windows 原生） | ~20 ms |

> 同引擎下 C++ 与 Python 推理耗时一致；收益在启动速度（0.2 s 模型加载 vs 秒级 import）、
> 无 GIL 真并行、单二进制部署、板端 BPU C++ 直调。WSL2 数字受虚拟化影响偏高，仅供参考。

## 📚 文档

- [docs/architecture.md](docs/architecture.md) — 模块布局、双后端差异、线程模型、**竞赛语义规范**、关键陷阱
- [docs/development.md](docs/development.md) — 构建/测试/运行命令、WSL2 环境坑、RDK X5 部署、性能基线

## 📄 许可证

随本项目仓库分发；ONNX Runtime 遵循 MIT，模型文件遵循 Ultralytics AGPL。
