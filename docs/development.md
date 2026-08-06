# 开发手册：构建、测试、运行、部署

## 依赖

x86 Linux / WSL2：g++ ≥ 9、cmake ≥ 3.16、OpenCV 4（`libopencv-dev`）、yaml-cpp（`libyaml-cpp-dev`）、预编译 ONNX Runtime（脚本自动下载到 `tools/onnxruntime`）。

RDK X5 板端：系统镜像自带 libDNN（`/usr/lib/hbbpu/libdnn.so`）与 OpenCV。

## 构建与测试

```bash
# x86 Linux / WSL2
bash scripts/setup_onnxruntime.sh   # 仅首次：下载预编译 ORT 1.23.2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# 等价一键脚本（内部 cd 项目根）
bash scripts/build_wsl.sh

# 单个测试
ctest --test-dir build -R logic          # logic / serial / golden_onnx

# RDK X5 板端（装依赖 + 编译；自动检测 libDNN 决定真后端/桩）
bash scripts/build_on_rdk.sh
```

### 测试说明

| 测试 | 内容 |
|---|---|
| `logic` | 逻辑补全 / 任务生成（移植 Python 全部测试向量 + 边界线用例） |
| `serial` | 帧编解码/校验、START 解析（分包/突发）、socketpair 回环 20 Hz 发送 |
| `golden_onnx` | 静态图 detections 对拍 Python onnxruntime 参考值 + 端到端 payload |

START 门控集成测试（需要串口，用 pty 伪造 UART）：

```bash
python3 tests/test_start_gate.py     # 在 WSL 构建环境内运行，全程 ~40s
```

验证：无 START 不推理 → START 打开 → 停 START 关断（sticky 保留）→ DEBUG 强制 → DEBUG 关闭回门控。用 `/api/status` 的 `recognition` 字段作瞬时信号（fps 是 5s 速率窗口，不适合做门控断言）。

`golden_onnx` 的容差（坐标 1.5px、置信度 0.005）是为吸收 OpenCV 4.5 vs 5.0 的 resize 亚像素差异，不要随意收紧。

## 运行

```bash
# 竞赛模式（RDK X5：BPU 后端 + 双相机 + 串口 + Web UI）
sudo ./build/crane_vision --backend bpu --web-port 8080

# x86 全流程验证（mock 相机 + Web 校准 UI，无串口）
./build/crane_vision --mock-cameras --no-serial --web-port 8080
# 浏览器打开 http://<host>:8080
# 底部控制栏右侧 DEBUG 键：强制识别（绕过下位机 START），用于标定/调试；
# 状态栏 REC:ON/OFF 实时显示识别门控是否打开。

# 静态图片单次扫描（无需相机/串口，端到端冒烟）
./build/crane_vision --once --bean-image tests/data/beans_2.jpg \
                     --digit-image tests/data/num_2.jpg
# 期望输出：SCAN RESULT C1-C3=1 3 2 N4-N8=1 4 5 3 2
#           FRAME A5 5A 01 03 02 01 04 05 03 02 0D 0A

# 性能基准
./build/crane_vision --bench 30 --bean-image tests/data/beans_2.jpg \
                     --digit-image tests/data/num_2.jpg
```

完整 CLI 见 `./build/crane_vision --help`。后端选择 `--backend auto|onnx|bpu`（auto：RDK X5 → bpu，其他 → onnx）。非 RDK 主机上 BPU 后端链接桩实现，调用时清晰报错。

## WSL2 环境坑（本机实测）

从 Windows Git Bash 调 WSL 的固定姿势：

```bash
export MSYS2_ARG_CONV_EXCL='*'          # 否则 /mnt/d/... 被 MSYS 转成 D:/Program Files/Git/mnt/...
wsl -d Ubuntu-22.04 bash -c "..."       # 必须显式指定发行版！
bash scripts/wsl_run.sh <cmd>           # WSL 内跑二进制：cd 项目根 + 设 LD_LIBRARY_PATH
```

- **默认 WSL 发行版是 Ubuntu-18.04**（cmake 3.10，过不了 `cmake_minimum_required(3.16)`），裸 `wsl` 会进错环境。构建用的是 Ubuntu-22.04（gcc 11.4 / cmake 3.22 / OpenCV 4.5.4 / yaml-cpp 0.7）。
- WSL 继承 Windows 侧 cwd，且容易卡在子目录——WSL 命令一律用绝对路径。
- 无 sudo 可用（密码未知），不要尝试需要 root 的安装路径。
- WSL2 推理耗时约为 Windows 原生 2 倍（ORT ~35ms vs ~20ms），性能对比必须同环境背靠背，WSL2 数字仅供参考。

## RDK X5 部署

- 目标板：`root@192.168.128.10`，工作区 `/home/sunrise/Crane_WS`。
- 增量同步：只传变更文件，**绝不覆盖 `assets/*.bin`** 等大型资产。
- BPU 后端在 x86 只能编译 + 桩链接验证；NV12 输入与反量化路径是按头文件/文档盲写的，**首次上板必须用 `--backend bpu` 实测首帧输出是否合理**。
- vendored 头文件在 `tools/rdk_dnn/include/dnn/`，与板端 `/usr/include/dnn` 一致（来源 D-Robotics/hobot-dnn）。

## 性能参考（Ryzen 9 7945HX，YOLO11n @ 640，含前后处理）

| 路径 | 单帧耗时 |
|---|---|
| C++ ONNX Runtime（WSL2） | ~35 ms（letterbox 1.5 + 推理 33.6 + 后处理 0.1） |
| Python onnxruntime（WSL2） | ~36 ms |
| Python Ultralytics .pt（Windows 原生 CPU） | ~28 ms |
| Python onnxruntime（Windows 原生） | ~20 ms |

同引擎下 C++ 与 Python 推理耗时一致；收益在启动速度（0.2s 模型加载 vs 秒级 import）、无 GIL 真并行、单二进制部署、板端 BPU C++ 直调。
