# Crane 视觉模块运行时说明

本仓库 `models/` / `scripts/` / `configs/` 下的代码组成 2026 物流技术创意赛视觉识别模块。

## 启动方式

```bash
# RDK X5 校准 UI（默认，调节辅助线、查看实时识别）
python3 scripts/main_arm64.py

# RDK X5 竞赛模式（看门狗会自动调用）
python3 scripts/main_arm64.py --no-ui

# 查看可用相机
python3 scripts/main_arm64.py --list-cameras

# 指定配置文件
python3 scripts/main_arm64.py --config /path/to/vision.yaml

# 无相机 UI 开发与测试（合成帧）
python3 scripts/main_arm64.py --mock-cameras
```

> `--ui` 仍是默认校准模式的向后兼容别名，与不带参数效果相同。`--mock-cameras` 使用合成帧替代真实相机；在 RDK X5 上仍会加载 BPU 模型并执行真实推理，仅相机帧为合成，常用于无真实相机时的算法验证。

## 配置字段

见 `configs/vision.yaml`，主要包含：

- `serial`：串口设备、波特率、读超时。
- `protocol`：START 命令超时（默认 300 ms）、结果帧发送间隔（默认 0.1 s，对应 10 Hz）、轮询间隔、扫描重试间隔。
- `cameras.bean` / `cameras.digit`：两个 USB 相机的设备号、分辨率、缓冲区大小。
- `models.bean` / `models.digit`：`.bin` 模型路径与对应 `_rdkx5_config.yaml`。
- `inference`：`imgsz`、`conf_thres`、`iou_thres`、`pad`。
- `scan`：连续扫描帧数、最大重试次数、总超时。
- `lines`：豆子/数字辅助线位置，会被 UI 自动改写。
- `mapping.bean_to_digit`：黄豆→1、绿豆→2、白芸豆→3（同时作为结果帧颜色编码）。

## 状态机

```text
INIT → WAIT_START → SCANNING → SENDING → SILENT → (resume) SENDING
```

1. **INIT**：加载配置，初始化串口、两个 Camera、两个 BPU 模型，预热推理。
2. **WAIT_START**：循环读取串口，检测下位机发送的 `A5 5A 01` START 命令。未收到前视觉端保持静默，不发送任何数据。
3. **SCANNING**：收到 START 后进入识别状态，连续采集 `scan.frames` 帧，分别对豆子/数字相机做 BPU 推理，通过 `TaskGenerator.generate_frame()` 生成 8 字节结果。
4. **SENDING**：获得完整合法结果后，以 `protocol.send_interval_sec` 间隔（默认 0.1 s，即 10 Hz）重复发送 12 字节结果帧 `A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`。
5. **SILENT**：若连续 300 ms 未收到新的 START 命令，停止发送，清空缓存结果，回到 WAIT_START 等待。

> **赛前准备阶段视觉保持串口静默**，只有收到 START 命令后才开始识别并回传结果帧。

## 看门狗

```bash
./scripts/watchdog.sh
```

`main_arm64.py --no-ui` 退出后会自动在 2 秒后重启。

## x86 开发与完整流程测试

x86 主机没有 `hobot_dnn`，但 `main_amd64.py` 可以通过 `ultralytics` 加载 `assets/*.pt` 进行真实 PyTorch 推理，验证相机 → 模型 → 任务生成 → 结果打印的完整数据流。`crane-clean` 环境已安装 GPU 版依赖：

```bash
# 全量单元测试（无需 hobot_dnn / ultralytics）
conda run --prefix /d/conda_envs/crane-clean python -m pytest test/ -v

# 交互式预览：按 SPACE / s 扫描，ESC 退出（需要 ultralytics）
conda run --prefix /d/conda_envs/crane-clean python scripts/main_amd64.py --mock-cameras

# 单次扫描并退出
conda run --prefix /d/conda_envs/crane-clean python scripts/main_amd64.py --once --mock-cameras
```

在 `--mock-cameras` 且没有 `ultralytics` 的环境下，`main_amd64.py` 会进入 camera-only 模式，扫描后输出 `SCAN FAILED: task generator unavailable`，证明数据链路已打通。

## 部署清单

1. 将 `models/`、`scripts/`、`configs/`、`test/` 与 `assets/*.bin`、`assets/*_rdkx5_config.yaml` 一起拷贝到 RDKx5。
2. 上电后先运行 `python3 scripts/main_arm64.py` 校准辅助线。
3. 确认 `configs/vision.yaml` 已保存。
4. 使用 `./scripts/watchdog.sh` 启动竞赛模式。
5. 从下位机循环发送 START 命令 `A5 5A 01`，验证返回 12 字节结果帧。

## 记录

- 实现日期：2026-07-07
- 协议更新日期：2026-07-11
- 测试环境：Windows 11 x86（pytest）+ RDKx5（待实测）
