# Crane 视觉模块

2026 物流技术创意赛 RDK X5 视觉识别模块：双 USB 相机 + 两路 YOLO11 BPU 模型 + 串口通信。

## 功能

- 豆子相机：识别取货区 ①–③ 的豆子种类（白芸豆 / 黄豆 / 绿豆）。
- 数字相机：识别放置区 ④–⑧ 货箱侧面的数字 1–5。
- 内部映射：黄豆→1、绿豆→2、白芸豆→3。
- 反向查找：确定数字 1/2/3 分别位于哪个放置区。
- 串口输出：收到下位机 START 帧 `AA 55 01 0D 0A` 后，发送 7 字节 TASK 帧 `BB 66 p1 p2 p3 1B 1A`。

## 目录说明

```text
configs/vision.yaml              # 运行时配置：串口、相机、模型、辅助线、映射
models/                          # 视觉模块源码
  __init__.py
  yolo11_postprocess.py          # letterbox / DFL 解码 / NMS / 坐标回映
  bpu_loader.py                  # hobot_dnn 封装，x86 上抛 EnvironmentError
  yolo11_model.py                # 共享 BPU 推理流程
  bean_model.py / digit_model.py # 豆子/数字模型薄封装
  camera.py                      # OpenCV 相机封装
  serial_port.py                 # START/TASK 串口帧
  mapping.py                     # 豆子类别 → 目标数字
  task_generator.py              # 辅助线分区 + 反向查找
  ui.py                          # tkinter 辅助线校准 UI
scripts/
  main.py                        # 竞赛入口（状态机 + --ui 调试模式）
  watchdog.sh                    # 开机自启动 / 崩溃重启
test/                            # x86 可测单元测试
```

## 快速开始

### 使用推荐虚拟环境

本项目在 `D:\conda_envs\crane-clean` 维护了一个干净的 Miniforge 环境（Python 3.11，USTC 镜像），用于避免 `opencv-python` / `opencv-python-headless` 等包冲突。所有 x86 命令均建议通过该环境执行：

```bash
conda run --prefix /d/conda_envs/crane-clean python ...
```

### 安装依赖

```bash
conda run --prefix /d/conda_envs/crane-clean pip install --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple -r requirements.txt
```

### x86 测试

```bash
conda run --prefix /d/conda_envs/crane-clean python -m pytest test/ -v
```

### 板端使用

1. 把 `models/`、`scripts/`、`configs/`、`test/` 与 `assets/*.bin`、`assets/*_rdkx5_config.yaml` 拷贝到 RDKx5。
2. 上电校准：
   ```bash
   python3 scripts/main.py
   ```
   > 在 x86 开发机上没有相机时，可运行 `conda run --prefix /d/conda_envs/crane-clean python scripts/main.py --mock-cameras` 预览校准 UI。
3. 启动看门狗竞赛模式：
   ```bash
   ./scripts/watchdog.sh
   ```
4. 下位机发送 START 帧，视觉返回 TASK 帧。

## 文档

- [docs/vision_runtime.md](docs/vision_runtime.md) — 运行时说明、状态机、配置字段
- [docs/vision_mapping.md](docs/vision_mapping.md) — 辅助线分区与映射规则
- [docs/RDKx5_Bin_Inference.md](docs/RDKx5_Bin_Inference.md) — YOLO11 `.bin` 板端调用细节

## 约束

- `hobot_dnn` 仅在 RDKx5 上可用；x86 侧通过 mock 做单元测试。
- 默认输入尺寸 640×640，letterbox pad 值 127。
- 串口参数：115200/8/N/1。

## 许可证

随本项目仓库一起分发。
