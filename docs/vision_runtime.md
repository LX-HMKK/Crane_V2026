# Crane 视觉模块运行时说明

本仓库 `models/` / `scripts/` / `configs/` 下的代码组成 2026 物流技术创意赛视觉识别模块。

## 启动方式

```bash
# 竞赛模式（看门狗会自动调用）
python3 scripts/main.py

# 校准 UI（调节辅助线、查看实时识别）
python3 scripts/main.py --ui

# 查看可用相机
python3 scripts/main.py --list-cameras

# 指定配置文件
python3 scripts/main.py --config /path/to/vision.yaml
```

## 配置字段

见 `configs/vision.yaml`，主要包含：

- `serial`：串口设备、波特率、读超时。
- `cameras.bean` / `cameras.digit`：两个 USB 相机的设备号、分辨率、缓冲区大小。
- `models.bean` / `models.digit`：`.bin` 模型路径与对应 `_rdkx5_config.yaml`。
- `inference`：`imgsz`、`conf_thres`、`iou_thres`、`pad`。
- `scan`：连续扫描帧数、最大重试次数、总超时。
- `lines`：豆子/数字辅助线位置，会被 UI 自动改写。
- `mapping.bean_to_digit`：黄豆→1、绿豆→2、白芸豆→3。

## 状态机

```text
INIT → WAIT_START → SCANNING → READY → SEND_TASK → DONE（回到 WAIT_START）
```

1. **INIT**：加载配置，初始化串口、两个 Camera、两个 BPU 模型，预热推理。
2. **WAIT_START**：循环读取串口，滑动窗口匹配 `AA 55 01 0D 0A`。
3. **SCANNING**：收到 START 后连续采集 `scan.frames` 帧，分别对豆子/数字相机做 BPU 推理。
4. **READY**：按辅助线把检测框归入区域，取每个区域置信度最高者，查表得到 3 个目标放置区。
5. **SEND_TASK**：发送固定 7 字节帧 `BB 66 p1 p2 p3 1B 1A`。
6. **DONE**：回到 WAIT_START，等待下一轮 START。

> **赛前准备阶段视觉保持串口静默**，只有收到 START 帧后才扫描并发送 TASK 帧。

## 看门狗

```bash
./scripts/watchdog.sh
```

`main.py` 退出后会自动在 2 秒后重启。

## 开发测试

```bash
# x86 全量单元测试（无需 hobot_dnn）
python -m pytest test/ -v
```

## 部署清单

1. 将 `models/`、`scripts/`、`configs/`、`test/` 与 `assets/*.bin`、`assets/*_rdkx5_config.yaml` 一起拷贝到 RDKx5。
2. 上电后先运行 `python3 scripts/main.py --ui` 校准辅助线。
3. 确认 `configs/vision.yaml` 已保存。
4. 使用 `./scripts/watchdog.sh` 启动竞赛模式。
5. 从下位机发送 START 帧，验证返回 TASK 帧。

## 记录

- 实现日期：2026-07-07
- 测试环境：Windows 11 x86（pytest）+ RDKx5（待实测）
