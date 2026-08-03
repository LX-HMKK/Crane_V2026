# RDKx5 BPU `.bin` 模型板端调用指南

本文档说明如何把 `hb_mapper` 编译出的 YOLO11 6 头 `.bin` 模型部署到 RDKx5 开发板上推理。`.bin` 的产生过程见仓库根目录 `CLAUDE.md` 的“训练检测器 / 导出 ONNX”以及 `tools/yolov5/` 下的训练导出脚本。

## 1. 概述

- **模型**：YOLO11 6 头解耦输出（P3/P4/P5 各一组 bbox + cls）。本文档假设你已经拥有 YOLO11 6-head ONNX 模型，并已完成 `hb_mapper makertbin` 编译为 BPU `.bin`。
- **输入**：`images`，`input_type_rt = nv12`、`input_type_train = rgb`、`input_layout_train = NCHW`，shape `1×3×640×640`
- **归一化**：`norm_type = data_scale`、`scale_value = 1/255`，已编译进模型，板端喂 uint8 `[0,255]` 即可，不再除 255
- **输出**：6 × FLOAT（3 个 bbox 头 `1×H×W×64` + 3 个 cls 头 `1×H×W×nc`）
- **后处理**：DFL 解码 + 还原 xyxy + sigmoid + NMS，由 `scripts/main_arm64.py` 在板端调用

## 2. 前置条件

### 2.1 运行环境

- **仅限 RDKx5 板端**：脚本依赖 `hobot_dnn`，Windows / 普通 x86 开发机无法运行（见 CLAUDE.md 常见坑 9）
- Python3 + 依赖：
  ```bash
  pip3 install opencv-python pyyaml numpy pyserial Pillow
  # hobot_dnn 随 RDKx5 出厂镜像自带，无需单独安装
  ```

### 2.2 文件清单

把以下文件拷到板子上（建议放同一目录）：

| 文件 | 来源 | 作用 |
|------|------|------|
| `tools/99-crane-cameras.rules` | 本项目 | USB 摄像头 udev 绑定规则（复制到 `/etc/udev/rules.d/`） |
| `assets/beans_v6.bin` | 本项目 | 豆子成熟度 BPU 模型 |
| `assets/num_v6.bin` | 本项目 | 数字检测 BPU 模型 |
| `assets/beans.yaml` | 本项目 | 豆子模型配置（nc、names、imgsz、reg_max、strides） |
| `assets/num.yaml` | 本项目 | 数字模型配置 |
| `models/` | 本项目 | 视觉运行时源码包 |
| `scripts/main_arm64.py` | 本项目 | 板端竞赛入口 / 校准 UI |
| `scripts/watchdog.sh` | 本项目 | 自启动 / 崩溃重启 |
| `configs/vision.yaml` | 本项目（上板后可编辑） | 串口、相机、辅助线、映射配置 |

从开发机拷到板端示例：
```bash
scp -r models/ scripts/ configs/ tools/99-crane-cameras.rules \
    assets/beans_v6.bin assets/num_v6.bin \
    assets/beans.yaml assets/num.yaml \
    rdkx5:/home/root/crane/
```

> 板端单图预览/校准请使用 `python3 scripts/main_arm64.py`（默认即校准 UI），无需额外单图推理脚本。

## 3. 配置文件说明

`<name>.yaml` 部署配置字段（以 `beans.yaml` 为例）：

```yaml
imgsz: 640              # 输入尺寸（int 或 [h,w]）
nc: 3                   # 类别数
names:                  # 类别 id -> 名称
  0: white
  1: green
  2: yellow
reg_max: 16             # DFL 回归分箱数
strides: [8, 16, 32]    # P3/P4/P5 步长
task: detect
```

脚本读取的字段：`names`、`nc`、`imgsz`、`reg_max`、`strides`，均有默认值（`nc=5 / imgsz=640 / reg_max=16 / strides=[8,16,32]`）。`model`、`onnx`、`note` 等字段板端推理时不使用。

## 4. 命令行调用

校准 UI（默认）：

```bash
python3 scripts/main_arm64.py
```

也可显式使用向后兼容的 `--ui` 别名：

```bash
python3 scripts/main_arm64.py --ui
```

在 UI 中按 `1` / `2` 切换豆子/数字相机，按 ESC 退出。辅助线位置会自动保存到 `configs/vision.yaml`。

指定配置文件：

```bash
python3 scripts/main_arm64.py --config /path/to/vision.yaml
```

竞赛模式由看门狗调用，无需手动执行：

```bash
./scripts/watchdog.sh
```

### 4.4 不接串口测试 main

当前未接 STM32 串口时，仍可用以下方式验证 `main_arm64.py` 在板端能正常加载模型、读取双摄像头并进入主循环。

#### 带 UI 的校准预览（默认即 UI，会打开串口）

```bash
ssh root@192.168.128.10
cd /home/sunrise/Crane_WS
python3 scripts/main_arm64.py
```

按 `1` / `2` 切换豆子/数字相机，按 `ESC` 退出；辅助线位置自动保存到 `configs/vision.yaml`。

> 需要 RDK X5 接显示器，或 SSH 带 X11 转发（Windows 先开 X server，再用 `ssh -X root@192.168.128.10`）。
> 若未接 STM32 串口，可临时注释掉 `configs/vision.yaml` 中的 `serial` 段，或将其 `port` 指向 PTY。

#### 无 UI 竞赛主循环（用 PTY 替代串口）

```bash
# 1. 创建虚拟串口
ssh root@192.168.128.10
python3 - <<'PY'
import os, pty, signal, sys, time
def cleanup(s, f): sys.exit(0)
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)
master, slave = pty.openpty()
slave_path = os.ttyname(slave)
link = '/tmp/crane_ttyS1'
if os.path.islink(link) or os.path.exists(link):
    os.remove(link)
os.symlink(slave_path, link)
print(f'PTY {link} -> {slave_path}', flush=True)
os.close(slave)
while True: time.sleep(1)
PY

# 2. 另开窗口运行 main
ssh root@192.168.128.10
cd /home/sunrise/Crane_WS
sed 's|port: /dev/ttyS1|port: /tmp/crane_ttyS1|' configs/vision.yaml > /tmp/crane_vision.yaml
python3 scripts/main_arm64.py --no-ui --config /tmp/crane_vision.yaml
```

> 测试结束后杀掉 PTY 进程并删除 `/tmp/crane_ttyS1`。

## 5. 推理流程

`scripts/main_arm64.py` 内部的处理链：

1. **加载模型**：`from hobot_dnn import pyeasy_dnn as dnn`，然后 `dnn.load(model)` → 取 `models[0]`
2. **读图**：`cv2.imread`（BGR）
3. **Letterbox**：等比缩放到 640×640，空白处填 **127（灰）**，居中放置
4. **预处理**：BGR → RGB，保持 uint8 `[0,255]`，转 NCHW，加 batch 维
   - 归一化（÷255）已编译进模型（`data_scale`），故输入保持 uint8，不再缩放
5. **前向**：由于 BPU 模型 `tensor_type` 为 **NV12**，`models/bpu_loader.py` 会把 RGB NCHW 张量转换为 flat NV12 缓冲区（`H×W×1.5` 字节）后再调用 `m.forward(...)`
6. **输出解包**：`pyeasy_dnn` 返回 6 个 `PyDNNTensor` 对象，代码解包 `.buffer` 得到 numpy 数组
7. **后处理**：`postprocess()` —— DFL 解码、还原 xyxy、sigmoid、NMS
8. **坐标回映**：letterbox 坐标还原回原图（减 `pad`、除 `scale`）
9. **输出**：打印每个检测 `类别: 置信度 at (x1,y1)-(x2,y2)`

## 6. 输出格式与后处理

### 6.1 模型 6 个输出

| 序号 | 名称 | shape（imgsz=640） | 含义 |
|------|------|-------|------|
| 0 | s_bbox | `1×80×80×64` | P3 (stride 8) bbox，64 = 4×reg_max(16) |
| 1 | m_bbox | `1×40×40×64` | P4 (stride 16) bbox |
| 2 | l_bbox | `1×20×20×64` | P5 (stride 32) bbox |
| 3 | s_cls | `1×80×80×nc` | P3 类别分数 |
| 4 | m_cls | `1×40×40×nc` | P4 类别分数 |
| 5 | l_cls | `1×20×20×nc` | P5 类别分数 |

> H×W 随 `imgsz` 变化；640 时为 80 / 40 / 20。`outputs[:3]` 是 bbox，`outputs[3:]` 是 cls。

### 6.2 后处理步骤

1. **DFL 解码**：`bbox (...,64)` → reshape `(...,4,16)` → softmax → 与 `[0..15]` 加权求和 → `(...,4)` 回归值
2. **还原 xyxy**：YOLO11 回归输出到网格中心的 `l,t,r,b` 距离，结合网格中心 `(x+0.5, y+0.5)` × stride 还原到输入尺度 xyxy
3. **类别分数**：cls 做 sigmoid，最大值为 `conf`、argmax 为 `cls_id`，按 `conf_thres` 过滤
4. **NMS**：按类别分组执行 `cv2.dnn.NMSBoxes`，仅抑制同一类内的重叠框
5. **回映原图**：`(coord - pad) / scale`

最终每行检测：`[x1, y1, x2, y2, conf, cls_id]`（原图像素坐标）。

## 7. 注意事项

1. **`hobot_dnn` 依赖**：仅 RDKx5 出厂镜像可用，开发机跑不了；上板前可在开发机用 ONNX 跑通后处理逻辑（用 ONNX Runtime 代替 `m.forward`）。
2. **NV12 输入的坑**：模型 `input_type_rt = nv12`，但当前脚本走的是 **RGB NCHW uint8** 路径（直接 `m.forward`），并未显式构造 NV12。脚本注释提示"真 NV12 模型应使用 hobot_dnn 的 helper"。若板端 `m.forward` 不接受 RGB uint8、或精度异常，需改用 `hobot_dnn` 的 NV12 / YUV420SP 输入接口（参考 RDK Model Zoo 的 yolov11 示例）。**上板后先单图验证精度再批量跑。**
3. **pad 值 = 127**：letterbox 空白填 127（灰），与导出时校准数据预处理保持一致；改预处理要同步改校准数据。
4. **`--config` 命名**：默认使用 `configs/vision.yaml`；若自定义路径请显式传 `--config <path>`。
5. **类别数 `nc`**：脚本读取 `_rdkx5_config.yaml` 中的 `nc`；豆子模型 `nc=3`、数字模型 `nc=10`，必须靠 config yaml 覆盖，否则 cls 解码维度错。
6. **不输出可视化**：当前脚本只 print 检测结果，不画框、不保存。需要可视化可自行加 `cv2.rectangle` + `cv2.imwrite`。

## 8. 常见问题

| 现象 | 排查 |
|------|------|
| `ModuleNotFoundError: hobot_dnn` | 不在 RDKx5 板端环境；该脚本不能在开发机跑 |
| `FileNotFoundError: ..._config.yaml` | `--config` 默认读取 `configs/vision.yaml` 中指定的 `_rdkx5_config.yaml`；检查文件是否已拷到板端 |
| 检测全无 / 置信度极低 | 检查 `nc` 是否匹配；检查 NV12 输入路径（注意事项 2）；检查 letterbox pad 值 |
| 坐标偏移 | 确认 `strides` / `reg_max` 与模型一致；确认坐标回映的 `scale` / `pad` 未被改动 |
| 类别错乱 | `names` dict 的 key 必须是 int（yaml 里 `0:` 会被解析为 int）；手改 yaml 注意别写成字符串 key |

## 9. 相关文档

- `CLAUDE.md` —— 仓库根目录的常用命令与项目结构说明
- `tools/yolov5/` —— 训练与 ONNX 导出脚本
- `scripts/main_arm64.py` —— 本文描述的板端竞赛入口/校准 UI
- CLAUDE.md 常见坑 9 —— `hobot_dnn` 板端依赖说明
