# RDKx5 BPU `.bin` 模型板端调用指南

本文档说明如何把 `hb_mapper` 编译出的 YOLO11 6 头 `.bin` 模型部署到 RDKx5 开发板上推理。`.bin` 的产生过程见 [RDKx5 导出工作流](RDKx5_Export_Workflow.md)。

## 1. 概述

- **模型**：YOLO11 6 头解耦输出（P3/P4/P5 各一组 bbox + cls），由 `export_rdkx5_onnx.py` 导出 ONNX，再经 `hb_mapper makertbin` 编译为 BPU `.bin`
- **输入**：`images`，`input_type_rt = nv12`、`input_type_train = rgb`、`input_layout_train = NCHW`，shape `1×3×640×640`
- **归一化**：`norm_type = data_scale`、`scale_value = 1/255`，已编译进模型，板端喂 uint8 `[0,255]` 即可，不再除 255
- **输出**：6 × FLOAT（3 个 bbox 头 `1×H×W×64` + 3 个 cls 头 `1×H×W×nc`）
- **后处理**：DFL 解码 + 还原 xyxy + sigmoid + NMS，由 `scripts/deployment/infer_rdkx5_yolo11.py` 完成

## 2. 前置条件

### 2.1 运行环境

- **仅限 RDKx5 板端**：脚本依赖 `hobot_dnn`，Windows / 普通 x86 开发机无法运行（见 CLAUDE.md 常见坑 9）
- Python3 + 依赖：
  ```bash
  pip3 install opencv-python pyyaml numpy
  # hobot_dnn 随 RDKx5 出厂镜像自带，无需单独安装
  ```

### 2.2 文件清单

把以下文件拷到板子上（建议放同一目录）：

| 文件 | 来源 | 作用 |
|------|------|------|
| `<name>_bayese_640x640_nv12.bin` | `assets/<exp>/rdkx5/` | BPU 编译后的模型 |
| `<name>_rdkx5_config.yaml` | `assets/<exp>/rdkx5/` | 类别名、nc、imgsz、reg_max、strides |
| `infer_rdkx5_yolo11.py` | `scripts/deployment/` | 板端推理 + 后处理脚本 |
| 测试图片 | 自备 | cv2 能读的常见格式 |

从开发机拷到板端示例：
```bash
scp assets/beans_v4/rdkx5/beans_v4_rdkx5_bayese_640x640_nv12.bin \
    assets/beans_v4/rdkx5/beans_v4_rdkx5_config.yaml \
    scripts/deployment/infer_rdkx5_yolo11.py \
    rdkx5:/home/root/beans_v4/
```

## 3. 配置文件说明

`<name>_rdkx5_config.yaml` 字段（以 `beans_v4` 为例）：

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

```bash
python3 infer_rdkx5_yolo11.py \
  --model beans_v4_rdkx5_bayese_640x640_nv12.bin \
  --cfg   beans_v4_rdkx5_config.yaml \
  --img   test.jpg \
  --conf  0.25 --iou 0.45
```

参数：

| 参数 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `--model` | 是 | — | `.bin` 模型路径 |
| `--img` | 是 | — | 测试图片路径 |
| `--cfg` | 否 | `<model>_config.yaml` | 配置 YAML；不传则把 `--model` 的 `.bin` 换成 `_config.yaml` |
| `--conf` | 否 | `0.25` | 置信度阈值 |
| `--iou` | 否 | `0.45` | NMS IoU 阈值 |

> **`--cfg` 命名坑**：默认推导会把 `beans_v4_rdkx5_bayese_640x640_nv12.bin` 解析成 `beans_v4_rdkx5_bayese_640x640_nv12_config.yaml`，而实际配置文件叫 `beans_v4_rdkx5_config.yaml`，**对不上**。建议始终显式传 `--cfg`。

## 5. 推理流程

`infer_rdkx5_yolo11.py` 的处理链：

1. **加载模型**：`dnn.load(model)` → 取 `models[0]`
2. **读图**：`cv2.imread`（BGR）
3. **Letterbox**：等比缩放到 640×640，空白处填 **127（灰）**，居中放置
4. **预处理**：BGR → RGB，保持 uint8 `[0,255]`，转 NCHW，加 batch 维
   - 归一化（÷255）已编译进模型（`data_scale`），故输入保持 uint8，不再缩放
5. **前向**：`m.forward(input_tensor)` → 6 个输出 tensor
6. **后处理**：`postprocess()` —— DFL 解码、还原 xyxy、sigmoid、NMS
7. **坐标回映**：letterbox 坐标还原回原图（减 `pad`、除 `scale`）
8. **输出**：打印每个检测 `类别: 置信度 at (x1,y1)-(x2,y2)`

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
4. **NMS**：`cv2.dnn.NMSBoxes` 按 `iou_thres` 去重
5. **回映原图**：`(coord - pad) / scale`

最终每行检测：`[x1, y1, x2, y2, conf, cls_id]`（原图像素坐标）。

## 7. 注意事项

1. **`hobot_dnn` 依赖**：仅 RDKx5 出厂镜像可用，开发机跑不了；上板前可在开发机用 ONNX 跑通后处理逻辑（用 ONNX Runtime 代替 `m.forward`）。
2. **NV12 输入的坑**：模型 `input_type_rt = nv12`，但当前脚本走的是 **RGB NCHW uint8** 路径（直接 `m.forward`），并未显式构造 NV12。脚本注释提示"真 NV12 模型应使用 hobot_dnn 的 helper"。若板端 `m.forward` 不接受 RGB uint8、或精度异常，需改用 `hobot_dnn` 的 NV12 / YUV420SP 输入接口（参考 RDK Model Zoo 的 yolov11 示例）。**上板后先单图验证精度再批量跑。**
3. **pad 值 = 127**：letterbox 空白填 127（灰），与导出时校准数据预处理保持一致；改预处理要同步改校准数据。
4. **`--cfg` 命名**：见第 4 节，默认推导名与实际文件名不一致，建议显式传 `--cfg`。
5. **类别数 `nc`**：脚本默认 `nc=5`（数字检测遗留）；豆子模型 `nc=3`，必须靠 config yaml 覆盖，否则 cls 解码维度错。
6. **不输出可视化**：当前脚本只 print 检测结果，不画框、不保存。需要可视化可自行加 `cv2.rectangle` + `cv2.imwrite`。

## 8. 常见问题

| 现象 | 排查 |
|------|------|
| `ModuleNotFoundError: hobot_dnn` | 不在 RDKx5 板端环境；该脚本不能在开发机跑 |
| `FileNotFoundError: ..._config.yaml` | `--cfg` 默认推导名与实际文件不符，显式传 `--cfg <path>` |
| 检测全无 / 置信度极低 | 检查 `nc` 是否匹配；检查 NV12 输入路径（注意事项 2）；检查 letterbox pad 值 |
| 坐标偏移 | 确认 `strides` / `reg_max` 与模型一致；确认坐标回映的 `scale` / `pad` 未被改动 |
| 类别错乱 | `names` dict 的 key 必须是 int（yaml 里 `0:` 会被解析为 int）；手改 yaml 注意别写成字符串 key |

## 9. 相关文档

- [RDKx5 导出工作流](RDKx5_Export_Workflow.md) —— `.pt` → ONNX → `.bin` 的完整流程与环境搭建
- `scripts/deployment/infer_rdkx5_yolo11.py` —— 本文描述的板端推理脚本
- CLAUDE.md 常见坑 9 —— `hobot_dnn` 板端依赖说明
