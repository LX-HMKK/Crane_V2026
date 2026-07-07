# 2026 物流技术创意赛视觉模块设计文档

> **日期**：2026-07-07  
> **主题**：crane 仓库全项目视觉模块设计  
> **状态**：已确认，待实施

---

## 1. 项目目标

为 RDK X5 开发一套双相机视觉识别模块，完成以下竞赛任务：

- **豆子相机**：识别取货区 ①–③ 号位的豆子种类（white / green / yellow）。
- **数字相机**：识别放置区 ④–⑧ 号位货箱侧面的数字 1–5。
- **内部映射**：黄豆→1、绿豆→2、白芸豆→3。
- **反向查找**：确定数字 1/2/3 分别出现在 ④–⑧ 的哪个位置。
- **串口输出**：收到下位机 START 帧后，发送 7 字节 TASK 帧 `BB 66 p1 p2 p3 1B 1A`。

---

## 2. 设计约束

- 现有模型均为 **YOLO11 6-head**，已通过 `hb_mapper` 编译为 `.bin`：
  - `assets/num.bin`：识别数字 1–5（`num_rdkx5_config.yaml`）
  - `assets/beans_v4.bin`：识别白/绿/黄豆（`beans_v4_rdkx5_config.yaml`）
- `tools/yolov5/` 是 vendored 代码；`tools/yolov5/export_rdkx5_onnx.py` 是 YOLOv5 遗留脚本，与本次开发无关，不改动。
- 板端运行依赖 `hobot_dnn`，仅在 RDKx5 上可用；x86 侧通过 mock 做单元测试。
- `models/`、`scripts/`、`configs/` 是用户已定好的框架，本次开发严格复用。

---

## 3. 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│  RDK X5 板端                                                │
│                                                             │
│  scripts/main.py  (状态机 + 竞赛主循环)                     │
│        │                                                    │
│        ▼                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ models/      │  │ models/      │  │ models/         │   │
│  │ serial_port  │  │ camera       │  │ bpu_loader      │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│        │                    │                    │          │
│        ▼                    ▼                    ▼          │
│  下位机 START/TASK 帧    双 USB 相机取流      两路 BPU 推理 │
│                                                             │
│        │                    │                    │          │
│        └────────────────────┴────────────────────┘          │
│                              ▼                              │
│              models/yolo11_postprocess.py                   │
│                              ▼                              │
│              models/task_generator.py                       │
│                              ▼                              │
│                        TASK 帧发送                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 目录布局

```text
crane/
├── assets/                          # .pt/.onnx/.bin + *_rdkx5_config.yaml
│   ├── num_rdkx5_config.yaml
│   ├── beans_v4_rdkx5_config.yaml
│   └── ...
├── configs/
│   └── vision.yaml                  # 运行时配置：串口、相机、辅助线、阈值
├── models/                          # 视觉模块源码
│   ├── __init__.py
│   ├── yolo11_postprocess.py        # letterbox / DFL / NMS / 坐标回映
│   ├── bpu_loader.py                # hobot_dnn 封装
│   ├── bean_model.py                # 豆子模型推理封装
│   ├── digit_model.py               # 数字模型推理封装
│   ├── camera.py                    # 双相机取流封装（合并为一个文件）
│   ├── serial_port.py               # START/TASK 串口帧
│   ├── mapping.py                   # 豆子类别 → 目标数字
│   ├── task_generator.py            # 反向查找 + 输出放置区编号
│   └── ui.py                        # OpenCV 校准/调试 UI
├── scripts/                         # 入口 + 看门狗
│   ├── main.py                      # 竞赛主程序（支持 --ui 调试模式）
│   └── watchdog.sh                  # 开机自启动 + 崩溃重启
├── test/                            # 测试用例
│   ├── test_serial.py
│   ├── test_camera.py
│   └── test_model.py
├── docs/
│   ├── RDKx5_Bin_Inference.md       # 修正 YOLOv5 遗留描述
│   ├── vision_runtime.md            # 新增：运行时说明
│   ├── vision_mapping.md            # 新增：辅助线与映射规则
│   └── README.md                    # 更新
└── tools/yolov5/                    # vendored，不动
```

---

## 5. 关键模块接口

### 5.1 `models/yolo11_postprocess.py`

- `letterbox(img, imgsz=640, pad=127)` → `(tensor NCHW uint8, scale, pad)`
- `postprocess(outputs, cfg)` → `[[x1, y1, x2, y2, conf, cls_id], ...]`
  - `outputs`：6 个 numpy 数组，顺序为 `[s_bbox, m_bbox, l_bbox, s_cls, m_cls, l_cls]`
  - `cfg`：包含 `imgsz`、`nc`、`reg_max`、`strides`、`conf_thres`、`iou_thres`
- `scale_coords(dets, scale, pad)` → 原图坐标

### 5.2 `models/bpu_loader.py::BpuModel`

- `__init__(bin_path, cfg_path)`
- `infer(rgb_nchw_uint8)` → 6 个 numpy 数组
- 若 `hobot_dnn` 不存在，抛出 `EnvironmentError`，便于 x86 mock 测试。

### 5.3 `models/camera.py::Camera`

- `__init__(device_index, width, height, buffer_size=1)`
- `capture()` → `(success, bgr_image)`
- `release()`

### 5.4 `models/serial_port.py::SerialPort`

- `wait_start(timeout=None)` → `bool`：滑动窗口匹配 `AA 55 01 0D 0A`
- `send_task(p1, p2, p3)`：发送 `BB 66 p1 p2 p3 1B 1A`

### 5.5 `models/task_generator.py`

- `generate_task(bean_detections, digit_detections, mapping, line_cfg)` → `(p1, p2, p3)` 或 `None`
- 输入检测框先按辅助线归入区域，再取各区域置信度最高者。
- 若任一目标区域缺失，返回 `None` 并附带缺失原因，由主程序决定重试或报错。

### 5.6 `models/ui.py`

- `CalibrationUI(cfg_path, camera, bean_model, digit_model)`
- 按键 `1`/`2` 切换两个相机视角。
- 显示实时识别框、类别、置信度。
- 用鼠标或 Trackbar 拖动辅助线，自动写回 `configs/vision.yaml`。

---

## 6. 辅助线规则

辅助线位置保存在 `configs/vision.yaml` 中。

### 6.1 豆子相机（取货位 ①–③）

两条竖线 `bean_left_x`、`bean_right_x` 将画面分成三列：

| 中心 x 范围 | 取货位 |
|-------------|--------|
| `cx < left` | ① |
| `left <= cx < right` | ② |
| `cx >= right` | ③ |

### 6.2 数字相机（放置区 ④–⑧）

- 横线 `digit_horizontal_y` 分隔前排（4、8）和后排（5、6、7）。
- 后排两条竖线 `digit_left_x`、`digit_right_x` 分隔 5/6/7。
- 前排用两条竖线的中点分隔 4/8。

| 位置 | 中心条件 | 放置区 |
|------|----------|--------|
| 前排 | `cy < horizontal_y` 且 `cx < midpoint` | ④ |
| 前排 | `cy < horizontal_y` 且 `cx >= midpoint` | ⑧ |
| 后排 | `cy >= horizontal_y` 且 `cx < left_x` | ⑤ |
| 后排 | `cy >= horizontal_y` 且 `left_x <= cx < right_x` | ⑥ |
| 后排 | `cy >= horizontal_y` 且 `cx >= right_x` | ⑦ |

> 若实际画面方向相反（前排在下），用户只需在 UI 中调整 `horizontal_y`，代码不做方向硬编码。

---

## 7. 状态机

```text
[INIT]
  加载 configs/vision.yaml
  初始化串口、两个 Camera、两个 BpuModel
  模型预热（跑若干空帧或真实帧）
  串口静默
    │
    ▼
[WAIT_START]
  循环读取串口，滑动窗口匹配 AA 55 01 0D 0A
  匹配成功 → 进入 SCANNING
    │
    ▼
[SCANNING]
  连续采集 N 帧（默认 5，可配置）
  每帧并行：豆子模型推理 + 数字模型推理
  后处理并归区域，记录各区域最高置信度结果
  全部完成 → 进入 READY
    │
    ▼
[READY]
  豆子类别 → 目标数字（white→3, green→2, yellow→1）
  反向查找目标数字所在放置区
  若任一结果缺失 → 报错/重试，不发送 TASK
  否则 → 进入 SEND_TASK
    │
    ▼
[SEND_TASK]
  构造并发送 BB 66 p1 p2 p3 1B 1A
  日志记录
    │
    ▼
[DONE]
  串口静默，等待比赛结束或下一次 START（根据配置）
```

---

## 8. UI 设计

- 框架：**OpenCV 原生 GUI**（`cv2.imshow` + 鼠标/Trackbar）。
- 启动：`python scripts/main.py --ui`
- 功能：
  1. 按键 `1` / `2` 切换两个相机视角。
  2. 实时显示识别框、类别、置信度、区域编号。
  3. 在当前视角上绘制辅助线。
  4. 鼠标拖动或 Trackbar 调整辅助线，松开后自动保存到 `configs/vision.yaml`。
- 竞赛模式：`python scripts/main.py`（无窗口）。

---

## 9. 未识别增强策略

1. 连续采集 `scan_frames` 帧（默认 5）。
2. 每帧分别对豆子相机和数字相机做推理。
3. 按辅助线把每个检测框归入对应区域。
4. 每个区域保留**置信度最高**的检测结果。
5. 扫描结束后：
   - 若 3 个取货位和数字 1/2/3 所在放置区全部存在 → 生成 TASK。
   - 任一区域缺失 → 记录日志，主程序重试 `max_retries` 次；仍失败则进入错误状态，不发送非法 TASK 帧。

---

## 10. 实施阶段

| 阶段 | 交付物 | 验证 |
|------|--------|------|
| 0. 仓库骨架 | `configs/vision.yaml`、空 `test/`、`.gitignore` 更新 | 文件存在、可解析 |
| 1. 后处理库 | `models/yolo11_postprocess.py` | `test/test_postprocess.py` 通过 |
| 2. 板端基础 | `models/bpu_loader.py`、`models/camera.py`、`models/serial_port.py` | mock 测试通过 |
| 3. 业务逻辑 | `models/mapping.py`、`models/task_generator.py` | 合成输入输出正确 |
| 4. 模型封装 | `models/bean_model.py`、`models/digit_model.py` | 依赖注入可测试 |
| 5. UI | `models/ui.py` | 可在 x86 上显示模拟画面、调节线并保存 |
| 6. 主程序 | `scripts/main.py` | mock 状态机端到端通过 |
| 7. 看门狗 | `scripts/watchdog.sh` | 可手动启动、kill 后自动重启 |
| 8. 文档 | README、vision_runtime.md、vision_mapping.md | 命令可拷贝执行 |
| 9. 板上实测 | 部署到 RDKx5 | START→TASK 正确、延迟 <30s |

---

## 11. 风险与应对

| 风险 | 应对 |
|------|------|
| `hobot_dnn` 在 x86 上不可用 | 所有板端模块支持 mock 注入；x86 只做逻辑测试，板端做集成测试。 |
| 相机枚举顺序漂移 | `configs/vision.yaml` 记录 `bean_device` / `digit_device`；提供 `scripts/main.py --list-cameras`。 |
| pad 值不一致 | 统一 `pad=127`，运行时断言。 |
| 数字 4/5 为空箱，可能误识别 | 只关心数字 1–3 的位置；4/5 识别结果忽略但可显示。 |
| 扫描超时 | 模型预热 + 并行推理 + 超时日志；`scan_frames`、`max_retries` 可配置。 |
| 辅助线位置随场地变化 | UI 校准 + 自动保存 YAML，比赛前可快速重调。 |

---

## 12. 文档清单

- `docs/RDKx5_Bin_Inference.md`：修正文中“由 `export_rdkx5_onnx.py` 导出”等 YOLOv5 遗留描述，改为 YOLO11 导出说明。
- `docs/vision_runtime.md`：新增，描述 `scripts/main.py` 用法、配置字段、状态机、日志。
- `docs/vision_mapping.md`：新增，描述辅助线规则、豆子→数字→放置区映射、缺失处理。
- `README.md`：更新快速开始、目录说明、部署清单。

---

**下一步**：调用 `writing-plans` 生成详细实施计划并进入编码阶段。
