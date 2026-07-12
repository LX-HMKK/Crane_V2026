# 2026 物流技术创意赛 — 视觉识别模块开发需求文档

> **注意**：本文档描述的 START 帧 / TASK 帧格式（`AA 55 01 0D 0A`、`BB 66 ... 1B 1A`）及输出语义已更新。当前实现以 `docs/reference/CRANE_VISION_PROTOCOL.md` 为准：START 命令为 `A5 5A 01`，视觉端发送 12 字节结果帧 `A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`。下文保留仅供参考，但不应作为实现依据。

> **目标**：RDK X5 + 双相机 + 已转 BPU 模型，识别豆子/数字，通过串口把识别结果发给 C 语言下位机。  
> **版本**：v6.0（上下位机使用不同帧头帧尾）

---

## 1. 硬件与模型

| 资产 | 说明 |
|------|------|
| **主控** | RDK X5（地平线 BPU 推理） |
| **豆子相机** | 对准取货区 ①~③ 号位，识别箱内豆子种类 |
| **数字相机** | 对准放置区 ④~⑧ 号位，识别货箱侧面数字 |
| **豆子模型** | 已转 BPU，识别 **白芸豆 / 黄豆 / 绿豆** |
| **数字模型** | 已转 BPU，识别数字 **1~5** |
| **通信** | `pyserial` 串口 → C 语言下位机 |

---

## 2. 视觉任务

### 2.1 取货区（①~③号位）— 豆子相机
- 3 个货箱，分别装三种豆子（赛前抽签决定哪个位置装哪种）。
- **货箱无数字编号**，靠豆子相机识别箱内豆子种类。

### 2.2 放置区（④~⑧号位）— 数字相机
- 5 个空货箱，贴有数字 1~5（赛前抽签决定摆放顺序）。
- **固定映射规则**：
  - 数字 **1** → 接收 **黄豆**
  - 数字 **2** → 接收 **绿豆**
  - 数字 **3** → 接收 **白芸豆**
  - 数字 **4、5** → 空货箱

### 2.3 视觉内部逻辑

```
豆子相机识别 ①~③ 号位豆子种类
       ↓
查表：黄豆→1、绿豆→2、白芸豆→3
       ↓
数字相机识别 ④~⑧ 号位货箱上的数字 1~5
       ↓
反向查找：数字 1 在 ④~⑧ 的哪个位置？数字 2 在哪个？数字 3 在哪个？
       ↓
输出 3 个放置区编号，分别对应 1/2/3 号取货位该去的位置
```

**示例**：
- 1 号取货位是黄豆 → 目标数字 1 → 在 5 号放置区 → 输出 **5**
- 2 号取货位是绿豆 → 目标数字 2 → 在 6 号放置区 → 输出 **6**
- 3 号取货位是白芸豆 → 目标数字 3 → 在 4 号放置区 → 输出 **4**

---

## 3. 串口通信协议

### 3.1 物理层

| 参数 | 值 |
|------|-----|
| 波特率 | 115200 |
| 数据位 | 8 |
| 停止位 | 1 |
| 校验 | 无 |
| 流控 | 无 |

### 3.2 帧格式

两帧均为**固定长度**，靠**不同帧头帧尾**区分方向。

| 方向 | 帧头 | 数据域 | 帧尾 | 总长度 |
|------|------|--------|------|--------|
| 下位机 → 视觉 | `AA 55` | 1 字节 | `0D 0A` | **5 字节** |
| 视觉 → 下位机 | `BB 66` | 3 字节 | `1B 1A` | **7 字节** |

> 数据域为放置区编号（4~8），不会出现帧头帧尾字节，固定长度即可可靠解析。

#### 3.2.1 START 帧（下位机 → 视觉）

通知视觉开始扫描。

| 字节 | 内容 | 值 |
|------|------|-----|
| 0 | HEAD0 | `0xAA` |
| 1 | HEAD1 | `0x55` |
| 2 | CMD | `0x01` |
| 3 | TAIL0 | `0x0D` |
| 4 | TAIL1 | `0x0A` |

**C 语言结构体**（下位机发送）：

```c
#pragma pack(1)
typedef struct {
    uint8_t head[2];   /* 0xAA, 0x55 */
    uint8_t cmd;       /* 0x01 */
    uint8_t tail[2];   /* 0x0D, 0x0A */
} StartFrame;

#define START_FRAME_SIZE    5
#define START_HEAD0         0xAA
#define START_HEAD1         0x55
#define START_TAIL0         0x0D
#define START_TAIL1         0x0A
```

**Python 接收**（视觉端）：

```python
START_FRAME = bytes([0xAA, 0x55, 0x01, 0x0D, 0x0A])

def wait_start(ser):
    buf = bytearray()
    while True:
        buf += ser.read(1)
        if len(buf) >= 5 and buf[-5:] == START_FRAME:
            return True
        if len(buf) > 10:
            buf.pop(0)
```

#### 3.2.2 TASK 帧（视觉 → 下位机）

视觉完成扫描后，一次性发送 3 个放置区编号。

| 字节 | 内容 | 值 | 说明 |
|------|------|-----|------|
| 0 | HEAD0 | `0xBB` | |
| 1 | HEAD1 | `0x66` | |
| 2 | PICK1 | 4~8 | 1 号取货位 → 放置区编号 |
| 3 | PICK2 | 4~8 | 2 号取货位 → 放置区编号 |
| 4 | PICK3 | 4~8 | 3 号取货位 → 放置区编号 |
| 5 | TAIL0 | `0x1B` | |
| 6 | TAIL1 | `0x1A` | |

**C 语言结构体**（下位机接收）：

```c
#pragma pack(1)
typedef struct {
    uint8_t head[2];   /* 0xBB, 0x66 */
    uint8_t pick1;     /* 1号取货位目标放置区: 4~8 */
    uint8_t pick2;     /* 2号取货位目标放置区: 4~8 */
    uint8_t pick3;     /* 3号取货位目标放置区: 4~8 */
    uint8_t tail[2];   /* 0x1B, 0x1A */
} TaskFrame;

#define TASK_FRAME_SIZE     7
#define TASK_HEAD0          0xBB
#define TASK_HEAD1          0x66
#define TASK_TAIL0          0x1B
#define TASK_TAIL1          0x1A
```

**C 语言接收示例**（下位机串口中断）：

```c
static uint8_t rx_buf[7];
static uint8_t rx_idx = 0;

void USART_IRQHandler(void) {
    uint8_t b = USART_DR;
    rx_buf[rx_idx++] = b;

    if (rx_idx >= 2 && (rx_buf[0] != TASK_HEAD0 || rx_buf[1] != TASK_HEAD1)) {
        rx_buf[0] = rx_buf[1];
        rx_idx = 1;  /* 帧头不对，滑动 */
    }

    if (rx_idx >= TASK_FRAME_SIZE) {
        if (rx_buf[5] == TASK_TAIL0 && rx_buf[6] == TASK_TAIL1) {
            uint8_t p1 = rx_buf[2];  /* 1号→p1 */
            uint8_t p2 = rx_buf[3];  /* 2号→p2 */
            uint8_t p3 = rx_buf[4];  /* 3号→p3 */
            /* 执行搬运 */
        }
        rx_idx = 0;
    }
}
```

**Python 发送**（视觉端）：

```python
import struct

def send_task(ser, pick1_place, pick2_place, pick3_place):
    frame = struct.pack(
        '>2B3B2B',
        0xBB, 0x66,           /* head */
        pick1_place, pick2_place, pick3_place,  /* data */
        0x1B, 0x1A            /* tail */
    )
    ser.write(frame)

# 示例：1号→5, 2号→6, 3号→4
# send_task(ser, 5, 6, 4)
# 串口发送: BB 66 05 06 04 1B 1A
```

---

## 4. 状态机

```text
[INIT] 加载模型，初始化相机，初始化串口
   ↓
[WAIT_START] 串口静默，等待 START 帧（AA 55 01 0D 0A）
   ↓ 收到 START
[SCANNING] 启动豆子相机 + 数字相机，并行推理
   ↓ 两边都识别完成
[READY] 内部查表生成 3 个放置区编号
   ↓ 发送 TASK 帧（BB 66 [p1] [p2] [p3] 1B 1A）
[DONE] 串口静默，等待比赛结束
```

> **注意**：赛前准备阶段（3 分钟）视觉可以初始化相机、加载模型、预热 BPU，但**不发送任何串口数据**。收到 START 后才启动扫描。

---

## 5. 场地关键参数

| 参数 | 数值 | 备注 |
|------|------|------|
| 场地地面 | 4 m × 2 m | PVC 塑胶地板，斑点纹浅灰色 |
| 取货区 | ①~③ 号位 | 场地左侧 |
| 放置区 | ④~⑧ 号位 | 场地右侧 |
| 货箱外尺寸 | 280×200×120 mm | 8104 型手提塑料保鲜盒（无盖） |
| 货箱内尺寸 | 250×170×120 mm | 有效装载空间 |
| 数字标签 | 100×100 mm | 贴于两个长侧面（280×120 mm 面） |
| 豆子重量 | 白 1.8 kg / 黄 2.1 kg / 绿 2.5 kg | 初始装载量 |

---

## 6. 代码模块建议

```
vision/
├── main.py              # 入口：串口监听 + 状态机
├── config.py            # 串口配置、相机参数、映射规则
├── models/
│   ├── bpu_loader.py    # 地平线 BPU 模型加载
│   ├── bean_model.py    # 豆子推理（输入图像→输出种类）
│   └── digit_model.py   # 数字推理（输入图像→输出数字）
├── cameras/
│   ├── bean_cam.py      # 豆子相机取流
│   └── digit_cam.py     # 数字相机取流
├── logic/
│   └── task_generator.py # 生成 3 个放置区编号
└── comm/
    └── serial_port.py   # pyserial 封装，发送 TASK 帧
```

---

## 7. 快速自查清单

- [ ] 收到 START 帧（`AA 55 01 0D 0A`）前，串口静默不发送。
- [ ] 豆子相机只识别 ①~③ 号位的豆子种类（白/黄/绿）。
- [ ] 数字相机只识别 ④~⑧ 号位的数字编号（1~5）。
- [ ] 内部映射正确：黄豆→1、绿豆→2、白芸豆→3。
- [ ] 反向查找正确：数字 1/2/3 在 ④~⑧ 中的位置。
- [ ] 发送 TASK 帧：`BB 66 [p1] [p2] [p3] 1B 1A`，p1/p2/p3 为 4~8。
- [ ] 扫描+发送阶段控制在 30 秒内。
- [ ] 串口参数：115200/8/N/1。

---

**文档结束**
