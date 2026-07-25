# 起重机视觉串口协议与硬件连接

## 硬件连接

RDK X5 通过 40PIN 物理引脚 8/10 与下位机通信：

| 物理引脚 | 功能 | 设备节点 |
|---|---|---|
| 8 | UART1_TXD | `/dev/ttyS1` |
| 10 | UART1_RXD | `/dev/ttyS1` |

下位机（STM32）侧使用 **USART6**，对应引脚：

| STM32 引脚 | 功能 |
|---|---|
| PG9 | USART6_RX |
| PG14 | USART6_TX |

- 视觉端 TX（引脚 8）→ 下位机 RX（PG9），视觉端 RX（引脚 10）→ 下位机 TX（PG14），GND → GND。
- IO 电压 3.3V TTL。
- `/dev/ttyS0` 是系统调试串口，**不要用于下位机通信**。

## 串口参数

```text
设备：/dev/ttyS1
波特率：921600
格式：8N1，无流控
```

## 控制板发送的开始命令

控制板处于自动准备或自动运行模式时，每 100 ms 向视觉端发送一次：

```text
A5 5A 01
```

其中 `A5 5A` 是帧头，`01` 是开始标志。退出自动模式后停止发送。视觉端在收到该命令前不要发送识别结果；收到后开始识别，并持续回传下面的 12 字节结果帧。如果连续 300 ms 没再收到开始命令，视觉端停止回传。

## 固定 12 字节帧

```text
A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A
```

| 字节 | 内容 |
| --- | --- |
| `A5 5A` | 固定帧头 |
| `C1` | 取货区 1 的颜色 |
| `C2` | 取货区 2 的颜色 |
| `C3` | 取货区 3 的颜色 |
| `N4` | 放置区 4 上的数字 |
| `N5` | 放置区 5 上的数字 |
| `N6` | 放置区 6 上的数字 |
| `N7` | 放置区 7 上的数字 |
| `N8` | 放置区 8 上的数字 |
| `0D 0A` | 固定帧尾 |

颜色编码：

```text
01 = 黄豆
02 = 绿豆
03 = 白芸豆
```

注意：字段发送的是单字节数值，不是 ASCII 字符。例如颜色 1 必须发送十六进制 `01`，不能发送字符 `'1'`（`31`）。

## 发送要求

- 取货区三个颜色必须是 `01、02、03` 各一次，不能重复。
- 放置区五个数字必须是 `01~05` 各一次，不能重复。
- 只有所有区域识别完成后才发送，未完成时不要发送残缺结果。
- 收到控制板的 `A5 5A 01` 后才开始发送识别结果。
- 等待起重机启动期间，以 5~10 Hz 重复发送同一个最新结果。
- 控制板连续收到两帧完全相同且合法的数据后才采用结果，用来抵抗无 CRC 时的单次串口干扰。
- 自动启动时，控制板要求 1 秒内收到过合法帧。
- 不需要 CRC、置信度、序号或 ACK。

## 示例

取货区 1~3 依次是黄、绿、白，放置区 4~8 的数字依次是 `3、1、5、2、4`：

```text
A5 5A 01 02 03 03 01 05 02 04 0D 0A
```

Python 示例：

```python
import serial
import time

port = serial.Serial("COM5", 921600, timeout=0.1)
frame = bytes([
    0xA5, 0x5A,
    1, 2, 3,       # 取货区 1、2、3
    3, 1, 5, 2, 4, # 放置区 4、5、6、7、8
    0x0D, 0x0A,
])

receive_buffer = bytearray()
last_start_time = 0.0

while True:
    receive_buffer.extend(port.read(port.in_waiting or 1))
    if b"\xA5\x5A\x01" in receive_buffer:
        last_start_time = time.monotonic()
        receive_buffer.clear()

    if time.monotonic() - last_start_time < 0.3:
        port.write(frame)
        time.sleep(0.1)
```

## STM32 联调

Keil Watch 观察：

```c
crane_vision_uart_data.rx_byte_count
crane_vision_uart_data.valid_frame_count
crane_vision_uart_data.accepted_result_count
crane_vision_uart_data.invalid_frame_count
crane_vision_uart_data.start_tx_count
crane_vision_uart_data.start_tx_error_count
crane_vision_uart_data.last_source_material[0]
crane_vision_uart_data.last_source_material[1]
crane_vision_uart_data.last_source_material[2]
crane_vision_uart_data.last_place_box_number[0]
crane_vision_uart_data.last_place_box_number[1]
crane_vision_uart_data.last_place_box_number[2]
crane_vision_uart_data.last_place_box_number[3]
crane_vision_uart_data.last_place_box_number[4]
```

正常时 `rx_byte_count`、`valid_frame_count` 和 `accepted_result_count` 持续增加，`invalid_frame_count` 不增加。
