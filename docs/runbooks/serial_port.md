# 串口

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

- 默认配置为 **921600 8/N/1**，无流控，IO 电压 3.3V。
- `/dev/ttyS0` 是系统调试串口，不建议用于下位机通信。
- 与下位机连接需要交叉接线：RDK TX（引脚 8）→ 下位机 RX（PG9），RDK RX（引脚 10）→ 下位机 TX（PG14），GND → GND。

`scripts/main_arm64.py` 通过 `models/serial_port.open_serial_port()` 打开串口，参数来自 `configs/vision.yaml` 的 `serial` 段。`models/serial_port.py` 封装了 START 命令检测（`A5 5A 01`）和 12 字节结果帧发送（`A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`）。

竞赛模式下，视觉端在收到 START 命令后才开始识别并以 5–10 Hz 重复发送结果帧；若连续 300 ms 未收到 START 命令，则停止回传。
