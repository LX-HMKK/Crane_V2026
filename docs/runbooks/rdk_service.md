# RDK X5 服务运维

生产运行由 `crane-vision.service` 管理。它在 XFCE 桌面就绪前先启动结果
通路，并在 X11 可用后显示 Tk UI。服务运行时，不要再启动
`scripts/watchdog.sh` 或第二个 `main_arm64.py` 进程，否则会争用两路相机和
`/dev/ttyS1`。

## 登录与状态

```bash
ssh root@192.168.128.10
systemctl status crane-vision.service --no-pager
systemctl is-active crane-vision.service
```

显示 `active` 表示视觉进程正在运行。服务默认已启用，RDK 每次上电后会自动
启动。

## 启动、重启与关闭

```bash
# 立即启动已停止的服务。
systemctl start crane-vision.service

# 更新 Python 代码或 vision.yaml 后重启。
systemctl restart crane-vision.service

# 立即停止；下一次上电仍会自动启动。
systemctl stop crane-vision.service

# 立即停止，并关闭后续上电自启。
systemctl disable --now crane-vision.service

# 恢复上电自启，并立即启动。
systemctl enable --now crane-vision.service
```

已配置 `Restart=on-failure`。进程异常退出时 systemd 会自动重启；显式执行
`systemctl stop` 被视为主动停止，不会自动拉起。

## 日志与启动耗时

```bash
# 实时跟随日志，按 Ctrl+C 结束查看。
journalctl -u crane-vision.service -f

# 查看本次开机日志和冷启动耗时。
journalctl -b -u crane-vision.service -o short-monotonic --no-pager

# 服务已停止或失败时，查看最新 100 行。
journalctl -u crane-vision.service -n 100 --no-pager
```

## Inference Throughput

The process writes an `Inference throughput` entry every five seconds. It reports completed inference work, rather than camera capture rate.

```bash
# Follow per-model and aggregate throughput live
journalctl -u crane-vision.service -f | grep "Inference throughput"

# Show recent throughput windows
journalctl -u crane-vision.service -n 200 --no-pager | grep "Inference throughput"
```

`bean` and `digit` are the completed rates for the two models; `total` is their sum. Cameras are configured for 30 fps. Each input retains only its newest frame, so insufficient inference capacity drops stale frames instead of accumulating latency.

冷启动日志中会出现 `runtime imports completed`、`cameras opened`、
`models loaded`、`first valid result ready` 和 `Calibration UI started` 等关键
阶段。

## 更新服务

将新版本同步到 `/home/sunrise/Crane_WS` 后，以 root 身份重新安装 unit。这会
保留由服务独占的相机和串口配置。

```bash
cd /home/sunrise/Crane_WS
bash scripts/install_crane_service.sh
systemctl restart crane-vision.service
```

若只修改了 `/etc/systemd/system/crane-vision.service`，则先重载 systemd 再
重启：

```bash
systemctl daemon-reload
systemctl restart crane-vision.service
```

## 校准与手动调试

打开校准工具或任何手动相机/BPU 进程前，必须先停止服务。完成后再启动服务。

```bash
systemctl stop crane-vision.service
cd /home/sunrise/Crane_WS
python3 tests/calibrate.py
systemctl start crane-vision.service
```

服务失败时，先查看上述日志，再清除失败状态并重新启动：

```bash
systemctl reset-failed crane-vision.service
systemctl start crane-vision.service
```
