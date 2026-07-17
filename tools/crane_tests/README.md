# crane_tests

本目录存放 Crane Vision 的**模块级单元测试**，由 pytest 驱动。

运行方式：

```bash
pytest tools/crane_tests/ -v
```

这些测试覆盖 `models/` 中各模块的独立行为，不依赖真实相机、BPU 模型或串口硬件。

独立功能测试入口请见仓库根目录 `test/` 下的脚本：

- `test/test_camera_view.py` — 双相机画面检查
- `test/test_inference_ui.py` — 双相机推理检查
- `test/test_serial_loopback.py` — 串口完整收发测试
