# crane_tests

本目录存放 Crane Vision 的**模块级单元测试**，由 pytest 驱动。

运行方式：

```bash
pytest tools/crane_tests/ -v
```

这些测试覆盖 `models/` 中各模块的独立行为，不依赖真实相机、BPU 模型或串口硬件。mock 与技术拆分的约定见仓库根目录 `CLAUDE.md` 的“测试入口约定”。

独立功能测试入口请见仓库根目录 `tests/` 下的脚本（如 `tests/test_camera_view.py`、`tests/test_inference_ui.py`、`tests/test_serial_loopback.py`、`tests/calibrate.py`）。
