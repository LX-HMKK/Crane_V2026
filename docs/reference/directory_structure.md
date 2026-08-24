# 仓库目录结构

2026 物流技术创意赛视觉目标检测工作空间的目录布局。从 CLAUDE.md 下沉的低频参考。

```text
assets/            # 训练好的模型产物（.pt、.onnx、.bin）
                   #   num.* / num_v7.* – 数字检测器
                   #   beans_v7.* – 豆子成熟度检测器
                   #   *.yaml（beans.yaml / num.yaml）– BPU 部署配置
configs/           # 运行时配置
                   #   vision.yaml – 串口、相机、模型、辅助线、映射
                   #   vision_1024x600_test.yaml – 1024×600 备选配置
                   #   .gitkeep
data/              # 数据集（YOLO 格式：images/ + labels/ + dataset.yaml）
                   #   dataset3/       – 原始数字图片
                   #   output_gray/    – 处理后的数字数据集 + split_dataset/
                   #   beans/          – 豆子图片 + beans_split/
docs/              # 部署与运行时文档
                   #   runbooks/
                   #     commands.md             – 完整命令清单
                   #     rdk_service.md          – crane-vision.service 系统服务运维
                   #   reference/
                   #     CRANE_VISION_PROTOCOL.md – 串口硬件与通信协议
                   #     directory_structure.md  – 本文件（目录布局）
                   #     vision_config.md        – vision.yaml 配置字段参考
                   #     commit_guidelines.md    – 提交规范
                   #   guides/
                   #     RDKx5_Bin_Inference.md   – YOLO11 BPU 板端工作流
                   #     vision_runtime.md        – 运行时入口、状态机与分区映射规则
models/            # 视觉运行时源码
                   #   application/runtime.py, competition_controller.py
                   #     – 入口共享助手、双相机结果配对与发送调度
                   #   config/config_binding.py – 类型化配置绑定
                   #   devices/camera.py, serial_port.py – 相机与串口访问
                   #   domain/mapping.py, zones.py, task_generator.py, logic_completion.py
                   #     – 业务映射、分区聚合、区域布局、逻辑补全
                   #   inference/bpu_loader.py, yolo11_model.py,
                   #     yolo11_postprocess.py, bean_model.py, digit_model.py,
                   #     pt_model.py – BPU / PyTorch 推理
                   #   ui/viewer.py – CameraViewer 基类
                   #   ui/calibration.py – CalibrationUI：校准、设置、推理叠加
                   #   ui/competition.py – CompetitionUI：采集、状态机、串口结果
scripts/           # 视觉运行时入口
                   #   main_arm64.py – RDK X5 竞赛模式入口
                   #   main_amd64.py – x86 双相机 PyTorch 推理 + 逻辑补全（CompetitionUI）
                   #   watchdog.sh – main_arm64.py 自动重启脚本
                   #   install_crane_service.sh – systemd 竞赛模式服务安装
                   #   .gitkeep
tests/              # 独立功能测试脚本（x86 / RDK X5 均可运行）
                   #   calibrate.py               – 校准 UI 入口
                   #   send_test_frame.py         – 串口测试帧发送工具
                   #   test_camera_view.py        – 双相机画面检查
                   #   test_inference_ui.py       – 双相机推理检查
                   #   test_serial_loopback.py    – 串口回环/协议测试
                   #   test_logic_completion.py   – 逻辑补全模块测试
                   #   test_settings_controls.py  – 校准控件单元测试
                   #   test_ui_completion.py      – 推断标签渲染测试
requirements.txt   # 运行时/测试依赖
README.md          # 项目概览（侧重视觉模块）
tools/crane_tests/           # 模块级单元测试（pytest 驱动，mock hobot_dnn）
tools/99-crane-cameras.rules # RDK X5 相机 udev 规则
tools/crane-vision.service   # systemd 竞赛模式服务
```

`data/*` 已加入 `.gitignore`，仅保留 `data/.gitkeep`。大型资产和数据集只保存在本地工作区，不提交到仓库。

## 数据集与资产命名规范

### 数据集目录结构

每个数据集目录应包含：

- `images/train/` 和 `images/val/`
- `labels/train/` 和 `labels/val/`，文件为 YOLO 格式 `.txt`（`class x_center y_center width height`，已归一化）
- `dataset.yaml`（或 `<name>.yaml`），包含 `path`、`train`、`val`、`names`、`nc`

现有数据集 YAML 使用 `path: .` 和 `train: .`，因此需要在该 YAML 所在目录下调用。新建数据集配置时，建议使用绝对路径或仓库根目录相对路径，以避免路径歧义。

### 模型资产命名

- `.pt`  – PyTorch 权重（Ultralytics 格式）
- `.onnx` – x86 验证用 ONNX 导出
- `.bin`  – RDKx5 部署用 BPU 二进制（由板级 `hb_mapper` 生成）

每个 `.bin` 基名都应配有一个部署配置 YAML（例如 `beans.yaml` / `num.yaml`），包含 `imgsz`、`nc`、`names`、`reg_max`、`strides`。额外字段（如 `task`、`model`、`onnx`）不影响板端推理。
