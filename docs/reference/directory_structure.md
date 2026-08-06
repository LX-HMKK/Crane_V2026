# 仓库目录结构

2026 物流技术创意赛视觉目标检测工作空间的目录布局。从 CLAUDE.md 下沉的低频参考。

```text
assets/            # 训练好的模型产物（.pt、.onnx、.bin）
                   #   num.* / num_v6.* – 数字检测器
                   #   beans_v6.* – 豆子成熟度检测器
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
                   #   reference/
                   #     conventions.md          – 数据集与资产命名规范
                   #     CRANE_VISION_PROTOCOL.md – 串口硬件与通信协议
                   #     directory_structure.md  – 本文件（目录布局）
                   #     vision_config.md        – vision.yaml 配置字段参考
                   #     commit_guidelines.md    – 提交规范
                   #   guides/
                   #     RDKx5_Bin_Inference.md   – YOLO11 BPU 板端工作流
                   #     vision_runtime.md        – 运行时入口、状态机与分区映射规则
models/            # 视觉运行时源码
                   #   camera_viewer.py – CameraViewer 基类
                   #   ui.py – CalibrationUI(CameraViewer) + CompetitionUI(CalibrationUI)
                   #   yolo11_postprocess.py, yolo11_model.py, bpu_loader.py
                   #   bean_model.py, digit_model.py, camera.py, pt_model.py
                   #   serial_port.py  – SerialPort + SerialThread（后台串口）
                   #   task_generator.py – 分区聚合 + 逻辑补全
                   #   logic_completion.py – 置信度排列评分 + 缺区推断
                   #   mapping.py, runtime.py – 入口共享助手
scripts/           # 视觉运行时入口
                   #   main_arm64.py – RDK X5 竞赛模式入口
                   #   main_amd64.py – x86 双相机 PyTorch 推理 + 逻辑补全（CompetitionUI）
                   #   watchdog.sh – main_arm64.py 自动重启脚本
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
tools/yolov5/      # 内嵌 YOLOv5 仓库（train.py、detect.py、val.py、export.py 等）
tools/crane_tests/ # 模块级单元测试（pytest 驱动，mock hobot_dnn）
tools/onnx/        # 本地 ONNX 导出；已加入 .gitignore
```

`data/*` 和 `tools/onnx/*` 已加入 `.gitignore`，仅保留 `data/.gitkeep`。大型资产和数据集只保存在本地工作区，不提交到仓库。
