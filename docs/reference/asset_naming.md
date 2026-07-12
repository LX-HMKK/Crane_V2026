# 资产命名规范

- `.pt`  – PyTorch 权重（YOLOv5 / Ultralytics 格式）
- `.onnx` – x86 验证用 ONNX 导出
- `.bin`  – RDKx5 部署用 BPU 二进制

每个 `.bin` 基名都应配有一个部署配置 YAML（例如 `beans_v4_rdkx5_config.yaml`），包含 `imgsz`、`nc`、`names`、`reg_max`、`strides`。额外字段（如 `task`、`model`、`onnx`）不影响板端推理。
