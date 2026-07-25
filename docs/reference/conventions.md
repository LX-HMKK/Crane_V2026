# 数据集与资产命名规范

## 数据集目录结构

每个数据集目录应包含：

- `images/train/` 和 `images/val/`
- `labels/train/` 和 `labels/val/`，文件为 YOLO 格式 `.txt`（`class x_center y_center width height`，已归一化）
- `dataset.yaml`（或 `<name>.yaml`），包含 `path`、`train`、`val`、`names`、`nc`

现有数据集 YAML 使用 `path: .` 和 `train: .`，因此需要在该 YAML 所在目录下调用。新建数据集配置时，建议使用绝对路径或仓库根目录相对路径，以避免路径歧义。

## 模型资产命名

- `.pt`  – PyTorch 权重（YOLOv5 / Ultralytics 格式）
- `.onnx` – x86 验证用 ONNX 导出
- `.bin`  – RDKx5 部署用 BPU 二进制

每个 `.bin` 基名都应配有一个部署配置 YAML（例如 `beans_v4_rdkx5_config.yaml`），包含 `imgsz`、`nc`、`names`、`reg_max`、`strides`。额外字段（如 `task`、`model`、`onnx`）不影响板端推理。
