# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a computer-vision object-detection workspace centered on a vendored copy of [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5). It contains datasets, trained weights, ONNX/BPU `.bin` exports, and deployment notes for an RDKx5 board. As of the latest checkout, `scripts/`, `configs/`, and `models/` are empty placeholders (only `.gitkeep`), so most training/inference/export work is done by calling scripts directly inside `tools/yolov5/`.

> **Note:** `docs/RDKx5_Bin_Inference.md` describes a YOLO11-style 6-head deployment workflow, but the vendored training code under `tools/yolov5/` is standard YOLOv5. Treat the deployment doc as the source of truth for board-side inference and the YOLOv5 tree as the source of truth for training/export.

## Repository layout

```text
assets/            # Trained model artifacts (.pt, .onnx, .bin)
                   #   num.*      – digit detector
                   #   beans_v4.* – bean maturity detector
configs/           # Empty placeholder; intended for future training/model configs
data/              # Datasets (YOLO format: images/ + labels/ + dataset.yaml)
                   #   dataset3/       – raw digit images
                   #   output_gray/    – processed digit dataset + split_dataset/
                   #   beans/          – bean images + beans_split/
docs/              # Deployment documentation (RDKx5_Bin_Inference.md)
models/            # Empty placeholder; intended for future model definitions
scripts/           # Empty placeholder; intended for future project-specific scripts
tools/yolov5/      # Vendored YOLOv5 repository (train.py, detect.py, val.py, export.py, ...)
tools/onnx/        # Empty; ignored by .gitignore
```

`data/*` is ignored by `.gitignore` except for `data/.gitkeep`; `tools/onnx/*` is also ignored. Large assets and datasets are expected to live on disk but are not committed.

## Common commands

All YOLOv5 commands run from the repository root and reference scripts inside `tools/yolov5/`.

### Install dependencies

```bash
pip install -r tools/yolov5/requirements.txt
```

### Train a detector

```bash
python tools/yolov5/train.py \
  --data data/output_gray/images/dataset.yaml \
  --weights yolov5n.pt \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --project runs/train \
  --name exp
```

Use `data/beans/beans_split/beans_split.yaml` for the bean dataset. Weights are downloaded automatically on first use.

### Validate a model

```bash
python tools/yolov5/val.py \
  --data data/output_gray/images/dataset.yaml \
  --weights assets/num.pt \
  --img 640
```

### Run inference

```bash
python tools/yolov5/detect.py \
  --weights assets/num.pt \
  --source data/dataset3/images/val/ \
  --img 640 \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --project runs/detect \
  --name exp
```

### Export to ONNX

```bash
python tools/yolov5/export.py \
  --weights assets/num.pt \
  --include onnx \
  --imgsz 640
```

### Run YOLOv5 tests

```bash
python -m pytest tools/yolov5/tests/
```

Run a single test file:

```bash
python -m pytest tools/yolov5/tests/test_invariant_export.py -v
```

### RDKx5 board-side inference

See `docs/RDKx5_Bin_Inference.md` for the full workflow. The board-side script referenced by that doc (`scripts/deployment/infer_rdkx5_yolo11.py`) does not currently exist in this checkout; deploy by copying the `.bin`, its `_config.yaml`, and the inference script together to the RDKx5 board. Key points from the doc:

- Input: RGB NCHW `uint8` tensor of shape `1×3×640×640`; normalization (`/255`) is baked into the model.
- Letterbox padding uses gray value `127`.
- Outputs are 6 tensors: `s_bbox`, `m_bbox`, `l_bbox`, `s_cls`, `m_cls`, `l_cls`.
- Post-processing is DFL decode → xyxy restore → sigmoid → NMS.
- The script depends on `hobot_dnn`, which is only available on the RDKx5 board.

## Dataset conventions

Each dataset directory should contain:

- `images/train/` and `images/val/`
- `labels/train/` and `labels/val/` with YOLO-format `.txt` files (`class x_center y_center width height`, normalized)
- A `dataset.yaml` (or `<name>.yaml`) with `path`, `train`, `val`, `names`, and `nc`

Existing dataset YAMLs use `path: .` and `train: .`, so they are invoked from inside their own directory. Prefer absolute or repo-root-relative paths when creating new dataset configs to avoid path confusion.

## Asset naming

- `.pt`  – PyTorch weights (YOLOv5 / Ultralytics format)
- `.onnx` – ONNX export for x86 validation
- `.bin`  – BPU binary for RDKx5 deployment

Each basename should have a matching config YAML for board deployment (e.g. `beans_v4_rdkx5_config.yaml`) with `imgsz`, `nc`, `names`, `reg_max`, and `strides`.

## 提交代码

提交信息按 Angular 规范写中文主题。在 PowerShell 中，避免把 `@` 解析成数组，使用 here-string 写入临时文件再提交：

```powershell
# 写入提交信息
@'
fix(rdkx5): 修复导出脚本路径问题

- 修复 ONNX 路径使用 POSIX 斜杠
- 校准数据 pad 值对齐 Ultralytics
'@ | Out-File -Encoding utf8 .git-msg

# 提交并删除临时文件
git commit -F .git-msg
Remove-Item .git-msg
```

## Important caveats

- Do not run `scripts/deployment/infer_rdkx5_yolo11.py` on a development machine; it requires `hobot_dnn` from the RDKx5 image.
- `README.md` is currently empty; treat `docs/RDKx5_Bin_Inference.md` as the only written documentation.
- Before committing new scripts or configs, check whether they belong in the empty top-level `scripts/`/`configs/` directories or inside `tools/yolov5/`.
