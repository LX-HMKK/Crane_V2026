# 常用命令

> RDK X5 日常启停、日志、更新和校准操作请见
> [RDK X5 服务运维](rdk_service.md)。

以下命令均在仓库根目录执行，面向视觉运行时、板端入口脚本与增量同步。

## 开发/测试虚拟环境

本项目使用独立的 Miniforge 虚拟环境，避免与系统或其他项目的 `opencv-python` / `opencv-python-headless` 等包冲突：

- 环境路径：`D:\conda_envs\crane-clean`
- Python：3.11
- 包源：USTC PyPI 镜像

所有 x86 开发/测试命令均通过该环境执行：

```bash
conda run --prefix /d/conda_envs/crane-clean python ...
```

### 安装运行时/测试依赖

```bash
conda run --prefix /d/conda_envs/crane-clean pip install --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple -r requirements.txt
```

### 运行单元测试

```bash
conda run --prefix /d/conda_envs/crane-clean python -m pytest tools/crane_tests/ -v
```

## 视觉运行时命令

这些命令用于操作 RDK X5 竞赛运行时。校准视图和单元测试可在 x86 上运行，但完整 BPU 推理需要 RDK X5 板端的 `hobot_dnn`。

### 运行竞赛模式（仅限 RDK X5）

```bash
python3 scripts/main_arm64.py
```

`main_arm64.py` 现在只负责竞赛模式主循环：等待串口 START 命令、扫描两路相机、向下位机发送 12 字节结果帧。

### 使用实时预览校准辅助线

```bash
conda run --prefix /d/conda_envs/crane-clean python tests/calibrate.py
```

按 `1` / `2` 切换豆子/数字相机视图，通过控制面板滑块调整辅助线，按 `ESC` 保存到 `configs/vision.yaml`。在没有相机的 x86 开发机上，可配合 `--mock-cameras` 预览校准 UI。

### 无相机测试校准 UI

```bash
conda run --prefix /d/conda_envs/crane-clean python tests/calibrate.py --mock-cameras
```

使用合成帧启动校准 UI，用于无相机环境下的 UI 开发与辅助线布局测试。按 `ESC` 保存的辅助线位置同样写回 `configs/vision.yaml`。

### 列出可用相机

```bash
conda run --prefix /d/conda_envs/crane-clean python tests/calibrate.py --list-cameras
```

### 独立功能测试脚本

#### 双相机画面检查（无模型、无串口）

```bash
conda run --prefix /d/conda_envs/crane-clean python tests/test_camera_view.py
```

仅显示两路相机画面和辅助线，按 `1` / `2` 切换视图。适合机械安装后确认相机位置和辅助线分区。

#### 双相机推理检查（加载 BPU 模型，无串口）

```bash
conda run --prefix /d/conda_envs/crane-clean python tests/test_inference_ui.py
```

加载豆子/数字 BPU 模型，在画面上叠加检测框与分区编号，验证模型输出与辅助线分区是否匹配。

#### 串口回环/协议测试

```bash
# TX-RX 短接的回环测试
conda run --prefix /d/conda_envs/crane-clean python tests/test_serial_loopback.py --mode loopback --count 100

# 使用 SerialPort 发送结果帧并等待 START
conda run --prefix /d/conda_envs/crane-clean python tests/test_serial_loopback.py --mode protocol --count 10
```

### 自动重启竞赛模式

```bash
bash scripts/watchdog.sh
```

> 生产 `crane-vision.service`（systemd）运行时不要同时启动 `watchdog.sh`，两者会争夺两路相机与 `/dev/ttyS1`；服务自带 `Restart=on-failure`。

### RDK X5 上电快速启动

使用 systemd 服务，无界面生成串口结果并在 XFCE 就绪后自动显示竞赛 UI。安装/启停/日志/校准前停服务释放资源等运维操作，见 [RDK X5 服务运维](rdk_service.md)。

### x86 完整流程测试（CompetitionUI + PyTorch，无串口）

`main_amd64.py` 启动 CompetitionUI：双相机同时 PyTorch 推理 + 逻辑补全，支持校准 UI 的全部三页（`1`=豆子、`2`=数字、`3`=settings）。无需串口：

```bash
conda run --prefix /d/conda_envs/crane-clean python scripts/main_amd64.py --mock-cameras
```

使用静态测试图片代替相机：

```bash
conda run --prefix /d/conda_envs/crane-clean python scripts/main_amd64.py \
  --bean-image data/image_for_test/beans_2.jpg \
  --digit-image data/image_for_test/num_2.jpg
```

Settings 页（`3`）除相机曝光/模型置信度外，还包含 **BEAN INFER MAX**（0–3）和 **DIGIT INFER MAX**（0–5）滑块，控制逻辑补全推断上限。

单次扫描并退出（无 UI）：

```bash
conda run --prefix /d/conda_envs/crane-clean python scripts/main_amd64.py --once --mock-cameras
```

## RDKx5 板端推理

完整流程见 `docs/guides/RDKx5_Bin_Inference.md`。板端入口包括 `scripts/main_arm64.py`（竞赛模式）、`tests/calibrate.py`（校准）、`tests/test_camera_view.py`、`tests/test_inference_ui.py`、`tests/test_serial_loopback.py`（独立测试）；部署时需把 `models/`、`scripts/`、`configs/`、`tests/`、`tools/99-crane-cameras.rules`、`assets/*.bin` 和 `assets/*.yaml` 一起拷贝到 RDKx5。使用 `python3 tests/calibrate.py` 进行校准，使用 `python3 scripts/main_arm64.py` 运行竞赛模式，使用 `./scripts/watchdog.sh` 以看门狗方式运行竞赛模式。`hobot_dnn` 导入、NV12 缓冲、6 头输出、后处理等板端细节见 [RDKx5 BPU 板端工作流](../guides/RDKx5_Bin_Inference.md)。

## 增量同步修改到 RDK X5

日常调试时只应把**自上次提交以来修改过的文件**同步到板端，避免全量拷贝模型与数据集：

1. 在 x86 开发机查看变更文件：

   ```bash
   git diff --name-only HEAD
   ```

2. 用 `scp`（本机无 `rsync`）按原目录结构逐文件复制到 RDK X5：`models/`、`scripts/`、`configs/`、`tests/` 等源码与配置目录。例如：

   ```bash
   scp configs/vision.yaml root@192.168.128.10:/home/sunrise/Crane_WS/configs/
   scp tests/calibrate.py root@192.168.128.10:/home/sunrise/Crane_WS/scripts/
   scp tests/test_serial_loopback.py root@192.168.128.10:/home/sunrise/Crane_WS/tests/
   ```

3. 若新增文件或目录，同样用 `scp -r` 或先 `ssh` 到板端创建目录后再复制。

> **规则：** 绝不覆盖 `assets/*.bin` 等大型资产，除非本次提交确实更新了模型；同步前先 `git diff --name-only HEAD` 确认变更范围。

## 在 RDK X5 上测试入口脚本（不接串口）

### 1. 校准预览

```bash
ssh root@192.168.128.10
cd /home/sunrise/Crane_WS
python3 tests/calibrate.py
```

`calibrate.py` 会按 `configs/vision.yaml` 的 `serial` 段打开串口。若未接串口，可临时注释掉 `serial` 段或改用 PTY。按 `1` / `2` 切换豆子/数字相机视图，拖动面板上的滑块调整辅助线，按 `ESC` 退出，位置自动保存到 `configs/vision.yaml`。

> 需要 RDK X5 已接显示器，或 SSH 带 X11 转发。Windows 上先开启 X server（如 VcXsrv / MobaXterm），再用 `ssh -X root@192.168.128.10`。

### 2. 双相机画面检查

```bash
ssh root@192.168.128.10
cd /home/sunrise/Crane_WS
python3 tests/test_camera_view.py
```

无模型、无串口、无滑块，仅确认相机画面和辅助线分区。

### 3. 双相机推理检查

```bash
ssh root@192.168.128.10
cd /home/sunrise/Crane_WS
python3 tests/test_inference_ui.py
```

加载 BPU 模型并叠加检测框，不打开串口，适合验证模型输出。

### 4. 无 UI 单帧扫描验证

```bash
ssh root@192.168.128.10
cd /home/sunrise/Crane_WS
python3 - <<'PY'
import logging, sys
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)
from models.application.runtime import (
    load_vision_cfg, load_models, create_task_generator,
    open_camera, scan_once_dets,
)
cfg = load_vision_cfg('configs/vision.yaml')
bean_model, digit_model = load_models(cfg)
task_gen = create_task_generator(cfg, bean_model, digit_model)
cam_bean = open_camera(cfg['cameras']['bean'])
cam_digit = open_camera(cfg['cameras']['digit'])
bean_dets, digit_dets, reason = scan_once_dets(
    cam_bean, cam_digit, bean_model, digit_model, cfg
)
print('reason :', reason)
print('bean   :', bean_dets)
print('digit  :', digit_dets)
if reason == 'ok':
    print('frame  :', task_gen.generate_frame(bean_dets, digit_dets))
cam_bean.release(); cam_digit.release()
PY
```

### 5. 无串口测试竞赛主循环

可用临时 PTY 替代 `/dev/ttyS1` 验证 `main_arm64.py` 加载模型并进入等待 START 状态。完整步骤（建虚拟串口 + `--config` 指向临时端口）见 [RDKx5 BPU 板端工作流](../guides/RDKx5_Bin_Inference.md) 的「无串口测试」小节。
