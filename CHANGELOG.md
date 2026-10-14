# 变更日志

本文件记录项目的所有重要变更，格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)。

## [Unreleased] - 2026-08-20

### 新增

- START 门控推理：仅在下位机发送 START 心跳时运行推理，空闲模型保持加载待命；UI 增加 DEBUG 按钮用于显示-only 调试，headless 增加等价的 `debug.force_inference` 开关。
- 共享 START 门控辅助函数 `inference_gate_open()` / `serial_start_active()`，统一 UI 与 headless 的门控判断。
- 竞赛结果控制器 `CompetitionResultController`，将协议发布状态从 Tkinter UI 层独立出来，便于 headless 与 UI 共用同一套逻辑补全和发送限速策略。
- 串口 START 心跳状态切换日志：收到 START 和超时时各打印一次，便于现场确认下位机是否真正发送命令。
- 运行时模块分层重构：`models/runtime.py` 拆分为 `models/application/runtime.py` 与 `models/application/competition_controller.py`；`models/ui.py` 拆分为 `models/ui/viewer.py`、`models/ui/calibration.py`、`models/ui/competition.py`；`models/serial_port.py` 迁移至 `models/devices/serial_port.py`。
- 豆子/数字检测权重更新至 v7（beans_v7 / num_v7）。

### 变更

- `protocol.start_timeout_sec` 默认值由 `0.3` 秒调整为 `3.0` 秒，收到单帧 START 后保持 active 更久，兼容下位机只发单帧或心跳偶发丢失的场景；仍可按需调小以恢复严格短超时。
- README 美化，新增界面预览与运行流程图；CLAUDE.md 低频参考内容下沉到 `docs/runbooks/` 与 `docs/reference/`。
- README 精简：删除手写目录、装饰性徽章、冗余 Mermaid 配色与重复段落（START 门控 / 串口参数 / 开发机限制），修正许可证章节为 Apache 2.0，板端部署步骤补上 udev 规则；首页导语与运行环境改为加粗导语 + 规格表，强化信息层级。
- README 运行流程图由 Mermaid 代码块改为已渲染的 `docs/images/pipeline.png`，避免在不支持 Mermaid 的 Markdown 渲染器中退化为源码。

### 修复

- START 门控陈旧帧发送风险：START 停止后清空串口缓存，恢复后以新推理结果续发；DEBUG 强制推理结果不再写入串口缓存和失败计数器。
- 相机 V4L2 setter 测试缺少 `_io_lock` 的问题。

## [0.1.0] - 2026-08-03

### 新增

- 相机/串口掉线自动重连：设备连续失败达到配置次数后自动重开，UI 状态栏显示 BEAN/DIGIT/SERIAL 离线状态，离线相机以占位画面提示。
- RDK X5 early-start `crane-vision.service`、安装脚本和服务运维文档；结果通路先启动，X11 就绪后接管同一 UI。
- 推理吞吐日志与验收命令，可分别观察 bean、digit 和总完成帧率。

- 竞赛状态机主程序，搭配崩溃自动重启看门狗
- 串口 START/TASK 帧编码：921600 波特率，12 字节结果帧
- 后台串口线程：200 Hz 轮询 START 帧，20 Hz 发送结果帧
- 逻辑补全模块：基于置信度排列评分的缺区推断与重复检测
- 竞赛 UI：双相机同时捕获，检测框与推断框叠加，结果帧输出
- 辅助线校准 UI：滑块控件旁置、推理 overlay、防抖保存、触摸优化
- 运行时配置支持按模型覆盖 `conf_thres` / `iou_thres`
- 豆子和数字 BPU 模型推理封装，YOLO11 6 头后处理（letterbox / DFL / NMS）
- `hobot_dnn` BPU 封装，x86 上优雅回退
- OpenCV 相机封装：帧率控制、自动曝光、灰度输出
- 豆子类别 → 协议颜色编码映射与分区任务生成
- RDK X5 相机 udev 规则（`/dev/crane_cam_beans`、`/dev/crane_cam_digit`）
- 独立校准入口 `calibrate.py` 与串口测试帧发送工具
- 双相机预览、推理检查、串口回环诊断脚本
- 运行时配置文件 `configs/vision.yaml`（串口、相机、模型、辅助线、映射、逻辑补全）

### 变更

- 提取 ConfigBinding 类，用类型化绑定替换字典操作 setter。
- 豆子/数字检测权重更新至 v6（beans_v6 / num_v6）。
- 竞赛模式改为两路独立相机采集线程和两路独立模型推理线程；每路仅保留最新帧，UI 刷新不再限制推理输入。
- 任务组合按串口发送周期限速，结果日志按窗口限速，降低重复计算和日志 I/O 对实时路径的影响。
- RDK 运行时改用 headless OpenCV，减少冷启动期间不需要的 GUI 原生库加载。

- 串口协议迁移至 `A5 5A` 起始帧头、12 字节结果格式、921600 波特率
- 校准 UI 从 OpenCV highgui 重构为 tkinter，控件旁置，适配 1024×600 触摸屏
- 紧凑化校准控件尺寸与间距，最大化可见视频区域
- 统一 amd64/arm64 校准流程，数字分区改为四线八区
- 改为手动保存（SAVE/EXIT 触摸按钮），取消自动保存以减少 SD 卡写入
- 竞赛模式默认启用串口
- 抽取 CameraViewer 基类供校准与诊断脚本复用

### 修复

- 修复慢推理或慢 UI 时旧相机帧形成等待的问题，始终以最新检测结果更新串口状态。

- 串口通信中 START 帧缓冲区清理与超时可靠性
- `hobot_dnn.pyeasy_dnn` 适配器与 NV12 输入转换
- BPU 加载器健壮性：路径校验、导入错误处理、输入张量校验、序列防护
- 逐类 NMS、坐标裁剪、配置校验、串口写入 flush
- 竞赛模式资源清理与启动异常安全加固
- 校准 UI 保存错误处理（ERROR 视觉反馈）与回调堆积
- 校准与任务生成器边界情况：无效输入、重复检测、辅助线比较、无头回退
- 看门狗启动命令参数错误
- 测试中校准导入路径
