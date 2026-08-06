# vision.yaml 配置参考

`configs/vision.yaml` 是运行时唯一事实来源，驱动竞赛逻辑。本文覆盖映射规则、结果帧格式与常用配置字段（辅助线之外）。从 CLAUDE.md 下沉的低频参考。

## 视觉运行时映射规则

- `mapping.bean_to_digit`：将豆子类别映射为协议颜色编码（默认黄豆→1、绿豆→2、白芸豆→3），该编码同时作为结果帧 `C1-C3` 的颜色值。
- `lines.bean.left_x` / `lines.bean.right_x`：竖直辅助线，将豆子相机画面划分为取货区 ①–③。
- `lines.digit.x1` / `x2` / `x3` / `x4`：竖直辅助线，将数字相机画面划分为放置区 ④–⑧。

结果帧格式为 `A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`：

- `C1-C3`：取货区 1–3 的豆子颜色（01=黄豆，02=绿豆，03=白芸豆）。
- `N4-N8`：放置区 4–8 上的数字（01–05）。

运行 `python tests/calibrate.py` 可在实时预览上通过滑块调整辅助线，点击 **SAVE** 按钮手动写回 `configs/vision.yaml`。具体分区规则见 [docs/guides/vision_runtime.md](../guides/vision_runtime.md)。

## 相机与模型配置字段

除辅助线外，`configs/vision.yaml` 还经常需要调整以下字段：

- `cameras.bean` / `cameras.digit`：
  - `device`：V4L2 设备节点，可用 `tests/calibrate.py --list-cameras` 列出。
  - `width` / `height`：捕获分辨率。
  - `fps`：请求摄像头帧率（例如 `15`），实际帧率以设备返回为准。
  - `buffer_size`：OpenCV 内部缓冲，竞赛模式建议设为 `1` 降低延迟。
  - `auto_exposure`：`true` 启用自动曝光，`false` 改为手动曝光。
  - `grayscale`：`true` 时把输出强制转成单通道后再扩为 3 通道 BGR；数字相机若使用灰度数据集训练，可设为 `true`。

- `models.bean` / `models.digit`：
  - `bin` / `cfg`：RDK X5 BPU 模型与部署配置。
  - `pt`：`main_amd64.py` 在 x86 上使用的 Ultralytics 权重。
  - `conf_thres` / `iou_thres`：可选，覆盖全局 `inference.conf_thres` / `inference.iou_thres`，用于按模型调参。

- `ui.controls`：
  - `touch_mode`：触摸屏布局开关。
  - `slider_width` / `slider_height` / `button_height` / `padding` / `font_scale`：校准控件尺寸。
  - `inference_overlay`：`true` 时在校准画面上叠加推理框与分区编号，方便观察模型输出。

- `inference.start_gated`：`true`（默认）时仅在下位机发送 START 时运行推理（空闲模型保持加载待命）；`false` 关闭门控（无串口的 x86 开发路径视为恒开启）。
- `debug.force_inference`：headless 无 UI 时强制推理的现场调试开关（等价于 UI 的 DEBUG 按钮）。
