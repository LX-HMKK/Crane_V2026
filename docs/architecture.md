# 架构与竞赛语义规范

## 模块布局

**硬规则：`src/` 只放 `main.cpp` 入口，全部模块在 `tasks/` 下。** 新增模块一律进 `tasks/`（或其子目录），不要往 `src/` 加文件。

```text
tasks/
  app.*              # 竞赛编排：采集×2 / 推理×2 / 任务合并 / 串口 / Web（根上唯一模块）
  common/            # types.hpp（Detection/FramePayload 等）+ logging.hpp + config.*
  io/
    camera.*         # ICamera / Camera(V4L2) / MockCamera / StaticImageCamera
    serial_port.*    # termios UART + 协议 + 后台线程
  logic/
    mapping.*        # 类别→协议编码映射
    task_generator.* # 分区聚合 + 严格/补全两种结果帧生成
    logic_completion.* # 置信度排列评分推断
  inference/
    yolo11.*         # letterbox / DFL / sigmoid / 逐类 NMS / 坐标回映 / NV12
    model.*          # Yolo11Model：预处理 → 后端 → 后处理 → 回映
    onnx_backend.*   # ONNX Runtime 后端（x86 验证）
    bpu_backend.*    # RDK X5 BPU 后端（板端）
    bpu_stub.c       # 非 RDK 主机的 libDNN 桩（保证全平台可编译链接）
  web/               # 内嵌 HTTP server(keep-alive) + MJPEG(降采样≤960宽) + REST
                     # + 校准页 index.html（布局对齐 Python 参考：视频/控制/底部状态栏；settings 隐藏视频）
```

include 根是 `tasks/`：跨模块引用写全路径（`"io/camera.hpp"`），同目录内写短名（`"mapping.hpp"`）。

## 双推理后端

`Yolo11Model::predict` = letterbox → backend → postprocess → scale_coords。两条后端输出格式根本不同，后处理分两条路径（都在 `yolo11.cpp`）：

| | ONNX 后端（x86） | BPU 后端（RDK X5） |
|---|---|---|
| 模型产物 | `.onnx`（DFL/sigmoid 融合在图内） | `.bin` + `.yaml` 部署配置 |
| 输出 | 融合张量 `[1,4+nc,8400]` | 6 头原始输出：s/m/l × bbox(64ch) + cls(nc) |
| 后处理 | `postprocess_fused`：转置 → argmax → 阈值 → 逐类 NMS | `postprocess_heads`：先分类过滤，再对幸存者 DFL 解码 |
| 输入 | `blobFromImage`（swapRB, 1/255） | NV12（`nv12_from_bgr`），hbSys 缓存内存 |
| 反量化 | 无（float） | 按输出张量 SCALE（per-tensor / per-channel） |

- ORT 用**共享 `Ort::Env` + 全局线程池**（`onnx_backend.cpp` 的 `shared_env()`），不要给每个 session 单独建线程池。
- 非 RDK 主机上 BPU 后端链接桩实现、调用时清晰报错；真实推理行为只能上板验证。
- **letterbox pad 必须保持 127**，与 BPU 校准时一致；letterbox 取整用 `nearbyint`（对齐 Python round 的银行家舍入），不要随意改成 `int()` 截断。

## 线程模型

```text
采集线程 ×2（每相机一个）   latest-slot 覆盖，无帧队列；失败计数 ≥5 判掉线
      ↓ 最新帧
推理线程 ×2（每模型一个）   条件变量等待新帧；识别门控：START 活跃才消费帧
                          model.predict()（DEBUG 强制、无串口 fail-open；
                          50ms 定时兜底唤醒，解耦 START 与采集速率）
      ↓ 最新检测结果
任务合并（app 主循环侧）    双相机健康才更新；去抖 50ms；sticky result；
                          连续 10 次生成失败才清空串口帧
      ↓ FramePayload
串口线程                  200Hz poll_start()；START 活跃才 20Hz 发帧；掉线重连
HTTP 连接线程             每连接一个线程；keep-alive 复用（WSL2 localhost 每新建连接 ~200ms，
                          滑块 POST 卡顿主因，必须复用）；stop() 时 shutdown 全部 fd 并排空
```

生命周期注意（都是审查修过的坑）：退出路径上所有 `std::thread` 必须 join（detached 线程 + 析构 = UAF）；跨线程共享的 `SerialPort` 用 `shared_ptr` + 互斥换指针；HTTP 连接线程入口必须套 try/catch（畸形请求抛异常 = terminate）。

## 竞赛语义规范（本仓库自身的行为定义，改动必须回归 golden 测试）

- **分区**：检测框中心 `cx < line` 归左区，**边界线归右侧区**（豆子 ①–③ / 数字 ④–⑧）。每区取最高置信度检测。
- **逻辑补全**：检测缺区/重复时，全局排列评分推断（`std::next_permutation` 字典序、**严格 `>`** 才替换——平局保字典序首个）。推断上限 bean≤1 / digit≤2，**超限拒发**整帧。
- **Sticky result**：结果不变不更新串口；下位机停发 START 保留缓存结果，恢复立即续发；相机掉线不清空。
- **串口协议**：`/dev/ttyS1`，921600 8/N/1。下位机发 `A5 5A 01`（START），300ms 无 START 停发；结果帧 12 字节 `A5 5A C1 C2 C3 N4 N5 N6 N7 N8 0D 0A`（C1-C3 为 1,2,3 排列、N4-N8 为 1..5 排列，编码时校验）。**禁用 `/dev/ttyS0`**（系统调试串口）。
- **START 门控识别**：START 不只是串口发送闸门，还门控推理入口——只有 START 活跃时才把两路相机帧输入模型（`app.cpp` 的 `recognition_enabled()`）。待机时模型**保持加载但不推理**（不卸载、不空转，x86 首次 START 即满速；RDK BPU 若出现 DVFS 冷启动掉速，可后续加低频率空转预热）。门控规则：`DEBUG 强制开` ∨ `无串口后端 fail-open 恒识别` ∨ `串口连接但 START 活跃`。串口启动打开失败、运行中断线重开窗口都按无串口处理（fail-open，不冻结感知）；healthy-but-quiet 的串口严格 gate。识别关断瞬间在途的一帧会发布为 sticky 结果（语义：保留最后一个有效结果）。Web UI 的 DEBUG 开关是运行时状态，**不写回 `vision.yaml`**；`/api/status` 暴露 `debug` / `recognition` 字段。
- **相机**：MJPG + buffer_size=1；v4l2-ctl 优先的曝光控制；失败重连（3 次 / 1s 间隔）。
- **配置**：`configs/vision.yaml` 是运行时唯一事实来源；Web UI 滑块实时生效（REST POST /api/config），SAVE 才写回文件。

## 测试基线

`tests/test_golden_onnx` 用 `tests/data/{beans_2,num_2}.jpg` 对拍**记录在测试内的参考 detections**（坐标容差 1.5px、置信度 0.005，为吸收 OpenCV 版本间 resize 差异），并验证端到端结果帧。改预处理/后处理代码后若 golden 挂了，先确认是行为变化还是数值噪声，再决定修代码还是更新参考值。

## 关键陷阱

- **yaml-cpp 0.7**：`node = node[key]` 是内容赋值（`AssignNode`），会把子树内容覆盖进父节点、原地摧毁文档（症状：第一次 get 成功，之后全部 fallback）。逐键下钻必须用 `node.reset(node[key])`——`config.cpp` 的 get/set/get_int_map 全是这个写法，新增代码照抄。
- **头文件约定**：C++ 用 `.hpp`；`.h` 仅用于 C 源（`bpu_stub.c`）和 `tools/` vendored 头文件。
- `Ort::AllocatedStringPtr` 无默认构造（explicit ctor），成员初始化必须放 ctor 初始化列表。
