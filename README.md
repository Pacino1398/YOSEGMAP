# YosegMap

基于 YOLOv5 分割结果的栅格化与 D* Lite 路径规划工程，支持 `PT / ONNX / RKNN` 三条链路。  
当前主流程：在原始视频帧上实时叠加障碍物、目标点和规划路径。

## 当前版本

- 版本命名：`v0.6.0-rk3588-async6hz`
- 关键变更：
  - `process_stream_source` 重构为三级流水线（Capture / Inference / Post-process）
  - 本地摄像头优先走 GStreamer + MPP 硬解路径
  - ROS2 发布改为异步缓存发布（不阻塞 NPU 推理链路）
  - 后处理 mask 裁剪向量化，降低 CPU 周期占用
  - 推荐实时更新目标：`>= 6 Hz`（优先新地图，不持续复用旧地图）
  - 新增 `--ros-lite-mode`：仅发布高度语义 `OccupancyGrid`，废弃点云依赖

## 功能

- PT 离线分割与落盘
- ONNX 实时分割 + 路径规划
- RKNN（RK3588）实时分割 + 路径规划
- 本机窗口 / 远端 MJPEG / 双显示 / 无显示
- 支持将 `octomap` 占据图以 ROS2 图像话题发布

## 目录

```text
.
├─app/
│  ├─inference/         # PT/ONNX/RKNN 推理入口
│  ├─mapping/           # mask -> 栅格/2.5D 地图
│  └─planning/          # 路径规划与渲染
├─data/                 # 类别配置
├─tools/                # 导出脚本
├─weights/              # 模型权重
└─yolo/                 # vendored YOLOv5
```

## 环境准备

### x86

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### RK3588

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip ffmpeg libgl1 libglib2.0-0
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements-rk3588.txt
pip install rknn-toolkit-lite2
```

## 快速开始

### 1) PT 分割（离线）

```bash
python app/inference/segmentation.py \
  --source ./test_input \
  --weights ./weights/0414_qy++.pt \
  --data ./data/my.yaml \
  --project ./runs/segment \
  --device cpu \
  --load-masks
```

### 2) 实时路径规划（ONNX/RKNN）

ONNX：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.onnx \
  --backend onnx \
  --data ./data/my.yaml \
  --device cpu \
  --display local
```

RKNN：

```bash
python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.rknn \
  --backend rknn \
  --data ./data/my.yaml \
  --display local
```

RK3588（推荐实时发布配置，目标 >=6Hz）：

```bash
YOSEGMAP_PLAN_EVERY_N_FRAMES=2 python app/planning/realtime_pathplan.py \
  --source 0 \
  --weights ./weights/0414_qy++.rknn \
  --backend rknn \
  --data ./data/my.yaml \
  --display none \
  --ros-publish-2p5d \
  --ros-lite-mode \
  --ros-rate 6 \
  --ros-occ-topic /octomap/occupancy \
  --z-max-cap 12.0 \
  --z-step 0.8 \
  --cloud-mode edge
```

若发布仍低于 6Hz，可降载：

```bash
YOSEGMAP_PLAN_EVERY_N_FRAMES=2 python app/planning/realtime_pathplan.py ...
```

性能日志（每 10 帧统计）：

```bash
python app/planning/realtime_pathplan.py ... --perf-log
```

输出示例：

```text
[perf] prep=xx.xxms infer=xx.xxms post=xx.xxms postprocess=xx.xxms planning=xx.xxms render=xx.xxms total=xx.xxms (n=10)
```

远端预览（MJPEG）：

```bash
python app/planning/realtime_pathplan.py ... --display remote --remote-port 8080
```

浏览器访问：`http://<board-ip>:8080/stream.mjpg`

## ROS2 话题发布（octomap 图像）

`app/mapping/octomap.py` 已支持将占据图发布为 `sensor_msgs/Image`（`mono8`）。

```bash
python -m app.mapping.octomap \
  --mask-dir runs/segment/exp2/masks \
  --ros2-publish \
  --ros-topic /octomap/image \
  --ros-frame-id map \
  --ros-rate 2
```

更多参数与订阅示例见：`docs/ROS2_OCTOMAP.md`

## 模型导出

PT -> ONNX：

```bash
python yolo/export.py --weights ./weights/0414_qy++.pt --include onnx --imgsz 640 640
```

ONNX -> RKNN：

```bash
python tools/export_rknn.py --onnx ./weights/0414_qy++.onnx --output ./weights/0414_qy++.rknn --target rk3588
```

## 常用说明

- 显示模式：`--display local|remote|both|none`
- 关键类别语义见：`data/my.yaml`、`app/mapping/grid_map.py`
- RKNN 推理缺包：安装 `rknn-toolkit-lite2`
- 导出 `.rknn`：安装 `rknn-toolkit2`
- RK3588 建议：发布侧尽量与推理侧隔离（例如 `taskset` 绑核），并优先使用 `--cloud-mode edge --z-step 0.8`
- RViz2 建议：添加 `Map` 插件并订阅 `/octomap/occupancy`，按 costmap 语义观察高度分层（1~100）


