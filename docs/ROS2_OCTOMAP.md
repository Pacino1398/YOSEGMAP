# ROS2 Octomap 2.5D 发布说明

## 功能

`app/mapping/octomap.py` 支持发布 2.5D 地图话题：

- `/octomap/occupancy`：`nav_msgs/msg/OccupancyGrid`（下视障碍分布）
- `/octomap/points`：`sensor_msgs/msg/PointCloud2`（障碍高度点云）
- （可选）`/octomap/image`：`sensor_msgs/msg/Image`（mono8 占据图）

## 发布 2.5D 地图

```bash
python -m app.mapping.octomap \
  --mask-dir runs/segment/exp2/masks \
  --ros-publish-2p5d \
  --ros-frame-id map \
  --ros-rate 2 \
  --cell-size 1.0 \
  --z-step 0.5
```

常用参数：

- `--ros-occ-topic`：`OccupancyGrid` 话题（默认 `/octomap/occupancy`）
- `--ros-cloud-topic`：`PointCloud2` 话题（默认 `/octomap/points`）
- `--cell-size`：每个栅格对应实际米数
- `--z-step`：点云在 z 方向采样步长（米）

## 仅发布图像（兼容模式）

```bash
python -m app.mapping.octomap \
  --mask-dir runs/segment/exp2/masks \
  --ros2-publish \
  --ros-topic /octomap/image
```

## 订阅验证

```bash
ros2 topic list | grep octomap
ros2 topic echo /octomap/occupancy
ros2 topic echo /octomap/points
```

## 发布正确性验证（推荐流程）

终端 1（发布）：

```bash
python -m app.mapping.octomap \
  --mask-dir runs/segment/exp2/masks \
  --ros-publish-2p5d \
  --ros-frame-id map \
  --ros-rate 2 \
  --cell-size 1.0 \
  --z-step 0.5
```

终端 2（验证）：

```bash
ros2 topic list | grep octomap
ros2 topic info /octomap/occupancy
ros2 topic info /octomap/points
ros2 topic hz /octomap/occupancy
ros2 topic hz /octomap/points
ros2 topic echo /octomap/occupancy --once
ros2 topic echo /octomap/points --once
```

本地可视化检查：

```bash
python tools/ros2_map_viewer.py \
  --occ-topic /octomap/occupancy \
  --cloud-topic /octomap/points \
  --scale 6
```

判定标准：

- 能看到 `/octomap/occupancy` 与 `/octomap/points`
- `hz` 与 `--ros-rate` 接近
- `occupancy` 的 `width/height/resolution` 合理且 `data` 非全 0
- `points` 的 `width` 大于 0，`header.frame_id` 正确（例如 `map`）

## RK3588 本地查看效果

查看脚本已移到 `tools/ros2_map_viewer.py`（不再放在 `app/mapping`）。

终端 1（发布）：

```bash
python -m app.mapping.octomap \
  --mask-dir runs/segment/exp2/masks \
  --ros-publish-2p5d
```

终端 2（订阅显示）：

```bash
python tools/ros2_map_viewer.py \
  --occ-topic /octomap/occupancy \
  --cloud-topic /octomap/points \
  --scale 6
```

按 `q` 或 `Esc` 退出。

## 注意

- 运行前先 `source` ROS2 环境（如 `/opt/ros/<distro>/setup.bash`）。
- 若提示 `rclpy`、`nav_msgs`、`sensor_msgs` 缺失，先安装对应 ROS2 组件。
