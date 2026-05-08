from __future__ import annotations

import argparse
import struct
from typing import Optional

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 2.5D map viewer for RK3588.")
    parser.add_argument("--occ-topic", type=str, default="/octomap/occupancy", help="OccupancyGrid topic.")
    parser.add_argument("--cloud-topic", type=str, default="/octomap/points", help="PointCloud2 topic.")
    parser.add_argument("--window", type=str, default="octomap_2p5d_view", help="OpenCV window name.")
    parser.add_argument("--scale", type=int, default=6, help="Display scale factor.")
    parser.add_argument("--show-grid", action="store_true", help="Draw grid lines.")
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip image vertically before display (legacy behavior).",
    )
    return parser.parse_args()


class MapViewerNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("octomap_2p5d_viewer")
        self.window_name = args.window
        self.scale = max(1, int(args.scale))
        self.show_grid = bool(args.show_grid)
        self.flip_y = bool(args.flip_y)

        self.latest_occ: Optional[OccupancyGrid] = None
        self.latest_cloud: Optional[PointCloud2] = None

        self.create_subscription(OccupancyGrid, args.occ_topic, self._on_occ, 10)
        self.create_subscription(PointCloud2, args.cloud_topic, self._on_cloud, 10)
        self.timer = self.create_timer(0.1, self._render)
        self.get_logger().info(f"Subscribed: occ={args.occ_topic}, cloud={args.cloud_topic}")

    def _on_occ(self, msg: OccupancyGrid) -> None:
        self.latest_occ = msg

    def _on_cloud(self, msg: PointCloud2) -> None:
        self.latest_cloud = msg

    @staticmethod
    def _occ_to_image(msg: OccupancyGrid) -> np.ndarray:
        h = int(msg.info.height)
        w = int(msg.info.width)
        if h <= 0 or w <= 0:
            return np.zeros((1, 1), dtype=np.uint8)
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        img = np.zeros((h, w), dtype=np.uint8)
        img[data < 0] = 40
        img[(data >= 0) & (data < 50)] = 0
        img[data >= 50] = 255
        return img

    @staticmethod
    def _cloud_to_height_map(
        msg: PointCloud2,
        h: int,
        w: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> np.ndarray:
        height_map = np.zeros((h, w), dtype=np.float32)
        if msg.point_step < 12 or not msg.data:
            return height_map

        point_step = int(msg.point_step)
        count = int(msg.width) * max(1, int(msg.height))
        safe_resolution = max(float(resolution), 1e-6)

        for index in range(count):
            offset = index * point_step
            if offset + 12 > len(msg.data):
                break
            px, py, pz = struct.unpack_from("<fff", msg.data, offset)
            gx = int((px - origin_x) / safe_resolution)
            gy = int((py - origin_y) / safe_resolution)
            if 0 <= gx < w and 0 <= gy < h and pz > height_map[gy, gx]:
                height_map[gy, gx] = pz
        return height_map

    def _render(self) -> None:
        if self.latest_occ is None:
            return

        occ = self.latest_occ
        occ_img = self._occ_to_image(occ)
        h, w = occ_img.shape
        resolution = float(occ.info.resolution) if occ.info.resolution > 0 else 1.0
        origin_x = float(occ.info.origin.position.x)
        origin_y = float(occ.info.origin.position.y)

        color = cv2.cvtColor(occ_img, cv2.COLOR_GRAY2BGR)

        if self.latest_cloud is not None:
            height_map = self._cloud_to_height_map(
                self.latest_cloud,
                h,
                w,
                resolution,
                origin_x,
                origin_y,
            )
            max_h = float(np.max(height_map))
            if max_h > 1e-6:
                normalized = np.clip((height_map / max_h) * 255.0, 0, 255).astype(np.uint8)
                heat = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                mask = normalized > 0
                color[mask] = cv2.addWeighted(color, 0.25, heat, 0.75, 0)[mask]

        if self.show_grid and self.scale >= 4:
            for y in range(0, h, 5):
                cv2.line(color, (0, y), (w - 1, y), (32, 32, 32), 1)
            for x in range(0, w, 5):
                cv2.line(color, (x, 0), (x, h - 1), (32, 32, 32), 1)

        show = cv2.flip(color, 0) if self.flip_y else color
        if self.scale != 1:
            show = cv2.resize(show, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(self.window_name, show)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.get_logger().info("Exit requested by keyboard.")
            rclpy.shutdown()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = MapViewerNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

