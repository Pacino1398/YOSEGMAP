from __future__ import annotations

import argparse
import hashlib
import time

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class NewMapRateNode(Node):
    def __init__(self, topic: str, window_sec: float) -> None:
        super().__init__("newmap_rate_meter")
        self._topic = topic
        self._window_sec = max(float(window_sec), 0.5)
        self._last_hash: str | None = None
        self._pub_count = 0
        self._new_count = 0
        self._t0 = time.perf_counter()
        self._sub = self.create_subscription(OccupancyGrid, topic, self._on_msg, 10)
        self.get_logger().info(f"Listening topic: {topic}, window={self._window_sec:.1f}s")

    def _on_msg(self, msg: OccupancyGrid) -> None:
        self._pub_count += 1
        raw = bytes(msg.data)
        digest = hashlib.blake2b(raw, digest_size=8).hexdigest()
        if digest != self._last_hash:
            self._new_count += 1
            self._last_hash = digest

        dt = time.perf_counter() - self._t0
        if dt >= self._window_sec:
            publish_hz = self._pub_count / dt
            newmap_hz = self._new_count / dt
            print(
                f"publish_hz={publish_hz:.2f}, "
                f"newmap_hz={newmap_hz:.2f}, "
                f"window={dt:.2f}s"
            )
            self._pub_count = 0
            self._new_count = 0
            self._t0 = time.perf_counter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure publish rate vs real new-map rate for OccupancyGrid.")
    parser.add_argument("--topic", default="/octomap/occupancy", help="OccupancyGrid topic name")
    parser.add_argument("--window", type=float, default=2.0, help="Statistics window in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)
    node = NewMapRateNode(topic=args.topic, window_sec=args.window)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
