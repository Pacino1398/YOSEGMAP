from __future__ import annotations

import argparse
import importlib
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.mapping.grid_map import GridMapHandler, load_mask_entries
from app.paths import resolve_path
from app.planning.pathplan_batch import get_latest_segmentation_run_dir

Cell2D = tuple[int, int]
DEFAULT_UAV_ALTITUDE = 12.0
TREE_CANOPY_DOWNWARD_EXPANSION = 1.0
CANOPY_CLASSES = {4, 6}
GROUNDED_CLASSES = {1, 3, 5, 9}
HOUSE_CLASS = 9


def _to_ros_image_msg(image: np.ndarray, frame_id: str):
    sensor_msgs_msg = importlib.import_module("sensor_msgs.msg")
    std_msgs_msg = importlib.import_module("std_msgs.msg")
    Image = getattr(sensor_msgs_msg, "Image")
    Header = getattr(std_msgs_msg, "Header")

    if image.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported for ROS2 image publishing.")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)

    height, width = image.shape
    msg = Image()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height = int(height)
    msg.width = int(width)
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = int(width)
    msg.data = image.tobytes()
    return msg


def _to_occupancy_grid_msg(occ2d: np.ndarray, *, frame_id: str, cell_size: float):
    nav_msgs_msg = importlib.import_module("nav_msgs.msg")
    std_msgs_msg = importlib.import_module("std_msgs.msg")
    OccupancyGrid = getattr(nav_msgs_msg, "OccupancyGrid")
    Header = getattr(std_msgs_msg, "Header")

    if occ2d.ndim != 2:
        raise ValueError("Occupancy grid expects 2D array.")
    grid = occ2d.astype(np.uint8, copy=False)
    height, width = grid.shape

    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.info.resolution = float(max(cell_size, 1e-6))
    msg.info.width = int(width)
    msg.info.height = int(height)
    msg.info.origin.position.x = 0.0
    msg.info.origin.position.y = 0.0
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0

    # ROS OccupancyGrid expects int8 values in [-1, 100]. Here use 0 (free) / 100 (occupied).
    data = np.where(grid > 0, 100, 0).astype(np.int8, copy=False)
    msg.data = data.reshape(-1).tolist()
    return msg


def _to_pointcloud2_msg_from_columns(
    columns: Mapping[Cell2D, "ColumnState"],
    *,
    frame_id: str,
    cell_size: float,
    z_step: float,
):
    sensor_msgs_msg = importlib.import_module("sensor_msgs.msg")
    std_msgs_msg = importlib.import_module("std_msgs.msg")
    PointCloud2 = getattr(sensor_msgs_msg, "PointCloud2")
    PointField = getattr(sensor_msgs_msg, "PointField")
    Header = getattr(std_msgs_msg, "Header")

    points: list[tuple[float, float, float]] = []
    safe_z_step = max(float(z_step), 0.05)
    safe_cell = max(float(cell_size), 1e-6)

    for (x, y), column in columns.items():
        z_lo = float(column.collision_base_z)
        z_hi = float(column.top_z)
        if z_hi <= z_lo:
            z_hi = z_lo + 0.05
        wx = (float(x) + 0.5) * safe_cell
        wy = (float(y) + 0.5) * safe_cell
        z = z_lo
        while z <= z_hi + 1e-6:
            points.append((wx, wy, z))
            z += safe_z_step

    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = b"".join(struct.pack("<fff", px, py, pz) for px, py, pz in points)
    return msg


@dataclass(slots=True)
class ColumnState:
    cell: Cell2D
    height: float
    class_id: int | None = None
    observed: bool = True
    display_base_z: float = 0.0
    collision_base_z: float = 0.0
    terrain_penalty: float = 0.0
    mode: str = "grounded"

    @property
    def top_z(self) -> float:
        return self.display_base_z + self.height


class OctoMap:
    def __init__(self, grid_w: int, grid_h: int, grid_scale: int):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.grid_scale = grid_scale
        self.grid_handler = GridMapHandler(grid_w, grid_h, grid_scale)
        self._reset_state()

    def build_occ2d(self, *, occupied_value: int = 255, use_columns: bool = False) -> np.ndarray:
        """Build a dense 2D occupancy image in grid coordinates.

        Args:
            occupied_value: Value to set for occupied cells (uint8).
            use_columns: If True, mark occupancy by whether a column exists (more inclusive).
                If False, mark occupancy by blocked_obstacles (more conservative).

        Returns:
            occ2d: uint8[grid_h, grid_w], 0=free, occupied_value=occupied.
        """

        occ = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        val = np.uint8(max(0, min(255, int(occupied_value))))
        if use_columns:
            cells = self.columns.keys()
        else:
            cells = self.blocked_obstacles
        for x, y in cells:
            if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                occ[y, x] = val
        return occ

    def build_z_band2d(self) -> np.ndarray:
        """Build a dense 2D z-band map in grid coordinates.

        For each occupied cell, export the dangerous vertical band [collision_base_z, top_z].

        Returns:
            z_band2d: float32[grid_h, grid_w, 2]
                [...,0]=z_lo (collision_base_z)
                [...,1]=z_hi (top_z)
            Empty cells are zeros.
        """

        z_band = np.zeros((self.grid_h, self.grid_w, 2), dtype=np.float32)
        for cell, column in self.columns.items():
            x, y = cell
            if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                z_band[y, x, 0] = float(column.collision_base_z)
                z_band[y, x, 1] = float(column.top_z)
        return z_band

    @staticmethod
    def infer_grid_size(mask_dir: str | Path, grid_scale: int) -> tuple[int, int]:
        mask_path = Path(mask_dir)
        first_mask = None
        if mask_path.exists():
            for mask_file in sorted(mask_path.glob("*.png")):
                first_mask = plt.imread(mask_file)
                if first_mask is not None:
                    break

        if first_mask is None:
            return 64, 64

        if getattr(first_mask, "ndim", 0) == 3:
            mask_h, mask_w = first_mask.shape[:2]
        else:
            mask_h, mask_w = first_mask.shape
        return max(1, mask_w // grid_scale), max(1, mask_h // grid_scale)

    def _reset_state(self):
        self.obstacles: set[Cell2D] = set()
        self.blocked_obstacles: set[Cell2D] = set()
        self.traversable_obstacles: set[Cell2D] = set()
        self.terrain_penalties: dict[Cell2D, float] = {}
        self.obstacle_heights: dict[Cell2D, int] = {}
        self.obstacle_class_ids: dict[Cell2D, int] = {}
        self.mask_instances: list[dict[str, object]] = []
        self.mask_instance_tiles: list[dict[str, object]] = []
        self.target_point: Cell2D | None = None
        self.columns: dict[Cell2D, ColumnState] = {}
        self.max_z = 0.0
        self.revision = 0

    def _sync_from_grid_handler(self):
        self.obstacles = set(self.grid_handler.obstacles)
        self.blocked_obstacles = set(self.grid_handler.blocked_obstacles)
        self.traversable_obstacles = set(self.grid_handler.traversable_obstacles)
        self.terrain_penalties = dict(self.grid_handler.terrain_penalties)
        self.obstacle_heights = dict(self.grid_handler.obstacle_heights)
        self.obstacle_class_ids = dict(self.grid_handler.obstacle_class_ids)
        self.mask_instances = list(self.grid_handler.mask_instances)
        self.mask_instance_tiles = list(getattr(self.grid_handler, "mask_instance_tiles", []))
        self.target_point = self.grid_handler.target_point

    def _normalize_column_state(self, cell: Cell2D, value: ColumnState | Mapping[str, Any] | int | float | None) -> ColumnState | None:
        if value is None:
            return None

        if isinstance(value, ColumnState):
            column = ColumnState(
                cell=cell,
                height=max(0.0, float(value.height)),
                class_id=value.class_id,
                observed=bool(value.observed),
                display_base_z=float(value.display_base_z),
                collision_base_z=float(value.collision_base_z),
                terrain_penalty=float(value.terrain_penalty),
                mode=str(value.mode),
            )
        elif isinstance(value, Mapping):
            class_id = value.get("class_id")
            column = ColumnState(
                cell=cell,
                height=max(0.0, float(value.get("height", 1.0))),
                class_id=None if class_id is None else int(class_id),
                observed=bool(value.get("observed", True)),
                display_base_z=float(value.get("display_base_z", 0.0)),
                collision_base_z=float(value.get("collision_base_z", 0.0)),
                terrain_penalty=float(value.get("terrain_penalty", 0.0)),
                mode=str(value.get("mode", "grounded")),
            )
        else:
            column = ColumnState(cell=cell, height=max(0.0, float(value)))

        if column.height <= 0.0:
            return None
        column.display_base_z = max(0.0, column.display_base_z)
        column.collision_base_z = max(0.0, min(column.collision_base_z, column.top_z))
        return column

    def _build_column_for_cell(self, cell: Cell2D, class_id: int | None, height: int | float) -> ColumnState:
        top_z = max(1.0, float(height))
        terrain_penalty = float(self.terrain_penalties.get(cell, 0.0))

        if class_id in CANOPY_CLASSES:
            collision_base_z = max(0.0, top_z - TREE_CANOPY_DOWNWARD_EXPANSION)
            mode = "canopy"
        else:
            collision_base_z = 0.0
            mode = "grounded"

        if class_id == HOUSE_CLASS:
            collision_base_z = 0.0
            mode = "grounded"

        return ColumnState(
            cell=cell,
            height=top_z,
            class_id=class_id,
            observed=True,
            display_base_z=0.0,
            collision_base_z=collision_base_z,
            terrain_penalty=terrain_penalty,
            mode=mode,
        )

    def _build_columns_from_synced_state(self):
        self.columns = {}
        cells = set(self.obstacles) | set(self.obstacle_heights)
        for cell in cells:
            class_id = self.obstacle_class_ids.get(cell)
            height = self.obstacle_heights.get(cell, 1)
            self.columns[cell] = self._build_column_for_cell(cell, class_id, height)

    def _build_columns_from_obstacle_set(self, obstacle_set):
        self.columns = {}
        for cell in set(obstacle_set or set()):
            self.columns[cell] = ColumnState(cell=cell, height=1.0, observed=True)

    def _rebuild_local_views(self):
        self.obstacles = set()
        self.blocked_obstacles = set()
        self.traversable_obstacles = set()
        self.terrain_penalties = {}
        self.obstacle_heights = {}
        self.obstacle_class_ids = {}
        self.max_z = 0.0

        for cell, column in self.columns.items():
            if column.height <= 0.0:
                continue
            self.obstacles.add(cell)
            self.blocked_obstacles.add(cell)
            self.obstacle_heights[cell] = int(round(column.height))
            if column.class_id is not None:
                self.obstacle_class_ids[cell] = column.class_id
            if column.terrain_penalty > 0.0:
                self.terrain_penalties[cell] = column.terrain_penalty
            if column.top_z > self.max_z:
                self.max_z = column.top_z

    def masks_to_obstacle(self, mask_list):
        obstacle_set, target_point = self.grid_handler.batch_masks_to_obs(mask_list or [])
        self._sync_from_grid_handler()
        self.build_octomap(obstacle_set)
        if not mask_list:
            print("地图构建：未检测到掩码，障碍物集合为空")
            return set(), None

        print(f"地图构建：生成 {len(obstacle_set)} 个局部 2.5D 障碍栅格")
        return set(self.blocked_obstacles), target_point

    def build_octomap(self, obstacle_set):
        print("开始构建局部 2.5D 地图...")
        has_synced_state = bool(self.obstacles or self.obstacle_heights or self.obstacle_class_ids)
        if has_synced_state:
            self._build_columns_from_synced_state()
        else:
            self._build_columns_from_obstacle_set(obstacle_set)
            self.mask_instances = []
            self.target_point = None

        self.revision += 1
        self._rebuild_local_views()
        print("局部 2.5D 地图构建完成！")
        return set(self.blocked_obstacles)

    def get_bounds(self) -> tuple[int, int, float]:
        return self.grid_w, self.grid_h, self.max_z

    def get_blocked_obstacles_2d(self) -> set[Cell2D]:
        return set(self.blocked_obstacles)

    def get_obstacle_heights_2d(self) -> dict[Cell2D, int]:
        return dict(self.obstacle_heights)

    def export_planner_snapshot(self) -> dict[str, object]:
        return {
            "bounds": self.get_bounds(),
            "grid_scale": self.grid_scale,
            "target_point": self.target_point,
            "blocked_obstacles": self.get_blocked_obstacles_2d(),
            "obstacle_heights": self.get_obstacle_heights_2d(),
            "obstacle_class_ids": dict(self.obstacle_class_ids),
            "columns": dict(self.columns),
            "mask_instances": list(self.mask_instances),
            "mask_instance_tiles": list(self.mask_instance_tiles),
            "revision": self.revision,
        }

    def show_local_map_3d(
        self,
        elev: float = 28.0,
        azim: float = -58.0,
        uav_altitude: float = DEFAULT_UAV_ALTITUDE,
        *,
        edge_only: bool = True,
        edge_mode: str = "4n",
        edge_stride: int = 1,
        edge_max_bars: int | None = None,
        top_cloth: bool = False,
        top_cloth_stride: int = 2,
        top_cloth_alpha: float = 0.18,
    ):
        """Display local 2.5D pillar map.

        Notes
        -----
        - `edge_only=True` will render only boundary columns to speed up matplotlib 3D.
        - `edge_mode`:
            - "4n": 4-neighborhood boundary (recommended)
            - "8n": 8-neighborhood boundary (more aggressive removal of interior columns)
        - `edge_stride`: keep 1 column for every `edge_stride` cells along X/Y to further thin boundary bars.
        - `edge_max_bars`: hard cap for rendered bars (applied after boundary + stride).
        - `top_cloth`: if True, render a lightweight "height-field top cloth" for grounded obstacles via `plot_surface`.
        - `top_cloth_stride`: downsample stride for the height field grid.
        - `top_cloth_alpha`: alpha for the cloth surface.
        """
        figure = plt.figure(figsize=(10, 8))
        axis = figure.add_subplot(111, projection="3d")

        grounded_columns = sorted(
            (column for column in self.columns.values() if column.mode == "grounded"),
            key=lambda column: column.cell,
        )
        canopy_columns = sorted(
            (column for column in self.columns.values() if column.mode == "canopy"),
            key=lambda column: column.cell,
        )

        def _filter_boundary(columns: list[ColumnState]) -> list[ColumnState]:
            if not columns:
                return []

            occ = {c.cell for c in columns}
            if edge_mode == "8n":
                nbs = (
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                )
            else:
                nbs = ((1, 0), (-1, 0), (0, 1), (0, -1))

            out: list[ColumnState] = []
            for c in columns:
                x, y = c.cell
                is_edge = False
                for dx, dy in nbs:
                    nb = (x + dx, y + dy)
                    if nb not in occ:
                        is_edge = True
                        break
                if is_edge:
                    out.append(c)
            return out

        def _stride_filter(columns: list[ColumnState]) -> list[ColumnState]:
            if not columns:
                return []
            stride = max(1, int(edge_stride))
            if stride == 1:
                return columns
            return [c for c in columns if (c.cell[0] % stride == 0 and c.cell[1] % stride == 0)]

        def _cap(columns: list[ColumnState]) -> list[ColumnState]:
            if not columns:
                return []
            if edge_max_bars is None:
                return columns
            cap = int(edge_max_bars)
            if cap <= 0 or len(columns) <= cap:
                return columns
            # deterministic thinning: keep evenly-spaced samples from sorted columns
            step = max(1, len(columns) // cap)
            return columns[::step][:cap]

        if edge_only:
            grounded_columns = _cap(_stride_filter(_filter_boundary(grounded_columns)))
            canopy_columns = _cap(_stride_filter(_filter_boundary(canopy_columns)))

        if grounded_columns:
            xs = [column.cell[0] for column in grounded_columns]
            ys = [column.cell[1] for column in grounded_columns]
            zs = [column.display_base_z for column in grounded_columns]
            dx = [1.0] * len(grounded_columns)
            dy = [1.0] * len(grounded_columns)
            dz = [column.height for column in grounded_columns]
            axis.bar3d(xs, ys, zs, dx, dy, dz, color="#4c4c4c", alpha=0.88, shade=True)

        if top_cloth:
            # Height-field "top cloth" for grounded obstacles.
            # Build a grid Z(x,y)=top_z for grounded columns (0 where empty), then plot a downsampled surface.
            stride = max(1, int(top_cloth_stride))
            Z = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            for column in (c for c in self.columns.values() if c.mode == "grounded"):
                x, y = column.cell
                if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                    if column.top_z > Z[y, x]:
                        Z[y, x] = float(column.top_z)

            if stride > 1:
                Zs = Z[::stride, ::stride]
                ys = np.arange(0, self.grid_h, stride, dtype=np.float32)
                xs = np.arange(0, self.grid_w, stride, dtype=np.float32)
            else:
                Zs = Z
                ys = np.arange(0, self.grid_h, 1, dtype=np.float32)
                xs = np.arange(0, self.grid_w, 1, dtype=np.float32)

            if Zs.size > 0 and float(Zs.max()) > 0.0:
                X, Y = np.meshgrid(xs, ys)
                axis.plot_surface(
                    X,
                    Y,
                    Zs,
                    color="#4c4c4c",
                    alpha=float(top_cloth_alpha),
                    linewidth=0,
                    antialiased=False,
                    shade=False,
                )

        if canopy_columns:
            xs = [column.cell[0] for column in canopy_columns]
            ys = [column.cell[1] for column in canopy_columns]
            zs = [column.collision_base_z for column in canopy_columns]
            dx = [1.0] * len(canopy_columns)
            dy = [1.0] * len(canopy_columns)
            dz = [max(column.top_z - column.collision_base_z, 0.1) for column in canopy_columns]
            axis.bar3d(xs, ys, zs, dx, dy, dz, color="#2ca02c", alpha=0.55, shade=True)

        if self.target_point is not None:
            tx, ty = self.target_point
            axis.scatter([tx + 0.5], [ty + 0.5], [0.5], color="red", s=70, depthshade=False)

        uav_x = self.grid_w / 2.0
        uav_y = self.grid_h / 2.0
        uav_z = max(0.0, float(uav_altitude))
        axis.scatter([uav_x], [uav_y], [uav_z], color="#1f77b4", s=90, depthshade=False)

        axis.set_xlim(0, max(self.grid_w, 1))
        axis.set_ylim(0, max(self.grid_h, 1))
        axis.set_zlim(0, max(float(self.max_z), uav_z + 1.0, 1.0))
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.set_title(f"Local 2.5D Map | UAV z={uav_z:.1f}")
        axis.view_init(elev=elev, azim=azim)

        legend_items = [
            Line2D([0], [0], marker="s", color="w", label="grounded obstacle", markerfacecolor="#4c4c4c", markersize=10),
            Line2D([0], [0], marker="s", color="w", label="canopy obstacle", markerfacecolor="#2ca02c", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="uav", markerfacecolor="#1f77b4", markersize=8),
        ]
        if self.target_point is not None:
            legend_items.append(
                Line2D([0], [0], marker="o", color="w", label="target", markerfacecolor="red", markersize=8)
            )
        axis.legend(handles=legend_items, loc="upper right")
        plt.tight_layout()
        plt.show()

    def update_columns(self, added_or_changed: Mapping[Cell2D, ColumnState | Mapping[str, Any] | int | float | None], removed: set[Cell2D] | None = None):
        for cell in removed or set():
            self.columns.pop(cell, None)

        for cell, value in added_or_changed.items():
            column = self._normalize_column_state(cell, value)
            if column is None:
                self.columns.pop(cell, None)
            else:
                self.columns[cell] = column

        self.revision += 1
        self._rebuild_local_views()

    def clear_column(self, cell: Cell2D):
        self.update_columns({}, removed={cell})


def get_default_mask_dir() -> Path:
    latest_run_dir = get_latest_segmentation_run_dir(DEFAULT_CONFIG.runs_dir / "segment")
    return (latest_run_dir / "masks").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and display a local 2.5D map view from segmentation masks.")
    parser.add_argument("--mask-dir", type=str, default=None, help="Directory containing mask PNG files.")
    parser.add_argument("--grid-scale", type=float, default=DEFAULT_CONFIG.default_grid_scale, help="Pixel-to-grid scale.")

    # Optional: pixel-domain tiling (xywh in pixel coordinates) for each mask instance.
    parser.add_argument("--tile-w-px", type=int, default=None, help="Tile width in pixels for xywh tiling export.")
    parser.add_argument("--tile-h-px", type=int, default=None, help="Tile height in pixels for xywh tiling export.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.3,
        help="Min foreground coverage ratio within a tile to keep it (0~1).",
    )

    parser.add_argument("--uav-altitude", type=float, default=DEFAULT_UAV_ALTITUDE, help="Default UAV altitude in the local 2.5D view.")
    parser.add_argument("--elev", type=float, default=28.0, help="3D camera elevation angle.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth angle.")
    parser.add_argument("--ros2-publish", action="store_true", help="Publish octomap occupancy image via ROS2 topic.")
    parser.add_argument("--ros-topic", type=str, default="/octomap/image", help="ROS2 image topic name.")
    parser.add_argument("--ros-frame-id", type=str, default="map", help="ROS2 image frame_id.")
    parser.add_argument("--ros-rate", type=float, default=2.0, help="ROS2 publish rate (Hz).")
    parser.add_argument("--occupied-value", type=int, default=255, help="Occupancy pixel value (0~255).")
    parser.add_argument("--use-columns", action="store_true", help="Use all columns for occupancy map instead of blocked obstacles only.")
    parser.add_argument("--ros-publish-2p5d", action="store_true", help="Publish 2.5D map topics (OccupancyGrid + PointCloud2).")
    parser.add_argument("--ros-occ-topic", type=str, default="/octomap/occupancy", help="ROS2 OccupancyGrid topic.")
    parser.add_argument("--ros-cloud-topic", type=str, default="/octomap/points", help="ROS2 PointCloud2 topic.")
    parser.add_argument("--cell-size", type=float, default=1.0, help="Grid cell size in meters for ROS map/point projection.")
    parser.add_argument("--z-step", type=float, default=0.5, help="Vertical sampling step (m) for 2.5D point cloud.")
    return parser.parse_args()


def publish_occ2d_ros2(
    occ2d: np.ndarray,
    *,
    topic: str,
    frame_id: str,
    rate_hz: float,
) -> None:
    try:
        rclpy = importlib.import_module("rclpy")
        rclpy_node = importlib.import_module("rclpy.node")
        sensor_msgs_msg = importlib.import_module("sensor_msgs.msg")
        Node = getattr(rclpy_node, "Node")
        Image = getattr(sensor_msgs_msg, "Image")
    except Exception as exc:
        raise RuntimeError("ROS2 dependencies not available. Please install/source ROS2 Python environment.") from exc

    class OctoMapImagePublisher(Node):
        def __init__(self):
            super().__init__("octomap_image_publisher")
            self.publisher = self.create_publisher(Image, topic, 10)
            self.msg = _to_ros_image_msg(occ2d, frame_id)
            period = 1.0 / max(float(rate_hz), 0.1)
            self.timer = self.create_timer(period, self._on_timer)

        def _on_timer(self):
            self.msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(self.msg)

    rclpy.init()
    node = OctoMapImagePublisher()
    node.get_logger().info(f"Publishing occupancy image on topic: {topic}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def publish_map_2p5d_ros2(
    occ2d: np.ndarray,
    columns: Mapping[Cell2D, ColumnState],
    *,
    frame_id: str,
    occ_topic: str,
    cloud_topic: str,
    rate_hz: float,
    cell_size: float,
    z_step: float,
) -> None:
    try:
        rclpy = importlib.import_module("rclpy")
        rclpy_node = importlib.import_module("rclpy.node")
        nav_msgs_msg = importlib.import_module("nav_msgs.msg")
        sensor_msgs_msg = importlib.import_module("sensor_msgs.msg")
        Node = getattr(rclpy_node, "Node")
        OccupancyGrid = getattr(nav_msgs_msg, "OccupancyGrid")
        PointCloud2 = getattr(sensor_msgs_msg, "PointCloud2")
    except Exception as exc:
        raise RuntimeError("ROS2 dependencies not available. Please install/source ROS2 Python environment.") from exc

    class OctoMap2p5DPublisher(Node):
        def __init__(self):
            super().__init__("octomap_2p5d_publisher")
            self.occ_pub = self.create_publisher(OccupancyGrid, occ_topic, 10)
            self.cloud_pub = self.create_publisher(PointCloud2, cloud_topic, 10)
            self.occ_msg = _to_occupancy_grid_msg(occ2d, frame_id=frame_id, cell_size=cell_size)
            self.cloud_msg = _to_pointcloud2_msg_from_columns(
                columns,
                frame_id=frame_id,
                cell_size=cell_size,
                z_step=z_step,
            )
            period = 1.0 / max(float(rate_hz), 0.1)
            self.timer = self.create_timer(period, self._on_timer)

        def _on_timer(self):
            stamp = self.get_clock().now().to_msg()
            self.occ_msg.header.stamp = stamp
            self.occ_msg.info.map_load_time = stamp
            self.cloud_msg.header.stamp = stamp
            self.occ_pub.publish(self.occ_msg)
            self.cloud_pub.publish(self.cloud_msg)

    rclpy.init()
    node = OctoMap2p5DPublisher()
    node.get_logger().info(
        f"Publishing 2.5D map: occ={occ_topic}, cloud={cloud_topic}, frame={frame_id}"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    args = parse_args()
    mask_dir = resolve_path(args.mask_dir, get_default_mask_dir())
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    grid_w, grid_h = OctoMap.infer_grid_size(mask_dir, args.grid_scale)
    octomap = OctoMap(grid_w=grid_w, grid_h=grid_h, grid_scale=args.grid_scale)
    mask_list = load_mask_entries(mask_dir, octomap.grid_handler)

    # Build grid + columns; optionally attach pixel-domain tiles to grid_handler and sync into octomap snapshot.
    octomap.grid_handler.batch_masks_to_obs(
        mask_list,
        tile_w_px=args.tile_w_px,
        tile_h_px=args.tile_h_px,
        min_coverage=args.min_coverage,
    )
    octomap._sync_from_grid_handler()
    octomap.build_octomap(octomap.grid_handler.blocked_obstacles)

    occ2d = octomap.build_occ2d(occupied_value=args.occupied_value, use_columns=args.use_columns)

    if args.ros_publish_2p5d:
        publish_map_2p5d_ros2(
            occ2d,
            octomap.columns,
            frame_id=args.ros_frame_id,
            occ_topic=args.ros_occ_topic,
            cloud_topic=args.ros_cloud_topic,
            rate_hz=args.ros_rate,
            cell_size=args.cell_size,
            z_step=args.z_step,
        )
        return

    if args.ros2_publish:
        publish_occ2d_ros2(
            occ2d,
            topic=args.ros_topic,
            frame_id=args.ros_frame_id,
            rate_hz=args.ros_rate,
        )
        return

    octomap.show_local_map_3d(elev=args.elev, azim=args.azim, uav_altitude=args.uav_altitude)


if __name__ == "__main__":
    main()

# python -m app.mapping.octomap --mask-dir runs/segment/exp2/masks
# python -m app.mapping.octomap --mask-dir runs/segment/exp2/masks --tile-w-px 16 --tile-h-px 16 --min-coverage 0.3
