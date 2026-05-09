'''
实时规划
'''
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.mapping.grid_map import (
    CLASS_HEIGHTS,
    GridMapHandler,
    TRAVERSABLE_CLASSES,
    load_mask_entries,
)
from app.planning.dstar_lite import DStarLite
from app.planning.pathplan_batch import (
    INFO_PANEL_EDGE_COLOR,
    PLANNER_GOAL_COLOR,
    PLANNER_GRID_COLOR,
    PLANNER_PATH_COLOR,
    PLANNER_START_COLOR,
    TRAVERSABLE_OVERLAY_COLOR,
    build_class_annotations,
    create_pathplan_run_dir,
    draw_obstacles,
    draw_plan_overlay,
    get_annotation_text_color,
    get_default_pathplan_project_dir,
    get_latest_segmentation_run_dir,
    get_mask_canvas_shape,
    get_obstacle_facecolor,
    load_class_names,
)

MANUAL_MASK_DIR: str | Path | None = r"D:\\qingyu\\YosegMap\\runs\\segment\\exp2\\masks"


def get_default_mask_dir() -> Path:
    if MANUAL_MASK_DIR is not None:
        mask_dir = Path(MANUAL_MASK_DIR).expanduser()
        if not mask_dir.is_absolute():
            mask_dir = DEFAULT_CONFIG.repo_root / mask_dir
        return mask_dir.resolve()

    try:
        latest_run_dir = get_latest_segmentation_run_dir(DEFAULT_CONFIG.runs_dir / "segment")
        return (latest_run_dir / "masks").resolve()
    except FileNotFoundError:
        return DEFAULT_CONFIG.default_mask_dir


class PathPlanner:
    def __init__(
        self,
        grid_scale: int = 10,
        mask_dir: str | Path | None = None,
        output_project: str | Path | None = None,
        data_yaml: str | Path | None = None,
    ):
        self.grid_scale = grid_scale
        self.mask_dir = Path(mask_dir) if mask_dir else get_default_mask_dir()
        self.output_project = Path(output_project) if output_project else get_default_pathplan_project_dir()
        self.output_dir = create_pathplan_run_dir(self.output_project)
        self.grid_w, self.grid_h = self._adapt_map_size()
        self.obs = set()
        self.display_obs = set()
        self.traversable_obs = set()
        self.terrain_penalties = {}
        self.obstacle_heights = {}
        self.obstacle_class_ids = {}
        self.mask_instances = []
        self.class_names = self._load_class_names(data_yaml)
        self.target_point = None
        self.start = (self.grid_w // 2, self.grid_h // 2)
        self.goal = None
        self.planner = None
        self.path = []
        self.fig = None
        self.ax = None
        self.obs_patches = []
        self.class_annotations = []
        self.class_info_text = None
        self.path_line = None
        self.start_dot = None
        self.goal_dot = None
        self._init_grid_map()
        self._init_planner()
        self._init_plot()
        self.save_outputs()

    def _load_class_names(self, data_yaml: str | Path | None = None) -> dict[int, str]:
        if data_yaml is not None:
            try:
                with Path(data_yaml).open("r", encoding="utf-8") as file:
                    data = yaml.safe_load(file) or {}
            except Exception:
                return {-1: "manual"}

            names = data.get("names", {})
            class_names = {-1: "manual"}
            if isinstance(names, dict):
                for key, value in names.items():
                    try:
                        class_id = int(key)
                    except (TypeError, ValueError):
                        continue
                    class_names[class_id] = str(value)
            elif isinstance(names, list):
                for class_id, value in enumerate(names):
                    class_names[class_id] = str(value)
            return class_names

        return load_class_names()

    def _adapt_map_size(self):
        first_mask = None
        if self.mask_dir.exists():
            for mask_file in sorted(self.mask_dir.glob("*.png")):
                first_mask = cv2.imread(str(mask_file), 0)
                if first_mask is not None:
                    break

        if first_mask is not None:
            mask_h, mask_w = first_mask.shape
            grid_w = max(1, mask_w // self.grid_scale)
            grid_h = max(1, mask_h // self.grid_scale)
            print(f"地图自适应完成 | 栅格尺寸:{grid_w}x{grid_h}")
        else:
            grid_w = 64
            grid_h = 64
            print(f"使用默认栅格尺寸:{grid_w}x{grid_h}")
        return grid_w, grid_h

    def _init_grid_map(self):
        grid_handler = GridMapHandler(self.grid_w, self.grid_h, self.grid_scale)
        self.obs, self.target_point = grid_handler.batch_masks_to_obs(load_mask_entries(self.mask_dir, grid_handler))
        self.display_obs = set(grid_handler.obstacles)
        self.traversable_obs = set(grid_handler.traversable_obstacles)
        self.terrain_penalties = dict(grid_handler.terrain_penalties)
        self.obstacle_heights = dict(grid_handler.obstacle_heights)
        self.obstacle_class_ids = dict(grid_handler.obstacle_class_ids)
        self.mask_instances = list(grid_handler.mask_instances)

        if self.target_point is not None:
            self.goal = self.target_point
            print(f"已自动设置终点为投递点：{self.goal}")
        else:
            self.goal = (self.grid_w - 5, self.grid_h - 5)
            print(f"未检测到投递点，使用默认终点：{self.goal}")

    def _init_planner(self):
        self.planner = DStarLite(
            self.start,
            self.goal,
            self.obs,
            self.grid_w,
            self.grid_h,
            passable_obs=self.traversable_obs,
            terrain_penalties=self.terrain_penalties,
        )

    def _init_plot(self):
        plt.rcParams["figure.facecolor"] = "white"
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.subplots_adjust(right=0.8)
        self.ax.set_xlim(0, self.grid_w)
        self.ax.set_ylim(0, self.grid_h)
        self.ax.invert_yaxis()
        self.ax.grid(True, color=PLANNER_GRID_COLOR, linewidth=0.2)
        self.ax.set_aspect("equal")
        self.ax.set_title(
            "D* Lite Dynamic Planner\nLeft:Add Obstacle | Middle:Set Start | Right:Set Goal",
            fontsize=12,
        )
        (self.path_line,) = self.ax.plot([], [], linewidth=3, color=PLANNER_PATH_COLOR, label="D*Lite Path")
        (self.start_dot,) = self.ax.plot([], [], "o", markersize=8, color=PLANNER_START_COLOR, label="Start")
        (self.goal_dot,) = self.ax.plot([], [], "o", markersize=8, color=PLANNER_GOAL_COLOR, label="Goal")
        self.class_info_text = self.ax.text(
            1.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": INFO_PANEL_EDGE_COLOR, "alpha": 0.9},
        )
        self.ax.legend(loc="lower left")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def replan(self):
        try:
            self.path = self.planner.plan()
        except Exception:
            self.path = []

    def _get_obstacle_color(self, cell: tuple[int, int]):
        return get_obstacle_facecolor(cell, self.obstacle_heights)

    def _get_manual_obstacle_height(self) -> int:
        if CLASS_HEIGHTS:
            return max(CLASS_HEIGHTS.values())
        return 10

    def _format_class_line(self, class_id: int) -> str:
        class_name = self.class_names.get(class_id, f"class_{class_id}")
        if class_id == -1:
            height = self._get_manual_obstacle_height()
            return f"manual | height {height}"
        height = CLASS_HEIGHTS.get(class_id, self.obstacle_heights.get(class_id, 0))
        if class_id in TRAVERSABLE_CLASSES:
            return f"{class_id}: {class_name} | h={height} | passable"
        return f"{class_id}: {class_name} | h={height}"

    def _format_mask_instance_label(self, instance: dict[str, object]) -> str:
        class_id = int(instance.get("class_id", -1))
        class_name = self.class_names.get(class_id, f"class_{class_id}")
        mask_index = instance.get("mask_index")
        if isinstance(mask_index, int):
            return f"{class_name}_{mask_index}"
        return class_name

    def _get_annotation_color(self, cell: tuple[int, int]):
        color = self._get_obstacle_color(cell)
        if isinstance(color, tuple):
            gray = color[0]
            return "white" if gray < 0.45 else "black"
        return "white"

    def _build_class_annotations(self):
        return build_class_annotations(
            self.mask_instances,
            self.obstacle_heights,
            self.class_names,
        )

    def _update_class_info(self):
        if self.class_info_text is None:
            return

        present_class_ids = sorted(set(self.obstacle_class_ids.values()), key=lambda class_id: (class_id == -1, class_id))
        if not present_class_ids:
            self.class_info_text.set_text("calss: none")
            return

        lines = ["obstacle classes:"]
        for class_id in present_class_ids:
            lines.append(self._format_class_line(class_id))
        self.class_info_text.set_text("\n".join(lines))

    def get_canvas_shape(self) -> tuple[int, int]:
        canvas_shape = get_mask_canvas_shape(load_mask_entries(self.mask_dir))
        if canvas_shape is not None:
            return canvas_shape
        return self.grid_h * self.grid_scale, self.grid_w * self.grid_scale

    def save_outputs(self):
        canvas_shape = self.get_canvas_shape()
        grid_handler = GridMapHandler(self.grid_w, self.grid_h, self.grid_scale)
        grid_handler.obstacles = set(self.display_obs)
        grid_handler.blocked_obstacles = set(self.obs)
        grid_handler.traversable_obstacles = set(self.traversable_obs)
        grid_handler.terrain_penalties = dict(self.terrain_penalties)
        grid_handler.obstacle_heights = dict(self.obstacle_heights)
        grid_handler.obstacle_class_ids = dict(self.obstacle_class_ids)
        grid_handler.mask_instances = list(self.mask_instances)

        obstacle_image = draw_obstacles(
            canvas_shape,
            grid_handler,
            self.grid_scale,
            class_names=self.class_names,
            show_labels=True,
        )
        plan_image = draw_plan_overlay(
            canvas_shape,
            grid_handler,
            self.path,
            self.start,
            self.goal,
            self.grid_scale,
            class_names=self.class_names,
            show_labels=True,
        )
        cv2.imwrite(str(self.output_dir / "obstacles.png"), obstacle_image)
        cv2.imwrite(str(self.output_dir / "planned.png"), plan_image)

    def update_plot(self):
        try:
            for patch in self.obs_patches:
                patch.remove()
            self.obs_patches.clear()
            for annotation in self.class_annotations:
                annotation.remove()
            self.class_annotations.clear()
            for x, y in self.display_obs:
                rect = plt.Rectangle((x, y), 1, 1, color=self._get_obstacle_color((x, y)))
                self.ax.add_patch(rect)
                self.obs_patches.append(rect)
            for x, y in self.traversable_obs:
                overlay = plt.Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=TRAVERSABLE_OVERLAY_COLOR,
                    edgecolor="none",
                    alpha=0.60,
                )
                self.ax.add_patch(overlay)
                self.obs_patches.append(overlay)
            for (x, y), label, text_color in self._build_class_annotations():
                annotation = self.ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.35, "pad": 0.8},
                )
                self.class_annotations.append(annotation)
            self._update_class_info()
            if self.path:
                xs = [point[0] + 0.5 for point in self.path]
                ys = [point[1] + 0.5 for point in self.path]
                self.path_line.set_data(xs, ys)
            else:
                self.path_line.set_data([], [])
            self.start_dot.set_data([self.start[0] + 0.5], [self.start[1] + 0.5])
            self.goal_dot.set_data([self.goal[0] + 0.5], [self.goal[1] + 0.5])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.save_outputs()
        except Exception:
            pass

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x = int(event.xdata)
        y = int(event.ydata)
        if not (0 <= x < self.grid_w and 0 <= y < self.grid_h):
            return
        if event.button == 1:
            if (x, y) not in self.obs:
                self.obs.add((x, y))
                self.display_obs.add((x, y))
                self.obstacle_heights[(x, y)] = self._get_manual_obstacle_height()
                self.obstacle_class_ids[(x, y)] = -1
                self.planner.update_obstacles({(x, y)})
        elif event.button == 2:
            self.start = (x, y)
            self.planner.update_start(self.start)
        elif event.button == 3:
            self.goal = (x, y)
            self.planner = DStarLite(
                self.start,
                self.goal,
                self.obs,
                self.grid_w,
                self.grid_h,
                passable_obs=self.traversable_obs,
                terrain_penalties=self.terrain_penalties,
            )
        self.replan()
        self.update_plot()

    def on_motion(self, event):
        if event.button != 1 or not event.inaxes:
            return
        try:
            x = int(event.xdata)
            y = int(event.ydata)
            if 0 <= x < self.grid_w and 0 <= y < self.grid_h and (x, y) not in self.obs:
                self.obs.add((x, y))
                self.display_obs.add((x, y))
                self.obstacle_heights[(x, y)] = self._get_manual_obstacle_height()
                self.obstacle_class_ids[(x, y)] = -1
                self.planner.update_obstacles({(x, y)})
                self.replan()
                self.update_plot()
        except Exception:
            pass

    def run(self):
        self.replan()
        self.update_plot()
        print(f"\n规划器启动！\nmask 来源: {self.mask_dir}\n输出目录: {self.output_dir}")
        print("操作说明：")
        print("  - 左键点击/拖动：添加障碍物")
        print("  - 中键点击：设置起点")
        print("  - 右键点击：设置终点")
        plt.show(block=True)


def parse_args():
    parser = argparse.ArgumentParser(description="根据分割 mask 启动 D* Lite 交互式路径规划。")
    parser.add_argument("--mask-dir", type=Path, default=get_default_mask_dir(), help="mask 目录，默认读取 runs/segment 下最新 exp*/masks")
    parser.add_argument("--project", type=Path, default=get_default_pathplan_project_dir(), help="路径规划输出根目录")
    parser.add_argument("--data", type=Path, default=DEFAULT_CONFIG.data_yaml, help="数据配置 yaml")
    parser.add_argument("--grid-scale", type=int, default=DEFAULT_CONFIG.default_grid_scale, help="栅格缩放")
    return parser.parse_args()


def main():
    args = parse_args()
    planner = PathPlanner(
        grid_scale=args.grid_scale,
        mask_dir=args.mask_dir,
        output_project=args.project,
        data_yaml=args.data,
    )
    planner.run()


if __name__ == "__main__":
    main()
