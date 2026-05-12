from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

MaskEntry = list
TARGET_CLASS = 0

# “可通行但有代价”的类别（不应出现在 batch_masks_to_obs 的返回集合 obs 中）
TRAVERSABLE_CLASSES: set[int] = {
    2,
    3,
    4,
    5,
    6,
}
TRAVERSABLE_CLASS_PENALTIES: dict[int, float] = {
    2: 0,
    3: 4.0,
    4: 4.0,
    5: 1.0,
    6: 4.0,
}
CLASS_HEIGHTS: dict[int, int] = {
    1: 3,   # car
    #2: 0,   # cover
    3: 5,   # road_sign
    4: 7,   # tree
    5: 2,   # person
    6: 8,   # forest
    #7: 0,   # road
    #8: 0,   # road_base
    9: 10,  # house
}


def parse_mask_filename(filename: str) -> dict[str, int | str | None]:
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return {"filename": filename, "image_stem": stem, "mask_index": None}

    body = parts[1]
    image_stem = body
    mask_index = None
    if "_" in body:
        maybe_stem, maybe_index = body.rsplit("_", 1)
        if maybe_index.isdigit():
            image_stem = maybe_stem
            mask_index = int(maybe_index)

    return {
        "filename": filename,
        "image_stem": image_stem,
        "mask_index": mask_index,
    }


def load_label_confidences(run_dir: str | Path) -> dict[str, float]:
    labels_dir = Path(run_dir) / "labels"
    confidences: dict[str, float] = {}
    if not labels_dir.exists():
        return confidences

    for label_file in sorted(labels_dir.glob("*.txt")):
        image_stem = label_file.stem
        try:
            with label_file.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle, delimiter=" ")
                for mask_index, row in enumerate(reader):
                    values = [value for value in row if value]
                    if len(values) < 2:
                        continue
                    try:
                        confidence = float(values[-1])
                    except ValueError:
                        continue
                    confidences[f"{image_stem}:{mask_index}"] = confidence
        except OSError:
            continue

    return confidences


class GridMapHandler:
    def __init__(self, grid_w: int = 64, grid_h: int = 64, grid_scale: int = 10):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.grid_scale = grid_scale

        # Optional: keep sparse pixel-domain tiling result for each mask instance.
        # Each item is a list[dict] with keys: x,y,w,h,coverage
        self.mask_instance_tiles: list[dict[str, object]] = []

        self.obstacles: set[tuple[int, int]] = set()
        self.blocked_obstacles: set[tuple[int, int]] = set()
        self.traversable_obstacles: set[tuple[int, int]] = set()
        self.terrain_penalties: dict[tuple[int, int], float] = {}
        self.obstacle_heights: dict[tuple[int, int], int] = {}
        self.obstacle_class_ids: dict[tuple[int, int], int] = {}
        self.mask_instances: list[dict[str, object]] = []
        self.target_point: tuple[int, int] | None = None
        self.OBSTACLE_CLASSES = set(CLASS_HEIGHTS.keys())
        self.TRAVERSABLE_CLASSES = set(TRAVERSABLE_CLASSES)
        self.TARGET_CLASS = TARGET_CLASS

    def batch_masks_to_obs(
        self,
        mask_list: Sequence[MaskEntry],
        *,
        tile_w_px: int | None = None,
        tile_h_px: int | None = None,
        min_coverage: float = 0.3,
    ):
        from app.mapping.pixel_tiles import mask_to_tiles_xywh

        full_obs: set[tuple[int, int]] = set()
        blocked_obs: set[tuple[int, int]] = set()
        traversable_obs: set[tuple[int, int]] = set()
        terrain_penalties: dict[tuple[int, int], float] = {}
        obstacle_heights: dict[tuple[int, int], int] = {}
        obstacle_class_ids: dict[tuple[int, int], int] = {}
        mask_instances: list[dict[str, object]] = []
        mask_instance_tiles: list[dict[str, object]] = []
        target_point = None

        if not mask_list:
            print("无有效障碍物掩码")
            self.obstacles = full_obs
            self.blocked_obstacles = blocked_obs
            self.traversable_obstacles = traversable_obs
            self.terrain_penalties = terrain_penalties
            self.obstacle_heights = obstacle_heights
            self.obstacle_class_ids = obstacle_class_ids
            self.mask_instances = mask_instances
            self.mask_instance_tiles = mask_instance_tiles
            self.target_point = target_point
            return blocked_obs, target_point

        print(f"开始处理 {len(mask_list)} 个【障碍物】掩码\n")

        for item in mask_list:
            try:
                cls_id = int(item[1])
                confidence = float(item[2])
                mask = item[3]
            except (TypeError, ValueError, IndexError):
                continue

            metadata = item[4] if len(item) > 4 and isinstance(item[4], dict) else {}

            if mask is None or not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
                continue

            ys, xs = np.where(mask > 0)
            if len(xs) < 50:
                continue

            # IMPORTANT: target (class 0) should NOT be treated as an obstacle.
            if cls_id == self.TARGET_CLASS:
                avg_x = float(xs.mean())
                avg_y = float(ys.mean())
                gx = int(avg_x // self.grid_scale)
                gy = int(avg_y // self.grid_scale)
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    target_point = (gx, gy)
                    print(f"找到投递点：{target_point}")
                continue

            if cls_id in self.OBSTACLE_CLASSES:
                height = CLASS_HEIGHTS[cls_id]
                is_traversable = cls_id in self.TRAVERSABLE_CLASSES
                print(f"障碍物 | 类别:{cls_id} | 高度:{height} | 像素:{len(xs)}")
                instance_cells: set[tuple[int, int]] = set()
                for x, y in zip(xs, ys):
                    gx = x // self.grid_scale
                    gy = y // self.grid_scale
                    if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                        cell = (gx, gy)
                        instance_cells.add(cell)
                        full_obs.add(cell)
                        if is_traversable:
                            if cell not in blocked_obs:
                                traversable_obs.add(cell)
                                terrain_penalties[cell] = max(
                                    terrain_penalties.get(cell, 0.0),
                                    TRAVERSABLE_CLASS_PENALTIES.get(cls_id, 0.0),
                                )
                        else:
                            blocked_obs.add(cell)
                            traversable_obs.discard(cell)
                            terrain_penalties.pop(cell, None)
                        previous_height = obstacle_heights.get(cell, 0)
                        if height > previous_height:
                            obstacle_heights[cell] = height
                            obstacle_class_ids[cell] = cls_id

                if instance_cells:
                    avg_x = sum(cell[0] for cell in instance_cells) / len(instance_cells)
                    avg_y = sum(cell[1] for cell in instance_cells) / len(instance_cells)
                    center_cell = min(instance_cells, key=lambda cell: (cell[0] - avg_x) ** 2 + (cell[1] - avg_y) ** 2)
                    mask_instances.append(
                        {
                            "class_id": cls_id,
                            "confidence": confidence,
                            "mask_index": metadata.get("mask_index"),
                            "image_stem": metadata.get("image_stem"),
                            "filename": metadata.get("filename"),
                            "cells": tuple(sorted(instance_cells)),
                            "center_cell": center_cell,
                        }
                    )

                    # Optional: tile the pixel region with fixed (w,h) pixels and store sparse xywh.
                    if tile_w_px is not None and tile_h_px is not None:
                        tiles = mask_to_tiles_xywh(
                            mask,
                            tile_w_px=int(tile_w_px),
                            tile_h_px=int(tile_h_px),
                            min_coverage=float(min_coverage),
                        )
                        mask_instance_tiles.append(
                            {
                                "class_id": cls_id,
                                "confidence": confidence,
                                "mask_index": metadata.get("mask_index"),
                                "image_stem": metadata.get("image_stem"),
                                "filename": metadata.get("filename"),
                                "tiles": tuple(
                                    {
                                        "x": t.x,
                                        "y": t.y,
                                        "w": t.w,
                                        "h": t.h,
                                        "coverage": t.coverage,
                                    }
                                    for t in tiles
                                ),
                            }
                        )
            elif cls_id == self.TARGET_CLASS:
                cx = int(np.mean(xs) // self.grid_scale)
                cy = int(np.mean(ys) // self.grid_scale)
                target_point = (cx, cy)
                print(f"找到投递点：{target_point}")

        self.obstacles = full_obs
        self.blocked_obstacles = blocked_obs
        self.traversable_obstacles = traversable_obs
        self.terrain_penalties = terrain_penalties
        self.obstacle_heights = obstacle_heights
        self.obstacle_class_ids = obstacle_class_ids
        self.mask_instances = mask_instances
        self.mask_instance_tiles = mask_instance_tiles
        self.target_point = target_point
        print(f"\n栅格地图完成 | 障碍物栅格：{len(full_obs)}")

        # Return semantics (per tests):
        # - If the frame contains any non-traversable obstacles, return those (blocked only).
        # - If the frame contains only traversable obstacles (e.g. tree/forest), return empty set.
        # - Full footprint is always available as `self.obstacles`.
        if blocked_obs:
            return blocked_obs, target_point
        return set(), target_point

    def show_map(self):
        plt.figure(figsize=(8, 8))
        plt.xlim(0, self.grid_w)
        plt.ylim(0, self.grid_h)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.title("Pure Obstacle Grid Map", fontsize=14)

        for gx, gy in self.obstacles:
            rect = plt.Rectangle((gx, gy), 1, 1, color="black")
            plt.gca().add_patch(rect)

        if self.target_point:
            tx, ty = self.target_point
            plt.plot(tx + 0.5, ty + 0.5, "ro", markersize=10, label="Target")
            plt.legend()

        plt.axis("equal")
        plt.show()


def load_mask_entries(
    mask_dir: str | Path,
    grid_handler: GridMapHandler | None = None,
    confidence_map: dict[str, float] | None = None,
) -> List[MaskEntry]:
    mask_path = Path(mask_dir)
    if not mask_path.exists():
        return []

    obstacle_classes = None if grid_handler is None else grid_handler.OBSTACLE_CLASSES
    target_class = None if grid_handler is None else grid_handler.TARGET_CLASS
    current_confidence_map = confidence_map or load_label_confidences(mask_path.parent)

    mask_list: List[MaskEntry] = []
    for filename in sorted(os.listdir(mask_path)):
        if not filename.endswith(".png"):
            continue

        try:
            cls_id = int(filename.split("_")[0])
        except ValueError:
            continue

        if obstacle_classes is not None and target_class is not None:
            if cls_id not in obstacle_classes and cls_id != target_class:
                continue

        metadata = parse_mask_filename(filename)
        mask = cv2.imread(str(mask_path / filename), 0)
        if mask is None:
            continue

        confidence = 1.0
        image_stem = metadata.get("image_stem")
        mask_index = metadata.get("mask_index")
        if isinstance(image_stem, str) and isinstance(mask_index, int):
            confidence = current_confidence_map.get(f"{image_stem}:{mask_index}", 1.0)

        mask_list.append([None, cls_id, confidence, mask, metadata])

    return mask_list


def group_mask_entries_by_stem(mask_list: Sequence[MaskEntry]) -> dict[str, list[MaskEntry]]:
    grouped_masks: dict[str, list[MaskEntry]] = {}
    for item in mask_list:
        metadata = item[4] if len(item) > 4 and isinstance(item[4], dict) else {}
        image_stem = metadata.get("image_stem")
        if not isinstance(image_stem, str) or not image_stem:
            filename = metadata.get("filename")
            if isinstance(filename, str) and filename:
                image_stem = Path(filename).stem
            else:
                continue

        grouped_masks.setdefault(image_stem, []).append(item)

    return {image_stem: grouped_masks[image_stem] for image_stem in sorted(grouped_masks)}


def load_grouped_mask_entries(mask_dir: str | Path, grid_handler: GridMapHandler | None = None) -> dict[str, list[MaskEntry]]:
    return group_mask_entries_by_stem(load_mask_entries(mask_dir, grid_handler))
