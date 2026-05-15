from __future__ import annotations

import importlib
import struct
import threading
import time
from dataclasses import dataclass
from http import server

import cv2
import numpy as np

from app.inference.onnx_realtime import (
    detections_to_mask_entries,
    extract_prediction_and_proto,
    postprocess_segmentation_outputs,
)
from app.inference.rknn_realtime import RknnRealtimeSegmenter
from app.mapping.grid_map import GridMapHandler
from app.planning.pathplan_batch import render_plan_on_frame


@dataclass
class MappingResult:
    planned_frame: np.ndarray
    grid_handler: object
    plan_result: dict[str, object]
    post_ms: float
    postprocess_ms: float
    planning_ms: float
    render_ms: float


class PerfMeter:
    def __init__(self, window: int = 10) -> None:
        self.window = max(1, int(window))
        self._acc: dict[str, float] = {}
        self._count = 0

    def push(self, **values: float) -> str | None:
        for k, v in values.items():
            self._acc[k] = self._acc.get(k, 0.0) + float(v)
        self._count += 1
        if self._count < self.window:
            return None
        parts = [f"{k}={self._acc[k]/self._count:.2f}ms" for k in ("prep", "infer", "post", "postprocess", "planning", "render", "total") if k in self._acc]
        msg = f"[perf] {' '.join(parts)} (n={self._count})"
        self._acc.clear()
        self._count = 0
        return msg


class MjpegFrameStore:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.payload: bytes | None = None
        self.sequence = 0

    def update(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        payload = encoded.tobytes()
        with self.condition:
            self.payload = payload
            self.sequence += 1
            self.condition.notify_all()

    def wait_for_frame(self, last_sequence: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        with self.condition:
            if self.sequence == last_sequence:
                self.condition.wait(timeout=timeout)
        return self.sequence, self.payload


class ThreadingMjpegServer(server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_cls, frame_store: MjpegFrameStore, stream_path: str):
        super().__init__(server_address, handler_cls)
        self.frame_store = frame_store
        self.stream_path = stream_path


class MjpegRequestHandler(server.BaseHTTPRequestHandler):
    server: ThreadingMjpegServer

    def do_GET(self) -> None:
        if self.path not in {self.server.stream_path, "/"}:
            self.send_error(404)
            return
        if self.path == "/":
            body = (
                f"<html><body><img src=\"{self.server.stream_path}\" style=\"max-width:100%;height:auto;\"></body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        sequence = 0
        try:
            while True:
                sequence, payload = self.server.frame_store.wait_for_frame(sequence)
                if payload is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args) -> None:
        return


class MjpegStreamServer:
    def __init__(self, host: str, port: int, stream_path: str):
        self.frame_store = MjpegFrameStore()
        self.server = ThreadingMjpegServer((host, port), MjpegRequestHandler, self.frame_store, stream_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        self.frame_store.update(frame)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.0)


class AsyncMapPublisher:
    def __init__(
        self,
        *,
        frame_id: str,
        occ_topic: str,
        marker_topic: str,
        rate_hz: float,
        cell_size: float,
        z_max_cap: float,
        cloud_mode: str,
        edge_mode: str,
        ros_lite_mode: bool = False,
        marker_div: int = 5,
    ) -> None:
        self.enabled = False
        self._period = 1.0 / max(float(rate_hz), 0.1)
        self._cell_size = max(float(cell_size), 1e-6)
        self._z_max_cap = max(float(z_max_cap), 0.0)
        self._cloud_mode = cloud_mode
        self._edge_mode = edge_mode
        self._ros_lite_mode = bool(ros_lite_mode)
        self._marker_div = max(1, int(marker_div))
        self._frame_id = frame_id
        self._lock = threading.Lock()
        self._grid_cache = None
        self._occ_cache = None
        self._marker_cache = None
        self._stamp_ns_cache = 0
        self._frame_index_cache = 0
        self.is_dirty = False
        self._stop_event = threading.Event()
        self._error_reported = False
        self._publisher_thread: threading.Thread | None = None
        try:
            self._rclpy = importlib.import_module("rclpy")
            rclpy_node = importlib.import_module("rclpy.node")
            nav_msgs_msg = importlib.import_module("nav_msgs.msg")
            std_msgs_msg = importlib.import_module("std_msgs.msg")
            visualization_msgs_msg = importlib.import_module("visualization_msgs.msg")
            builtin_time_msg = importlib.import_module("builtin_interfaces.msg")
            self._Node = getattr(rclpy_node, "Node")
            self._OccupancyGrid = getattr(nav_msgs_msg, "OccupancyGrid")
            self._Header = getattr(std_msgs_msg, "Header")
            self._Marker = getattr(visualization_msgs_msg, "Marker")
            self._MarkerArray = getattr(visualization_msgs_msg, "MarkerArray")
            self._BuiltinTime = getattr(builtin_time_msg, "Time")
            self._rclpy.init(args=None)
            self._node = self._Node("realtime_map_publisher")
            self._occ_pub = self._node.create_publisher(self._OccupancyGrid, occ_topic, 10)
            self._marker_pub = self._node.create_publisher(self._MarkerArray, marker_topic, 10)
            self.enabled = True
            self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self._publisher_thread.start()
        except Exception as exc:
            print(f"ROS2 发布器初始化失败，已禁用发布: {exc}")

    def _stamp_from_ns(self, stamp_ns: int):
        if stamp_ns <= 0:
            return self._node.get_clock().now().to_msg()
        msg = self._BuiltinTime()
        msg.sec = int(stamp_ns // 1_000_000_000)
        msg.nanosec = int(stamp_ns % 1_000_000_000)
        return msg

    def update_data(self, grid_handler, stamp_ns: int, frame_index: int = 0) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._grid_cache = grid_handler
            self._stamp_ns_cache = int(stamp_ns)
            self._frame_index_cache = int(frame_index)
            self.is_dirty = True

    def _publish_loop(self) -> None:
        next_deadline = time.perf_counter()
        while not self._stop_event.is_set():
            next_deadline += self._period
            try:
                self._rclpy.spin_once(self._node, timeout_sec=0.0)
                with self._lock:
                    dirty = self.is_dirty
                    grid_handler = self._grid_cache
                    stamp_ns = self._stamp_ns_cache
                    frame_index = self._frame_index_cache
                    if dirty:
                        self.is_dirty = False
                if dirty and grid_handler is not None:
                    stamp = self._stamp_from_ns(stamp_ns)
                    occ_msg = self._build_occ_msg(grid_handler, stamp)
                    marker_msg = None
                    if (not self._ros_lite_mode) and frame_index > 0 and frame_index % self._marker_div == 0:
                        marker_msg = self._build_marker_msg(grid_handler, stamp)
                    with self._lock:
                        self._occ_cache = occ_msg
                        if marker_msg is not None:
                            self._marker_cache = marker_msg
                with self._lock:
                    occ_cached = self._occ_cache
                    marker_cached = self._marker_cache
                if occ_cached is not None:
                    self._occ_pub.publish(occ_cached)
                if (not self._ros_lite_mode) and marker_cached is not None and dirty:
                    self._marker_pub.publish(marker_cached)
            except Exception as exc:
                if not self._error_reported:
                    print(f"ROS2 发布失败（后续不再重复提示）: {exc}")
                    self._error_reported = True
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.perf_counter()

    def _build_occ_msg(self, grid_handler, stamp):
        h = int(grid_handler.grid_h)
        w = int(grid_handler.grid_w)
        occ = np.zeros((h, w), dtype=np.int8)
        blocked = getattr(grid_handler, "blocked_obstacles", set())
        if blocked:
            points = np.asarray(list(blocked), dtype=np.int32)
            xs_all, ys_all = points[:, 0], points[:, 1]
            valid = (xs_all >= 0) & (xs_all < w) & (ys_all >= 0) & (ys_all < h)
            xs, ys = xs_all[valid], ys_all[valid]
            if xs.size > 0:
                heights = getattr(grid_handler, "obstacle_heights", {})
                z_vals = np.asarray([float(heights.get((int(x), int(y)), 0.0)) for x, y in zip(xs.tolist(), ys.tolist())], dtype=np.float32)
                safe_cap = max(self._z_max_cap, 1e-6)
                scaled = np.minimum((z_vals / safe_cap) * 100.0, 100.0)
                costs = np.clip(scaled.astype(np.int16, copy=False), 0, 100).astype(np.int8, copy=False)
                costs = np.where((z_vals <= 0.0) & (costs == 0), 1, costs).astype(np.int8, copy=False)
                occ[ys, xs] = costs
        msg = self._OccupancyGrid()
        msg.header = self._Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.info.map_load_time = stamp
        msg.info.resolution = self._cell_size
        msg.info.width = w
        msg.info.height = h
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = np.ascontiguousarray(occ).reshape(-1).tolist()
        return msg

    def _build_marker_msg(self, grid_handler, stamp):
        marker_array = self._MarkerArray()
        blocked = getattr(grid_handler, "blocked_obstacles", set())
        if not blocked:
            return marker_array
        heights = getattr(grid_handler, "obstacle_heights", {})
        safe_cap = max(self._z_max_cap, 1e-6)
        cells = set(blocked)
        if self._cloud_mode == "edge":
            nbs = ((1, 0), (-1, 0), (0, 1), (0, -1)) if self._edge_mode != "8n" else ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
            edge_cells = set()
            for x, y in cells:
                for dx, dy in nbs:
                    if (x + dx, y + dy) not in cells:
                        edge_cells.add((x, y))
                        break
            cells = edge_cells
        for i, (x_i, y_i) in enumerate(list(cells)):
            z = max(0.01, min(float(heights.get((x_i, y_i), 0.0)), self._z_max_cap))
            ratio = max(0.0, min(1.0, z / safe_cap))
            m = self._Marker()
            m.header = self._Header()
            m.header.stamp = stamp
            m.header.frame_id = self._frame_id
            m.ns = "height_cells"
            m.id = i
            m.type = self._Marker.CUBE
            m.action = self._Marker.ADD
            m.pose.position.x = (float(x_i) + 0.5) * self._cell_size
            m.pose.position.y = (float(y_i) + 0.5) * self._cell_size
            m.pose.position.z = z * 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = self._cell_size
            m.scale.y = self._cell_size
            m.scale.z = z
            m.color.a = 0.75
            m.color.r = ratio
            m.color.g = 1.0 - abs(2.0 * ratio - 1.0)
            m.color.b = 1.0 - ratio
            marker_array.markers.append(m)
        return marker_array

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self._stop_event.set()
            if self._publisher_thread is not None:
                self._publisher_thread.join(timeout=1.0)
            self._node.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass
        self.enabled = False


RealtimeRos2MapPublisher = AsyncMapPublisher


def map_from_outputs(
    frame: np.ndarray,
    frame_stem: str,
    outputs: list[np.ndarray],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]],
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    map_only: bool = False,
    planner_state: dict[str, object] | None = None,
) -> MappingResult:
    backend_name = "RKNN" if isinstance(segmenter, RknnRealtimeSegmenter) else "ONNX"
    t_post0 = time.perf_counter()
    prediction, proto = extract_prediction_and_proto(outputs, backend_name)
    detections, masks = postprocess_segmentation_outputs(
        prediction,
        proto,
        frame.shape[:2],
        segmenter.imgsz,
        ratio_pad,
        segmenter.conf_thres,
        segmenter.iou_thres,
        segmenter.max_det,
        segmenter.classes,
        segmenter.agnostic_nms,
    )
    postprocess_ms = (time.perf_counter() - t_post0) * 1000.0
    t_plan0 = time.perf_counter()
    mask_entries = detections_to_mask_entries(detections, masks, frame_stem)
    frame_h, frame_w = frame.shape[:2]
    grid_w = max(1, frame_w // grid_scale)
    grid_h = max(1, frame_h // grid_scale)
    grid_handler = GridMapHandler(grid_w=grid_w, grid_h=grid_h, grid_scale=grid_scale)
    obs, target_point = grid_handler.batch_masks_to_obs(mask_entries)
    start = (grid_w // 2, grid_h // 2)
    goal = target_point if target_point is not None else (max(0, grid_w - 5), max(0, grid_h - 5))
    path: list[tuple[int, int]] = []
    obs_set = obs if isinstance(obs, set) else set(obs)
    passable_raw = grid_handler.traversable_obstacles
    passable_set = passable_raw if isinstance(passable_raw, set) else set(passable_raw)
    terrain_raw = grid_handler.terrain_penalties
    terrain_map = terrain_raw if isinstance(terrain_raw, dict) else dict(terrain_raw)
    if not map_only:
        from app.planning.dstar_lite import DStarLite

        planner = None
        reusable = False
        if planner_state is not None:
            prev = planner_state.get("planner")
            if prev is not None:
                same_shape = planner_state.get("grid_w") == grid_w and planner_state.get("grid_h") == grid_h
                same_passable = planner_state.get("passable_obs") == passable_set
                same_terrain = planner_state.get("terrain_penalties") == terrain_map
                if same_shape and same_passable and same_terrain:
                    prev_obs = planner_state.get("obs_set")
                    if not isinstance(prev_obs, set):
                        prev_obs = set(prev_obs or ())
                    removed = prev_obs - obs_set
                    if not removed:
                        planner = prev
                        if planner_state.get("start") != start:
                            planner.update_start(start)
                        if planner_state.get("goal") != goal:
                            planner.update_goal(goal)
                        added = obs_set - prev_obs
                        if added:
                            planner.update_obstacles(added)
                        reusable = True

        if planner is None:
            planner = DStarLite(
                start,
                goal,
                obs_set,
                grid_w,
                grid_h,
                passable_obs=passable_set,
                terrain_penalties=terrain_map,
            )

        if planner_state is not None:
            planner_state["planner"] = planner
            planner_state["grid_w"] = grid_w
            planner_state["grid_h"] = grid_h
            planner_state["start"] = start
            planner_state["goal"] = goal
            planner_state["obs_set"] = obs_set
            planner_state["passable_obs"] = passable_set
            planner_state["terrain_penalties"] = terrain_map
            planner_state["planner_reused"] = reusable
        try:
            path = planner.plan()
        except Exception:
            path = []
    plan_result = {
        "grid_handler": grid_handler,
        "path": path,
        "start": start,
        "goal": goal,
    }
    planning_ms = (time.perf_counter() - t_plan0) * 1000.0
    t_render0 = time.perf_counter()
    rendered = render_plan_on_frame(
        frame,
        plan_result["grid_handler"],
        plan_result["path"],
        plan_result["start"],
        plan_result["goal"],
        grid_scale,
        class_names=class_names,
        show_labels=True,
    )
    render_ms = (time.perf_counter() - t_render0) * 1000.0
    post_ms = postprocess_ms + planning_ms + render_ms
    return MappingResult(
        planned_frame=np.ascontiguousarray(rendered),
        grid_handler=plan_result["grid_handler"],
        plan_result=plan_result,
        post_ms=post_ms,
        postprocess_ms=postprocess_ms,
        planning_ms=planning_ms,
        render_ms=render_ms,
    )
