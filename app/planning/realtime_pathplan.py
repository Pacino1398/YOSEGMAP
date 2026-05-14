from __future__ import annotations

import argparse
import importlib
import os
import queue
import struct
import sys
import threading
import time
from http import server
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.inference.onnx_realtime import (
    OnnxRealtimeSegmenter,
    detections_to_mask_entries,
    extract_prediction_and_proto,
    get_default_data_yaml,
    get_default_realtime_weights,
    postprocess_segmentation_outputs,
)
from app.inference.rknn_realtime import RknnRealtimeSegmenter
from app.inference.segmentation import get_default_source_dir
from app.paths import resolve_path
from app.planning.pathplan_batch import (
    build_plan_result,
    create_pathplan_run_dir,
    get_default_pathplan_project_dir,
    get_frame_stem,
    is_image_file,
    is_video_file,
    iter_source_media,
    load_class_names,
    render_plan_on_frame,
)

STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")
WINDOW_NAME = "realtime_pathplan"
DEFAULT_REMOTE_PATH = "/stream.mjpg"
# Deployment notes:
# 1) Recommend pinning ROS publish process to LITTLE cores:
#    taskset -c 0-3 python app/planning/realtime_pathplan.py ...
#    and keep big cores (4-7) for inference.
# 2) For lower PointCloud2 serialization overhead, prefer:
#    --cloud-mode edge --z-step 0.8


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


class AsyncMapPublisher:
    def __init__(
        self,
        *,
        frame_id: str,
        occ_topic: str,
        cloud_topic: str,
        rate_hz: float,
        cell_size: float,
        z_step: float,
        z_max_cap: float,
        xy_spread: float,
        xy_samples: int,
        cloud_mode: str,
        edge_mode: str,
        z_style: str,
        ros_publish_occ_only: bool = False,
    ) -> None:
        self.enabled = False
        self._period = 1.0 / max(float(rate_hz), 0.1)
        self._cell_size = max(float(cell_size), 1e-6)
        self._z_step = max(float(z_step), 0.05)
        self._z_max_cap = max(float(z_max_cap), 0.0)
        self._xy_spread = max(float(xy_spread), 0.0)
        self._xy_samples = max(1, int(xy_samples))
        self._cloud_mode = cloud_mode
        self._edge_mode = edge_mode
        self._z_style = z_style
        self._ros_publish_occ_only = bool(ros_publish_occ_only)
        self._frame_id = frame_id
        self._occ_topic = occ_topic
        self._cloud_topic = cloud_topic
        self._error_reported = False
        self._lock = threading.Lock()
        self._grid_cache = None
        self._occ_cache = None
        self._cloud_cache = None
        self._stamp_ns_cache = 0
        self.is_dirty = False
        self._stop_event = threading.Event()
        self._publisher_thread: threading.Thread | None = None

        try:
            self._rclpy = importlib.import_module("rclpy")
            rclpy_node = importlib.import_module("rclpy.node")
            nav_msgs_msg = importlib.import_module("nav_msgs.msg")
            sensor_msgs_msg = importlib.import_module("sensor_msgs.msg")
            std_msgs_msg = importlib.import_module("std_msgs.msg")
            builtin_time_msg = importlib.import_module("builtin_interfaces.msg")

            self._Node = getattr(rclpy_node, "Node")
            self._OccupancyGrid = getattr(nav_msgs_msg, "OccupancyGrid")
            self._PointCloud2 = getattr(sensor_msgs_msg, "PointCloud2")
            self._PointField = getattr(sensor_msgs_msg, "PointField")
            self._Header = getattr(std_msgs_msg, "Header")
            self._BuiltinTime = getattr(builtin_time_msg, "Time")

            self._rclpy.init(args=None)
            self._node = self._Node("realtime_pathplan_2p5d_publisher")
            self._occ_pub = self._node.create_publisher(self._OccupancyGrid, self._occ_topic, 10)
            self._cloud_pub = self._node.create_publisher(self._PointCloud2, self._cloud_topic, 10)
            self.enabled = True
            self._node.get_logger().info(
                f"ROS2 2.5D publisher enabled: occ={self._occ_topic}, cloud={self._cloud_topic}, frame={self._frame_id}"
            )
            self._publisher_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self._publisher_thread.start()
        except Exception as exc:
            print(f"ROS2 发布器初始化失败，已禁用发布: {exc}")
            self.enabled = False

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

    def _stamp_from_ns(self, stamp_ns: int):
        if stamp_ns <= 0:
            return self._node.get_clock().now().to_msg()
        msg = self._BuiltinTime()
        msg.sec = int(stamp_ns // 1_000_000_000)
        msg.nanosec = int(stamp_ns % 1_000_000_000)
        return msg

    def update_data(self, grid_handler, stamp_ns: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._grid_cache = grid_handler
            self._stamp_ns_cache = int(stamp_ns)
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
                    if dirty:
                        self.is_dirty = False
                if dirty and grid_handler is not None:
                    stamp = self._stamp_from_ns(stamp_ns)
                    occ_msg = self._build_occ_msg(grid_handler, stamp)
                    cloud_msg = None if self._ros_publish_occ_only else self._build_cloud_msg(grid_handler, stamp)
                    with self._lock:
                        self._occ_cache = occ_msg
                        self._cloud_cache = cloud_msg
                with self._lock:
                    occ_cached = self._occ_cache
                    cloud_cached = self._cloud_cache
                if occ_cached is not None:
                    self._occ_pub.publish(occ_cached)
                # Dirty Bit policy: unchanged frames only heartbeat OccupancyGrid.
                if dirty and (not self._ros_publish_occ_only) and cloud_cached is not None:
                    self._cloud_pub.publish(cloud_cached)
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
            xs_all = points[:, 0]
            ys_all = points[:, 1]
            valid = (xs_all >= 0) & (xs_all < w) & (ys_all >= 0) & (ys_all < h)
            xs = xs_all[valid]
            ys = ys_all[valid]
            if xs.size > 0:
                heights = getattr(grid_handler, "obstacle_heights", {})
                if heights:
                    z_vals = np.asarray(
                        [float(heights.get((int(x), int(y)), 0.0)) for x, y in zip(xs.tolist(), ys.tolist())],
                        dtype=np.float32,
                    )
                else:
                    z_vals = np.zeros((xs.shape[0],), dtype=np.float32)
                safe_cap = max(self._z_max_cap, 1e-6)
                scaled = np.minimum((z_vals / safe_cap) * 100.0, 100.0)
                costs = scaled.astype(np.int16, copy=False)
                costs = np.clip(costs, 0, 100).astype(np.int8, copy=False)
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
        # RViz2 tip:
        # Add "Map" display -> topic: /octomap/occupancy
        # In Map display, choose Costmap-style color interpretation to highlight [1..100] height costs.
        return msg

    def _build_cloud_msg(self, grid_handler, stamp):
        points: list[tuple[float, float, float, float]] = []
        heights = getattr(grid_handler, "obstacle_heights", {})
        all_cells = set(getattr(grid_handler, "blocked_obstacles", set()))
        cells = all_cells
        if self._cloud_mode == "edge":
            if self._edge_mode == "8n":
                nbs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
            else:
                nbs = ((1, 0), (-1, 0), (0, 1), (0, -1))
            edge_cells = set()
            for x, y in all_cells:
                for dx, dy in nbs:
                    if (x + dx, y + dy) not in all_cells:
                        edge_cells.add((x, y))
                        break
            cells = edge_cells
        safe_cap = max(self._z_max_cap, 1e-6)

        def _pack_rgb(r: int, g: int, b: int) -> float:
            rgb_u32 = ((r & 255) << 16) | ((g & 255) << 8) | (b & 255)
            return struct.unpack("<f", struct.pack("<I", rgb_u32))[0]

        def _height_rgb(z_val: float) -> float:
            ratio = max(0.0, min(1.0, z_val / safe_cap))
            r = int(255.0 * ratio)
            g = int(255.0 * (1.0 - abs(2.0 * ratio - 1.0)))
            b = int(255.0 * (1.0 - ratio))
            return _pack_rgb(r, g, b)

        if self._xy_samples <= 1 or self._xy_spread <= 1e-6:
            xy_offsets = [(0.0, 0.0)]
        else:
            side = self._xy_samples
            step = (2.0 * self._xy_spread) / max(side - 1, 1)
            xy_offsets = []
            for ix in range(side):
                for iy in range(side):
                    ox = -self._xy_spread + ix * step
                    oy = -self._xy_spread + iy * step
                    xy_offsets.append((ox, oy))

        for x, y in cells:
            z_hi = min(float(heights.get((x, y), 1.0)), self._z_max_cap)
            z_lo = 0.0
            cx = (float(x) + 0.5) * self._cell_size
            cy = (float(y) + 0.5) * self._cell_size
            if self._z_style == "top":
                z_values = [z_hi]
            else:
                z_values = []
                z = z_lo
                while z <= z_hi + 1e-6:
                    z_values.append(z)
                    z += self._z_step
                if not z_values:
                    z_values = [z_hi]
            for z_val in z_values:
                rgb = _height_rgb(z_val)
                for ox, oy in xy_offsets:
                    points.append((cx + ox, cy + oy, z_val, rgb))

        msg = self._PointCloud2()
        msg.header = self._Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            self._PointField(name="x", offset=0, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="y", offset=4, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="z", offset=8, datatype=self._PointField.FLOAT32, count=1),
            self._PointField(name="rgb", offset=12, datatype=self._PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = b"".join(struct.pack("<ffff", px, py, pz, prgb) for px, py, pz, prgb in points)
        return msg

    def publish(self, grid_handler) -> None:
        self.update_data(grid_handler, time.time_ns())


RealtimeRos2MapPublisher = AsyncMapPublisher


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
                f"<html><body><img src=\"{self.server.stream_path}\" "
                f"style=\"max-width:100%;height:auto;\"></body></html>"
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
        self.host = host
        self.port = port
        self.stream_path = stream_path

    def start(self) -> None:
        self.thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        self.frame_store.update(frame)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.0)


def is_stream_source(source: str) -> bool:
    return source.isdigit() or source.lower().startswith(STREAM_SCHEMES)


def resolve_source(source: str | Path | None) -> str | Path:
    if source is None:
        return get_default_source_dir()
    if isinstance(source, Path):
        return resolve_path(source, get_default_source_dir())
    if is_stream_source(source):
        return source
    return resolve_path(source, get_default_source_dir())


def get_source_stem(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.stem

    if source.isdigit():
        return f"camera{source}"

    if source.lower().startswith(STREAM_SCHEMES):
        parsed = urlparse(source)
        stream_stem = Path(parsed.path).stem
        return stream_stem or "stream"

    return Path(source).stem


def resolve_backend(backend: str, weights: str | Path | None) -> str:
    if backend != "auto":
        return backend
    if weights is None:
        default_weights = get_default_realtime_weights()
        return "rknn" if default_weights.suffix.lower() == ".rknn" else "onnx"
    suffix = Path(weights).suffix.lower()
    if suffix == ".rknn":
        return "rknn"
    if suffix == ".onnx":
        return "onnx"
    raise ValueError(f"无法根据权重后缀自动判断 realtime 后端: {weights}")


def normalize_display_mode(display: str | None, view: bool) -> str:
    if display is not None:
        return display
    return "local" if view else "none"


def should_show_local(display: str) -> bool:
    return display in {"local", "both"}


def should_stream_remote(display: str) -> bool:
    return display in {"remote", "both"}


def create_segmenter(
    backend: str,
    weights: str | Path | None,
    data_yaml: str | Path | None,
    device: str | None,
    imgsz: int | tuple[int, int],
    conf_thres: float | None,
    iou_thres: float,
    dnn: bool,
    half: bool,
):
    if backend == "rknn":
        return RknnRealtimeSegmenter(
            weights=weights,
            data_yaml=data_yaml,
            device=device,
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            dnn=dnn,
            half=half,
        )
    return OnnxRealtimeSegmenter(
        weights=weights,
        data_yaml=data_yaml,
        device=device,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        dnn=dnn,
        half=half,
    )


def render_planned_frame(
    frame: np.ndarray,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    frame_stem: str,
) -> tuple[np.ndarray, dict[str, object]]:
    mask_entries = segmenter.predict_frame(frame, frame_stem)
    plan_result = build_plan_result(frame.shape[:2], mask_entries, grid_scale)
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
    return rendered, plan_result


def _render_planned_frame_from_outputs(
    frame: np.ndarray,
    frame_stem: str,
    outputs: list[np.ndarray],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]],
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
) -> tuple[np.ndarray, dict[str, object]]:
    backend_name = "RKNN" if isinstance(segmenter, RknnRealtimeSegmenter) else "ONNX"
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
    mask_entries = detections_to_mask_entries(detections, masks, frame_stem)
    plan_result = build_plan_result(frame.shape[:2], mask_entries, grid_scale)
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
    return np.ascontiguousarray(rendered), plan_result


def maybe_show_frame(frame: np.ndarray, enabled: bool, imshow_state: dict[str, bool] | None = None) -> bool:
    if not enabled:
        return False

    try:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        return key in {27, ord("q")}
    except cv2.error as exc:
        if imshow_state is not None and not imshow_state.get("warned", False):
            print(f"本机显示不可用，已自动关闭 local 显示: {exc}")
            imshow_state["warned"] = True
        if imshow_state is not None:
            imshow_state["enabled"] = False
        return False


def to_gray_view_frame(frame: np.ndarray, gray_view: bool) -> np.ndarray:
    if not gray_view:
        return frame
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _build_mpp_mjpg_gstreamer(device: str, width: int = 1280, height: int = 720, fps: int = 30) -> str:
    return (
        f"v4l2src device={device} io-mode=4 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        "jpegparse ! mppjpegdec ! "
        "videoconvert n-threads=2 ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class _LatestFrameGrabber:
    def __init__(self, capture: cv2.VideoCapture) -> None:
        self.capture = capture
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._latest_ts_ns = 0
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> "_LatestFrameGrabber":
        self._thread.start()
        return self

    def _worker(self) -> None:
        while not self._stopped.is_set():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue
            with self._lock:
                self._latest = frame
                self._latest_ts_ns = time.time_ns()

    def read_latest(self) -> tuple[bool, np.ndarray | None, int]:
        with self._lock:
            if self._latest is None:
                return False, None, 0
            return True, self._latest.copy(), self._latest_ts_ns

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=1.0)


def process_image_source(
    image_path: Path,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    show_local: bool,
    imshow_state: dict[str, bool],
    remote_server: MjpegStreamServer | None,
    ros2_publisher: RealtimeRos2MapPublisher | None = None,
    gray_view: bool = False,
) -> Path | None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    planned, plan_result = render_planned_frame(frame, segmenter, class_names, grid_scale, image_path.stem)
    if ros2_publisher is not None:
        ros2_publisher.publish(plan_result["grid_handler"])
    display_frame = to_gray_view_frame(planned, gray_view)
    if remote_server is not None:
        remote_server.update_frame(display_frame)
    output_path = None
    if run_dir is not None:
        output_path = run_dir / f"{image_path.stem}_planned.png"
        cv2.imwrite(str(output_path), planned)
    if maybe_show_frame(display_frame, show_local and imshow_state["enabled"], imshow_state):
        return output_path
    return output_path


def process_video_capture(
    capture: cv2.VideoCapture,
    source_name: str,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    show_local: bool,
    imshow_state: dict[str, bool],
    remote_server: MjpegStreamServer | None,
    ros2_publisher: RealtimeRos2MapPublisher | None = None,
    gray_view: bool = False,
    fps: float | None = None,
) -> Path | None:
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取视频流: {source_name}")

    frame_h, frame_w = frame.shape[:2]
    current_fps = fps if fps is not None else capture.get(cv2.CAP_PROP_FPS)
    if not current_fps or current_fps <= 0:
        current_fps = 30.0

    output_path = run_dir / f"{source_name}_planned.mp4" if run_dir is not None else None
    writer = None
    if output_path is not None:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            current_fps,
            (frame_w, frame_h),
        )

    frame_index = 1
    try:
        while True:
            frame_stem = get_frame_stem(Path(f"{source_name}.mp4"), frame_index)
            planned, plan_result = render_planned_frame(frame, segmenter, class_names, grid_scale, frame_stem)
            if ros2_publisher is not None:
                ros2_publisher.publish(plan_result["grid_handler"])
            display_frame = to_gray_view_frame(planned, gray_view)
            if remote_server is not None:
                remote_server.update_frame(display_frame)
            if writer is not None:
                writer.write(planned)
            if maybe_show_frame(display_frame, show_local and imshow_state["enabled"], imshow_state):
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
    finally:
        if writer is not None:
            writer.release()

    return output_path


def process_video_source(
    video_path: Path,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    show_local: bool,
    imshow_state: dict[str, bool],
    remote_server: MjpegStreamServer | None,
    ros2_publisher: RealtimeRos2MapPublisher | None = None,
    gray_view: bool = False,
) -> Path | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    try:
        return process_video_capture(
            capture,
            video_path.stem,
            segmenter,
            class_names,
            grid_scale,
            run_dir,
            show_local,
            imshow_state,
            remote_server,
            ros2_publisher,
            gray_view,
        )
    finally:
        capture.release()


def process_stream_source(
    stream_source: str,
    segmenter,
    class_names: dict[int, str],
    grid_scale: int,
    run_dir: Path | None,
    show_local: bool,
    imshow_state: dict[str, bool],
    remote_server: MjpegStreamServer | None,
    ros2_publisher: RealtimeRos2MapPublisher | None = None,
    gray_view: bool = False,
) -> Path | None:
    capture_target: int | str = int(stream_source) if stream_source.isdigit() else stream_source
    capture: cv2.VideoCapture
    if isinstance(capture_target, int):
        gst = _build_mpp_mjpg_gstreamer(device=f"/dev/video{capture_target}")
        capture = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            capture = cv2.VideoCapture(capture_target)
    else:
        capture = cv2.VideoCapture(capture_target)

    if not capture.isOpened():
        raise RuntimeError(f"无法打开流: {stream_source}")

    source_name = get_source_stem(stream_source)
    output_path = None
    writer = None
    frame_index = 1
    grabber = _LatestFrameGrabber(capture).start()
    stop_event = threading.Event()
    prep_queue: "queue.Queue[tuple[np.ndarray, str, np.ndarray, tuple[tuple[float, float], tuple[float, float]], int, int]]" = queue.Queue(maxsize=2)
    post_queue: "queue.Queue[tuple[np.ndarray, str, list[np.ndarray], tuple[tuple[float, float], tuple[float, float]], int, int]]" = queue.Queue(maxsize=2)
    display_queue: "queue.Queue[tuple[np.ndarray, np.ndarray, dict[str, object]]]" = queue.Queue(maxsize=1)
    planner_every_n = max(1, int(os.getenv("YOSEGMAP_PLAN_EVERY_N_FRAMES", "1")))

    def _preprocess_worker() -> None:
        local_index = 1
        while not stop_event.is_set():
            ok_local, frame_local, stamp_ns_local = grabber.read_latest()
            if not ok_local or frame_local is None:
                time.sleep(0.001)
                continue
            frame_stem_local = get_frame_stem(Path(f"{source_name}.mp4"), local_index)
            input_tensor_local, ratio_pad_local = segmenter.preprocess_frame(frame_local)
            input_tensor_local = np.ascontiguousarray(input_tensor_local)
            item = (frame_local, frame_stem_local, input_tensor_local, ratio_pad_local, stamp_ns_local, local_index)
            if prep_queue.full():
                try:
                    prep_queue.get_nowait()
                except queue.Empty:
                    pass
            prep_queue.put_nowait(item)
            local_index += 1

    def _postprocess_worker() -> None:
        while not stop_event.is_set():
            try:
                frame_local, frame_stem_local, outputs_local, ratio_pad_local, stamp_ns_local, frame_idx_local = post_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame_idx_local % planner_every_n != 0:
                continue
            planned_local, plan_result_local = _render_planned_frame_from_outputs(
                frame_local,
                frame_stem_local,
                outputs_local,
                ratio_pad_local,
                segmenter,
                class_names,
                grid_scale,
            )
            if ros2_publisher is not None:
                ros2_publisher.update_data(plan_result_local["grid_handler"], stamp_ns_local)
            display_frame_local = to_gray_view_frame(planned_local, gray_view)
            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except queue.Empty:
                    pass
            display_queue.put_nowait((planned_local, display_frame_local, plan_result_local))

    preprocess_thread = threading.Thread(target=_preprocess_worker, daemon=True)
    postprocess_thread = threading.Thread(target=_postprocess_worker, daemon=True)
    preprocess_thread.start()
    postprocess_thread.start()

    try:
        while True:
            try:
                frame, frame_stem, input_tensor, ratio_pad, stamp_ns, frame_idx = prep_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            outputs = segmenter._run_inference(input_tensor)

            if post_queue.full():
                try:
                    post_queue.get_nowait()
                except queue.Empty:
                    pass
            post_queue.put_nowait((frame, frame_stem, outputs, ratio_pad, stamp_ns, frame_idx))

            while True:
                try:
                    planned, display_frame, _ = display_queue.get_nowait()
                except queue.Empty:
                    break

                if writer is None and run_dir is not None:
                    frame_h, frame_w = planned.shape[:2]
                    current_fps = capture.get(cv2.CAP_PROP_FPS)
                    if not current_fps or current_fps <= 0:
                        current_fps = 30.0
                    output_path = run_dir / f"{source_name}_planned.mp4"
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        current_fps,
                        (frame_w, frame_h),
                    )

                if remote_server is not None:
                    remote_server.update_frame(display_frame)
                if writer is not None:
                    writer.write(planned)
                if maybe_show_frame(display_frame, show_local and imshow_state["enabled"], imshow_state):
                    stop_event.set()
                    break
                frame_index += 1
            if stop_event.is_set():
                break
    finally:
        stop_event.set()
        preprocess_thread.join(timeout=1.0)
        postprocess_thread.join(timeout=1.0)
        grabber.stop()
        if writer is not None:
            writer.release()
        capture.release()

    return output_path


def run_realtime_pathplan(
    source: str | Path | None = None,
    weights: str | Path | None = None,
    data_yaml: str | Path | None = None,
    project: str | Path | None = None,
    device: str | None = None,
    conf_thres: float | None = None,
    iou_thres: float = 0.45,
    imgsz: int | tuple[int, int] = 640,
    grid_scale: int | None = None,
    view: bool = False,
    save: bool = False,
    dnn: bool = False,
    half: bool = False,
    backend: str = "auto",
    display: str | None = None,
    remote_host: str = "0.0.0.0",
    remote_port: int = 8080,
    remote_path: str = DEFAULT_REMOTE_PATH,
    ros_publish_2p5d: bool = False,
    ros_publish_occ_only: bool = False,
    ros_frame_id: str = "map",
    ros_rate: float = 2.0,
    ros_occ_topic: str = "/octomap/occupancy",
    ros_cloud_topic: str = "/octomap/points",
    cell_size: float = 1.0,
    z_step: float = 0.5,
    z_max_cap: float = 12.0,
    xy_spread: float = 0.0,
    xy_samples: int = 1,
    cloud_mode: str = "edge",
    edge_mode: str = "4n",
    z_style: str = "top",
    gray_view: bool = False,
) -> Path | None:
    source_value = resolve_source(source)
    current_grid_scale = grid_scale if grid_scale is not None else DEFAULT_CONFIG.default_grid_scale
    selected_backend = resolve_backend(backend, weights)
    selected_weights = weights if weights is not None else get_default_realtime_weights()
    display_mode = normalize_display_mode(display, view)
    show_local = should_show_local(display_mode)
    enable_remote = should_stream_remote(display_mode)
    normalized_remote_path = remote_path if remote_path.startswith("/") else f"/{remote_path}"
    run_dir = None
    if save:
        project_path = resolve_path(project, get_default_pathplan_project_dir())
        run_dir = create_pathplan_run_dir(project_path)
        print(f"路径规划输出目录: {run_dir}")

    segmenter = create_segmenter(
        selected_backend,
        selected_weights,
        data_yaml,
        device,
        imgsz,
        conf_thres,
        iou_thres,
        dnn,
        half,
    )
    print(f"实时推理后端: {selected_backend} | 权重: {selected_weights}")
    class_names = load_class_names(resolve_path(data_yaml, get_default_data_yaml()))
    imshow_state = {"enabled": show_local, "warned": False}
    remote_server = MjpegStreamServer(remote_host, remote_port, normalized_remote_path) if enable_remote else None
    ros2_publisher = (
        RealtimeRos2MapPublisher(
            frame_id=ros_frame_id,
            occ_topic=ros_occ_topic,
            cloud_topic=ros_cloud_topic,
            rate_hz=ros_rate,
            cell_size=cell_size,
            z_step=z_step,
            z_max_cap=z_max_cap,
            xy_spread=xy_spread,
            xy_samples=xy_samples,
            cloud_mode=cloud_mode,
            edge_mode=edge_mode,
            z_style=z_style,
            ros_publish_occ_only=ros_publish_occ_only,
        )
        if ros_publish_2p5d
        else None
    )
    if remote_server is not None:
        remote_server.start()
        print(f"MJPEG 预览地址: http://{remote_host if remote_host != '0.0.0.0' else '127.0.0.1'}:{remote_port}{normalized_remote_path}")

    try:
        if isinstance(source_value, Path):
            if source_value.is_dir():
                for media_path in iter_source_media(source_value):
                    if is_image_file(media_path):
                        output_path = process_image_source(
                            media_path,
                            segmenter,
                            class_names,
                            current_grid_scale,
                            run_dir,
                            show_local,
                            imshow_state,
                            remote_server,
                            ros2_publisher,
                            gray_view,
                        )
                    elif is_video_file(media_path):
                        output_path = process_video_source(
                            media_path,
                            segmenter,
                            class_names,
                            current_grid_scale,
                            run_dir,
                            show_local,
                            imshow_state,
                            remote_server,
                            ros2_publisher,
                            gray_view,
                        )
                    else:
                        continue
                    if output_path is not None:
                        print(f"已保存规划结果: {output_path}")
                return run_dir

            if is_image_file(source_value):
                output_path = process_image_source(
                    source_value,
                    segmenter,
                    class_names,
                    current_grid_scale,
                    run_dir,
                    show_local,
                    imshow_state,
                    remote_server,
                    ros2_publisher,
                    gray_view,
                )
                if output_path is not None:
                    print(f"已保存规划结果: {output_path}")
                return run_dir if run_dir is not None else output_path

            output_path = process_video_source(
                source_value,
                segmenter,
                class_names,
                current_grid_scale,
                run_dir,
                show_local,
                imshow_state,
                remote_server,
                ros2_publisher,
                gray_view,
            )
            if output_path is not None:
                print(f"已保存规划结果: {output_path}")
            return run_dir if run_dir is not None else output_path

        output_path = process_stream_source(
            source_value,
            segmenter,
            class_names,
            current_grid_scale,
            run_dir,
            show_local,
            imshow_state,
            remote_server,
            ros2_publisher,
            gray_view,
        )
        if output_path is not None:
            print(f"已保存规划结果: {output_path}")
        return run_dir if run_dir is not None else output_path
    finally:
        close = getattr(segmenter, "close", None)
        if callable(close):
            close()
        if remote_server is not None:
            remote_server.close()
        if ros2_publisher is not None:
            ros2_publisher.close()
        if imshow_state["enabled"]:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="使用 ONNX 或 RKNN 分割结果直接做实时路径规划，不经过 mask 落盘中转。")
    parser.add_argument("--source", default=str(get_default_source_dir()), help="输入图片、视频、目录、摄像头索引或流地址")
    parser.add_argument("--weights", default=str(get_default_realtime_weights()), help="实时分割权重路径，支持 .onnx 或 .rknn")
    parser.add_argument("--backend", choices=("auto", "onnx", "rknn"), default="auto", help="实时推理后端")
    parser.add_argument("--data", default=str(get_default_data_yaml()), help="数据配置 yaml")
    parser.add_argument("--project", type=Path, default=get_default_pathplan_project_dir(), help="路径规划输出根目录")
    parser.add_argument("--device", default=DEFAULT_CONFIG.default_device, help="推理设备")
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONFIG.default_conf_thres, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640], help="推理尺寸，支持 --imgsz 640 或 --imgsz 640 640")
    parser.add_argument("--grid-scale", type=int, default=DEFAULT_CONFIG.default_grid_scale, help="栅格缩放")
    parser.add_argument("--display", choices=("local", "remote", "both", "none"), default=None, help="显示目标：本机窗口、上位机 MJPEG、同时显示或都不显示")
    parser.add_argument("--remote-host", default="0.0.0.0", help="MJPEG 服务绑定地址")
    parser.add_argument("--remote-port", type=int, default=8080, help="MJPEG 服务端口")
    parser.add_argument("--remote-path", default=DEFAULT_REMOTE_PATH, help="MJPEG 预览路径")
    parser.add_argument("--view", action="store_true", help="兼容旧参数，等价于 --display local")
    parser.add_argument("--ros-publish-2p5d", action="store_true", help="发布 ROS2 2.5D 地图（OccupancyGrid + PointCloud2）")
    parser.add_argument("--ros-publish-occ-only", action="store_true", help="仅发布 OccupancyGrid（高度语义编码），不发布 PointCloud2")
    parser.add_argument("--ros-frame-id", default="map", help="ROS2 frame_id")
    parser.add_argument("--ros-rate", type=float, default=10.0, help="ROS2 发布频率 Hz")
    parser.add_argument("--ros-occ-topic", default="/octomap/occupancy", help="OccupancyGrid 话题名")
    parser.add_argument("--ros-cloud-topic", default="/octomap/points", help="PointCloud2 话题名")
    parser.add_argument("--cell-size", type=float, default=1.0, help="栅格尺寸（米）")
    parser.add_argument("--z-step", type=float, default=0.5, help="点云高度采样步长（米）")
    parser.add_argument("--z-max-cap", type=float, default=12.0, help="点云灌注最大高度上限（米）")
    parser.add_argument("--xy-spread", type=float, default=0.0, help="点云加粗半径（米，0表示不加粗）")
    parser.add_argument("--xy-samples", type=int, default=2, help="点云加粗采样边长（>=1，3表示3x3扩点）")
    parser.add_argument("--cloud-mode", choices=("full", "edge"), default="full", help="点云发布模式：全量或边缘")
    parser.add_argument("--edge-mode", choices=("4n", "8n"), default="4n", help="边缘提取邻域模式")
    parser.add_argument("--z-style", choices=("top", "band"), default="top", help="高度发布方式：仅顶面或整段")
    parser.add_argument("--gray-view", action="store_true", help="本机显示与远端预览使用灰度图，减轻可视化负载")
    parser.add_argument("--nosave", action="store_true", help="只显示不保存输出（默认已不保存）")
    parser.add_argument("--dnn", action="store_true", help="使用 OpenCV DNN 加载 ONNX")
    parser.add_argument("--half", action="store_true", help="启用 FP16")
    return parser.parse_args()


def normalize_imgsz(values: list[int]) -> int | tuple[int, int]:
    if len(values) == 1:
        return values[0]
    return values[0], values[1]


def main():
    args = parse_args()
    run_realtime_pathplan(
        source=args.source,
        weights=args.weights,
        data_yaml=args.data,
        project=args.project,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        imgsz=normalize_imgsz(args.imgsz),
        grid_scale=args.grid_scale,
        view=args.view,
        save=not args.nosave,
        dnn=args.dnn,
        half=args.half,
        backend=args.backend,
        display=args.display,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_path=args.remote_path,
        ros_publish_2p5d=args.ros_publish_2p5d,
        ros_publish_occ_only=args.ros_publish_occ_only,
        ros_frame_id=args.ros_frame_id,
        ros_rate=args.ros_rate,
        ros_occ_topic=args.ros_occ_topic,
        ros_cloud_topic=args.ros_cloud_topic,
        cell_size=args.cell_size,
        z_step=args.z_step,
        z_max_cap=args.z_max_cap,
        xy_spread=args.xy_spread,
        xy_samples=args.xy_samples,
        cloud_mode=args.cloud_mode,
        edge_mode=args.edge_mode,
        z_style=args.z_style,
        gray_view=args.gray_view,
    )


if __name__ == "__main__":
    main()
