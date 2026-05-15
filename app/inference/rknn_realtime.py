from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.paths import resolve_path
from app.inference.realtime_common import (
    detections_to_mask_entries,
    ensure_rknn_weights_path,
    extract_prediction_and_proto,
    get_default_data_yaml,
    get_default_rknn_weights,
    postprocess_segmentation_outputs,
    _letterbox,
    _resolve_imgsz,
)


class RknnRealtimeSegmenter:
    def __init__(
        self,
        weights: str | Path | None = None,
        data_yaml: str | Path | None = None,
        device: str | None = None,
        imgsz: int | tuple[int, int] = 640,
        conf_thres: float | None = None,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        classes: Sequence[int] | None = None,
        agnostic_nms: bool = False,
        target: str = "rk3588",
    ):
        weights_path = ensure_rknn_weights_path(resolve_path(weights, get_default_rknn_weights()))
        data_yaml_path = resolve_path(data_yaml, get_default_data_yaml())

        self.weights = weights_path
        self.data_yaml = data_yaml_path
        self.device = device or DEFAULT_CONFIG.default_device
        self.imgsz = _resolve_imgsz(imgsz)
        self.conf_thres = conf_thres if conf_thres is not None else DEFAULT_CONFIG.default_conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = set(classes) if classes is not None else None
        self.agnostic_nms = agnostic_nms
        self.target = target
        self._input_ring_size = 3
        self._input_ring: list[np.ndarray] = []
        self._input_ring_index = 0

        runtime_backend = None
        import_error = None
        try:
            from rknnlite.api import RKNNLite  # type: ignore

            runtime_backend = "lite"
            self.runtime = RKNNLite()
            status = self.runtime.load_rknn(str(self.weights))
            if status != 0:
                raise RuntimeError(f"RKNNLite 加载模型失败，返回码: {status}")

            init_kwargs: dict[str, object] = {}
            if hasattr(RKNNLite, "NPU_CORE_ALL"):
                init_kwargs["core_mask"] = RKNNLite.NPU_CORE_ALL
            elif hasattr(RKNNLite, "NPU_CORE_0_1_2"):
                init_kwargs["core_mask"] = RKNNLite.NPU_CORE_0_1_2
            elif hasattr(RKNNLite, "NPU_CORE_AUTO"):
                init_kwargs["core_mask"] = RKNNLite.NPU_CORE_AUTO
            else:
                init_kwargs["core_mask"] = 7
            status = self.runtime.init_runtime(**init_kwargs)
            if status != 0:
                raise RuntimeError(f"RKNNLite 初始化运行时失败，返回码: {status}")
        except ModuleNotFoundError as exc:
            import_error = exc
        except Exception:
            raise

        if runtime_backend is None:
            try:
                from rknn.api import RKNN  # type: ignore

                runtime_backend = "toolkit"
                self.runtime = RKNN(verbose=False)
                status = self.runtime.load_rknn(str(self.weights))
                if status != 0:
                    raise RuntimeError(f"RKNN Toolkit 加载模型失败，返回码: {status}")
                init_kwargs: dict[str, object] = {"target": self.target}
                status = self.runtime.init_runtime(**init_kwargs)
                if status != 0:
                    raise RuntimeError(f"RKNN Toolkit 初始化运行时失败，返回码: {status}")
            except ModuleNotFoundError:
                if import_error is not None:
                    raise ModuleNotFoundError(
                        "RKNN realtime 推理需要先安装 rknn-toolkit2 或板端 rknn-toolkit-lite2。"
                    ) from import_error
                raise ModuleNotFoundError(
                    "RKNN realtime 推理需要先安装 rknn-toolkit2 或板端 rknn-toolkit-lite2。"
                )

        self.runtime_backend = runtime_backend

    def _next_input_buffer(self, input_tensor: np.ndarray) -> np.ndarray:
        if not self._input_ring:
            for _ in range(self._input_ring_size):
                self._input_ring.append(np.empty_like(input_tensor, dtype=np.float32))
        self._input_ring_index = (self._input_ring_index + 1) % self._input_ring_size
        buf = self._input_ring[self._input_ring_index]
        np.copyto(buf, input_tensor, casting="no")
        return buf

    def preprocess_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
        import cv2

        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame 必须是 HxWxC 的 numpy.ndarray")

        letterboxed, ratio, pad = _letterbox(frame, self.imgsz)
        image = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32, copy=False)
        input_tensor = np.expand_dims(np.ascontiguousarray(image), axis=0)
        return input_tensor, (ratio, pad)

    def _run_inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        infer_input = self._next_input_buffer(np.ascontiguousarray(input_tensor, dtype=np.float32))
        if self.runtime_backend == "lite":
            outputs = self.runtime.inference(inputs=[infer_input], data_format=["nhwc"])
        else:
            outputs = self.runtime.inference(inputs=[infer_input], data_format=["nhwc"])
        if outputs is None:
            raise RuntimeError("RKNN 推理未返回输出。")
        return [np.asarray(output) for output in outputs]

    def run_inference_timed(self, input_tensor: np.ndarray) -> tuple[list[np.ndarray], float]:
        t0 = time.perf_counter()
        outputs = self._run_inference(input_tensor)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return outputs, infer_ms

    def predict_frame(self, frame: np.ndarray, frame_stem: str) -> list[list]:
        input_tensor, ratio_pad = self.preprocess_frame(frame)
        outputs = self._run_inference(input_tensor)
        prediction, proto = extract_prediction_and_proto(outputs, "RKNN")
        detections, masks = postprocess_segmentation_outputs(
            prediction,
            proto,
            frame.shape[:2],
            self.imgsz,
            ratio_pad,
            self.conf_thres,
            self.iou_thres,
            self.max_det,
            self.classes,
            self.agnostic_nms,
        )
        return detections_to_mask_entries(detections, masks, frame_stem)

    def predict_image(self, image_path: str | Path) -> list[list]:
        import cv2

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        return self.predict_frame(frame, image_path.stem)

    def close(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return
        release = getattr(runtime, "release", None)
        if callable(release):
            release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
