from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_CONFIG
from app.paths import resolve_path

MANUAL_ONNX_WEIGHTS: str | Path | None = None
MANUAL_DATA_YAML: str | Path | None = None
MANUAL_RKNN_WEIGHTS: str | Path | None = None


def _resolve_manual_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = DEFAULT_CONFIG.repo_root / path
    return path.resolve()


def get_default_onnx_weights() -> Path:
    if MANUAL_ONNX_WEIGHTS is not None:
        return _resolve_manual_path(MANUAL_ONNX_WEIGHTS, DEFAULT_CONFIG.default_weights.with_suffix(".onnx"))

    default_onnx = DEFAULT_CONFIG.default_weights.with_suffix(".onnx")
    if default_onnx.exists():
        return default_onnx.resolve()

    candidates = sorted(DEFAULT_CONFIG.default_weights.parent.glob("*.onnx"))
    if candidates:
        return candidates[0].resolve()

    return default_onnx.resolve()


def get_default_rknn_weights() -> Path:
    if MANUAL_RKNN_WEIGHTS is not None:
        return _resolve_manual_path(MANUAL_RKNN_WEIGHTS, DEFAULT_CONFIG.default_weights.with_suffix(".rknn"))

    default_rknn = DEFAULT_CONFIG.default_weights.with_suffix(".rknn")
    if default_rknn.exists():
        return default_rknn.resolve()

    candidates = sorted(DEFAULT_CONFIG.default_weights.parent.glob("*.rknn"))
    if candidates:
        return candidates[0].resolve()

    return default_rknn.resolve()


def get_default_realtime_weights() -> Path:
    for candidate in (get_default_rknn_weights(), get_default_onnx_weights()):
        if candidate.exists():
            return candidate.resolve()
    return get_default_rknn_weights().resolve()


def get_default_data_yaml() -> Path:
    return _resolve_manual_path(MANUAL_DATA_YAML, DEFAULT_CONFIG.data_yaml)


def ensure_onnx_weights_path(weights: str | Path) -> Path:
    weights_path = Path(weights)
    if weights_path.suffix.lower() != ".onnx":
        raise ValueError(f"realtime ONNX 后端要求 .onnx 权重，当前收到: {weights_path}")
    return weights_path


def ensure_rknn_weights_path(weights: str | Path) -> Path:
    weights_path = Path(weights)
    if weights_path.suffix.lower() != ".rknn":
        raise ValueError(f"realtime RKNN 后端要求 .rknn 权重，当前收到: {weights_path}")
    return weights_path


def build_mask_metadata(class_id: int, frame_stem: str, mask_index: int) -> dict[str, object]:
    return {
        "filename": f"{class_id}_{frame_stem}_{mask_index}.png",
        "image_stem": frame_stem,
        "mask_index": mask_index,
    }


def detections_to_mask_entries(
    detections: np.ndarray | Sequence[Sequence[float]],
    masks: np.ndarray,
    frame_stem: str,
) -> list[list]:
    detection_rows = detections.tolist() if isinstance(detections, np.ndarray) else [list(row) for row in detections]

    if not detection_rows:
        return []

    if masks.ndim == 2:
        masks = masks[:, :, None]
    if masks.ndim != 3:
        raise ValueError(f"masks 必须是 HxW 或 HxWxN，当前 shape={masks.shape}")
    if masks.shape[2] != len(detection_rows):
        raise ValueError(
            f"检测数量与 mask 数量不一致: detections={len(detection_rows)} masks={masks.shape[2]}"
        )

    mask_entries: list[list] = []
    for mask_index, det in enumerate(detection_rows):
        if len(det) < 6:
            continue

        class_id = int(det[5])
        confidence = float(det[4])
        mask = masks[:, :, mask_index]
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if not np.any(binary_mask):
            continue

        metadata = build_mask_metadata(class_id, frame_stem, mask_index)
        mask_entries.append([None, class_id, confidence, binary_mask, metadata])

    return mask_entries


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, max(width, 0))
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, max(height, 0))
    return clipped


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter_area

    iou = np.zeros_like(inter_area, dtype=np.float32)
    valid = union > 0
    iou[valid] = inter_area[valid] / union[valid]
    return iou


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0 and len(keep) < max_det:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        ious = _compute_iou(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_thres]

    return np.asarray(keep, dtype=np.int64)


def _crop_mask(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    cropped = np.zeros_like(mask, dtype=np.float32)
    x1 = max(int(np.floor(box[0])), 0)
    y1 = max(int(np.floor(box[1])), 0)
    x2 = min(int(np.ceil(box[2])), mask.shape[1])
    y2 = min(int(np.ceil(box[3])), mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return cropped
    cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return cropped


def _crop_masks_vectorized(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError(f"masks 必须是 NxHxW，当前 shape={masks.shape}")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes 必须是 Nx4，当前 shape={boxes.shape}")
    if masks.shape[0] != boxes.shape[0]:
        raise ValueError(f"masks 与 boxes 数量不一致: {masks.shape[0]} vs {boxes.shape[0]}")

    n, h, w = masks.shape
    if n == 0:
        return masks

    x1 = np.clip(np.floor(boxes[:, 0]).astype(np.int32), 0, max(w - 1, 0))
    y1 = np.clip(np.floor(boxes[:, 1]).astype(np.int32), 0, max(h - 1, 0))
    x2 = np.clip(np.ceil(boxes[:, 2]).astype(np.int32), 0, w)
    y2 = np.clip(np.ceil(boxes[:, 3]).astype(np.int32), 0, h)

    xs = np.arange(w, dtype=np.int32)[None, None, :]
    ys = np.arange(h, dtype=np.int32)[None, :, None]
    valid_x = (xs >= x1[:, None, None]) & (xs < x2[:, None, None])
    valid_y = (ys >= y1[:, None, None]) & (ys < y2[:, None, None])
    keep = valid_x & valid_y
    return masks * keep.astype(masks.dtype, copy=False)


def _normalize_prediction(prediction: np.ndarray, mask_dim: int) -> np.ndarray:
    if prediction.ndim != 3:
        raise ValueError(f"预测输出必须是 3 维，当前 shape={prediction.shape}")

    if prediction.shape[0] != 1:
        raise ValueError(f"当前只支持 batch=1，当前 shape={prediction.shape}")

    if prediction.shape[-1] < prediction.shape[1]:
        normalized = prediction[0]
    else:
        normalized = prediction[0].transpose(1, 0)

    if normalized.shape[1] <= 5 + mask_dim:
        raise ValueError(f"预测输出特征维度异常: shape={normalized.shape}, mask_dim={mask_dim}")

    return normalized.astype(np.float32, copy=False)


def _normalize_proto(proto: np.ndarray) -> np.ndarray:
    if proto.ndim != 4:
        raise ValueError(f"proto 输出必须是 4 维，当前 shape={proto.shape}")

    if proto.shape[0] != 1:
        raise ValueError(f"当前只支持 batch=1，当前 shape={proto.shape}")

    proto = proto[0]
    if proto.shape[0] <= proto.shape[1] and proto.shape[0] <= proto.shape[2]:
        normalized = proto
    elif proto.shape[2] <= proto.shape[0] and proto.shape[2] <= proto.shape[1]:
        normalized = np.moveaxis(proto, -1, 0)
    else:
        raise ValueError(f"无法识别 proto 输出布局: shape={proto.shape}")

    return normalized.astype(np.float32, copy=False)


def _resize_masks(masks: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    output_h, output_w = output_shape
    resized_masks: list[np.ndarray] = []
    for index in range(masks.shape[2]):
        resized = cv2.resize(masks[:, :, index], (output_w, output_h), interpolation=cv2.INTER_LINEAR)
        resized_masks.append(resized)

    if not resized_masks:
        return np.zeros((output_h, output_w, 0), dtype=np.float32)

    return np.stack(resized_masks, axis=2).astype(np.float32, copy=False)


def _resolve_imgsz(imgsz: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(imgsz, int):
        return imgsz, imgsz
    return int(imgsz[0]), int(imgsz[1])


def _letterbox(
    frame: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    shape = frame.shape[:2]
    ratio_value = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (ratio_value, ratio_value)

    new_unpad = int(round(shape[1] * ratio_value)), int(round(shape[0] * ratio_value))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return frame, ratio, (dw, dh)


def _scale_boxes_from_letterbox(
    boxes: np.ndarray,
    frame_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    scaled = boxes.copy().astype(np.float32, copy=False)
    gain = ratio_pad[0][0]
    pad_w, pad_h = ratio_pad[1]
    scaled[:, [0, 2]] -= pad_w
    scaled[:, [1, 3]] -= pad_h
    scaled[:, :4] /= gain
    return _clip_boxes(scaled, frame_shape[1], frame_shape[0])


def _scale_masks_from_letterbox(
    masks: np.ndarray,
    input_shape: tuple[int, int],
    frame_shape: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    if masks.ndim == 2:
        masks = masks[:, :, None]
    if masks.ndim != 3:
        raise ValueError(f"masks 必须是 HxW 或 HxWxN，当前 shape={masks.shape}")
    if masks.size == 0:
        return np.zeros((frame_shape[0], frame_shape[1], 0), dtype=np.float32)

    pad_w, pad_h = ratio_pad[1]
    top = max(int(pad_h), 0)
    left = max(int(pad_w), 0)
    bottom = min(int(input_shape[0] - pad_h), input_shape[0])
    right = min(int(input_shape[1] - pad_w), input_shape[1])

    cropped = masks[top:bottom, left:right]
    if cropped.size == 0:
        return np.zeros((frame_shape[0], frame_shape[1], masks.shape[2]), dtype=np.float32)

    return _resize_masks(cropped, frame_shape)


def extract_prediction_and_proto(outputs: Sequence[np.ndarray], backend_name: str) -> tuple[np.ndarray, np.ndarray]:
    prediction = next((np.asarray(output) for output in outputs if np.asarray(output).ndim == 3), None)
    proto = next((np.asarray(output) for output in outputs if np.asarray(output).ndim == 4), None)
    if prediction is None or proto is None:
        shapes = [tuple(np.asarray(output).shape) for output in outputs]
        raise ValueError(f"无法从 {backend_name} 输出中识别 prediction/proto，当前 outputs={shapes}")
    return prediction, proto


def postprocess_segmentation_outputs(
    prediction: np.ndarray,
    proto: np.ndarray,
    frame_shape: tuple[int, int],
    imgsz: tuple[int, int],
    ratio_pad: tuple[tuple[float, float], tuple[float, float]],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    classes: set[int] | None = None,
    agnostic_nms: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    proto = _normalize_proto(proto)
    mask_dim = int(proto.shape[0])
    prediction = _normalize_prediction(prediction, mask_dim)

    num_classes = prediction.shape[1] - 5 - mask_dim
    if num_classes <= 0:
        raise ValueError(f"无法从预测输出中解析类别数: shape={prediction.shape}, mask_dim={mask_dim}")

    boxes_xywh = prediction[:, :4]
    objectness = prediction[:, 4]
    class_scores = prediction[:, 5 : 5 + num_classes]
    mask_coeffs = prediction[:, 5 + num_classes :]

    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
    scores = objectness * class_conf

    keep = scores > conf_thres
    if classes is not None:
        keep &= np.isin(class_ids, list(classes))
    if not np.any(keep):
        return np.empty((0, 6), dtype=np.float32), np.zeros((frame_shape[0], frame_shape[1], 0), dtype=np.uint8)

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep].astype(np.float32, copy=False)
    class_ids = class_ids[keep].astype(np.float32, copy=False)
    mask_coeffs = mask_coeffs[keep]

    boxes = _xywh_to_xyxy(boxes_xywh)
    boxes = _clip_boxes(boxes, imgsz[1], imgsz[0])

    if agnostic_nms:
        keep_indices = _nms(boxes, scores, iou_thres, max_det)
    else:
        kept_parts: list[np.ndarray] = []
        for class_id in np.unique(class_ids.astype(np.int32)):
            class_mask = class_ids == float(class_id)
            class_indices = np.where(class_mask)[0]
            class_keep = _nms(boxes[class_mask], scores[class_mask], iou_thres, max_det)
            if class_keep.size:
                kept_parts.append(class_indices[class_keep])
        if kept_parts:
            keep_indices = np.concatenate(kept_parts)
            keep_indices = keep_indices[np.argsort(scores[keep_indices])[::-1][:max_det]]
        else:
            keep_indices = np.empty((0,), dtype=np.int64)

    if keep_indices.size == 0:
        return np.empty((0, 6), dtype=np.float32), np.zeros((frame_shape[0], frame_shape[1], 0), dtype=np.uint8)

    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]
    mask_coeffs = mask_coeffs[keep_indices]

    proto_h, proto_w = proto.shape[1:]
    mask_logits = mask_coeffs @ proto.reshape(mask_dim, -1)
    mask_logits = mask_logits.reshape((-1, proto_h, proto_w))
    masks = _sigmoid(mask_logits)

    scaled_boxes_for_proto = boxes.copy()
    scaled_boxes_for_proto[:, [0, 2]] *= proto_w / float(imgsz[1])
    scaled_boxes_for_proto[:, [1, 3]] *= proto_h / float(imgsz[0])

    cropped_masks_array = _crop_masks_vectorized(masks, scaled_boxes_for_proto)
    if cropped_masks_array.size == 0:
        return np.empty((0, 6), dtype=np.float32), np.zeros((frame_shape[0], frame_shape[1], 0), dtype=np.uint8)

    resized_masks = _resize_masks(np.moveaxis(cropped_masks_array, 0, -1), imgsz)
    scaled_masks = _scale_masks_from_letterbox(resized_masks, imgsz, frame_shape, ratio_pad)
    binary_masks = np.ascontiguousarray((scaled_masks > 0.5).astype(np.uint8, copy=False))

    scaled_boxes = _scale_boxes_from_letterbox(boxes, frame_shape, ratio_pad)
    detections = np.column_stack((scaled_boxes, scores, class_ids)).astype(np.float32, copy=False)
    return detections, binary_masks


class OnnxRealtimeSegmenter:
    def __init__(
        self,
        weights: str | Path | None = None,
        data_yaml: str | Path | None = None,
        device: str | None = None,
        imgsz: int | tuple[int, int] = 640,
        conf_thres: float | None = None,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        dnn: bool = False,
        half: bool = False,
        classes: Sequence[int] | None = None,
        agnostic_nms: bool = False,
    ):
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("ONNX realtime 推理需要先安装 onnxruntime。") from exc

        weights_path = ensure_onnx_weights_path(resolve_path(weights, get_default_onnx_weights()))
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
        self.dnn = dnn
        self.half = half
        self.ort = ort

        providers = self._select_providers(self.device)
        self.session = ort.InferenceSession(str(self.weights), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _select_providers(self, device: str | None) -> list[str]:
        available = set(self.ort.get_available_providers())
        device_name = (device or "cpu").lower()
        if device_name != "cpu" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def preprocess_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame 必须是 HxWxC 的 numpy.ndarray")

        letterboxed, ratio, pad = _letterbox(frame, self.imgsz)
        image = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1)).astype(np.float32, copy=False)
        image /= 255.0
        return np.expand_dims(np.ascontiguousarray(image), axis=0), (ratio, pad)

    def _run_inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return [np.asarray(output) for output in outputs]

    def predict_frame(self, frame: np.ndarray, frame_stem: str) -> list[list]:
        input_tensor, ratio_pad = self.preprocess_frame(frame)
        outputs = self._run_inference(input_tensor)
        prediction, proto = extract_prediction_and_proto(outputs, "ONNX")
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
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        return self.predict_frame(frame, image_path.stem)
