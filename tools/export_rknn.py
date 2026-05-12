'''
    PT -> ONNX：
    python yolo/export.py --weights ./weights/0512_5k.pt --include onnx --imgsz 640 640

    ONNX -> RKNN：
    ```
    请在 x86_64 Linux + rknn-toolkit2 环境（不要在板端）执行：
    python tools/export_rknn.py --onnx ./weights/0512_5k.onnx --output ./weights/0512_5k.rknn --target rk3588
    ```
    其中 --opset-check 可选，启用后会先打印 ONNX 模型的 IR 版本、opset 版本和输出节点信息，确认无误后继续导出 .rknn。

'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference.onnx_realtime import get_default_onnx_weights
from app.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 ONNX 分割模型转换为 RK3588 可部署的 .rknn 文件。")
    parser.add_argument("--onnx", type=Path, default=get_default_onnx_weights(), help="输入 ONNX 路径")
    parser.add_argument("--output", type=Path, default=None, help="输出 RKNN 路径，默认与 ONNX 同名")
    parser.add_argument("--target", default="rk3588", help="RKNN target platform")
    parser.add_argument("--quantize", action="store_true", help="启用量化")
    parser.add_argument("--dataset", type=Path, default=None, help="量化数据集 txt 路径，每行一个图片路径")
    parser.add_argument("--mean-values", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="预处理 mean，默认 0 0 0")
    parser.add_argument("--std-values", nargs=3, type=float, default=[255.0, 255.0, 255.0], help="预处理 std，默认 255 255 255")
    parser.add_argument("--quantized-dtype", default="asymmetric_quantized-8", help="量化 dtype")
    parser.add_argument("--opset-check", action="store_true", help="打印 ONNX 信息后继续转换")
    return parser.parse_args()


def resolve_output_path(onnx_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return onnx_path.with_suffix(".rknn")
    return output_path.expanduser().resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def print_onnx_hint(onnx_path: Path) -> None:
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        print(f"ONNX IR version: {model.ir_version}")
        print(f"ONNX opset imports: {[opset.version for opset in model.opset_import]}")
        print(f"ONNX graph outputs: {[output.name for output in model.graph.output]}")
    except ModuleNotFoundError:
        print("未安装 onnx，跳过 ONNX 信息检查。")


def export_rknn(
    onnx_path: Path,
    output_path: Path,
    target: str,
    quantize: bool,
    dataset: Path | None,
    mean_values: list[float],
    std_values: list[float],
    quantized_dtype: str,
) -> None:
    try:
        from rknn.api import RKNN  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "导出 .rknn 需要先安装 rknn-toolkit2。请在支持环境执行: pip install rknn-toolkit2"
        ) from exc

    if quantize and dataset is None:
        raise ValueError("启用量化时必须提供 --dataset")

    if dataset is not None and not dataset.exists():
        raise FileNotFoundError(f"量化数据集不存在: {dataset}")

    rknn = RKNN(verbose=False)
    try:
        config_status = rknn.config(
            target_platform=target,
            mean_values=[mean_values],
            std_values=[std_values],
            quantized_dtype=quantized_dtype,
        )
        if config_status != 0:
            raise RuntimeError(f"RKNN config 失败，返回码: {config_status}")

        load_status = rknn.load_onnx(model=str(onnx_path))
        if load_status != 0:
            raise RuntimeError(f"RKNN load_onnx 失败，返回码: {load_status}")

        build_status = rknn.build(do_quantization=quantize, dataset=str(dataset) if dataset is not None else None)
        if build_status != 0:
            raise RuntimeError(f"RKNN build 失败，返回码: {build_status}")

        export_status = rknn.export_rknn(str(output_path))
        if export_status != 0:
            raise RuntimeError(f"RKNN export_rknn 失败，返回码: {export_status}")
    finally:
        release = getattr(rknn, "release", None)
        if callable(release):
            release()


def main() -> None:
    args = parse_args()
    onnx_path = resolve_path(args.onnx, get_default_onnx_weights())
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"输入必须是 .onnx 文件: {onnx_path}")

    output_path = resolve_output_path(onnx_path, args.output)
    dataset_path = resolve_path(args.dataset, output_path.parent) if args.dataset is not None else None
    ensure_parent_dir(output_path)

    if args.opset_check:
        print_onnx_hint(onnx_path)

    export_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        target=args.target,
        quantize=args.quantize,
        dataset=dataset_path,
        mean_values=args.mean_values,
        std_values=args.std_values,
        quantized_dtype=args.quantized_dtype,
    )
    print(f"RKNN 导出完成: {output_path}")


if __name__ == "__main__":
    main()
