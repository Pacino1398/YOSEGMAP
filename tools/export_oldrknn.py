
'''
旧版 export_oldrknn.py

    按 weights/yolov5-best-rk3588.rknn 的结构导出：
    输入:
    images: [1,3,640,640]
    输出:
    output0: [1,45,80,80]
    332:     [1,45,40,40]
    334:     [1,45,20,20]

    导出 YOLOv5 检测头的三个原始特征层。
    输出是 3 个张量，对应 80x80、40x40、20x20 三个尺度。
    默认按参考模型做 int8 量化，需要 --dataset。
    更接近 Rockchip/旧 YOLOv5 RKNN 示例常用结构。
    后处理需要自己按 anchors/stride 解码三层输出。
    适合兼容 yolov5-best-rk3588.rknn 这类旧模型结构。


新版/普通版 export_rknn.py

    按 ONNX 原始暴露的最终输出导出：
    输入:
    images: [1,3,640,640]
    输出:
    output0: 通常是最终 concat 后的预测结果
    例如你现在的 0513_5k.onnx 类似：

    output0: [1,25200,16]   只导出 ONNX graph output。
    输出通常是 1 个最终 concat 张量。默认不量化。


export_rknn.py 是按 ONNX 当前最终输出导出；
export_oldrknn.py 是强行改成旧 RKNN 的三检测头输出结构。
'''
"""
    python tools/export_oldrknn.py \
    --make-dataset runs/0409_qy \
    --dataset-output runs/rknn_dataset.txt \
    --dataset-limit 200

    python tools/export_oldrknn.py \
    --onnx weights/0513_5k.onnx \
    --output weights/0513_5k_oldlayout.rknn \
    --dataset runs/rknn_dataset.txt \
    --target rk3588
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REFERENCE_RKNN = ROOT / "weights" / "yolov5-best-rk3588.rknn"
DEFAULT_OUTPUT_NODES = [
    "/model.24/m.0/Conv_output_0",
    "/model.24/m.1/Conv_output_0",
    "/model.24/m.2/Conv_output_0",
]
DEFAULT_RENAMED_OUTPUTS = ["output0", "332", "334"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default.resolve()
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def get_default_onnx_weights() -> Path:
    preferred = [ROOT / "weights" / "0513_5k.onnx", ROOT / "weights" / "0512_5k.onnx"]
    for candidate in preferred:
        if candidate.exists():
            return candidate.resolve()

    candidates = sorted((ROOT / "weights").glob("*.onnx"))
    if candidates:
        return candidates[0].resolve()
    return (ROOT / "weights" / "model.onnx").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 yolov5-best-rk3588.rknn 的三输出 int8 结构导出 RK3588 RKNN。"
    )
    parser.add_argument("--onnx", type=Path, default=get_default_onnx_weights(), help="输入 ONNX 路径")
    parser.add_argument("--output", type=Path, default=None, help="输出 RKNN 路径，默认 ONNX 同名加 _oldlayout")
    parser.add_argument("--target", default="rk3588", help="RKNN target platform，默认 rk3588")
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=DEFAULT_OUTPUT_NODES,
        help="ONNX 中要截取的三个检测头输出节点名",
    )
    parser.add_argument(
        "--rename-outputs",
        nargs="+",
        default=DEFAULT_RENAMED_OUTPUTS,
        help="导出前将三个输出重命名为参考 RKNN 的名字",
    )
    parser.add_argument("--input-name", default="images", help="ONNX 输入名，默认 images")
    parser.add_argument("--input-size-list", nargs=4, type=int, default=[1, 3, 640, 640], help="输入 shape")
    parser.add_argument("--mean-values", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="RKNN mean_values")
    parser.add_argument("--std-values", nargs=3, type=float, default=[255.0, 255.0, 255.0], help="RKNN std_values")
    parser.add_argument("--quantized-dtype", default="asymmetric_quantized-8", help="量化 dtype")
    parser.add_argument("--dataset", type=Path, default=None, help="量化校准 dataset txt，每行一个图片路径")
    parser.add_argument("--no-quantize", action="store_true", help="关闭量化，仅用于调试；参考结构应保持量化开启")
    parser.add_argument("--make-dataset", type=Path, default=None, help="从图片目录生成 dataset txt 后退出")
    parser.add_argument("--dataset-output", type=Path, default=ROOT / "runs" / "rknn_dataset.txt")
    parser.add_argument("--dataset-limit", type=int, default=200, help="生成 dataset txt 的最大图片数")
    parser.add_argument("--opset-check", action="store_true", help="打印 ONNX 输入输出和候选节点信息")
    parser.add_argument("--check-only", action="store_true", help="只检查参考结构和 ONNX 候选输出，不执行 RKNN 导出")
    return parser.parse_args()


def resolve_output_path(onnx_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return onnx_path.with_name(f"{onnx_path.stem}_oldlayout.rknn")
    return output_path.expanduser().resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_dataset(image_root: Path, output_path: Path, limit: int) -> None:
    image_root = image_root.expanduser().resolve()
    if not image_root.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_root}")

    images = [
        path.resolve()
        for path in sorted(image_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if limit > 0:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"未在目录中找到图片: {image_root}")

    ensure_parent_dir(output_path)
    output_path.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
    print(f"dataset 已生成: {output_path} ({len(images)} images)")


def print_reference_hint() -> None:
    if not REFERENCE_RKNN.exists():
        return
    data = REFERENCE_RKNN.read_bytes()
    tail = data[-1400:].decode("utf-8", errors="ignore")
    print("参考 RKNN 尾部结构摘要:")
    for line in (
        '"size":[1,3,640,640],"tensor_id":0,"url":"images"',
        '"size":[1,45,80,80],"tensor_id":1,"url":"output0"',
        '"size":[1,45,40,40],"tensor_id":2,"url":"332"',
        '"size":[1,45,20,20],"tensor_id":3,"url":"334"',
        '"output_num":3',
    ):
        print(f"  {'OK' if line in tail else '??'} {line}")


def print_onnx_hint(onnx_path: Path, outputs: list[str]) -> None:
    print_reference_hint()
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        print(f"ONNX IR version: {model.ir_version}")
        print(f"ONNX opset imports: {[opset.version for opset in model.opset_import]}")
        print(
            "ONNX inputs:",
            [
                (value.name, [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim])
                for value in model.graph.input
            ],
        )
        print(
            "ONNX graph outputs:",
            [
                (value.name, [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim])
                for value in model.graph.output
            ],
        )
        node_outputs = {output for node in model.graph.node for output in node.output}
        for output in outputs:
            print(f"candidate output {output!r}: {'FOUND' if output in node_outputs else 'NOT FOUND'}")
    except ModuleNotFoundError:
        data = onnx_path.read_bytes()
        print("未安装 onnx，仅用二进制字符串检查候选输出节点:")
        for output in outputs:
            print(f"candidate output {output!r}: {'FOUND' if output.encode() in data else 'NOT FOUND'}")


def patch_onnx_outputs(onnx_path: Path, outputs: list[str], renamed_outputs: list[str]) -> Path | None:
    if outputs == renamed_outputs:
        return None
    if len(outputs) != len(renamed_outputs):
        raise ValueError("--outputs 与 --rename-outputs 数量必须一致")

    try:
        import onnx
        from onnx import shape_inference
        from onnx.helper import make_node, make_tensor_value_info
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("重命名/改写 ONNX 输出需要安装 onnx。请先 pip install onnx") from exc

    model = onnx.load(str(onnx_path))
    inferred = shape_inference.infer_shapes(model)
    value_info = {value.name: value for value in inferred.graph.value_info}
    value_info.update({value.name: value for value in inferred.graph.output})

    # Some YOLOv5 ONNX files already use "output0" for the final concat output.
    # The reference RKNN also wants the first raw head to be named "output0".
    # Rename existing tensors that would collide before adding export identities.
    protected_sources = set(outputs)
    target_names = set(renamed_outputs)
    for node in model.graph.node:
        for index, name in enumerate(node.output):
            if name in target_names and name not in protected_sources:
                node.output[index] = f"{name}_final_unused"
    for graph_output in model.graph.output:
        if graph_output.name in target_names and graph_output.name not in protected_sources:
            graph_output.name = f"{graph_output.name}_final_unused"

    del model.graph.output[:]
    for source_name, target_name in zip(outputs, renamed_outputs):
        if source_name not in value_info:
            raise ValueError(f"ONNX 中找不到输出节点: {source_name}")

        source_info = value_info[source_name]
        elem_type = source_info.type.tensor_type.elem_type
        dims = [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in source_info.type.tensor_type.shape.dim
        ]
        model.graph.node.append(make_node("Identity", [source_name], [target_name], name=f"export_{target_name}"))
        model.graph.output.append(make_tensor_value_info(target_name, elem_type, dims))

    patched_path = onnx_path.with_name(f"{onnx_path.stem}_oldlayout_tmp.onnx")
    onnx.save(model, str(patched_path))
    return patched_path


def export_rknn(
    onnx_path: Path,
    output_path: Path,
    target: str,
    outputs: list[str],
    renamed_outputs: list[str],
    input_name: str,
    input_size_list: list[int],
    mean_values: list[float],
    std_values: list[float],
    quantized_dtype: str,
    quantize: bool,
    dataset: Path | None,
) -> None:
    try:
        from rknn.api import RKNN  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "导出 .rknn 需要 rknn-toolkit2。请在 x86_64 Linux 环境安装后执行。"
        ) from exc

    if quantize:
        if dataset is None:
            raise ValueError("参考 rknn 是 int8 量化模型；未使用 --no-quantize 时必须提供 --dataset")
        if not dataset.exists():
            raise FileNotFoundError(f"量化数据集不存在: {dataset}")

    patched_onnx = patch_onnx_outputs(onnx_path, outputs, renamed_outputs)
    model_for_export = patched_onnx or onnx_path

    rknn = RKNN(verbose=True)
    try:
        config_status = rknn.config(
            target_platform=target,
            mean_values=[mean_values],
            std_values=[std_values],
            quantized_dtype=quantized_dtype,
        )
        if config_status != 0:
            raise RuntimeError(f"RKNN config 失败，返回码: {config_status}")

        load_status = rknn.load_onnx(
            model=str(model_for_export),
            inputs=[input_name],
            input_size_list=[input_size_list],
            outputs=renamed_outputs,
        )
        if load_status != 0:
            raise RuntimeError(f"RKNN load_onnx 失败，返回码: {load_status}")

        build_status = rknn.build(do_quantization=quantize, dataset=str(dataset) if quantize else None)
        if build_status != 0:
            raise RuntimeError(f"RKNN build 失败，返回码: {build_status}")

        export_status = rknn.export_rknn(str(output_path))
        if export_status != 0:
            raise RuntimeError(f"RKNN export_rknn 失败，返回码: {export_status}")
    finally:
        release = getattr(rknn, "release", None)
        if callable(release):
            release()
        if patched_onnx is not None and patched_onnx.exists():
            patched_onnx.unlink()


def main() -> None:
    args = parse_args()

    if args.make_dataset is not None:
        make_dataset(
            resolve_path(args.make_dataset, ROOT),
            resolve_path(args.dataset_output, ROOT),
            args.dataset_limit,
        )
        return

    onnx_path = resolve_path(args.onnx, get_default_onnx_weights())
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"输入必须是 .onnx 文件: {onnx_path}")

    output_path = resolve_output_path(onnx_path, args.output)
    dataset_path = resolve_path(args.dataset, ROOT) if args.dataset is not None else None
    ensure_parent_dir(output_path)

    if args.opset_check:
        print_onnx_hint(onnx_path, list(args.outputs))
    if args.check_only:
        return

    export_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        target=args.target,
        outputs=list(args.outputs),
        renamed_outputs=list(args.rename_outputs),
        input_name=args.input_name,
        input_size_list=list(args.input_size_list),
        mean_values=list(args.mean_values),
        std_values=list(args.std_values),
        quantized_dtype=args.quantized_dtype,
        quantize=not args.no_quantize,
        dataset=dataset_path,
    )
    print(f"RKNN 导出完成: {output_path}")


if __name__ == "__main__":
    main()
