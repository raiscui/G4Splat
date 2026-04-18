#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "导出与 FreeFix camera trajectory 相近的 JSON。"
            "支持原有的 render-image + cameras.json 模式，也支持 "
            "COLMAP sparse + indices + resolution 直接导出。"
            "导出的 intrinsics 会遵循 G4Splat/3DGS 当前训练相机假设："
            "保留 fx/fy，并把主点放在输出分辨率中心。"
        )
    )
    parser.add_argument(
        "--render-image",
        type=Path,
        default=None,
        help="渲染图路径，例如 train/ours_7000/renders/00000.png。",
    )
    parser.add_argument(
        "--cameras-json",
        type=Path,
        default=None,
        help="G4Splat 导出的 cameras.json 路径。",
    )
    parser.add_argument(
        "--colmap-dir",
        type=Path,
        default=None,
        help=(
            "COLMAP 模型目录。可以传 scene 根目录、sparse 目录，或 sparse/0 目录。"
        ),
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help='索引列表，如 "0"、"0,1,2,3" 或 "[0,1,2,3]"。',
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help='目标分辨率，如 "1920x1080"、"1920,1080" 或 "1920 1080"。',
    )
    parser.add_argument(
        "--source-images-dir",
        type=Path,
        default=None,
        help="原始图像目录。默认按 cameras.json 或 COLMAP 场景结构自动推断。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="基础 JSON 输出路径。默认根据输入模式自动命名。",
    )
    parser.add_argument(
        "--unity-output",
        type=Path,
        default=None,
        help="Unity JSON 输出路径。默认自动生成 *_camera_trajectory_unity.json。",
    )
    return parser


def parse_indices_spec(indices_spec: str) -> list[int]:
    raw_spec = indices_spec.strip()
    if not raw_spec:
        raise ValueError("indices 不能为空。")

    values: list[Any]
    if raw_spec.startswith("["):
        values = json.loads(raw_spec)
        if not isinstance(values, list):
            raise TypeError(f"indices 必须是列表: {indices_spec}")
    else:
        values = [item for item in re.split(r"[\s,]+", raw_spec) if item]

    parsed: list[int] = []
    for value in values:
        index = int(value)
        if index < 0:
            raise ValueError(f"indices 里不能有负数: {indices_spec}")
        parsed.append(index)
    if not parsed:
        raise ValueError("indices 不能为空。")
    return parsed


def parse_resolution_spec(resolution_spec: str) -> tuple[int, int]:
    match = re.fullmatch(
        r"\s*(\d+)\s*(?:x|X|,|\s)\s*(\d+)\s*",
        resolution_spec,
    )
    if match is None:
        raise ValueError(
            f"无法解析分辨率: {resolution_spec}。请使用 1920x1080、1920,1080 或 1920 1080。"
        )
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"分辨率必须为正整数: {resolution_spec}")
    return width, height


def validate_args(args: argparse.Namespace) -> None:
    has_camera_source = args.cameras_json is not None or args.colmap_dir is not None
    if not has_camera_source:
        raise ValueError("必须提供 --cameras-json 或 --colmap-dir 之一。")
    if args.cameras_json is not None and args.colmap_dir is not None:
        raise ValueError("--cameras-json 和 --colmap-dir 只能二选一。")

    has_frame_source = args.render_image is not None or args.indices is not None
    if not has_frame_source:
        raise ValueError("必须提供 --render-image 或 --indices 之一。")
    if args.render_image is not None and args.indices is not None:
        raise ValueError("--render-image 和 --indices 只能二选一。")

    if args.render_image is None and args.resolution is None:
        raise ValueError("未提供 --render-image 时，必须提供 --resolution。")
    if args.render_image is not None and args.resolution is not None:
        raise ValueError("提供 --render-image 时，不需要再传 --resolution。")


def default_output_basename(selected_indices: list[int]) -> str:
    encoded = "_".join(f"{index:05d}" for index in selected_indices)
    return f"colmap_{encoded}_camera_trajectory.json"


def resolve_output_path(
    *,
    render_image: Path | None,
    output_path: Path | None,
    default_dir: Path,
    selected_indices: list[int],
) -> Path:
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
    elif render_image is not None:
        resolved = render_image.with_name(
            f"{render_image.stem}_camera_trajectory.json"
        ).resolve()
    else:
        resolved = (default_dir / default_output_basename(selected_indices)).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_unity_output_path(output_path: Path, unity_output_path: Path | None) -> Path:
    if unity_output_path is not None:
        resolved = unity_output_path.expanduser().resolve()
    elif output_path.stem.endswith("_camera_trajectory"):
        resolved = output_path.with_name(
            f"{output_path.stem.replace('_camera_trajectory', '_camera_trajectory_unity')}{output_path.suffix}"
        )
    else:
        resolved = output_path.with_name(f"{output_path.stem}_unity{output_path.suffix}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def infer_source_images_dir_from_cameras_json(cameras_json_path: Path) -> Path | None:
    scene_root = cameras_json_path.resolve().parent.parent
    candidates = [
        scene_root / "mast3r_sfm" / "images",
        cameras_json_path.resolve().parent / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_colmap_sparse_dir(colmap_dir: Path) -> Path:
    root = colmap_dir.expanduser().resolve()
    candidates = [
        root,
        root / "0",
        root / "sparse" / "0",
    ]
    for candidate in candidates:
        if any(
            (candidate / name).exists()
            for name in ("cameras.bin", "cameras.txt", "images.bin", "images.txt")
        ):
            return candidate
    raise FileNotFoundError(
        f"无法在 {root} 下找到 COLMAP 模型目录，请传 scene 根目录、sparse 目录，或 sparse/0。"
    )


def infer_scene_root_from_colmap_sparse_dir(colmap_sparse_dir: Path) -> Path:
    if colmap_sparse_dir.name == "0" and colmap_sparse_dir.parent.name == "sparse":
        return colmap_sparse_dir.parent.parent
    if colmap_sparse_dir.name == "sparse":
        return colmap_sparse_dir.parent
    return colmap_sparse_dir


def infer_source_images_dir_from_colmap(colmap_sparse_dir: Path) -> Path | None:
    scene_root = infer_scene_root_from_colmap_sparse_dir(colmap_sparse_dir)
    candidate = scene_root / "images"
    if candidate.exists():
        return candidate.resolve()
    return None


def load_camera_entries(cameras_json_path: Path) -> list[dict[str, Any]]:
    with cameras_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"cameras.json 不是列表: {cameras_json_path}")
    return sorted(data, key=lambda item: (str(item["img_name"]), int(item["id"])))


def load_colmap_loader_module() -> Any:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "2d-gaussian-splatting"
        / "scene"
        / "colmap_loader.py"
    )
    spec = importlib.util.spec_from_file_location("g4_colmap_loader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 COLMAP 读取模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_colmap_model(colmap_sparse_dir: Path) -> tuple[Any, Any, Any]:
    loader = load_colmap_loader_module()

    cameras_bin = colmap_sparse_dir / "cameras.bin"
    cameras_txt = colmap_sparse_dir / "cameras.txt"
    images_bin = colmap_sparse_dir / "images.bin"
    images_txt = colmap_sparse_dir / "images.txt"

    if cameras_bin.exists():
        intrinsics = loader.read_intrinsics_binary(cameras_bin)
    elif cameras_txt.exists():
        intrinsics = loader.read_intrinsics_text(cameras_txt)
    else:
        raise FileNotFoundError(f"COLMAP cameras 文件不存在: {colmap_sparse_dir}")

    if images_bin.exists():
        extrinsics = loader.read_extrinsics_binary(images_bin)
    elif images_txt.exists():
        extrinsics = loader.read_extrinsics_text(images_txt)
    else:
        raise FileNotFoundError(f"COLMAP images 文件不存在: {colmap_sparse_dir}")

    return loader, intrinsics, extrinsics


def focal_lengths_from_colmap_camera(camera: Any) -> tuple[float, float]:
    params = [float(value) for value in camera.params]
    if camera.model in {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "FOV",
    }:
        return params[0], params[0]
    if camera.model in {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "THIN_PRISM_FISHEYE",
    }:
        return params[0], params[1]
    raise ValueError(
        f"暂不支持的 COLMAP camera model: {camera.model}。"
        "请先做 undistort，或扩展 focal 提取逻辑。"
    )


def load_camera_entries_from_colmap(colmap_sparse_dir: Path) -> list[dict[str, Any]]:
    loader, cam_intrinsics, cam_extrinsics = read_colmap_model(colmap_sparse_dir)

    camera_entries: list[dict[str, Any]] = []
    for _, extrinsic in cam_extrinsics.items():
        intrinsic = cam_intrinsics[extrinsic.camera_id]
        fx, fy = focal_lengths_from_colmap_camera(intrinsic)

        rotation_world_to_cam = np.transpose(loader.qvec2rotmat(extrinsic.qvec))
        translation_world_to_cam = np.asarray(extrinsic.tvec, dtype=np.float64)

        rt = np.zeros((4, 4), dtype=np.float64)
        rt[:3, :3] = rotation_world_to_cam.transpose()
        rt[:3, 3] = translation_world_to_cam
        rt[3, 3] = 1.0

        camera_to_world = np.linalg.inv(rt)
        position = camera_to_world[:3, 3]
        rotation = camera_to_world[:3, :3]

        image_path = Path(extrinsic.name)
        camera_entries.append(
            {
                "id": int(extrinsic.id),
                "img_name": image_path.stem,
                "source_image_name": image_path.name,
                "width": int(intrinsic.width),
                "height": int(intrinsic.height),
                "position": position.astype(float).tolist(),
                "rotation": rotation.astype(float).tolist(),
                "fx": float(fx),
                "fy": float(fy),
            }
        )

    sorted_entries = sorted(
        camera_entries,
        key=lambda item: (str(item["img_name"]), int(item["id"])),
    )
    for index, camera_entry in enumerate(sorted_entries):
        camera_entry["id"] = index
    return sorted_entries


def render_index_from_path(render_image: Path) -> int:
    match = re.search(r"(\d+)$", render_image.stem)
    if match is None:
        raise ValueError(f"无法从 render 文件名解析索引: {render_image.name}")
    return int(match.group(1))


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat.astype(float).tolist()


def flatten_row_major(matrix: list[list[float]]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def flatten_column_major(matrix: list[list[float]]) -> list[float]:
    array = np.asarray(matrix, dtype=np.float64)
    return array.T.reshape(-1).astype(float).tolist()


def build_intrinsics(
    *,
    source_width: int,
    source_height: int,
    render_width: int,
    render_height: int,
    fx: float,
    fy: float,
) -> list[list[float]]:
    scale_x = render_width / source_width
    scale_y = render_height / source_height
    return [
        [float(fx * scale_x), 0.0, float(render_width / 2.0)],
        [0.0, float(fy * scale_y), float(render_height / 2.0)],
        [0.0, 0.0, 1.0],
    ]


def build_camera_to_world(rotation: np.ndarray, position: np.ndarray) -> list[list[float]]:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = position
    return c2w.astype(float).tolist()


def infer_iteration(render_image: Path | None) -> int | None:
    if render_image is None:
        return None
    for parent in render_image.parents:
        match = re.fullmatch(r"ours_(\d+)", parent.name)
        if match is not None:
            return int(match.group(1))
    return None


def resolve_source_image_name(camera_entry: dict[str, Any], source_images_dir: Path | None) -> str:
    explicit = camera_entry.get("source_image_name")
    if explicit:
        return str(explicit)

    img_name = str(camera_entry["img_name"])
    if source_images_dir is not None:
        matches = sorted(source_images_dir.glob(f"{img_name}.*"))
        if matches:
            return matches[0].name
    return f"{img_name}.jpg"


def resolve_source_image_path(
    *,
    source_images_dir: Path | None,
    source_image_name: str,
) -> str | None:
    if source_images_dir is None:
        return None
    candidate = source_images_dir / source_image_name
    if candidate.exists():
        return str(candidate.resolve())
    return None


def build_frame_record(
    *,
    frame_index: int,
    dataset_index: int,
    camera_entry: dict[str, Any],
    output_width: int,
    output_height: int,
    image_name: str,
    image_path: str | None,
    source_images_dir: Path | None,
) -> dict[str, Any]:
    source_width = int(camera_entry["width"])
    source_height = int(camera_entry["height"])
    source_image_name = resolve_source_image_name(camera_entry, source_images_dir)
    source_image_path = resolve_source_image_path(
        source_images_dir=source_images_dir,
        source_image_name=source_image_name,
    )

    rotation = np.asarray(camera_entry["rotation"], dtype=np.float64)
    position = np.asarray(camera_entry["position"], dtype=np.float64)
    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(rotation)
    quaternion_wxyz = [
        float(quaternion_xyzw[3]),
        float(quaternion_xyzw[0]),
        float(quaternion_xyzw[1]),
        float(quaternion_xyzw[2]),
    ]
    intrinsics = build_intrinsics(
        source_width=source_width,
        source_height=source_height,
        render_width=output_width,
        render_height=output_height,
        fx=float(camera_entry["fx"]),
        fy=float(camera_entry["fy"]),
    )
    camera_to_world = build_camera_to_world(rotation, position)

    return {
        "frame_index": frame_index,
        "time_sec": float(frame_index),
        "dataset_index": dataset_index,
        "parser_index": int(camera_entry["id"]),
        "image_name": image_name,
        "image_path": image_path,
        "image_size": [output_width, output_height],
        "source_image_name": source_image_name,
        "source_image_path": source_image_path,
        "source_image_size": [source_width, source_height],
        "position": position.astype(float).tolist(),
        "rotation_matrix": rotation.astype(float).tolist(),
        "quaternion_xyzw": quaternion_xyzw,
        "quaternion_wxyz": quaternion_wxyz,
        "camera_to_world": camera_to_world,
        "intrinsics": intrinsics,
    }


def build_virtual_frame_name(dataset_index: int) -> str:
    return f"{dataset_index:05d}.png"


def build_payloads(
    *,
    output_path: Path,
    unity_output_path: Path,
    camera_entries: list[dict[str, Any]],
    selected_indices: list[int],
    output_width: int,
    output_height: int,
    source_images_dir: Path | None,
    cameras_json_path: Path | None,
    colmap_sparse_dir: Path | None,
    render_image: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for frame_index, dataset_index in enumerate(selected_indices):
        if dataset_index < 0 or dataset_index >= len(camera_entries):
            raise IndexError(
                f"索引越界: index={dataset_index}, camera_count={len(camera_entries)}"
            )
        camera_entry = camera_entries[dataset_index]
        if render_image is not None:
            image_name = render_image.name
            image_path = str(render_image)
        else:
            image_name = build_virtual_frame_name(dataset_index)
            image_path = str((output_path.parent / image_name).resolve())

        frames.append(
            build_frame_record(
                frame_index=frame_index,
                dataset_index=dataset_index,
                camera_entry=camera_entry,
                output_width=output_width,
                output_height=output_height,
                image_name=image_name,
                image_path=image_path,
                source_images_dir=source_images_dir,
            )
        )

    if render_image is not None:
        video_path = str(render_image)
        video_name = render_image.name
        trajectory_source = "g4splat_train_render_camera"
    else:
        video_path = str(output_path.with_suffix(".mp4"))
        video_name = Path(video_path).name
        trajectory_source = "g4splat_colmap_camera_selection"

    load_step = infer_iteration(render_image)
    base_dir = (
        str(cameras_json_path.parent)
        if cameras_json_path is not None
        else str(colmap_sparse_dir)
        if colmap_sparse_dir is not None
        else None
    )
    data_dir = str(source_images_dir.parent) if source_images_dir is not None else None
    exp_name = (
        render_image.parents[1].name
        if render_image is not None and len(render_image.parents) > 1
        else None
    )

    source_payload = {
        "schema_version": 1,
        "exporter": "scripts.export_g4_render_camera",
        "trajectory_source": trajectory_source,
        "video": {
            "path": video_path,
            "name": video_name,
            "width": output_width,
            "height": output_height,
            "fps": 1.0,
            "frame_count": len(frames),
        },
        "refine": {
            "exp_cfg_path": None,
            "base_cfg_path": None,
            "base_dir": base_dir,
            "exp_name": exp_name,
            "data_dir": data_dir,
            "test_split": "train" if render_image is not None else None,
            "test_trans": [0.0, 0.0, 0.0],
            "test_rots": [0.0, 0.0, 0.0],
            "refine_start_idx": selected_indices[0],
            "refine_end_idx": selected_indices[-1] + 1,
            "selected_indices": selected_indices,
            "load_step": load_step,
        },
        "render": {
            "render_image": str(render_image) if render_image is not None else None,
            "render_index": selected_indices[0] if render_image is not None else None,
            "render_resolution": [output_width, output_height],
            "virtual_frames": render_image is None,
            "intrinsics_mode": "g4splat_centered_principal_point",
            "cameras_json": str(cameras_json_path) if cameras_json_path is not None else None,
            "colmap_dir": str(colmap_sparse_dir) if colmap_sparse_dir is not None else None,
            "camera_count": len(camera_entries),
            "camera_order": "sorted_by_img_name",
            "selected_indices": selected_indices,
        },
        "output": {
            "path": str(output_path),
        },
        "frames": frames,
    }

    unity_frames = []
    for frame in frames:
        unity_frames.append(
            {
                "frameIndex": frame["frame_index"],
                "timeSec": frame["time_sec"],
                "datasetIndex": frame["dataset_index"],
                "parserIndex": frame["parser_index"],
                "imageName": frame["image_name"],
                "imagePath": frame["image_path"],
                "imageSize": frame["image_size"],
                "sourceImageName": frame["source_image_name"],
                "sourceImagePath": frame["source_image_path"],
                "sourceImageSize": frame["source_image_size"],
                "position": frame["position"],
                "quaternionXyzw": frame["quaternion_xyzw"],
                "cameraToWorldRowMajor": flatten_row_major(frame["camera_to_world"]),
                "cameraToWorldColumnMajor": flatten_column_major(frame["camera_to_world"]),
                "intrinsicsRowMajor": flatten_row_major(frame["intrinsics"]),
            }
        )

    unity_payload = {
        "schemaVersion": 1,
        "exporter": "scripts.export_g4_render_camera",
        "sourceJson": str(output_path),
        "outputPath": str(unity_output_path),
        "coordinateSpace": "g4splat_colmap_world",
        "axisConversionApplied": False,
        "note": (
            "This payload keeps the original G4Splat/COLMAP world space. "
            "Intrinsics are scaled to the requested output resolution, and the "
            "principal point is fixed to the image center to match G4Splat cameras."
        ),
        "video": source_payload["video"],
        "refine": source_payload["refine"],
        "render": source_payload["render"],
        "frames": unity_frames,
    }

    return source_payload, unity_payload


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    render_image = (
        args.render_image.expanduser().resolve() if args.render_image is not None else None
    )
    cameras_json_path = (
        args.cameras_json.expanduser().resolve()
        if args.cameras_json is not None
        else None
    )
    colmap_sparse_dir = (
        resolve_colmap_sparse_dir(args.colmap_dir)
        if args.colmap_dir is not None
        else None
    )

    if render_image is not None and not render_image.exists():
        raise FileNotFoundError(f"render 图不存在: {render_image}")
    if cameras_json_path is not None and not cameras_json_path.exists():
        raise FileNotFoundError(f"cameras.json 不存在: {cameras_json_path}")

    if cameras_json_path is not None:
        camera_entries = load_camera_entries(cameras_json_path)
        inferred_source_images_dir = infer_source_images_dir_from_cameras_json(
            cameras_json_path
        )
        default_output_dir = cameras_json_path.parent
    else:
        assert colmap_sparse_dir is not None
        camera_entries = load_camera_entries_from_colmap(colmap_sparse_dir)
        inferred_source_images_dir = infer_source_images_dir_from_colmap(
            colmap_sparse_dir
        )
        default_output_dir = infer_scene_root_from_colmap_sparse_dir(colmap_sparse_dir)

    source_images_dir = (
        args.source_images_dir.expanduser().resolve()
        if args.source_images_dir is not None
        else inferred_source_images_dir
    )

    if render_image is not None:
        selected_indices = [render_index_from_path(render_image)]
        output_width, output_height = Image.open(render_image).size
    else:
        assert args.indices is not None
        assert args.resolution is not None
        selected_indices = parse_indices_spec(args.indices)
        output_width, output_height = parse_resolution_spec(args.resolution)

    output_path = resolve_output_path(
        render_image=render_image,
        output_path=args.output,
        default_dir=default_output_dir,
        selected_indices=selected_indices,
    )
    unity_output_path = resolve_unity_output_path(output_path, args.unity_output)

    source_payload, unity_payload = build_payloads(
        output_path=output_path,
        unity_output_path=unity_output_path,
        camera_entries=camera_entries,
        selected_indices=selected_indices,
        output_width=output_width,
        output_height=output_height,
        source_images_dir=source_images_dir,
        cameras_json_path=cameras_json_path,
        colmap_sparse_dir=colmap_sparse_dir,
        render_image=render_image,
    )

    output_path.write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unity_output_path.write_text(
        json.dumps(unity_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"camera_source: {'cameras_json' if cameras_json_path is not None else 'colmap'}")
    print(f"selected_indices: {selected_indices}")
    print(f"resolution: {output_width}x{output_height}")
    if render_image is not None:
        print(f"render_image: {render_image}")
    if cameras_json_path is not None:
        print(f"cameras_json: {cameras_json_path}")
    if colmap_sparse_dir is not None:
        print(f"colmap_dir: {colmap_sparse_dir}")
    print(f"output: {output_path}")
    print(f"unity_output: {unity_output_path}")


if __name__ == "__main__":
    main()
