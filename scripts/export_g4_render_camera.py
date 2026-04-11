#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
            "为 G4Splat 的单张 render 图导出与 FreeFix camera trajectory 相近的 JSON。"
        )
    )
    parser.add_argument(
        "--render-image",
        type=Path,
        required=True,
        help="渲染图路径，例如 train/ours_7000/renders/00000.png",
    )
    parser.add_argument(
        "--cameras-json",
        type=Path,
        required=True,
        help="G4Splat 导出的 cameras.json 路径。",
    )
    parser.add_argument(
        "--source-images-dir",
        type=Path,
        default=None,
        help="原始图像目录。默认尝试自动定位到场景根目录下的 mast3r_sfm/images。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="基础 JSON 输出路径。默认写到 render 同目录，文件名追加 _camera_trajectory.json。",
    )
    parser.add_argument(
        "--unity-output",
        type=Path,
        default=None,
        help="Unity JSON 输出路径。默认自动生成 *_camera_trajectory_unity.json。",
    )
    return parser


def resolve_output_path(render_image: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
    else:
        resolved = render_image.with_name(f"{render_image.stem}_camera_trajectory.json").resolve()
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


def infer_source_images_dir(cameras_json_path: Path) -> Path | None:
    scene_root = cameras_json_path.resolve().parent.parent
    candidates = [
        scene_root / "mast3r_sfm" / "images",
        cameras_json_path.resolve().parent / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_camera_entries(cameras_json_path: Path) -> list[dict[str, Any]]:
    with cameras_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"cameras.json 不是列表: {cameras_json_path}")
    return sorted(data, key=lambda item: (str(item["img_name"]), int(item["id"])))


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


def infer_iteration(render_image: Path) -> int | None:
    for parent in render_image.parents:
        match = re.fullmatch(r"ours_(\d+)", parent.name)
        if match is not None:
            return int(match.group(1))
    return None


def main() -> None:
    args = build_arg_parser().parse_args()

    render_image = args.render_image.expanduser().resolve()
    cameras_json_path = args.cameras_json.expanduser().resolve()
    output_path = resolve_output_path(render_image, args.output)
    unity_output_path = resolve_unity_output_path(output_path, args.unity_output)

    if not render_image.exists():
        raise FileNotFoundError(f"render 图不存在: {render_image}")
    if not cameras_json_path.exists():
        raise FileNotFoundError(f"cameras.json 不存在: {cameras_json_path}")

    source_images_dir = (
        args.source_images_dir.expanduser().resolve()
        if args.source_images_dir is not None
        else infer_source_images_dir(cameras_json_path)
    )

    camera_entries = load_camera_entries(cameras_json_path)
    render_index = render_index_from_path(render_image)
    if render_index < 0 or render_index >= len(camera_entries):
        raise IndexError(
            f"render 索引越界: index={render_index}, camera_count={len(camera_entries)}"
        )

    camera_entry = camera_entries[render_index]
    render_width, render_height = Image.open(render_image).size
    source_width = int(camera_entry["width"])
    source_height = int(camera_entry["height"])
    source_img_name = f"{camera_entry['img_name']}.jpg"
    source_img_path = (
        str((source_images_dir / source_img_name).resolve())
        if source_images_dir is not None
        else None
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
        render_width=render_width,
        render_height=render_height,
        fx=float(camera_entry["fx"]),
        fy=float(camera_entry["fy"]),
    )
    camera_to_world = build_camera_to_world(rotation, position)
    load_step = infer_iteration(render_image)

    frame_record = {
        "frame_index": 0,
        "time_sec": 0.0,
        "dataset_index": render_index,
        "parser_index": int(camera_entry["id"]),
        "image_name": render_image.name,
        "image_path": str(render_image),
        "image_size": [render_width, render_height],
        "source_image_name": source_img_name,
        "source_image_path": source_img_path,
        "source_image_size": [source_width, source_height],
        "position": position.astype(float).tolist(),
        "rotation_matrix": rotation.astype(float).tolist(),
        "quaternion_xyzw": quaternion_xyzw,
        "quaternion_wxyz": quaternion_wxyz,
        "camera_to_world": camera_to_world,
        "intrinsics": intrinsics,
    }

    source_payload = {
        "schema_version": 1,
        "exporter": "scripts.export_g4_render_camera",
        "trajectory_source": "g4splat_train_render_camera",
        "video": {
            "path": str(render_image),
            "name": render_image.name,
            "width": render_width,
            "height": render_height,
            "fps": 1.0,
            "frame_count": 1,
        },
        "refine": {
            "exp_cfg_path": None,
            "base_cfg_path": None,
            "base_dir": str(cameras_json_path.parent),
            "exp_name": render_image.parents[1].name,
            "data_dir": str(source_images_dir.parent) if source_images_dir is not None else None,
            "test_split": "train",
            "test_trans": [0.0, 0.0, 0.0],
            "test_rots": [0.0, 0.0, 0.0],
            "refine_start_idx": render_index,
            "refine_end_idx": render_index + 1,
            "load_step": load_step,
        },
        "render": {
            "render_image": str(render_image),
            "render_index": render_index,
            "cameras_json": str(cameras_json_path),
            "camera_count": len(camera_entries),
            "camera_order": "sorted_by_img_name",
        },
        "output": {
            "path": str(output_path),
        },
        "frames": [frame_record],
    }

    unity_frame = {
        "frameIndex": frame_record["frame_index"],
        "timeSec": frame_record["time_sec"],
        "datasetIndex": frame_record["dataset_index"],
        "parserIndex": frame_record["parser_index"],
        "imageName": frame_record["image_name"],
        "imagePath": frame_record["image_path"],
        "imageSize": frame_record["image_size"],
        "sourceImageName": frame_record["source_image_name"],
        "sourceImagePath": frame_record["source_image_path"],
        "sourceImageSize": frame_record["source_image_size"],
        "position": frame_record["position"],
        "quaternionXyzw": frame_record["quaternion_xyzw"],
        "cameraToWorldRowMajor": flatten_row_major(frame_record["camera_to_world"]),
        "cameraToWorldColumnMajor": flatten_column_major(frame_record["camera_to_world"]),
        "intrinsicsRowMajor": flatten_row_major(frame_record["intrinsics"]),
    }
    unity_payload = {
        "schemaVersion": 1,
        "exporter": "scripts.export_g4_render_camera",
        "sourceJson": str(output_path),
        "outputPath": str(unity_output_path),
        "coordinateSpace": "g4splat_colmap_world",
        "axisConversionApplied": False,
        "note": (
            "This payload keeps the original G4Splat/COLMAP world space. "
            "Intrinsics are scaled to the actual render image resolution."
        ),
        "video": source_payload["video"],
        "refine": source_payload["refine"],
        "render": source_payload["render"],
        "frames": [unity_frame],
    }

    output_path.write_text(
        json.dumps(source_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    unity_output_path.write_text(
        json.dumps(unity_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"render_image: {render_image}")
    print(f"render_index: {render_index}")
    print(f"source_image_name: {source_img_name}")
    print(f"output: {output_path}")
    print(f"unity_output: {unity_output_path}")


if __name__ == "__main__":
    main()
