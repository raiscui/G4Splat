#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


def parse_image_indices(raw: str) -> list[int]:
    tokens = raw.replace(",", " ").split()
    if not tokens:
        raise ValueError("image_indices cannot be empty")
    return [int(token) for token in tokens]


def read_camera_lines(path: Path) -> tuple[list[str], dict[int, str]]:
    header: list[str] = []
    camera_lines: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            header.append(line)
            continue
        parts = line.split()
        camera_id = int(parts[0])
        camera_lines[camera_id] = line
    return header, camera_lines


def read_image_pairs(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    header: list[str] = []
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            stripped = line.rstrip("\n")
            if not stripped or stripped.startswith("#"):
                header.append(stripped)
                continue
            next_line = handle.readline()
            if next_line == "":
                raise ValueError(f"Missing 2D points line after image line in {path}")
            pairs.append((stripped, next_line.rstrip("\n")))
    return header, pairs


def select_pairs(image_pairs: list[tuple[str, str]], image_indices: list[int]) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for idx in image_indices:
        if idx < 0 or idx >= len(image_pairs):
            raise IndexError(f"Image index {idx} out of range for {len(image_pairs)} image pairs")
        selected.append(image_pairs[idx])
    return selected


def write_text_model(
    out_dir: Path,
    camera_header: list[str],
    camera_lines: dict[int, str],
    image_header: list[str],
    selected_pairs: list[tuple[str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_camera_ids = []
    for image_line, _ in selected_pairs:
        parts = image_line.split()
        selected_camera_ids.append(int(parts[8]))

    selected_camera_ids = sorted(dict.fromkeys(selected_camera_ids))
    cameras_txt = out_dir / "cameras.txt"
    images_txt = out_dir / "images.txt"

    with cameras_txt.open("w", encoding="utf-8") as handle:
        for line in camera_header:
            handle.write(f"{line}\n")
        for camera_id in selected_camera_ids:
            handle.write(f"{camera_lines[camera_id]}\n")

    with images_txt.open("w", encoding="utf-8") as handle:
        for line in image_header:
            handle.write(f"{line}\n")
        for image_line, points_line in selected_pairs:
            handle.write(f"{image_line}\n")
            handle.write(f"{points_line}\n")


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def build_reduced_scene(
    source_scene: Path,
    output_scene: Path,
    image_indices: list[int],
    charts_data: Path | None,
) -> None:
    dense_root = source_scene / "dense-view-sparse" / "0"
    sparse_root = source_scene / "sparse" / "0"
    if not dense_root.exists():
        raise FileNotFoundError(f"Expected dense-view-sparse/0 under {source_scene}")
    if not sparse_root.exists():
        raise FileNotFoundError(f"Expected sparse/0 under {source_scene}")

    dense_cam_header, dense_cameras = read_camera_lines(dense_root / "cameras.txt")
    dense_img_header, dense_image_pairs = read_image_pairs(dense_root / "images.txt")
    sparse_cam_header, sparse_cameras = read_camera_lines(sparse_root / "cameras.txt")
    sparse_img_header, sparse_image_pairs = read_image_pairs(sparse_root / "images.txt")

    selected_dense_pairs = select_pairs(dense_image_pairs, image_indices)
    selected_sparse_pairs = select_pairs(sparse_image_pairs, image_indices)

    output_scene.mkdir(parents=True, exist_ok=True)
    (output_scene / "images").mkdir(parents=True, exist_ok=True)

    for image_line, _ in selected_dense_pairs:
        image_name = image_line.split()[9]
        symlink_or_copy(source_scene / "images" / image_name, output_scene / "images" / image_name)

    write_text_model(output_scene / "dense-view-sparse" / "0", dense_cam_header, dense_cameras, dense_img_header, selected_dense_pairs)
    write_text_model(output_scene / "sparse" / "0", sparse_cam_header, sparse_cameras, sparse_img_header, selected_sparse_pairs)

    for model_root_name in ("dense-view-sparse", "sparse"):
        model_root = output_scene / model_root_name / "0"
        source_root = source_scene / model_root_name / "0"
        for name in ("points3D.ply", "points3D.bin", "points3D.txt"):
            if (source_root / name).exists():
                symlink_or_copy(source_root / name, model_root / name)

    for optional_name in ("all-sparse", "pointmaps", "cameras.json", "points.ply", "chart_pcd.ply"):
        source_path = source_scene / optional_name
        if source_path.exists():
            symlink_or_copy(source_path, output_scene / optional_name)

    if charts_data is not None:
        shutil.copy2(charts_data, output_scene / "charts_data.npz")


def main():
    parser = argparse.ArgumentParser(description="Build a reduced COLMAP/mast3r scene from selected image indices.")
    parser.add_argument("--source_scene", required=True, type=str)
    parser.add_argument("--output_scene", required=True, type=str)
    parser.add_argument("--image_indices", required=True, type=str, help="Comma-separated 0-based indices in sorted image-pair order.")
    parser.add_argument("--charts_data", type=str, default=None, help="Optional charts_data.npz to copy into the reduced scene.")
    args = parser.parse_args()

    build_reduced_scene(
        source_scene=Path(args.source_scene).resolve(),
        output_scene=Path(args.output_scene).resolve(),
        image_indices=parse_image_indices(args.image_indices),
        charts_data=Path(args.charts_data).resolve() if args.charts_data else None,
    )
    print(f"Reduced scene written to {Path(args.output_scene).resolve()}")


if __name__ == "__main__":
    main()
