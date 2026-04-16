#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert float depth TIFF files into viewable PNGs using percentile-based normalization."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A depth TIFF file or a directory that contains depth TIFF files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="depth_frame*.tiff",
        help="Glob pattern used when input_path is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when input_path is a directory.",
    )
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used for normalization.",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used for normalization.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name used for PNG output.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_vis.png",
        help="Suffix appended to each source stem to form the output filename.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PNG files.",
    )
    return parser


def collect_depth_paths(input_path: Path, pattern: str, recursive: bool) -> list[Path]:
    resolved = input_path.expanduser()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input path not found: {resolved}")

    iterator = resolved.rglob(pattern) if recursive else resolved.glob(pattern)
    paths = sorted(path for path in iterator if path.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No files matched pattern '{pattern}' under: {resolved}"
        )
    return paths


def compute_normalized_depth(
    depth_map: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, float, float]:
    if depth_map.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth_map.shape}")

    finite_mask = np.isfinite(depth_map)
    finite_values = depth_map[finite_mask]
    if finite_values.size == 0:
        raise ValueError("Depth map does not contain any finite values.")

    low_value, high_value = np.percentile(
        finite_values, [low_percentile, high_percentile]
    ).astype(np.float32)
    if not np.isfinite(low_value) or not np.isfinite(high_value) or high_value <= low_value:
        low_value = np.float32(finite_values.min())
        high_value = np.float32(finite_values.max())

    normalized = np.zeros_like(depth_map, dtype=np.float32)
    if high_value > low_value:
        normalized[finite_mask] = np.clip(
            (depth_map[finite_mask] - low_value) / (high_value - low_value),
            0.0,
            1.0,
        )

    return normalized, float(low_value), float(high_value)


def convert_depth_file(
    src_path: Path,
    *,
    suffix: str,
    cmap: str,
    low_percentile: float,
    high_percentile: float,
    overwrite: bool,
) -> Path:
    output_path = src_path.with_name(f"{src_path.stem}{suffix}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    depth_map = np.asarray(Image.open(src_path), dtype=np.float32)
    normalized, low_value, high_value = compute_normalized_depth(
        depth_map,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    plt.imsave(output_path, normalized, cmap=cmap)
    print(
        f"{src_path} -> {output_path} "
        f"(p{low_percentile:g}={low_value:.6f}, p{high_percentile:g}={high_value:.6f})"
    )
    return output_path


def main() -> None:
    args = build_arg_parser().parse_args()

    if not (0.0 <= args.low_percentile <= 100.0):
        raise ValueError("--low-percentile must be in [0, 100].")
    if not (0.0 <= args.high_percentile <= 100.0):
        raise ValueError("--high-percentile must be in [0, 100].")
    if args.low_percentile >= args.high_percentile:
        raise ValueError("--low-percentile must be smaller than --high-percentile.")
    if not args.suffix.endswith(".png"):
        raise ValueError("--suffix must end with .png.")

    depth_paths = collect_depth_paths(
        args.input_path,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    for depth_path in depth_paths:
        convert_depth_file(
            depth_path,
            suffix=args.suffix,
            cmap=args.cmap,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
