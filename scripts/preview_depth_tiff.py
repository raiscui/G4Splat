#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import colormaps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a depth TIFF file into a PNG preview image."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Input TIFF path, for example refine_depth_frame000000.tiff",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <input_stem>_preview.png beside the TIFF.",
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used for robust normalization. Default: 1.0",
    )
    parser.add_argument(
        "--max-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used for robust normalization. Default: 99.0",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=None,
        help="Explicit lower bound. Overrides --min-percentile when set.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Explicit upper bound. Overrides --max-percentile when set.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name used for the preview. Default: viridis",
    )
    parser.add_argument(
        "--invalid-color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(0, 0, 0),
        help="RGB color for invalid pixels. Default: 0 0 0",
    )
    return parser.parse_args()


def load_depth_tiff(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.array(image, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError(f"Expected a single-channel TIFF, got shape {depth.shape}")

    return depth


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}_preview.png")


def compute_display_range(
    depth: np.ndarray,
    min_percentile: float,
    max_percentile: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[float, float]:
    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        raise ValueError("Depth TIFF has no finite pixels to visualize.")

    finite_values = depth[finite_mask]

    low = float(min_value) if min_value is not None else float(np.percentile(finite_values, min_percentile))
    high = float(max_value) if max_value is not None else float(np.percentile(finite_values, max_percentile))

    if high <= low:
        raise ValueError(
            f"Invalid display range: min={low}, max={high}. "
            "Adjust the percentile or explicit value settings."
        )

    return low, high


def depth_to_preview_rgb(
    depth: np.ndarray,
    low: float,
    high: float,
    colormap_name: str = "viridis",
    invalid_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    finite_mask = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=np.float32)

    # Keep invalid pixels black by default while normalizing only valid depth values.
    normalized[finite_mask] = np.clip((depth[finite_mask] - low) / (high - low), 0.0, 1.0)

    try:
        cmap = colormaps[colormap_name]
    except KeyError as exc:
        raise ValueError(f"Unknown matplotlib colormap: {colormap_name}") from exc

    preview_rgb = (cmap(normalized)[..., :3] * 255.0).astype(np.uint8)
    preview_rgb[~finite_mask] = np.asarray(invalid_color, dtype=np.uint8)
    return preview_rgb


def main():
    args = parse_args()

    if not args.input_path.is_file():
        raise FileNotFoundError(f"Input TIFF not found: {args.input_path}")

    if not (0.0 <= args.min_percentile <= 100.0 and 0.0 <= args.max_percentile <= 100.0):
        raise ValueError("Percentiles must be in [0, 100].")
    if args.max_percentile <= args.min_percentile and (
        args.min_value is None or args.max_value is None
    ):
        raise ValueError("--max-percentile must be greater than --min-percentile.")

    output_path = resolve_output_path(args.input_path, args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth = load_depth_tiff(args.input_path)
    low, high = compute_display_range(
        depth,
        min_percentile=args.min_percentile,
        max_percentile=args.max_percentile,
        min_value=args.min_value,
        max_value=args.max_value,
    )
    preview_rgb = depth_to_preview_rgb(
        depth,
        low=low,
        high=high,
        colormap_name=args.colormap,
        invalid_color=tuple(args.invalid_color),
    )

    Image.fromarray(preview_rgb).save(output_path)

    finite_values = depth[np.isfinite(depth)]
    print(f"Input: {args.input_path}")
    print(f"Output: {output_path}")
    print(f"Shape: {depth.shape[1]}x{depth.shape[0]}")
    print(f"Finite depth range: min={float(finite_values.min()):.6f}, max={float(finite_values.max()):.6f}")
    print(f"Display range: min={low:.6f}, max={high:.6f}")
    print(f"Colormap: {args.colormap}")


if __name__ == "__main__":
    main()
