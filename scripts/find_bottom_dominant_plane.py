#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find the dominant plane label in the lower part of each plane_mask_frame*.npy. "
            "Optionally zero that label out across the full frame."
        )
    )
    parser.add_argument(
        "--plane_root_path",
        type=str,
        required=True,
        help="Directory containing plane_mask_frame*.npy and optional plane_vis_frame*.png files.",
    )
    parser.add_argument(
        "--lower-start-ratio",
        type=float,
        default=0.5,
        help="Lower-region start ratio in [0, 1). 0.5 means lower half, 0.66 means lower third.",
    )
    parser.add_argument(
        "--apply-zero",
        action="store_true",
        help="Set the selected dominant label to 0 in each mask and overwrite the npy file.",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Backup directory for original npy files when --apply-zero is used. Defaults to <plane_root>/bottom_plane_backups.",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help="Optional directory to save preview PNGs with the selected plane highlighted.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to save a JSON report.",
    )
    parser.add_argument(
        "--min-bottom-pixels",
        type=int,
        default=1,
        help="Skip frames whose selected lower-region label has fewer than this many pixels in the lower region.",
    )
    return parser.parse_args()


def get_frame_stem(mask_path: Path) -> str:
    return mask_path.stem.replace("plane_mask_", "")


def make_preview(mask: np.ndarray, selected_label: int, vis_path: Path | None, out_path: Path):
    selected = mask == selected_label

    if vis_path is not None and vis_path.exists():
        base = np.array(Image.open(vis_path).convert("RGB"))
    else:
        normalized = np.zeros(mask.shape + (3,), dtype=np.uint8)
        positive = mask > 0
        if positive.any():
            vals = mask.astype(np.float32)
            vals = vals / vals.max()
            normalized[..., 1] = (vals * 255).astype(np.uint8)
            normalized[..., 2] = ((1.0 - vals) * 255).astype(np.uint8)
        base = normalized

    overlay = base.copy()
    overlay[selected] = np.array([255, 32, 32], dtype=np.uint8)
    blended = (0.4 * base + 0.6 * overlay).astype(np.uint8)
    Image.fromarray(blended).save(out_path)


def main():
    args = parse_args()

    if not (0.0 <= args.lower_start_ratio < 1.0):
        raise ValueError("--lower-start-ratio must be in [0, 1).")

    plane_root = Path(args.plane_root_path)
    if not plane_root.is_dir():
        raise FileNotFoundError(f"Plane root path not found: {plane_root}")

    mask_files = sorted(plane_root.glob("plane_mask_frame*.npy"))
    if not mask_files:
        raise FileNotFoundError(f"No plane_mask_frame*.npy files found in: {plane_root}")

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = None
    if args.apply_zero:
        backup_dir = Path(args.backup_dir) if args.backup_dir else plane_root / "bottom_plane_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

    report = []
    modified = 0

    for mask_path in mask_files:
        mask = np.load(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask in {mask_path}, got shape {mask.shape}")

        h, w = mask.shape
        lower_start = int(round(h * args.lower_start_ratio))
        lower_mask = mask[lower_start:, :]

        labels, counts = np.unique(lower_mask, return_counts=True)
        valid = labels != 0
        labels = labels[valid]
        counts = counts[valid]

        frame_info = {
            "mask_file": str(mask_path),
            "frame": get_frame_stem(mask_path),
            "height": int(h),
            "width": int(w),
            "lower_start_row": int(lower_start),
        }

        if labels.size == 0:
            frame_info.update(
                {
                    "selected_label": None,
                    "bottom_pixels": 0,
                    "full_pixels": 0,
                    "bottom_fraction_of_lower_region": 0.0,
                    "bottom_fraction_of_label": 0.0,
                    "modified": False,
                }
            )
            report.append(frame_info)
            continue

        best_idx = int(np.argmax(counts))
        selected_label = int(labels[best_idx])
        bottom_pixels = int(counts[best_idx])
        full_pixels = int(np.count_nonzero(mask == selected_label))
        lower_region_pixels = int(lower_mask.size)

        frame_info.update(
            {
                "selected_label": selected_label,
                "bottom_pixels": bottom_pixels,
                "full_pixels": full_pixels,
                "bottom_fraction_of_lower_region": float(bottom_pixels / max(lower_region_pixels, 1)),
                "bottom_fraction_of_label": float(bottom_pixels / max(full_pixels, 1)),
                "modified": False,
            }
        )

        if preview_dir is not None:
            vis_path = plane_root / f"plane_vis_{frame_info['frame']}.png"
            preview_path = preview_dir / f"bottom_dominant_{frame_info['frame']}.png"
            make_preview(mask, selected_label, vis_path, preview_path)
            frame_info["preview_path"] = str(preview_path)

        if args.apply_zero and bottom_pixels >= args.min_bottom_pixels:
            backup_path = backup_dir / mask_path.name
            if not backup_path.exists():
                np.save(backup_path, mask)
            mask[mask == selected_label] = 0
            np.save(mask_path, mask)
            frame_info["modified"] = True
            modified += 1

        report.append(frame_info)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    selected_frames = sum(1 for item in report if item["selected_label"] is not None)
    print(f"Processed {len(report)} mask files.")
    print(f"Frames with a non-zero lower dominant plane: {selected_frames}")
    if args.apply_zero:
        print(f"Modified {modified} mask files.")

    preview_items = [item for item in report if item.get("selected_label") is not None][:10]
    for item in preview_items:
        print(
            f"{item['frame']}: label={item['selected_label']} "
            f"bottom_pixels={item['bottom_pixels']} full_pixels={item['full_pixels']} "
            f"bottom_fraction_of_lower_region={item['bottom_fraction_of_lower_region']:.4f}"
        )


if __name__ == "__main__":
    main()
