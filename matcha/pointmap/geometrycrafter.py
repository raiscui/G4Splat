from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from matcha.pointmap.base import PointMap


_DEFAULT_VIEW_ORDER_FALLBACK = (0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9)
_GEOMETRYCRAFTER_PROCESSING_MULTIPLE = 64
_DEFAULT_GEOMETRYCRAFTER_MAX_RES = 1024
_SAM2_SEQUENCE_MODULE = None


def _load_sam2_sequence_module():
    global _SAM2_SEQUENCE_MODULE
    if _SAM2_SEQUENCE_MODULE is not None:
        return _SAM2_SEQUENCE_MODULE

    module_path = Path(__file__).resolve().parents[2] / "2d-gaussian-splatting" / "planes" / "sam2_sequence.py"
    if not module_path.exists():
        _SAM2_SEQUENCE_MODULE = False
        return None

    spec = importlib.util.spec_from_file_location("g4splat_sam2_sequence", module_path)
    if spec is None or spec.loader is None:
        _SAM2_SEQUENCE_MODULE = False
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SAM2_SEQUENCE_MODULE = module
    return module


def _get_default_view_order() -> tuple[int, ...]:
    module = _load_sam2_sequence_module()
    if module is not None and hasattr(module, "DEFAULT_VIEW_ORDER"):
        return tuple(int(v) for v in module.DEFAULT_VIEW_ORDER)
    return _DEFAULT_VIEW_ORDER_FALLBACK


def _get_interleaved_layout(num_views: int, view_order: tuple[int, ...]):
    module = _load_sam2_sequence_module()
    if module is not None and hasattr(module, "InterleavedSequenceLayout"):
        return module.InterleavedSequenceLayout(num_views=num_views, view_order=view_order)
    return _FallbackInterleavedSequenceLayout(num_views=num_views, view_order=view_order)


@dataclass(frozen=True)
class _FallbackInterleavedSequenceLayout:
    num_views: int
    view_order: tuple[int, ...]

    def __post_init__(self):
        if len(self.view_order) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} entries in view_order, got {len(self.view_order)}: {self.view_order}"
            )
        if len(set(self.view_order)) != len(self.view_order):
            raise ValueError(f"view_order must not contain duplicates: {self.view_order}")

    def split_index(self, index: int) -> tuple[int, int]:
        return index // self.num_views, index % self.num_views


@dataclass(frozen=True)
class GeometryCrafterFrame:
    local_index: int
    global_image_index: int
    time_index: int
    view_slot: int
    source_view_id: int
    image_path: Path


@dataclass(frozen=True)
class GeometryCrafterSequence:
    view_slot: int
    source_view_id: int
    frames: tuple[GeometryCrafterFrame, ...]


@dataclass(frozen=True)
class GeometryCrafterCacheEntry:
    sequence: GeometryCrafterSequence
    video_path: Path
    npz_path: Path
    manifest_path: Path
    cache_dir: Path


class PointMapGeometryCrafter(PointMap):
    def __init__(
        self,
        scene_cameras=None,
        scene_eval_cameras=None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scene_cameras = scene_cameras
        self.scene_eval_cameras = scene_eval_cameras
        self.metadata = metadata or {}


def parse_view_order(view_order: str | Sequence[int] | None, *, num_views: int = 12) -> tuple[int, ...]:
    if view_order is None:
        parsed = _get_default_view_order()
    elif isinstance(view_order, str):
        normalized = view_order.strip()
        if normalized.startswith("[") and normalized.endswith("]"):
            normalized = normalized[1:-1]
        if not normalized:
            raise ValueError("view_order string cannot be empty")
        parsed = tuple(int(token.strip()) for token in normalized.split(",") if token.strip())
    else:
        parsed = tuple(int(v) for v in view_order)

    if len(parsed) != num_views:
        raise ValueError(f"Expected {num_views} entries in view_order, got {len(parsed)}: {parsed}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"view_order must not contain duplicates: {parsed}")
    return parsed


def _coerce_image_indices(
    pointmap_cameras,
    *,
    image_indices: list[int] | None,
    n_images_in_pointmap: int | None,
) -> list[int]:
    if image_indices is not None:
        return [int(idx) for idx in image_indices]
    if n_images_in_pointmap is None:
        return list(range(len(pointmap_cameras)))
    if n_images_in_pointmap <= 0:
        raise ValueError(f"n_images_in_pointmap must be positive, got {n_images_in_pointmap}")
    if n_images_in_pointmap == 1:
        return [0]

    n_total_images = len(pointmap_cameras)
    frame_interval = max(1, n_total_images // max(1, n_images_in_pointmap - 1))
    selected = [step * frame_interval for step in range(n_images_in_pointmap)]
    return [min(idx, n_total_images - 1) for idx in selected]


def _resolve_scene_image_dir(scene_source_path: str | os.PathLike[str]) -> Path:
    scene_root = Path(scene_source_path)
    if not scene_root.exists():
        raise FileNotFoundError(f"Scene source path does not exist: {scene_root}")

    image_suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if any(child.suffix in image_suffixes for child in scene_root.iterdir() if child.is_file()):
        return scene_root

    image_dir = scene_root / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Could not find images directory under {scene_root}")
    return image_dir


def _resolve_image_extension(image_dir: Path) -> str:
    for candidate in sorted(image_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return candidate.suffix
    raise FileNotFoundError(f"Could not infer image extension from {image_dir}")


def _build_selected_image_paths(
    pointmap_cameras,
    camera_indices: Sequence[int],
    image_dir: Path,
) -> list[Path]:
    extension = _resolve_image_extension(image_dir)
    image_paths = []
    for camera_idx in camera_indices:
        image_name = pointmap_cameras.gs_cameras[camera_idx].image_name.split(".")[0]
        image_path = image_dir / f"{image_name}{extension}"
        if not image_path.exists():
            raise FileNotFoundError(f"Expected image for camera {image_name} at {image_path}")
        image_paths.append(image_path)
    return image_paths


def build_interleaved_view_sequences(
    image_paths: Sequence[str | os.PathLike[str]],
    *,
    image_indices: Sequence[int] | None = None,
    num_views: int = 12,
    view_order: str | Sequence[int] | None = None,
    layout: str = "auto",
) -> tuple[GeometryCrafterSequence, ...]:
    normalized_paths = tuple(Path(path) for path in image_paths)
    if not normalized_paths:
        raise ValueError("image_paths cannot be empty")

    if image_indices is None:
        normalized_indices = tuple(range(len(normalized_paths)))
    else:
        normalized_indices = tuple(int(idx) for idx in image_indices)
    if len(normalized_indices) != len(normalized_paths):
        raise ValueError("image_indices must match image_paths length")

    resolved_layout = layout
    if resolved_layout == "auto":
        resolved_layout = "single" if num_views <= 1 else "interleaved"
    if resolved_layout not in {"single", "interleaved"}:
        raise ValueError(f"Unsupported layout: {layout}")

    if resolved_layout == "single":
        sequence = GeometryCrafterSequence(
            view_slot=0,
            source_view_id=0,
            frames=tuple(
                GeometryCrafterFrame(
                    local_index=local_idx,
                    global_image_index=normalized_indices[local_idx],
                    time_index=local_idx,
                    view_slot=0,
                    source_view_id=0,
                    image_path=normalized_paths[local_idx],
                )
                for local_idx in range(len(normalized_paths))
            ),
        )
        return (sequence,)

    parsed_view_order = parse_view_order(view_order, num_views=num_views)
    layout_config = _get_interleaved_layout(num_views=num_views, view_order=parsed_view_order)
    grouped_frames: list[list[GeometryCrafterFrame]] = [[] for _ in range(num_views)]
    for local_idx, image_path in enumerate(normalized_paths):
        time_idx, view_slot = layout_config.split_index(local_idx)
        grouped_frames[view_slot].append(
            GeometryCrafterFrame(
                local_index=local_idx,
                global_image_index=normalized_indices[local_idx],
                time_index=time_idx,
                view_slot=view_slot,
                source_view_id=parsed_view_order[view_slot],
                image_path=image_path,
            )
        )

    return tuple(
        GeometryCrafterSequence(
            view_slot=view_slot,
            source_view_id=parsed_view_order[view_slot],
            frames=tuple(grouped_frames[view_slot]),
        )
        for view_slot in range(num_views)
        if grouped_frames[view_slot]
    )


def _hash_jsonable(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sequence_manifest_payload(
    sequence: GeometryCrafterSequence,
    *,
    video_path: Path,
    output_npz_path: Path,
    geometrycrafter_args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sequence": {
            "view_slot": sequence.view_slot,
            "source_view_id": sequence.source_view_id,
            "frame_paths": [str(frame.image_path) for frame in sequence.frames],
            "frame_mtimes_ns": [frame.image_path.stat().st_mtime_ns for frame in sequence.frames],
            "global_image_indices": [frame.global_image_index for frame in sequence.frames],
            "local_indices": [frame.local_index for frame in sequence.frames],
            "time_indices": [frame.time_index for frame in sequence.frames],
        },
        "video_path": str(video_path),
        "output_npz_path": str(output_npz_path),
        "geometrycrafter_args": geometrycrafter_args,
    }


def _load_manifest_if_valid(
    manifest_path: Path,
    expected_payload: dict[str, Any],
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        stored_payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    return stored_payload == expected_payload


def _write_video_from_frames(
    frame_paths: Sequence[Path],
    output_path: Path,
    *,
    fps: int,
) -> None:
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
            for frame_path in frame_paths:
                writer.append_data(np.asarray(Image.open(frame_path).convert("RGB")))
        return
    except Exception:
        pass

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Could not write GeometryCrafter cache video. Install imageio[ffmpeg] or opencv-python."
        ) from exc

    first_frame = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for frame_path in frame_paths:
            rgb = np.asarray(Image.open(frame_path).convert("RGB"))
            if rgb.shape[:2] != (height, width):
                raise ValueError("All frames in a cached sequence must share the same resolution")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _round_to_multiple(value: float, *, multiple: int) -> int:
    rounded = int(round(float(value) / float(multiple)) * multiple)
    return max(multiple, rounded)


def _resolve_geometrycrafter_processing_resolution(
    image_paths: Sequence[Path],
    *,
    requested_height: int | None,
    requested_width: int | None,
    max_res: int = _DEFAULT_GEOMETRYCRAFTER_MAX_RES,
    multiple: int = _GEOMETRYCRAFTER_PROCESSING_MULTIPLE,
) -> tuple[int, int, tuple[int, int]]:
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    if requested_height is not None and requested_height <= 0:
        raise ValueError(f"requested_height must be positive, got {requested_height}")
    if requested_width is not None and requested_width <= 0:
        raise ValueError(f"requested_width must be positive, got {requested_width}")
    if max_res <= 0:
        raise ValueError(f"max_res must be positive, got {max_res}")

    with Image.open(image_paths[0]) as first_image:
        source_width, source_height = first_image.size

    if requested_height is None and requested_width is None:
        scale = min(1.0, float(max_res) / float(max(source_height, source_width)))
        target_height = source_height * scale
        target_width = source_width * scale
    elif requested_height is None:
        target_width = float(requested_width)
        target_height = target_width * float(source_height) / float(source_width)
    elif requested_width is None:
        target_height = float(requested_height)
        target_width = target_height * float(source_width) / float(source_height)
    else:
        target_height = float(requested_height)
        target_width = float(requested_width)

    resolved_height = _round_to_multiple(target_height, multiple=multiple)
    resolved_width = _round_to_multiple(target_width, multiple=multiple)
    return resolved_height, resolved_width, (source_height, source_width)


def _normalize_geometrycrafter_args_for_images(
    image_paths: Sequence[Path],
    geometrycrafter_args: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    normalized_args = dict(geometrycrafter_args)
    resolved_height, resolved_width, (source_height, source_width) = _resolve_geometrycrafter_processing_resolution(
        image_paths,
        requested_height=normalized_args.get("height"),
        requested_width=normalized_args.get("width"),
    )

    original_height = normalized_args.get("height")
    original_width = normalized_args.get("width")
    normalized_args["height"] = resolved_height
    normalized_args["width"] = resolved_width

    if original_height is None and original_width is None:
        return (
            normalized_args,
            (
                "[INFO] GeometryCrafter processing resolution auto-resolved "
                f"from {source_width}x{source_height} to {resolved_width}x{resolved_height}."
            ),
        )

    if original_height != resolved_height or original_width != resolved_width:
        return (
            normalized_args,
            (
                "[INFO] GeometryCrafter processing resolution adjusted to "
                f"{resolved_width}x{resolved_height} from requested "
                f"{original_width}x{original_height} to satisfy aspect preservation and { _GEOMETRYCRAFTER_PROCESSING_MULTIPLE }-pixel alignment."
            ),
        )

    return normalized_args, None


def _prepare_sequence_cache_entries(
    sequences: Sequence[GeometryCrafterSequence],
    *,
    cache_root: Path,
    geometrycrafter_args: dict[str, Any],
    video_fps: int,
    force: bool,
) -> list[GeometryCrafterCacheEntry]:
    cache_root.mkdir(parents=True, exist_ok=True)
    entries: list[GeometryCrafterCacheEntry] = []
    for sequence in sequences:
        cache_dir = cache_root / f"view_{sequence.view_slot:02d}_src_{sequence.source_view_id:02d}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        video_path = cache_dir / f"view_{sequence.view_slot:02d}.mp4"
        npz_path = cache_dir / f"view_{sequence.view_slot:02d}.npz"
        manifest_path = cache_dir / "manifest.json"
        payload = _sequence_manifest_payload(
            sequence,
            video_path=video_path,
            output_npz_path=npz_path,
            geometrycrafter_args=geometrycrafter_args,
        )
        is_valid = _load_manifest_if_valid(manifest_path, payload)
        if force or not is_valid:
            npz_path.unlink(missing_ok=True)
        if force or not (is_valid and video_path.exists()):
            _write_video_from_frames([frame.image_path for frame in sequence.frames], video_path, fps=video_fps)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        entries.append(
            GeometryCrafterCacheEntry(
                sequence=sequence,
                video_path=video_path,
                npz_path=npz_path,
                manifest_path=manifest_path,
                cache_dir=cache_dir,
            )
        )
    return entries


def _default_geometrycrafter_python(geometrycrafter_root: Path) -> str:
    candidate = geometrycrafter_root / ".pixi" / "envs" / "default" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _write_sidecar_manifest(
    *,
    cache_root: Path,
    cache_entries: Sequence[GeometryCrafterCacheEntry],
    geometrycrafter_args: dict[str, Any],
    view_order: tuple[int, ...],
    layout: str,
) -> Path:
    manifest_path = cache_root / "geometrycrafter_sidecar_manifest.json"
    payload = {
        "cache_root": str(cache_root),
        "layout": layout,
        "view_order": list(view_order),
        "geometrycrafter_args": geometrycrafter_args,
        "sequences": [
            {
                "view_slot": entry.sequence.view_slot,
                "source_view_id": entry.sequence.source_view_id,
                "video_path": str(entry.video_path),
                "npz_path": str(entry.npz_path),
                "manifest_path": str(entry.manifest_path),
                "frame_count": len(entry.sequence.frames),
                "global_image_indices": [frame.global_image_index for frame in entry.sequence.frames],
            }
            for entry in cache_entries
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_path


def _geometrycrafter_cli_args(args: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in args.items():
        cli_key = f"--{key}"
        if isinstance(value, bool):
            cli_args.append(f"{cli_key}={str(value)}")
        elif value is None:
            continue
        elif isinstance(value, (list, tuple)):
            cli_args.append(f"{cli_key}={','.join(str(item) for item in value)}")
        else:
            cli_args.append(f"{cli_key}={value}")
    return cli_args


def _run_geometrycrafter_sequences(
    cache_entries: Sequence[GeometryCrafterCacheEntry],
    *,
    geometrycrafter_root: Path,
    python_executable: str,
    geometrycrafter_args: dict[str, Any],
    force: bool,
    parallel_sequences: int = 1,
) -> None:
    script_path = geometrycrafter_root / "run.py"
    if not script_path.exists():
        raise FileNotFoundError(f"GeometryCrafter entrypoint not found: {script_path}")

    pending_entries = []
    for entry in cache_entries:
        if not force and entry.npz_path.exists():
            continue
        pending_entries.append(entry)

    if not pending_entries:
        return

    def run_single_entry(entry: GeometryCrafterCacheEntry) -> None:
        command = [
            python_executable,
            str(script_path),
            str(entry.video_path),
            f"--save_folder={entry.cache_dir}",
        ]
        command.extend(_geometrycrafter_cli_args(geometrycrafter_args))
        subprocess.run(
            command,
            check=True,
            cwd=str(geometrycrafter_root),
        )
        generated_npz_path = entry.cache_dir / f"{entry.video_path.stem}.npz"
        if not generated_npz_path.exists():
            raise FileNotFoundError(
                f"GeometryCrafter finished without producing expected output: {generated_npz_path}"
            )
        if generated_npz_path != entry.npz_path:
            generated_npz_path.replace(entry.npz_path)

    worker_count = max(1, int(parallel_sequences))
    if worker_count == 1:
        for entry in pending_entries:
            run_single_entry(entry)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_single_entry, entry) for entry in pending_entries]
        for future in as_completed(futures):
            future.result()


def _resize_geometry_payload(
    point_map: np.ndarray,
    mask: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if point_map.shape[:2] == (target_height, target_width):
        return point_map.astype(np.float32, copy=False), mask.astype(bool, copy=False)

    point_map_tensor = torch.from_numpy(point_map.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    resized_point_map = torch.nn.functional.interpolate(
        point_map_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).cpu().numpy()
    mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized_mask = torch.nn.functional.interpolate(
        mask_tensor,
        size=(target_height, target_width),
        mode="nearest",
    )[0, 0].cpu().numpy() > 0.5
    return resized_point_map, resized_mask


def _resize_rgb_payload(
    image: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    if tuple(image.shape[-2:]) == (target_height, target_width):
        return image.detach().cpu().permute(1, 2, 0)

    resized_image = torch.nn.functional.interpolate(
        image.detach().cpu().unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized_image.permute(1, 2, 0)


def _camera_points_to_world(point_map: np.ndarray, pose_c2w: torch.Tensor) -> torch.Tensor:
    point_tensor = torch.from_numpy(point_map.astype(np.float32))
    rot = pose_c2w[:3, :3].detach().cpu().float()
    trans = pose_c2w[:3, 3].detach().cpu().float()
    return point_tensor @ rot.transpose(0, 1) + trans


def _camera_pose_from_gs_camera(gs_camera, device: torch.device | str = "cpu") -> torch.Tensor:
    pose_device = torch.device(device)
    R = gs_camera.R.detach().to(pose_device) if isinstance(gs_camera.R, torch.Tensor) else torch.tensor(gs_camera.R, device=pose_device)
    T = gs_camera.T.detach().to(pose_device) if isinstance(gs_camera.T, torch.Tensor) else torch.tensor(gs_camera.T, device=pose_device)
    Rt = torch.cat([R.transpose(-1, -2), T.view(3, 1)], dim=1)
    Rt = torch.cat([Rt, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=pose_device)], dim=0)
    return torch.linalg.inv(Rt).view(4, 4)


def _build_geometrycrafter_pointmap(
    *,
    pointmap_cameras,
    training_cameras,
    test_cameras,
    image_paths: Sequence[Path],
    local_camera_indices: Sequence[int],
    global_image_indices: Sequence[int],
    cache_entries: Sequence[GeometryCrafterCacheEntry],
    target_height: int | None = None,
    target_width: int | None = None,
    device: torch.device | str = "cpu",
) -> PointMapGeometryCrafter:
    from matcha.dm_utils.rendering import fov2focal

    build_device = torch.device("cpu")
    geometry_by_local_index: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for entry in cache_entries:
        data = np.load(entry.npz_path)
        point_map = np.asarray(data["point_map"])
        mask = np.asarray(data["mask"])
        if point_map.shape[0] != len(entry.sequence.frames):
            raise ValueError(
                f"GeometryCrafter output frame count mismatch for {entry.npz_path}: "
                f"expected {len(entry.sequence.frames)}, got {point_map.shape[0]}"
            )
        for sequence_frame_idx, frame in enumerate(entry.sequence.frames):
            geometry_by_local_index[frame.local_index] = (point_map[sequence_frame_idx], mask[sequence_frame_idx])

    img_paths = []
    images = []
    original_images = []
    focals = []
    poses = []
    points3d = []
    confidence = []
    masks = []

    for local_idx, (camera_idx, global_image_idx) in enumerate(zip(local_camera_indices, global_image_indices)):
        if local_idx not in geometry_by_local_index:
            raise KeyError(f"Missing GeometryCrafter output for local frame {local_idx} (global index {global_image_idx})")

        training_camera = training_cameras.gs_cameras[camera_idx]
        img_path_i = str(image_paths[local_idx])

        camera_space_point_map, valid_mask = geometry_by_local_index[local_idx]
        height, width = camera_space_point_map.shape[:2]
        if (
            target_height is not None
            and target_width is not None
            and (height, width) != (target_height, target_width)
        ):
            camera_space_point_map, valid_mask = _resize_geometry_payload(
                camera_space_point_map,
                valid_mask,
                target_height=target_height,
                target_width=target_width,
            )
            height, width = target_height, target_width

        image_i = _resize_rgb_payload(
            training_camera.original_image,
            target_height=height,
            target_width=width,
        )
        original_image_i = image_i.clone()

        fx = fov2focal(training_camera.FoVx, width)
        fy = fov2focal(training_camera.FoVy, height)
        focal_i = torch.tensor([fx, fy], device=build_device)

        pose_i = _camera_pose_from_gs_camera(training_camera, device=build_device)
        points3d_i = _camera_points_to_world(camera_space_point_map.astype(np.float32, copy=False), pose_i)
        confidence_i = torch.from_numpy(valid_mask.astype(np.float32, copy=False))
        mask_i = torch.from_numpy(valid_mask.astype(bool, copy=False))

        img_paths.append(img_path_i)
        images.append(image_i)
        original_images.append(original_image_i)
        focals.append(focal_i)
        poses.append(pose_i)
        points3d.append(points3d_i)
        confidence.append(confidence_i)
        masks.append(mask_i)

    metadata = {
        "mode": "hybrid-override-at-align-prep",
        "image_indices": list(global_image_indices),
        "cache_dirs": [str(entry.cache_dir) for entry in cache_entries],
        "view_slots": [entry.sequence.view_slot for entry in cache_entries],
        "confidence_surrogate": "binary_validity_mask",
    }

    return PointMapGeometryCrafter(
        scene_cameras=training_cameras,
        scene_eval_cameras=test_cameras,
        metadata=metadata,
        img_paths=img_paths,
        images=images,
        original_images=original_images,
        focals=focals,
        poses=poses,
        points3d=points3d,
        confidence=confidence,
        masks=masks,
        device=device,
    )


def run_geometrycrafter_sidecar_from_sfm_data(
    *,
    pointmap_cameras,
    training_cameras,
    test_cameras,
    image_dir: str | os.PathLike[str],
    image_indices: Sequence[int] | None = None,
    n_images_in_pointmap: int | None = None,
    cache_root: str | os.PathLike[str],
    geometrycrafter_root: str | os.PathLike[str] = "/home/rais/GeometryCrafter",
    python_executable: str | None = None,
    view_order: str | Sequence[int] | None = None,
    num_views: int = 12,
    layout: str = "auto",
    video_fps: int = 6,
    force: bool = False,
    parallel_sequences: int = 1,
    device: torch.device | str = "cpu",
    geometrycrafter_args: dict[str, Any] | None = None,
) -> tuple[PointMapGeometryCrafter, list[GeometryCrafterCacheEntry]]:
    resolved_cache_root = Path(cache_root).resolve()
    selected_indices = _coerce_image_indices(
        pointmap_cameras,
        image_indices=list(image_indices) if image_indices is not None else None,
        n_images_in_pointmap=n_images_in_pointmap,
    )
    local_camera_indices = list(range(len(selected_indices)))
    normalized_image_dir = Path(image_dir)
    image_paths = _build_selected_image_paths(pointmap_cameras, local_camera_indices, normalized_image_dir)
    sequences = build_interleaved_view_sequences(
        image_paths,
        image_indices=selected_indices,
        num_views=num_views,
        view_order=view_order,
        layout=layout,
    )

    geometrycrafter_args = dict(geometrycrafter_args or {})
    geometrycrafter_args, resolution_message = _normalize_geometrycrafter_args_for_images(
        image_paths,
        geometrycrafter_args,
    )
    if resolution_message is not None:
        print(resolution_message)
    if "cache_dir" not in geometrycrafter_args:
        geometrycrafter_args["cache_dir"] = str(Path(cache_root) / "hf_cache")
    cache_entries = _prepare_sequence_cache_entries(
        sequences,
        cache_root=resolved_cache_root,
        geometrycrafter_args=geometrycrafter_args,
        video_fps=video_fps,
        force=force,
    )
    _run_geometrycrafter_sequences(
        cache_entries,
        geometrycrafter_root=Path(geometrycrafter_root),
        python_executable=python_executable or _default_geometrycrafter_python(Path(geometrycrafter_root)),
        geometrycrafter_args=geometrycrafter_args,
        force=force,
        parallel_sequences=parallel_sequences,
    )
    parsed_view_order = parse_view_order(view_order, num_views=num_views) if num_views > 1 else (0,)
    sidecar_manifest_path = _write_sidecar_manifest(
        cache_root=resolved_cache_root,
        cache_entries=cache_entries,
        geometrycrafter_args=geometrycrafter_args,
        view_order=parsed_view_order,
        layout=layout,
    )
    scene_pm = _build_geometrycrafter_pointmap(
        pointmap_cameras=pointmap_cameras,
        training_cameras=training_cameras,
        test_cameras=test_cameras,
        image_paths=image_paths,
        local_camera_indices=local_camera_indices,
        global_image_indices=selected_indices,
        cache_entries=cache_entries,
        target_height=geometrycrafter_args.get("height"),
        target_width=geometrycrafter_args.get("width"),
        device=device,
    )
    scene_pm.metadata["cache_root"] = str(resolved_cache_root)
    scene_pm.metadata["sidecar_manifest_path"] = str(sidecar_manifest_path)
    return scene_pm, cache_entries


def get_pointmap_from_mast3r_scene_with_geometrycrafter(
    scene_source_path,
    n_images_in_pointmap,
    image_indices=None,
    white_background=False,
    eval_split=False,
    eval_split_interval=8,
    max_img_size=1600,
    pointmap_img_size=512,
    randomize_images=False,
    max_sfm_points=200_000,
    sfm_confidence_threshold=0.0,
    average_focal_distances=False,
    mast3r_scene_source_path=None,
    geometrycrafter_root: str = "/home/rais/GeometryCrafter",
    geometrycrafter_cache_root: str | None = None,
    geometrycrafter_view_order: str | Sequence[int] | None = None,
    geometrycrafter_num_views: int = 12,
    geometrycrafter_layout: str = "auto",
    geometrycrafter_video_fps: int = 6,
    geometrycrafter_force: bool = False,
    geometrycrafter_parallel_sequences: int = 1,
    geometrycrafter_args: dict[str, Any] | None = None,
    python_executable: str | None = None,
    sidecar_only: bool = False,
    device="cuda",
    return_sfm_data=False,
    return_mast3r_pointmap=False,
):
    from matcha.pointmap.mast3r import compute_mast3r_scene

    mast3r_output = compute_mast3r_scene(
        mast3r_scene_source_path=mast3r_scene_source_path,
        n_images_in_pointmap=n_images_in_pointmap,
        image_indices=image_indices,
        white_background=white_background,
        eval_split=eval_split,
        eval_split_interval=eval_split_interval,
        max_img_size=max_img_size,
        pointmap_img_size=pointmap_img_size,
        randomize_images=randomize_images,
        max_sfm_points=max_sfm_points,
        sfm_confidence_threshold=sfm_confidence_threshold,
        average_focal_distances=average_focal_distances,
        device=device,
        return_pointmap=return_mast3r_pointmap,
    )
    if return_mast3r_pointmap:
        mast3r_sfm_data, mast3r_pm = mast3r_output
    else:
        mast3r_sfm_data = mast3r_output
        mast3r_pm = None

    image_dir = _resolve_scene_image_dir(scene_source_path)
    cache_root = Path(
        geometrycrafter_cache_root
        or Path(mast3r_scene_source_path or scene_source_path) / "geometrycrafter_sidecar"
    )
    scene_pm, cache_entries = run_geometrycrafter_sidecar_from_sfm_data(
        pointmap_cameras=mast3r_sfm_data["pointmap_cameras"],
        training_cameras=mast3r_sfm_data["training_cameras"],
        test_cameras=mast3r_sfm_data["test_cameras"],
        image_dir=image_dir,
        image_indices=image_indices,
        n_images_in_pointmap=n_images_in_pointmap,
        cache_root=cache_root,
        geometrycrafter_root=geometrycrafter_root,
        python_executable=python_executable,
        view_order=geometrycrafter_view_order,
        num_views=geometrycrafter_num_views,
        layout=geometrycrafter_layout,
        video_fps=geometrycrafter_video_fps,
        force=geometrycrafter_force,
        parallel_sequences=geometrycrafter_parallel_sequences,
        device=device,
        geometrycrafter_args=geometrycrafter_args,
    )
    scene_pm.metadata["sidecar_only"] = sidecar_only
    metadata_view_order = (
        list(parse_view_order(geometrycrafter_view_order, num_views=geometrycrafter_num_views))
        if geometrycrafter_num_views > 1
        else [0]
    )
    scene_pm.metadata["cache_key"] = _hash_jsonable(
        {
            "cache_entries": [str(entry.cache_dir) for entry in cache_entries],
            "view_order": metadata_view_order,
            "layout": geometrycrafter_layout,
        }
    )

    if return_sfm_data:
        if return_mast3r_pointmap:
            return scene_pm, mast3r_sfm_data, mast3r_pm
        return scene_pm, mast3r_sfm_data
    return scene_pm
