from __future__ import annotations

import json
from pathlib import Path


def _normalize_camera_name(raw_name: str) -> str:
    return Path(str(raw_name)).stem


def load_artifact_camera_names(artifact_source_path: str | None) -> list[str] | None:
    if artifact_source_path is None:
        return None

    cameras_json_path = Path(artifact_source_path) / "cameras.json"
    if not cameras_json_path.exists():
        return None

    with open(cameras_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        filepaths = payload.get("filepaths")
        if filepaths:
            return [_normalize_camera_name(path) for path in filepaths]
        return None

    if isinstance(payload, list):
        ordered_names: list[str] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            raw_name = entry.get("img_name")
            if raw_name is None:
                continue
            ordered_names.append(_normalize_camera_name(raw_name))
        return ordered_names or None

    return None


def filter_cameras_to_artifact_subset(cameras, artifact_source_path: str | None):
    ordered_names = load_artifact_camera_names(artifact_source_path)
    if not ordered_names:
        return cameras

    camera_by_name = {}
    for camera in cameras:
        camera_by_name[_normalize_camera_name(camera.image_name)] = camera

    filtered_cameras = []
    missing_names = []
    for name in ordered_names:
        camera = camera_by_name.get(name)
        if camera is None:
            missing_names.append(name)
            continue
        filtered_cameras.append(camera)

    if missing_names:
        missing_preview = ", ".join(missing_names[:8])
        raise ValueError(
            "Could not find artifact subset cameras in the loaded camera set: "
            f"{missing_preview}"
        )

    return filtered_cameras
