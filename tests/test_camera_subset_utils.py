from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "2d-gaussian-splatting" / "utils" / "camera_subset_utils.py"
SPEC = importlib.util.spec_from_file_location("tmp_camera_subset_utils", MODULE_PATH)
CAMERA_SUBSET_UTILS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CAMERA_SUBSET_UTILS
assert SPEC.loader is not None
SPEC.loader.exec_module(CAMERA_SUBSET_UTILS)

filter_cameras_to_artifact_subset = CAMERA_SUBSET_UTILS.filter_cameras_to_artifact_subset
load_artifact_camera_names = CAMERA_SUBSET_UTILS.load_artifact_camera_names


class _FakeCamera:
    def __init__(self, image_name: str):
        self.image_name = image_name


class CameraSubsetUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="camera-subset-utils-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_artifact_camera_names_supports_mast3r_dict_payload(self):
        cameras_json = self.root / "cameras.json"
        with open(cameras_json, "w", encoding="utf-8") as handle:
            json.dump({"filepaths": ["images/000010.jpg", "images/000002.jpg"]}, handle)

        self.assertEqual(load_artifact_camera_names(str(self.root)), ["000010", "000002"])

    def test_filter_cameras_to_artifact_subset_reorders_full_scene_cameras(self):
        cameras_json = self.root / "cameras.json"
        with open(cameras_json, "w", encoding="utf-8") as handle:
            json.dump({"filepaths": ["images/000010.jpg", "images/000002.jpg"]}, handle)

        full_scene_cameras = [
            _FakeCamera("000001.jpg"),
            _FakeCamera("000002.jpg"),
            _FakeCamera("000010.jpg"),
        ]

        filtered = filter_cameras_to_artifact_subset(full_scene_cameras, str(self.root))
        self.assertEqual([camera.image_name for camera in filtered], ["000010.jpg", "000002.jpg"])


if __name__ == "__main__":
    unittest.main()
