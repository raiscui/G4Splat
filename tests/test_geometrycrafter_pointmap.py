from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from matcha.pointmap.geometrycrafter import (
    GeometryCrafterCacheEntry,
    GeometryCrafterFrame,
    GeometryCrafterSequence,
    _build_geometrycrafter_pointmap,
    _resolve_geometrycrafter_processing_resolution,
    build_interleaved_view_sequences,
    parse_view_order,
)


class _FakeCamera:
    def __init__(self, image_name: str, image_value: int, translation_x: float = 0.0):
        self.image_name = image_name
        self.original_image = torch.full((3, 2, 2), float(image_value), dtype=torch.float32)
        self.FoVx = math.pi / 3.0
        self.FoVy = math.pi / 3.0
        self.R = torch.eye(3, dtype=torch.float32)
        self.T = torch.tensor([-translation_x, 0.0, 0.0], dtype=torch.float32)


class _FakeCameraSet:
    def __init__(self, cameras):
        self.gs_cameras = list(cameras)

    def __len__(self):
        return len(self.gs_cameras)


class GeometryCrafterHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="geometrycrafter-helper-test-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_rgb(self, name: str, fill_value: int) -> Path:
        array = np.full((4, 5, 3), fill_value, dtype=np.uint8)
        path = self.root / name
        Image.fromarray(array).save(path)
        return path

    def _write_rgb_with_size(self, name: str, *, width: int, height: int, fill_value: int = 0) -> Path:
        array = np.full((height, width, 3), fill_value, dtype=np.uint8)
        path = self.root / name
        Image.fromarray(array).save(path)
        return path

    def test_parse_view_order_accepts_defaults_and_csv(self):
        default_order = parse_view_order(None)
        csv_order = parse_view_order("0,1,10,11,2,3,4,5,6,7,8,9")
        self.assertEqual(default_order, csv_order)
        self.assertEqual(csv_order[2], 10)

    def test_build_interleaved_view_sequences_uses_default_slots(self):
        image_paths = [self._write_rgb(f"{idx:06d}.png", idx) for idx in range(18)]
        sequences = build_interleaved_view_sequences(
            image_paths,
            image_indices=list(range(18)),
            num_views=12,
            layout="auto",
        )

        self.assertEqual(len(sequences), 12)
        self.assertEqual(sequences[0].frames[0].global_image_index, 0)
        self.assertEqual(sequences[0].frames[1].global_image_index, 12)
        self.assertEqual(sequences[2].source_view_id, 10)
        self.assertEqual(len(sequences[5].frames), 2)
        self.assertEqual(len(sequences[6].frames), 1)

    def test_resolve_geometrycrafter_processing_resolution_uses_app_style_default_cap(self):
        image_path = self._write_rgb_with_size("large.png", width=2560, height=1440)

        resolved_height, resolved_width, source_shape = _resolve_geometrycrafter_processing_resolution(
            [image_path],
            requested_height=None,
            requested_width=None,
        )

        self.assertEqual(source_shape, (1440, 2560))
        self.assertEqual((resolved_height, resolved_width), (576, 1024))

    def test_resolve_geometrycrafter_processing_resolution_fills_missing_dimension(self):
        image_path = self._write_rgb_with_size("medium.png", width=1000, height=500)

        resolved_height, resolved_width, _ = _resolve_geometrycrafter_processing_resolution(
            [image_path],
            requested_height=256,
            requested_width=None,
        )

        self.assertEqual((resolved_height, resolved_width), (256, 512))

    def test_build_geometrycrafter_pointmap_preserves_camera_ownership(self):
        image_paths = [
            self._write_rgb("000000.png", 10),
            self._write_rgb("000001.png", 20),
        ]
        cache_dir = self.root / "cache"
        cache_dir.mkdir()
        npz_path = cache_dir / "view_00.npz"
        camera_space_points = np.stack(
            [
                np.ones((2, 2, 3), dtype=np.float32) * np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.ones((2, 2, 3), dtype=np.float32) * np.array([1.0, 2.0, 3.0], dtype=np.float32),
            ],
            axis=0,
        )
        valid_mask = np.array(
            [
                [[True, False], [True, True]],
                [[True, True], [False, True]],
            ],
            dtype=np.bool_,
        )
        np.savez(npz_path, point_map=camera_space_points, mask=valid_mask)

        sequence = GeometryCrafterSequence(
            view_slot=0,
            source_view_id=0,
            frames=(
                GeometryCrafterFrame(
                    local_index=0,
                    global_image_index=0,
                    time_index=0,
                    view_slot=0,
                    source_view_id=0,
                    image_path=image_paths[0],
                ),
                GeometryCrafterFrame(
                    local_index=1,
                    global_image_index=1,
                    time_index=1,
                    view_slot=0,
                    source_view_id=0,
                    image_path=image_paths[1],
                ),
            ),
        )
        cache_entry = GeometryCrafterCacheEntry(
            sequence=sequence,
            video_path=cache_dir / "view_00.mp4",
            npz_path=npz_path,
            manifest_path=cache_dir / "manifest.json",
            cache_dir=cache_dir,
        )
        pointmap_cameras = _FakeCameraSet(
            [
                _FakeCamera("000000", 10, translation_x=0.0),
                _FakeCamera("000001", 20, translation_x=10.0),
            ]
        )
        training_cameras = _FakeCameraSet(
            [
                _FakeCamera("000000", 30, translation_x=0.0),
                _FakeCamera("000001", 40, translation_x=10.0),
            ]
        )

        scene_pm = _build_geometrycrafter_pointmap(
            pointmap_cameras=pointmap_cameras,
            training_cameras=training_cameras,
            test_cameras=None,
            image_paths=image_paths,
            local_camera_indices=[0, 1],
            global_image_indices=[0, 1],
            cache_entries=[cache_entry],
            device="cpu",
        )

        self.assertEqual(scene_pm.metadata["confidence_surrogate"], "binary_validity_mask")
        np.testing.assert_allclose(scene_pm.points3d[0][0, 0].numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(scene_pm.points3d[1][0, 0].numpy(), np.array([11.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_array_equal(scene_pm.confidence[0].numpy(), valid_mask[0].astype(np.float32))
        self.assertEqual(tuple(scene_pm.original_images[1].shape), (2, 2, 3))


if __name__ == "__main__":
    unittest.main()
