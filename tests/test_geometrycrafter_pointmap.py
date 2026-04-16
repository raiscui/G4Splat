from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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
    get_pointmap_from_colmap_scene_with_geometrycrafter,
    parse_view_order,
)


class _FakeCamera:
    def __init__(
        self,
        image_name: str,
        image_value: int,
        translation_x: float = 0.0,
        *,
        height: int = 2,
        width: int = 2,
    ):
        self.image_name = image_name
        self.original_image = torch.full((3, height, width), float(image_value), dtype=torch.float32)
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

    def test_resolve_geometrycrafter_processing_resolution_keeps_explicit_recommended_shape(self):
        image_path = self._write_rgb_with_size("recommended.png", width=1280, height=720)

        resolved_height, resolved_width, _ = _resolve_geometrycrafter_processing_resolution(
            [image_path],
            requested_height=576,
            requested_width=1024,
        )

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

    def test_build_geometrycrafter_pointmap_keeps_geometrycrafter_resolution(self):
        image_paths = [
            self._write_rgb("000000_hi.png", 10),
        ]
        cache_dir = self.root / "cache_hi"
        cache_dir.mkdir()
        npz_path = cache_dir / "view_00.npz"
        camera_space_points = np.ones((1, 4, 5, 3), dtype=np.float32)
        valid_mask = np.ones((1, 4, 5), dtype=np.bool_)
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
                _FakeCamera("000000", 10, height=2, width=2),
            ]
        )
        training_cameras = _FakeCameraSet(
            [
                _FakeCamera("000000", 30, height=8, width=10),
            ]
        )

        scene_pm = _build_geometrycrafter_pointmap(
            pointmap_cameras=pointmap_cameras,
            training_cameras=training_cameras,
            test_cameras=None,
            image_paths=image_paths,
            local_camera_indices=[0],
            global_image_indices=[0],
            cache_entries=[cache_entry],
            device="cpu",
        )

        self.assertEqual(tuple(scene_pm.points3d[0].shape), (4, 5, 3))
        self.assertEqual(tuple(scene_pm.images[0].shape), (4, 5, 3))
        self.assertEqual(tuple(scene_pm.original_images[0].shape), (4, 5, 3))

    def test_build_geometrycrafter_pointmap_can_resize_to_requested_resolution(self):
        image_paths = [
            self._write_rgb("000000_req.png", 10),
        ]
        cache_dir = self.root / "cache_req"
        cache_dir.mkdir()
        npz_path = cache_dir / "view_00.npz"
        camera_space_points = np.ones((1, 2, 3, 3), dtype=np.float32)
        valid_mask = np.ones((1, 2, 3), dtype=np.bool_)
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
            ),
        )
        cache_entry = GeometryCrafterCacheEntry(
            sequence=sequence,
            video_path=cache_dir / "view_00.mp4",
            npz_path=npz_path,
            manifest_path=cache_dir / "manifest.json",
            cache_dir=cache_dir,
        )
        pointmap_cameras = _FakeCameraSet([_FakeCamera("000000", 10, height=2, width=2)])
        training_cameras = _FakeCameraSet([_FakeCamera("000000", 30, height=8, width=10)])

        scene_pm = _build_geometrycrafter_pointmap(
            pointmap_cameras=pointmap_cameras,
            training_cameras=training_cameras,
            test_cameras=None,
            image_paths=image_paths,
            local_camera_indices=[0],
            global_image_indices=[0],
            cache_entries=[cache_entry],
            target_height=4,
            target_width=5,
            device="cpu",
        )

        self.assertEqual(tuple(scene_pm.points3d[0].shape), (4, 5, 3))
        self.assertEqual(tuple(scene_pm.images[0].shape), (4, 5, 3))
        self.assertEqual(tuple(scene_pm.original_images[0].shape), (4, 5, 3))

    @mock.patch("matcha.pointmap.geometrycrafter.run_geometrycrafter_sidecar_from_sfm_data")
    @mock.patch("matcha.pointmap.geometrycrafter.load_colmap_scene")
    def test_get_pointmap_from_colmap_scene_with_geometrycrafter_uses_colmap_anchors(
        self,
        mock_load_colmap_scene,
        mock_run_geometrycrafter_sidecar_from_sfm_data,
    ):
        image_dir = self.root / "images"
        image_dir.mkdir()
        self._write_rgb(str(Path("images") / "000000.png"), 10)

        training_cameras = _FakeCameraSet([_FakeCamera("000000.png", 30, height=8, width=10)])
        pointmap_cameras = _FakeCameraSet([_FakeCamera("000000.png", 20, height=4, width=5)])
        test_cameras = _FakeCameraSet([])
        sfm_xyz = torch.zeros((3, 3), dtype=torch.float32)
        sfm_col = torch.zeros((3, 3), dtype=torch.float32)
        image_sfm_points = {"000000": torch.arange(3, dtype=torch.long)}

        mock_load_colmap_scene.side_effect = [
            {
                "training_cameras": training_cameras,
                "test_cameras": test_cameras,
                "sfm_xyz": sfm_xyz,
                "sfm_col": sfm_col,
                "image_sfm_points": image_sfm_points,
            },
            {
                "training_cameras": pointmap_cameras,
                "test_cameras": test_cameras,
                "sfm_xyz": sfm_xyz,
                "sfm_col": sfm_col,
                "image_sfm_points": image_sfm_points,
            },
        ]

        fake_scene_pm = mock.Mock()
        fake_scene_pm.metadata = {"cache_root": str(self.root / "geometrycrafter_sidecar")}
        cache_dir = self.root / "gc_cache" / "view_00_src_00"
        cache_dir.mkdir(parents=True)
        fake_entry = GeometryCrafterCacheEntry(
            sequence=GeometryCrafterSequence(
                view_slot=0,
                source_view_id=0,
                frames=(),
            ),
            video_path=cache_dir / "view_00.mp4",
            npz_path=cache_dir / "view_00.npz",
            manifest_path=cache_dir / "manifest.json",
            cache_dir=cache_dir,
        )
        mock_run_geometrycrafter_sidecar_from_sfm_data.return_value = (fake_scene_pm, [fake_entry])

        scene_pm, sfm_data = get_pointmap_from_colmap_scene_with_geometrycrafter(
            colmap_source_path=str(self.root),
            scene_source_path=str(self.root),
            geometrycrafter_root="/tmp/GeometryCrafter",
            geometrycrafter_num_views=1,
            geometrycrafter_view_order=(0,),
            geometrycrafter_args={"height": 576, "width": 1024},
            return_sfm_data=True,
            device="cpu",
        )

        self.assertIs(scene_pm, fake_scene_pm)
        self.assertEqual(len(sfm_data["training_cameras"].gs_cameras), 1)
        self.assertEqual(len(sfm_data["pointmap_cameras"].gs_cameras), 1)
        self.assertEqual(sfm_data["training_cameras"].gs_cameras[0].image_name, "000000.png")
        self.assertEqual(sfm_data["pointmap_cameras"].gs_cameras[0].image_name, "000000.png")
        self.assertTrue(scene_pm.metadata["cache_root"].endswith("geometrycrafter_sidecar"))
        self.assertTrue(scene_pm.metadata["cache_key"])
        self.assertFalse(scene_pm.metadata["sidecar_only"])

        self.assertEqual(mock_load_colmap_scene.call_count, 2)
        _, kwargs = mock_run_geometrycrafter_sidecar_from_sfm_data.call_args
        self.assertEqual(len(kwargs["training_cameras"].gs_cameras), 1)
        self.assertEqual(len(kwargs["pointmap_cameras"].gs_cameras), 1)
        self.assertEqual(kwargs["training_cameras"].gs_cameras[0].image_name, "000000.png")
        self.assertEqual(kwargs["pointmap_cameras"].gs_cameras[0].image_name, "000000.png")
        self.assertEqual(kwargs["image_dir"], image_dir)

    @mock.patch("matcha.pointmap.geometrycrafter.run_geometrycrafter_sidecar_from_sfm_data")
    @mock.patch("matcha.pointmap.geometrycrafter.load_colmap_scene")
    def test_get_pointmap_from_colmap_scene_with_geometrycrafter_prefers_scene_camera_subset(
        self,
        mock_load_colmap_scene,
        mock_run_geometrycrafter_sidecar_from_sfm_data,
    ):
        image_dir = self.root / "images"
        image_dir.mkdir()
        self._write_rgb(str(Path("images") / "000000.png"), 10)
        self._write_rgb(str(Path("images") / "000001.png"), 20)
        self._write_rgb(str(Path("images") / "000002.png"), 30)

        scene_subset_root = self.root / "mast3r_sfm"
        scene_subset_root.mkdir()
        scene_subset_images_dir = scene_subset_root / "images"
        scene_subset_images_dir.mkdir()
        with open(scene_subset_root / "cameras.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "filepaths": [
                        str(scene_subset_root / "images" / "000000.png"),
                        str(scene_subset_root / "images" / "000002.png"),
                    ]
                },
                handle,
            )

        training_cameras = _FakeCameraSet(
            [
                _FakeCamera("000000.png", 10, height=8, width=10),
                _FakeCamera("000001.png", 20, height=8, width=10),
                _FakeCamera("000002.png", 30, height=8, width=10),
            ]
        )
        pointmap_cameras = _FakeCameraSet(
            [
                _FakeCamera("000000.png", 10, height=4, width=5),
                _FakeCamera("000001.png", 20, height=4, width=5),
                _FakeCamera("000002.png", 30, height=4, width=5),
            ]
        )
        test_cameras = _FakeCameraSet([])
        sfm_xyz = torch.zeros((3, 3), dtype=torch.float32)
        sfm_col = torch.zeros((3, 3), dtype=torch.float32)
        image_sfm_points = {"000000": torch.arange(3), "000001": torch.arange(3), "000002": torch.arange(3)}

        mock_load_colmap_scene.side_effect = [
            {
                "training_cameras": training_cameras,
                "test_cameras": test_cameras,
                "sfm_xyz": sfm_xyz,
                "sfm_col": sfm_col,
                "image_sfm_points": image_sfm_points,
            },
            {
                "training_cameras": pointmap_cameras,
                "test_cameras": test_cameras,
                "sfm_xyz": sfm_xyz,
                "sfm_col": sfm_col,
                "image_sfm_points": image_sfm_points,
            },
        ]

        fake_scene_pm = mock.Mock()
        fake_scene_pm.metadata = {"cache_root": str(self.root / "geometrycrafter_sidecar")}
        cache_dir = self.root / "gc_cache" / "view_00_src_00"
        cache_dir.mkdir(parents=True)
        fake_entry = GeometryCrafterCacheEntry(
            sequence=GeometryCrafterSequence(
                view_slot=0,
                source_view_id=0,
                frames=(),
            ),
            video_path=cache_dir / "view_00.mp4",
            npz_path=cache_dir / "view_00.npz",
            manifest_path=cache_dir / "manifest.json",
            cache_dir=cache_dir,
        )
        mock_run_geometrycrafter_sidecar_from_sfm_data.return_value = (fake_scene_pm, [fake_entry])

        get_pointmap_from_colmap_scene_with_geometrycrafter(
            colmap_source_path=str(self.root),
            scene_source_path=str(scene_subset_root),
            image_indices=[0, 2],
            geometrycrafter_root="/tmp/GeometryCrafter",
            geometrycrafter_num_views=1,
            geometrycrafter_view_order=(0,),
            geometrycrafter_args={"height": 576, "width": 1024},
            return_sfm_data=True,
            device="cpu",
        )

        _, kwargs = mock_run_geometrycrafter_sidecar_from_sfm_data.call_args
        self.assertEqual([cam.image_name for cam in kwargs["training_cameras"].gs_cameras], ["000000.png", "000002.png"])
        self.assertEqual([cam.image_name for cam in kwargs["pointmap_cameras"].gs_cameras], ["000000.png", "000002.png"])
        self.assertIsNone(kwargs["image_indices"])
        self.assertIsNone(kwargs["n_images_in_pointmap"])
        self.assertEqual(kwargs["image_dir"], scene_subset_images_dir)


if __name__ == "__main__":
    unittest.main()
