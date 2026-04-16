from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from matcha.dm_modules.matching_limits import (
    DEFAULT_MATCHING_MAX_PAIRWISE_ELEMENTS,
    estimate_matching_pairwise_elements,
    matching_loss_is_safe,
)
from matcha.pointmap.base import PointMap
from matcha.dm_trainers.charts_alignment import save_charts_data_npz
from scripts.align_charts import (
    _downsample_pointmap_for_alignment,
    _prepare_alignment_masks,
    _prepare_geometrycrafter_reference_depth_maps,
    _prepare_geometrycrafter_reference_point_clouds,
    _prepare_reference_depth_maps,
    _resize_boolean_maps,
    _resize_depth_maps,
)


class _FakeTransform:
    def __init__(self, depths: torch.Tensor):
        self._depths = depths

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        xy = torch.zeros((self._depths.numel(), 2), dtype=self._depths.dtype)
        return torch.cat([xy, self._depths[:, None]], dim=1)


class _FakeP3DCamera:
    def __init__(self, depths: torch.Tensor):
        self._transform = _FakeTransform(depths)

    def get_world_to_view_transform(self):
        return self._transform


class _IdentityTransform:
    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        return points


class _IdentityP3DCamera:
    def __init__(self):
        self._transform = _IdentityTransform()

    def get_world_to_view_transform(self):
        return self._transform


class _FakeGSCamera:
    def __init__(self, image_name: str, image_height: int, image_width: int):
        self.image_name = image_name
        self.image_height = image_height
        self.image_width = image_width


class _FakeCameraWrapper:
    def __init__(self, names, heights, widths, depth_maps):
        self.gs_cameras = [
            _FakeGSCamera(name, height, width)
            for name, height, width in zip(names, heights, widths)
        ]
        self.p3d_cameras = [
            _FakeP3DCamera(depth_map.reshape(-1))
            for depth_map in depth_maps
        ]

    def __len__(self):
        return len(self.gs_cameras)


class _IdentityCameraWrapper:
    def __init__(self, names, heights, widths):
        self.gs_cameras = [
            _FakeGSCamera(name, height, width)
            for name, height, width in zip(names, heights, widths)
        ]
        self.p3d_cameras = [
            _IdentityP3DCamera()
            for _ in names
        ]

    def __len__(self):
        return len(self.gs_cameras)


class _FakeScenePointMap:
    def __init__(self, n_charts: int, height: int, width: int):
        self.points3d = torch.zeros((n_charts, height, width, 3), dtype=torch.float32)
        self.masks = torch.zeros((n_charts, height, width), dtype=torch.bool)


class _FakeMast3RPointMap:
    def __init__(self, confidence: torch.Tensor):
        self.confidence = confidence


class AlignChartsResolutionTests(unittest.TestCase):
    def test_resize_depth_maps_preserves_when_same_shape(self):
        depth_maps = torch.arange(12, dtype=torch.float32).view(1, 3, 4)
        resized = _resize_depth_maps(depth_maps, 3, 4)
        self.assertTrue(torch.equal(resized, depth_maps))

    def test_prepare_reference_depth_maps_upsamples_lowres_maps(self):
        scene_pm = _FakeScenePointMap(n_charts=1, height=4, width=6)
        lowres_depth = torch.arange(6, dtype=torch.float32).view(2, 3)
        scaled_cameras = _FakeCameraWrapper(
            names=["000000"],
            heights=[2],
            widths=[3],
            depth_maps=[lowres_depth],
        )
        sfm_data = {
            "pointmap_cameras": _FakeCameraWrapper(
                names=["000000"],
                heights=[2],
                widths=[3],
                depth_maps=[lowres_depth],
            ),
            "sfm_xyz": torch.zeros((6, 3), dtype=torch.float32),
            "image_sfm_points": {"000000": torch.arange(6, dtype=torch.long)},
        }

        reference = _prepare_reference_depth_maps(
            scene_pm,
            sfm_data,
            scaled_cameras,
            scale_factor=1.0,
        )

        self.assertEqual(tuple(reference.shape), (1, 4, 6))
        self.assertAlmostEqual(reference[0, 0, 0].item(), 0.0)
        self.assertAlmostEqual(reference[0, -1, -1].item(), 5.0)

    def test_prepare_reference_depth_maps_requires_dense_pointmaps(self):
        scene_pm = _FakeScenePointMap(n_charts=1, height=4, width=6)
        lowres_depth = torch.arange(5, dtype=torch.float32)
        scaled_cameras = _FakeCameraWrapper(
            names=["000000"],
            heights=[2],
            widths=[3],
            depth_maps=[lowres_depth],
        )
        sfm_data = {
            "pointmap_cameras": _FakeCameraWrapper(
                names=["000000"],
                heights=[2],
                widths=[3],
                depth_maps=[lowres_depth],
            ),
            "sfm_xyz": torch.zeros((5, 3), dtype=torch.float32),
            "image_sfm_points": {"000000": torch.arange(5, dtype=torch.long)},
        }

        with self.assertRaises(RuntimeError):
            _prepare_reference_depth_maps(
                scene_pm,
                sfm_data,
                scaled_cameras,
                scale_factor=1.0,
            )

    def test_prepare_geometrycrafter_reference_depth_maps_uses_pointmap_depths(self):
        scene_pm = _FakeScenePointMap(n_charts=1, height=2, width=3)
        scene_pm.points3d[0, ..., 2] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        scaled_cameras = _IdentityCameraWrapper(
            names=["000000"],
            heights=[2],
            widths=[3],
        )

        reference = _prepare_geometrycrafter_reference_depth_maps(
            scene_pm,
            scaled_cameras,
            scale_factor=2.0,
        )

        expected = torch.tensor([[[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]])
        self.assertTrue(torch.equal(reference, expected))

    def test_prepare_geometrycrafter_reference_point_clouds_uses_sparse_points(self):
        scaled_cameras = _IdentityCameraWrapper(
            names=["000000", "000001"],
            heights=[2, 2],
            widths=[3, 3],
        )
        sfm_xyz = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        )
        sfm_data = {
            "sfm_xyz": sfm_xyz,
            "image_sfm_points": {
                "000000": torch.tensor([0, 2], dtype=torch.long),
                "000001": torch.tensor([1], dtype=torch.long),
            },
        }

        reference_points = _prepare_geometrycrafter_reference_point_clouds(
            sfm_data,
            scaled_cameras,
            scale_factor=2.0,
        )

        self.assertEqual(len(reference_points), 2)
        self.assertTrue(torch.equal(reference_points[0], torch.tensor([[2.0, 4.0, 6.0], [14.0, 16.0, 18.0]])))
        self.assertTrue(torch.equal(reference_points[1], torch.tensor([[8.0, 10.0, 12.0]])))

    def test_prepare_alignment_masks_uses_geometrycrafter_masks(self):
        scene_pm = _FakeScenePointMap(n_charts=1, height=2, width=2)
        scene_pm.masks = torch.tensor([[[True, False], [False, True]]])

        masks = _prepare_alignment_masks(
            scene_pm=scene_pm,
            mast3r_pm=None,
            masking_config={"use_masks_for_alignment": True, "sfm_mask_threshold": 0.25},
            use_geometrycrafter_masks=True,
        )

        self.assertTrue(torch.equal(masks, scene_pm.masks))

    def test_prepare_alignment_masks_uses_mast3r_confidence_for_legacy_path(self):
        scene_pm = _FakeScenePointMap(n_charts=1, height=2, width=2)
        mast3r_pm = _FakeMast3RPointMap(
            confidence=torch.tensor([[[0.5, 0.2], [0.1, 0.8]]], dtype=torch.float32)
        )

        masks = _prepare_alignment_masks(
            scene_pm=scene_pm,
            mast3r_pm=mast3r_pm,
            masking_config={"use_masks_for_alignment": True, "sfm_mask_threshold": 0.25},
            use_geometrycrafter_masks=False,
        )

        expected = torch.tensor([[[True, False], [False, True]]])
        self.assertTrue(torch.equal(masks, expected))

    def test_resize_boolean_maps_uses_nearest(self):
        boolean_maps = torch.tensor([[[True, False], [False, True]]])
        resized = _resize_boolean_maps(boolean_maps, 4, 4)
        self.assertEqual(tuple(resized.shape), (1, 4, 4))
        self.assertTrue(resized[0, 0, 0].item())
        self.assertTrue(resized[0, -1, -1].item())
        self.assertFalse(resized[0, 0, -1].item())

    def test_matching_guard_keeps_small_workloads_enabled(self):
        safe, pairwise_elements = matching_loss_is_safe(
            n_charts=12,
            height=128,
            width=128,
            max_pairwise_elements=DEFAULT_MATCHING_MAX_PAIRWISE_ELEMENTS,
        )

        self.assertTrue(safe)
        self.assertEqual(pairwise_elements, estimate_matching_pairwise_elements(12, 128, 128))

    def test_matching_guard_flags_large_workloads(self):
        safe, pairwise_elements = matching_loss_is_safe(
            n_charts=294,
            height=576,
            width=1024,
            max_pairwise_elements=DEFAULT_MATCHING_MAX_PAIRWISE_ELEMENTS,
        )

        self.assertFalse(safe)
        self.assertGreater(pairwise_elements, DEFAULT_MATCHING_MAX_PAIRWISE_ELEMENTS)

    def test_downsample_pointmap_for_alignment_scales_spatial_maps_and_focals(self):
        scene_pm = PointMap(
            img_paths=["000000"],
            images=torch.ones((1, 4, 6, 3), dtype=torch.float32),
            original_images=torch.ones((1, 8, 12, 3), dtype=torch.float32),
            focals=torch.tensor([[12.0, 18.0]], dtype=torch.float32),
            poses=torch.eye(4, dtype=torch.float32).view(1, 4, 4),
            points3d=torch.arange(1 * 4 * 6 * 3, dtype=torch.float32).view(1, 4, 6, 3),
            confidence=torch.ones((1, 4, 6), dtype=torch.float32),
            masks=torch.ones((1, 4, 6), dtype=torch.bool),
            device="cpu",
        )

        resized = _downsample_pointmap_for_alignment(scene_pm, 2)

        self.assertEqual(tuple(resized.images.shape), (1, 2, 3, 3))
        self.assertEqual(tuple(resized.original_images.shape), (1, 4, 6, 3))
        self.assertEqual(tuple(resized.points3d.shape), (1, 2, 3, 3))
        self.assertEqual(tuple(resized.confidence.shape), (1, 2, 3))
        self.assertEqual(tuple(resized.masks.shape), (1, 2, 3))
        self.assertTrue(torch.equal(resized.poses, scene_pm.poses))
        self.assertTrue(torch.allclose(resized.focals, torch.tensor([[6.0, 9.0]], dtype=torch.float32)))

    def test_save_charts_data_npz_writes_expected_arrays(self):
        with tempfile.TemporaryDirectory(prefix="align-charts-save-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_charts_data_npz(
                charts_data_path=str(tmp_path),
                prior_depths=torch.ones((1, 2, 3), dtype=torch.float32),
                depths=torch.full((1, 2, 3), 2.0, dtype=torch.float32),
                pts=torch.full((1, 2, 3, 3), 3.0, dtype=torch.float32),
                confs=torch.full((1, 2, 3), 4.0, dtype=torch.float32),
                scale_factor=0.5,
            )

            data = np.load(tmp_path / "charts_data.npz")
            self.assertEqual(tuple(data["prior_depths"].shape), (1, 2, 3))
            self.assertEqual(tuple(data["depths"].shape), (1, 2, 3))
            self.assertEqual(tuple(data["pts"].shape), (1, 2, 3, 3))
            self.assertEqual(tuple(data["confs"].shape), (1, 2, 3))
            self.assertAlmostEqual(float(data["scale_factor"]), 0.5)


if __name__ == "__main__":
    unittest.main()
