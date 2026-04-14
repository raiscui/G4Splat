from __future__ import annotations

import unittest

import torch

from scripts.align_charts import (
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


class _FakeScenePointMap:
    def __init__(self, n_charts: int, height: int, width: int):
        self.points3d = torch.zeros((n_charts, height, width, 3), dtype=torch.float32)


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

    def test_resize_boolean_maps_uses_nearest(self):
        boolean_maps = torch.tensor([[[True, False], [False, True]]])
        resized = _resize_boolean_maps(boolean_maps, 4, 4)
        self.assertEqual(tuple(resized.shape), (1, 4, 4))
        self.assertTrue(resized[0, 0, 0].item())
        self.assertTrue(resized[0, -1, -1].item())
        self.assertFalse(resized[0, 0, -1].item())


if __name__ == "__main__":
    unittest.main()
