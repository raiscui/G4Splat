from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

from matcha.dm_scene.parallel_aligner import ParallelAligner


class ParallelAlignerPointReferenceTests(unittest.TestCase):
    def test_loss_ignores_out_of_fov_sparse_points(self):
        aligner = types.SimpleNamespace(
            using_pts_as_reference=True,
            n_pm=1,
            reference_pts=[torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32)],
            cameras=types.SimpleNamespace(p3d_cameras=[object()]),
            use_learnable_confidence=False,
        )
        reference_depths = torch.tensor([1.0, 100.0], dtype=torch.float32)
        pred_depths = torch.ones((1, 2, 2), dtype=torch.float32)

        with mock.patch(
            "matcha.dm_scene.parallel_aligner.get_points_depth_in_depthmap",
            return_value=(
                torch.tensor([2.0, 0.0], dtype=torch.float32),
                torch.tensor([True, False]),
            ),
        ):
            loss = ParallelAligner.loss(aligner, reference_depths=reference_depths, pred_depths=pred_depths)

        self.assertAlmostEqual(float(loss.item()), 1.0, places=6)

    def test_loss_returns_zero_when_all_sparse_points_are_out_of_fov(self):
        aligner = types.SimpleNamespace(
            using_pts_as_reference=True,
            n_pm=1,
            reference_pts=[torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)],
            cameras=types.SimpleNamespace(p3d_cameras=[object()]),
            use_learnable_confidence=False,
        )
        reference_depths = torch.tensor([5.0], dtype=torch.float32)
        pred_depths = torch.ones((1, 2, 2), dtype=torch.float32, requires_grad=True)

        with mock.patch(
            "matcha.dm_scene.parallel_aligner.get_points_depth_in_depthmap",
            return_value=(
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([False]),
            ),
        ):
            loss = ParallelAligner.loss(aligner, reference_depths=reference_depths, pred_depths=pred_depths)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(pred_depths.grad)

    def test_optimize_accepts_dense_matching_reference_for_sparse_primary_reference(self):
        class _IdentityTransform:
            def transform_points(self, points):
                return points

        class _FakeP3DCameras:
            def get_world_to_view_transform(self):
                return _IdentityTransform()

        class _FakeCameras:
            def __init__(self):
                self.p3d_cameras = _FakeP3DCameras()

            def get_spatial_extent(self):
                return 1.0

        class _FakeOptimizer:
            def step(self):
                return None

            def zero_grad(self, set_to_none=True):
                return None

        matcher_instances = []

        class _FakeMatcher:
            def __init__(self, cameras, reference_depths):
                self.reference_depths = reference_depths
                self.reference_matches = torch.ones((1, 1, 1, 1), dtype=torch.float32)
                matcher_instances.append(self)

            def match(self, matching_thr):
                self.matching_thr = matching_thr

            def compute_reprojection_errors(self, depths):
                return torch.zeros((1, 1, 1, 1), dtype=depths.dtype), torch.ones((1, 1, 1, 1), dtype=depths.dtype)

            def update_references(self, reference_depths=None):
                self.reference_depths = reference_depths

        aligner = types.SimpleNamespace(
            n_pm=1,
            pm_h=1,
            pm_w=1,
            _depths=torch.ones((1, 1, 1), dtype=torch.float32),
            cameras=_FakeCameras(),
            use_learnable_confidence=False,
            prepare_for_optimization=lambda **kwargs: setattr(aligner, "optimizer", _FakeOptimizer()),
            loss=lambda reference_depths, pred_depths, masks=None: pred_depths.sum(),
            verts=torch.tensor([[[[0.0, 0.0, 1.0]]]], dtype=torch.float32, requires_grad=True),
            _deformed_verts=torch.zeros((1, 1, 1, 3), dtype=torch.float32),
        )

        reference_data = [torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)]
        matching_reference_depths = torch.ones((1, 1, 1), dtype=torch.float32)

        with mock.patch("matcha.dm_scene.parallel_aligner.Matcher3D", _FakeMatcher):
            ParallelAligner.optimize(
                aligner,
                reference_data=reference_data,
                matching_reference_depths=matching_reference_depths,
                n_iterations=50,
                use_gradient_loss=False,
                use_hessian_loss=False,
                use_normal_loss=False,
                use_curvature_loss=False,
                use_matching_loss=True,
                verbose=False,
                lr_update_iters=[],
            )

        self.assertEqual(len(matcher_instances), 1)
        self.assertTrue(torch.equal(matcher_instances[0].reference_depths, matching_reference_depths))


if __name__ == "__main__":
    unittest.main()
