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


if __name__ == "__main__":
    unittest.main()
