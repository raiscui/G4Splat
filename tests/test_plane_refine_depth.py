from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "2d-gaussian-splatting" / "planes" / "refine_depth_with_planes.py"
SPEC = importlib.util.spec_from_file_location("tmp_refine_depth_with_planes", MODULE_PATH)
REFINE_DEPTH_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REFINE_DEPTH_MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(REFINE_DEPTH_MODULE)

GeneralPlaneRegressor = REFINE_DEPTH_MODULE.GeneralPlaneRegressor
should_apply_aligned_depth = REFINE_DEPTH_MODULE.should_apply_aligned_depth
build_plane_replace_mask = REFINE_DEPTH_MODULE.build_plane_replace_mask


class PlaneRefineDepthTests(unittest.TestCase):
    def test_get_plane_params_uses_stable_closest_point_center(self):
        regressor = GeneralPlaneRegressor()
        regressor.coef_ = np.array([1.0, 0.0, 0.0, -3.0], dtype=np.float64)

        normal, center = regressor.get_plane_params()

        self.assertTrue(np.allclose(normal, np.array([1.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(center, np.array([3.0, 0.0, 0.0])))

    def test_get_plane_params_handles_non_unit_coefficients(self):
        regressor = GeneralPlaneRegressor()
        regressor.coef_ = np.array([0.0, 2.0, 0.0, -8.0], dtype=np.float64)

        normal, center = regressor.get_plane_params()

        self.assertTrue(np.allclose(normal, np.array([0.0, 1.0, 0.0])))
        self.assertTrue(np.allclose(center, np.array([0.0, 4.0, 0.0])))

    def test_should_apply_aligned_depth_accepts_high_quality_plane(self):
        original = torch.linspace(1.0, 2.0, 2048)
        aligned = original * 1.001
        mask = torch.ones_like(original, dtype=torch.bool)

        ok, stats = should_apply_aligned_depth(original, aligned, mask)

        self.assertTrue(ok)
        self.assertEqual(stats["reason"], "ok")

    def test_should_apply_aligned_depth_rejects_bad_plane(self):
        original = torch.linspace(1.0, 2.0, 2048)
        aligned = torch.flip(original, dims=[0]) + 2.0
        mask = torch.ones_like(original, dtype=torch.bool)

        ok, stats = should_apply_aligned_depth(original, aligned, mask)

        self.assertFalse(ok)
        self.assertEqual(stats["reason"], "quality_gate_failed")

    def test_build_plane_replace_mask_filters_large_local_depth_jumps(self):
        original = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32)
        aligned = torch.tensor([10.2, 10.4, 16.0, 9.9], dtype=torch.float32)
        candidate_mask = torch.tensor([True, True, True, False])

        replace_mask = build_plane_replace_mask(original, aligned, candidate_mask)

        expected = torch.tensor([True, True, False, False])
        self.assertTrue(torch.equal(replace_mask.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
