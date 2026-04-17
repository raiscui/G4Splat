from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from matcha.dm_utils.depth_trust import (
    build_depth_agreement_mask,
    erode_binary_mask,
    load_confident_mask_from_visibility,
)


class DepthTrustTests(unittest.TestCase):
    def test_build_depth_agreement_mask_keeps_only_close_depths(self):
        warp = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float32)
        aligned = torch.tensor([[10.4, 13.0, 8.7]], dtype=torch.float32)
        candidate = torch.tensor([[True, True, False]])

        mask = build_depth_agreement_mask(
            warp_depth=warp,
            aligned_depth=aligned,
            candidate_mask=candidate,
            max_relative_error=0.05,
            max_absolute_error=0.5,
        )

        expected = torch.tensor([[True, False, False]])
        self.assertTrue(torch.equal(mask.cpu(), expected))

    def test_load_confident_mask_from_visibility_prefers_saved_mask(self):
        with tempfile.TemporaryDirectory(prefix="depth-trust-") as tmp_dir:
            plane_root = Path(tmp_dir)
            np.save(plane_root / "visibility_frame000000.npy", np.array([[1, 0], [0, 1]], dtype=np.uint8))

            confident = load_confident_mask_from_visibility(
                plane_root_path=str(plane_root),
                view_id=0,
                fallback_shape=(2, 2),
            )

            self.assertTrue(np.array_equal(confident, np.array([[1, 0], [0, 1]], dtype=np.uint8)))

    def test_load_confident_mask_from_visibility_falls_back_to_ones(self):
        with tempfile.TemporaryDirectory(prefix="depth-trust-") as tmp_dir:
            plane_root = Path(tmp_dir)

            confident = load_confident_mask_from_visibility(
                plane_root_path=str(plane_root),
                view_id=0,
                fallback_shape=(2, 3),
            )

            self.assertTrue(np.array_equal(confident, np.ones((2, 3), dtype=np.uint8)))

    def test_erode_binary_mask_removes_boundary_ring(self):
        mask = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        eroded = erode_binary_mask(mask, radius=1)

        expected = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(eroded.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
