from __future__ import annotations

import unittest

import torch

from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps


class MeshMaskTests(unittest.TestCase):
    def test_get_manifold_meshes_from_pointmaps_skips_empty_masked_views(self):
        points3d = torch.tensor(
            [
                [
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                    [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
                [
                    [[10.0, 0.0, 1.0], [11.0, 0.0, 1.0]],
                    [[10.0, 1.0, 1.0], [11.0, 1.0, 1.0]],
                ],
            ],
            dtype=torch.float32,
        )
        imgs = torch.ones((2, 2, 2, 3), dtype=torch.float32)
        masks = torch.tensor(
            [
                [[True, True], [True, True]],
                [[False, False], [False, False]],
            ],
            dtype=torch.bool,
        )

        manifold, manifold_idx = get_manifold_meshes_from_pointmaps(
            points3d=points3d,
            imgs=imgs,
            masks=masks,
            return_single_mesh_object=True,
            return_manifold_idx=True,
            device=torch.device("cpu"),
        )

        self.assertEqual(int(manifold_idx.min().item()), 0)
        self.assertEqual(int(manifold_idx.max().item()), 0)
        self.assertGreater(manifold.verts_packed().shape[0], 0)


if __name__ == "__main__":
    unittest.main()
