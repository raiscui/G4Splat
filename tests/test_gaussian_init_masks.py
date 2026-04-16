from __future__ import annotations

import unittest
from unittest import mock

import torch

from matcha.dm_scene import charts as charts_module


class GaussianInitMaskTests(unittest.TestCase):
    @mock.patch.object(charts_module, "get_gaussian_surfel_parameters_from_mesh")
    @mock.patch.object(charts_module, "remove_faces_from_single_mesh")
    @mock.patch.object(charts_module, "get_manifold_meshes_from_pointmaps")
    def test_get_gaussian_parameters_from_pa_data_uses_visibility_masks(
        self,
        mock_get_manifold_meshes,
        mock_remove_faces,
        mock_get_gaussians,
    ):
        pa_points = torch.zeros((2, 4, 5, 3), dtype=torch.float32)
        images = torch.zeros((2, 4, 5, 3), dtype=torch.float32)
        visibility_masks = torch.tensor(
            [
                [[[1, 0, 1, 0, 1], [1, 1, 0, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1]]],
                [[[0, 1, 0, 1, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [1, 0, 1, 0, 1]]],
            ],
            dtype=torch.float32,
        )

        mock_mesh = mock.Mock()
        mock_faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        mock_verts = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        mock_mesh.faces_packed.return_value = mock_faces
        mock_mesh.verts_packed.return_value = mock_verts
        mock_get_manifold_meshes.return_value = mock_mesh
        mock_remove_faces.return_value = mock_mesh
        mock_get_gaussians.return_value = {"means": torch.zeros((1, 3))}

        result = charts_module.get_gaussian_parameters_from_pa_data(
            pa_points=pa_points,
            images=images,
            visibility_masks=visibility_masks,
        )

        self.assertIn("means", result)
        kwargs = mock_get_manifold_meshes.call_args.kwargs
        self.assertIn("masks", kwargs)
        self.assertEqual(tuple(kwargs["masks"].shape), (2, 4, 5))
        self.assertTrue(kwargs["masks"].dtype == torch.bool)


if __name__ == "__main__":
    unittest.main()
