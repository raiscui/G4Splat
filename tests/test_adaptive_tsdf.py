from __future__ import annotations

import unittest

import torch

from matcha.dm_extractors.adaptive_tsdf import get_interpolated_value_from_pixel_coordinates


class AdaptiveTSDFTests(unittest.TestCase):
    def test_get_interpolated_value_from_pixel_coordinates_handles_empty_queries_rgb(self):
        value_img = torch.ones((4, 6, 3), dtype=torch.float32)
        pix_coords = torch.empty((0, 2), dtype=torch.float32)

        interpolated = get_interpolated_value_from_pixel_coordinates(value_img, pix_coords)

        self.assertEqual(tuple(interpolated.shape), (0, 3))
        self.assertEqual(interpolated.dtype, value_img.dtype)

    def test_get_interpolated_value_from_pixel_coordinates_handles_empty_queries_scalar(self):
        value_img = torch.ones((4, 6, 1), dtype=torch.float32)
        pix_coords = torch.empty((0, 2), dtype=torch.float32)

        interpolated = get_interpolated_value_from_pixel_coordinates(value_img, pix_coords)

        self.assertEqual(tuple(interpolated.shape), (0, 1))
        self.assertEqual(interpolated.dtype, value_img.dtype)


if __name__ == "__main__":
    unittest.main()
