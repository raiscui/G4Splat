import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preview_depth_tiff.py"
SPEC = importlib.util.spec_from_file_location("preview_depth_tiff", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PreviewDepthTiffTests(unittest.TestCase):
    def test_compute_display_range_uses_finite_percentiles(self):
        depth = np.array(
            [
                [np.nan, 10.0, 20.0],
                [30.0, 40.0, np.inf],
            ],
            dtype=np.float32,
        )

        low, high = MODULE.compute_display_range(depth, min_percentile=0.0, max_percentile=100.0)

        self.assertEqual(low, 10.0)
        self.assertEqual(high, 40.0)

    def test_depth_to_preview_rgb_marks_invalid_pixels(self):
        depth = np.array(
            [
                [0.0, 5.0],
                [10.0, np.nan],
            ],
            dtype=np.float32,
        )

        preview = MODULE.depth_to_preview_rgb(
            depth,
            low=0.0,
            high=10.0,
            invalid_color=(1, 2, 3),
        )

        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(preview[1, 1], np.array([1, 2, 3], dtype=np.uint8)))
        self.assertFalse(np.array_equal(preview[0, 0], preview[1, 0]))


if __name__ == "__main__":
    unittest.main()
