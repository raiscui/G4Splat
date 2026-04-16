from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/vis_depth.py"
SPEC = importlib.util.spec_from_file_location("tmp_vis_depth", MODULE_PATH)
VIS_DEPTH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VIS_DEPTH
assert SPEC.loader is not None
SPEC.loader.exec_module(VIS_DEPTH)

collect_depth_paths = VIS_DEPTH.collect_depth_paths
compute_normalized_depth = VIS_DEPTH.compute_normalized_depth


class VisDepthTests(unittest.TestCase):
    def test_collect_depth_paths_returns_single_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "depth_frame000001.tiff"
            path.write_bytes(b"placeholder")
            self.assertEqual(
                collect_depth_paths(path, pattern="depth_frame*.tiff", recursive=False),
                [path],
            )

    def test_collect_depth_paths_matches_directory_pattern(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "depth_frame000001.tiff").write_bytes(b"a")
            (root / "depth_frame000002.tiff").write_bytes(b"b")
            (root / "mono_depth_frame000003.tiff").write_bytes(b"c")

            paths = collect_depth_paths(root, pattern="depth_frame*.tiff", recursive=False)

            self.assertEqual(
                [path.name for path in paths],
                ["depth_frame000001.tiff", "depth_frame000002.tiff"],
            )

    def test_compute_normalized_depth_clamps_to_percentiles(self):
        depth_map = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 100.0],
            ],
            dtype=np.float32,
        )

        normalized, low_value, high_value = compute_normalized_depth(
            depth_map,
            low_percentile=0.0,
            high_percentile=90.0,
        )

        self.assertAlmostEqual(low_value, 1.0)
        self.assertGreater(high_value, 5.0)
        self.assertLess(high_value, 100.0)
        self.assertAlmostEqual(float(normalized[0, 0]), 0.0)
        self.assertAlmostEqual(float(normalized[-1, -1]), 1.0)

    def test_compute_normalized_depth_returns_zero_map_for_constant_input(self):
        depth_map = np.full((2, 3), 7.5, dtype=np.float32)

        normalized, low_value, high_value = compute_normalized_depth(
            depth_map,
            low_percentile=1.0,
            high_percentile=99.0,
        )

        self.assertTrue(np.array_equal(normalized, np.zeros_like(depth_map)))
        self.assertAlmostEqual(low_value, 7.5)
        self.assertAlmostEqual(high_value, 7.5)


if __name__ == "__main__":
    unittest.main()
