from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/generate_view_split.py"
SPEC = importlib.util.spec_from_file_location("tmp_generate_view_split", MODULE_PATH)
GENERATE_VIEW_SPLIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATE_VIEW_SPLIT
assert SPEC.loader is not None
SPEC.loader.exec_module(GENERATE_VIEW_SPLIT)

build_dense_view_indices = GENERATE_VIEW_SPLIT.build_dense_view_indices


class DenseViewSamplingTests(unittest.TestCase):
    def test_default_grouped_sampling_keeps_whole_12_view_blocks(self):
        expected = list(range(12)) + list(range(24, 36))
        self.assertEqual(build_dense_view_indices(36, 2), expected)

    def test_offset_selects_the_other_interleaved_blocks(self):
        expected = list(range(12, 24)) + list(range(36, 48))
        self.assertEqual(build_dense_view_indices(48, 2, offset=2), expected)

    def test_incomplete_tail_group_is_kept_when_selected(self):
        expected = list(range(12)) + list(range(24, 30))
        self.assertEqual(build_dense_view_indices(30, 2), expected)

    def test_group_size_one_preserves_legacy_per_image_stride(self):
        self.assertEqual(build_dense_view_indices(6, 2, group_size=1), [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
