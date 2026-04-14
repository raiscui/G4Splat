from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "2d-gaussian-splatting/planes/sam2_sequence.py"
SPEC = importlib.util.spec_from_file_location("tmp_sam2_sequence", MODULE_PATH)
SAM2_SEQUENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAM2_SEQUENCE
assert SPEC.loader is not None
SPEC.loader.exec_module(SAM2_SEQUENCE)

build_sequences = SAM2_SEQUENCE.build_sequences
InterleavedSequenceLayout = SAM2_SEQUENCE.InterleavedSequenceLayout


class SequenceMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="sam2-sequence-test-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_frame_pair(self, frame_id: int):
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        rgb[..., 0] = frame_id
        Image.fromarray(rgb).save(self.root / f"rgb_frame{frame_id:06d}.png")
        np.save(self.root / f"mono_normal_frame{frame_id:06d}.npy", np.zeros((4, 5, 3), dtype=np.float32))

    def test_interleaved_mapping_round_trip(self):
        for frame_id in range(24):
            self._write_frame_pair(frame_id)

        group = build_sequences(
            self.root,
            num_views=12,
            view_order=(0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9),
            layout="interleaved",
        )

        first_sequence = group.sequences[0]
        third_sequence = group.sequences[2]
        last_sequence = group.sequences[11]
        self.assertEqual(group.layout, "interleaved")
        self.assertEqual(first_sequence[0].frame_idx, 0)
        self.assertEqual(first_sequence[1].frame_idx, 12)
        self.assertEqual(first_sequence[1].time_idx, 1)
        self.assertEqual(third_sequence[0].view_id, 10)
        self.assertEqual(last_sequence[0].frame_idx, 11)
        self.assertEqual(last_sequence[1].frame_idx, 23)

    def test_auto_layout_keeps_mast3r_interleaved_with_incomplete_last_group(self):
        mast3r_root = self.root / "mast3r_sfm" / "plane-refine-depths"
        mast3r_root.mkdir(parents=True)
        self.root = mast3r_root
        for frame_id in range(18):
            self._write_frame_pair(frame_id)

        group = build_sequences(mast3r_root, num_views=12, layout="auto")
        self.assertEqual(group.layout, "interleaved")
        self.assertEqual(len(group.sequences[0]), 2)
        self.assertEqual(len(group.sequences[5]), 2)
        self.assertEqual(len(group.sequences[6]), 1)
        self.assertEqual(len(group.sequences[11]), 1)

    def test_single_layout_for_select_gs_planes(self):
        select_root = self.root / "see3d_render" / "stage1" / "select-gs-planes"
        select_root.mkdir(parents=True)
        self.root = select_root
        for frame_id in range(6):
            self._write_frame_pair(frame_id)

        group = build_sequences(select_root, num_views=12, layout="auto")
        self.assertEqual(group.layout, "single")
        self.assertEqual(group.num_views, 1)
        self.assertEqual(len(group.sequences), 1)
        self.assertEqual(len(group.sequences[0]), 6)

    def test_invalid_layout_order_is_rejected(self):
        with self.assertRaises(ValueError):
            InterleavedSequenceLayout(num_views=12, view_order=(0, 1, 2))


if __name__ == "__main__":
    unittest.main()
