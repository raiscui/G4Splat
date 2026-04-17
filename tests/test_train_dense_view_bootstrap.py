from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "train.py"
SPEC = importlib.util.spec_from_file_location("tmp_train_module", MODULE_PATH)
TRAIN_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN_MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(TRAIN_MODULE)

load_or_create_dense_view_indices = TRAIN_MODULE.load_or_create_dense_view_indices
resolve_initial_view_selection = TRAIN_MODULE.resolve_initial_view_selection
serialize_image_indices = TRAIN_MODULE.serialize_image_indices
copy_sparse_point_files = TRAIN_MODULE.copy_sparse_point_files
should_skip_initial_chart_plane_refine = TRAIN_MODULE.should_skip_initial_chart_plane_refine


class TrainDenseViewBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="train-dense-view-bootstrap-")
        self.root = Path(self.temp_dir.name)
        images_dir = self.root / "images"
        images_dir.mkdir()
        for idx in range(3):
            Image.new("RGB", (4, 4), color=(idx, idx, idx)).save(images_dir / f"{idx:06d}.png")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_or_create_dense_view_indices_creates_full_view_fallback(self):
        dense_view_path, indices = load_or_create_dense_view_indices(str(self.root))

        self.assertEqual(indices, [0, 1, 2])
        with open(dense_view_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["train"], [0, 1, 2])

    def test_resolve_initial_view_selection_uses_dense_view_when_requested(self):
        _, dense_view_indices = load_or_create_dense_view_indices(str(self.root))

        n_images, image_idx_list, selection_source = resolve_initial_view_selection(
            use_view_config=False,
            source_path=str(self.root),
            config_view_num=10,
            n_images=None,
            image_idx=None,
            use_dense_view=True,
            dense_view_idx_list=dense_view_indices,
        )

        self.assertIsNone(n_images)
        self.assertEqual(image_idx_list, [0, 1, 2])
        self.assertEqual(selection_source, "dense_view_json")

    def test_resolve_initial_view_selection_keeps_explicit_image_idx_precedence(self):
        n_images, image_idx_list, selection_source = resolve_initial_view_selection(
            use_view_config=False,
            source_path=str(self.root),
            config_view_num=10,
            n_images=None,
            image_idx=[5, 7],
            use_dense_view=True,
            dense_view_idx_list=[0, 1, 2],
        )

        self.assertIsNone(n_images)
        self.assertEqual(image_idx_list, [5, 7])
        self.assertEqual(selection_source, "explicit_image_idx")

    def test_serialize_image_indices_uses_comma_separated_cli_format(self):
        self.assertEqual(serialize_image_indices([0, 12, 24]), "0,12,24")
        self.assertIsNone(serialize_image_indices(None))

    def test_copy_sparse_point_files_skips_missing_ply(self):
        source_root = self.root / "source_sparse"
        dest_root = self.root / "dest_sparse"
        source_root.mkdir()
        (source_root / "points3D.bin").write_bytes(b"bin")
        (source_root / "points3D.txt").write_text("txt", encoding="utf-8")

        copy_sparse_point_files(str(source_root), str(dest_root))

        self.assertTrue((dest_root / "points3D.bin").exists())
        self.assertTrue((dest_root / "points3D.txt").exists())
        self.assertFalse((dest_root / "points3D.ply").exists())

    def test_should_skip_initial_chart_plane_refine_only_for_sidecar_mode(self):
        self.assertFalse(should_skip_initial_chart_plane_refine("geometrycrafter", "hybrid-override-at-align-prep"))
        self.assertTrue(should_skip_initial_chart_plane_refine("geometrycrafter", "sidecar-only"))
        self.assertFalse(should_skip_initial_chart_plane_refine("depthanythingv2", "baseline"))


if __name__ == "__main__":
    unittest.main()
