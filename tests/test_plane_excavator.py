from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "2d-gaussian-splatting/planes/plane_excavator.py"


def _load_plane_excavator_module():
    plane_dir = str(MODULE_PATH.parent)
    if plane_dir not in sys.path:
        sys.path.insert(0, plane_dir)

    if "mask_generator" not in sys.modules:
        mask_generator = types.ModuleType("mask_generator")
        mask_generator.setup_sam = lambda *args, **kwargs: None
        sys.modules["mask_generator"] = mask_generator

    if "sam2_sequence" not in sys.modules:
        sam2_sequence = types.ModuleType("sam2_sequence")
        sam2_sequence.DEFAULT_VIEW_ORDER = (0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9)
        sam2_sequence.build_sequences = lambda *args, **kwargs: None
        sam2_sequence.materialize_sequence_jpgs = lambda *args, **kwargs: None
        sys.modules["sam2_sequence"] = sam2_sequence

    if "utils.general_utils" not in sys.modules:
        utils_module = types.ModuleType("utils")
        general_utils = types.ModuleType("utils.general_utils")
        general_utils.seed_everything = lambda *args, **kwargs: None
        sys.modules["utils"] = utils_module
        sys.modules["utils.general_utils"] = general_utils

    spec = importlib.util.spec_from_file_location("tmp_plane_excavator", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLANE_EXCAVATOR = _load_plane_excavator_module()
normals_cluster = PLANE_EXCAVATOR.normals_cluster


class PlaneExcavatorNormalsClusterTests(unittest.TestCase):
    def test_normals_cluster_ignores_sparse_nan_pixels(self):
        height, width = 128, 128
        normals = np.zeros((height, width, 3), dtype=np.float32)
        normals[:, : width // 2, 0] = 1.0
        normals[:, width // 2 :, 1] = 1.0

        normals[0, 0] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        normals[1, 1] = np.array([np.nan, 0.0, 1.0], dtype=np.float32)
        normals[2, 2] = np.array([0.0, np.nan, 1.0], dtype=np.float32)
        normals[3, 3] = np.array([0.0, 1.0, np.nan], dtype=np.float32)

        masks = normals_cluster(
            normals,
            (height, width),
            n_init_clusters=4,
            n_clusters=2,
            min_size_ratio=0.01,
        )

        self.assertGreaterEqual(len(masks), 2)
        self.assertTrue(all(mask.shape == (height, width) for mask in masks))
        self.assertTrue(any(mask[:, : width // 2].sum() > 1000 for mask in masks))
        self.assertTrue(any(mask[:, width // 2 :].sum() > 1000 for mask in masks))


if __name__ == "__main__":
    unittest.main()
