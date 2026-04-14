from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from matcha.pointmap.mast3r import get_pointmap_with_mast3r


class MASt3RPointmapSubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="mast3r-pointmap-subset-")
        self.root = Path(self.temp_dir.name)
        (self.root / "images").mkdir()
        (self.root / "pointmaps").mkdir()

        cameras = {
            "filepaths": [f"images/{idx:06d}.png" for idx in range(3)],
            "focals": [100.0, 110.0, 120.0],
            "cams2world": [
                np.eye(4, dtype=np.float32).tolist(),
                np.eye(4, dtype=np.float32).tolist(),
                np.eye(4, dtype=np.float32).tolist(),
            ],
        }
        (self.root / "cameras.json").write_text(json.dumps(cameras), encoding="utf-8")

        for idx in range(3):
            rgb = np.full((2, 2, 3), idx / 10.0, dtype=np.float32)
            points = np.full((2, 2, 3), float(idx + 1), dtype=np.float32)
            confs = np.full((2, 2), float(idx + 5), dtype=np.float32)
            pointmap_payload = {
                "rgb": rgb.tolist(),
                "points": points.reshape(-1, 3).tolist(),
                "confs": confs.tolist(),
            }
            (self.root / "pointmaps" / f"{idx:06d}.json").write_text(json.dumps(pointmap_payload), encoding="utf-8")

            image = np.full((2, 2, 3), idx * 20, dtype=np.uint8)
            Image.fromarray(image).save(self.root / "images" / f"{idx:06d}.png")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_explicit_image_indices_select_matching_pointmaps(self):
        pointmap = get_pointmap_with_mast3r(
            mast3r_scene_source_path=str(self.root),
            n_images_in_pointmap=2,
            image_indices=[0, 2],
            max_img_size=2,
            pointmap_img_size=2,
            device="cpu",
        )

        self.assertEqual(len(pointmap.points3d), 2)
        self.assertEqual(Path(pointmap.img_paths[0]).name, "000000.png")
        self.assertEqual(Path(pointmap.img_paths[1]).name, "000002.png")
        self.assertEqual(float(pointmap.points3d[0][0, 0, 0]), 1.0)
        self.assertEqual(float(pointmap.points3d[1][0, 0, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()
