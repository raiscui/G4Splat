import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_g4_render_camera.py"
SPEC = importlib.util.spec_from_file_location("export_g4_render_camera", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExportG4RenderCameraTests(unittest.TestCase):
    def test_colmap_indices_mode_uses_centered_principal_point(self):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            scene_root = Path(tmp_dir_str)
            images_dir = scene_root / "images"
            sparse_dir = scene_root / "sparse" / "0"
            output_path = scene_root / "trajectory.json"

            images_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)

            Image.new("RGB", (640, 480), color=(10, 20, 30)).save(images_dir / "00000.jpg")
            Image.new("RGB", (640, 480), color=(40, 50, 60)).save(images_dir / "00001.jpg")

            (sparse_dir / "cameras.txt").write_text(
                "\n".join(
                    [
                        "# Camera list with one line of data per camera:",
                        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                        "1 PINHOLE 640 480 800 700 123 234",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sparse_dir / "images.txt").write_text(
                "\n".join(
                    [
                        "# Image list with two lines of data per image:",
                        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME",
                        "# POINTS2D[] as (X, Y, POINT3D_ID)",
                        "1 1 0 0 0 0 0 0 1 00000.jpg",
                        "0 0 -1",
                        "2 1 0 0 0 1 0 0 1 00001.jpg",
                        "0 0 -1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--colmap-dir",
                    str(scene_root),
                    "--indices",
                    "[0,1]",
                    "--resolution",
                    "320x240",
                    "--output",
                    str(output_path),
                ],
                check=True,
                cwd=scene_root,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["trajectory_source"], "g4splat_colmap_camera_selection")
            self.assertEqual(payload["render"]["intrinsics_mode"], "g4splat_centered_principal_point")
            self.assertEqual(payload["render"]["selected_indices"], [0, 1])
            self.assertEqual(payload["video"]["width"], 320)
            self.assertEqual(payload["video"]["height"], 240)
            self.assertEqual(len(payload["frames"]), 2)

            first_frame = payload["frames"][0]
            second_frame = payload["frames"][1]

            self.assertEqual(first_frame["source_image_name"], "00000.jpg")
            self.assertEqual(first_frame["image_size"], [320, 240])
            self.assertEqual(
                first_frame["intrinsics"],
                [
                    [400.0, 0.0, 160.0],
                    [0.0, 350.0, 120.0],
                    [0.0, 0.0, 1.0],
                ],
            )
            self.assertEqual(first_frame["position"], [0.0, 0.0, 0.0])
            self.assertEqual(second_frame["position"], [-1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
