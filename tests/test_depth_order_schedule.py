from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from scripts.refine_free_gaussians import serialize_depth_order_schedule


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_WITH_REFINE_DEPTH_PATH = REPO_ROOT / "2d-gaussian-splatting" / "train_with_refine_depth.py"


def _load_train_with_refine_depth_module():
    spec = importlib.util.spec_from_file_location(
        "g4splat_train_with_refine_depth_test_module",
        TRAIN_WITH_REFINE_DEPTH_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {TRAIN_WITH_REFINE_DEPTH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DepthOrderScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_with_refine_depth = _load_train_with_refine_depth_module()

    def test_serialize_depth_order_schedule_supports_yaml_style_pairs(self):
        serialized = serialize_depth_order_schedule([[1500, 1.0], [3000, 0.1], [6000, 0.001]])
        self.assertEqual(serialized, "1500:1.0,3000:0.1,6000:0.001")

    def test_serialize_depth_order_schedule_supports_dict_items(self):
        serialized = serialize_depth_order_schedule(
            [
                {"iteration": 1500, "weight": 1.0},
                {"iteration": 3000, "weight": 0.1},
            ]
        )
        self.assertEqual(serialized, "1500:1.0,3000:0.1")

    def test_parse_depth_order_schedule_defaults_to_builtin_schedule(self):
        parsed = self.train_with_refine_depth.parse_depth_order_schedule(None)
        self.assertEqual(parsed, list(self.train_with_refine_depth.DEFAULT_DEPTH_ORDER_SCHEDULE))

    def test_parse_depth_order_schedule_parses_cli_string(self):
        parsed = self.train_with_refine_depth.parse_depth_order_schedule("6000:0.1,1500:1.0,3000:0.3")
        self.assertEqual(parsed, [(1500, 1.0), (3000, 0.3), (6000, 0.1)])

    def test_get_depth_order_lambda_uses_last_matching_schedule_entry(self):
        schedule = [(1500, 1.0), (3000, 0.3), (6000, 0.1), (14000, 0.03)]
        get_lambda = self.train_with_refine_depth.get_depth_order_lambda
        self.assertEqual(get_lambda(1000, schedule), 0.0)
        self.assertEqual(get_lambda(2000, schedule), 1.0)
        self.assertEqual(get_lambda(7000, schedule), 0.1)
        self.assertEqual(get_lambda(15000, schedule), 0.03)


if __name__ == "__main__":
    unittest.main()
