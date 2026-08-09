import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "Scripts" / "Dataset" / "ros1_bag_to_unified.py"

# These helper tests do not exercise image/YAML I/O. Lightweight import stubs
# allow them to run even in a minimal environment without ROS/OpenCV/PyYAML.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("yaml", types.ModuleType("yaml"))

SPEC = importlib.util.spec_from_file_location("ros1_bag_to_unified", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class ConverterHelperTests(unittest.TestCase):
    def test_synchronize_one_to_one_and_drops(self):
        left = [(100, "l0"), (200, "l1"), (300, "l2")]
        right = [(101, "r0"), (250, "drop"), (302, "r2")]
        pairs, dropped_left, dropped_right = converter.synchronize(left, right, tolerance_ns=3)
        self.assertEqual([(p[1], p[3]) for p in pairs], [(100, 101), (300, 302)])
        self.assertEqual(dropped_left, 1)
        self.assertEqual(dropped_right, 1)

    def test_fusionportable_extrinsic_matches_macvo_sensor_to_body(self):
        calib = {
            "quaternion_sensor_body_imu": {"data": [1.0, 0.0, 0.0, 0.0]},
            "translation_sensor_body_imu": {"data": [1.0, 2.0, 3.0]},
        }
        t_bs, found = converter.fusionportable_t_bs(calib)
        self.assertTrue(found)
        np.testing.assert_allclose(t_bs[:3, 3], [1.0, 2.0, 3.0])

    def test_groundtruth_seconds_to_nanoseconds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "gt.txt"
            source.write_text(
                "1.0 0 0 0 0 0 0 2\n"
                "1.5 1 0 0 0 0 0 4\n",
                encoding="utf-8",
            )
            destination = Path(directory) / "groundtruth.csv"
            self.assertEqual(converter.copy_groundtruth(source, destination, "s"), 2)
            rows = destination.read_text(encoding="utf-8").splitlines()
            self.assertTrue(rows[1].startswith("1000000000,"))
            self.assertTrue(rows[2].startswith("1500000000,"))
            self.assertTrue(rows[1].endswith(",1.0"))


if __name__ == "__main__":
    unittest.main()

