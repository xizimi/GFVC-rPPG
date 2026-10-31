import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "evaluate_visual_quality.py"
SPEC = importlib.util.spec_from_file_location("evaluate_visual_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeRunner:
    def score_batch(self, reference, generated):
        difference = np.mean(np.abs(reference.astype(float) - generated), axis=(1, 2, 3))
        return {"psnr_db": 100.0 - difference, "maniqa": 1.0 - difference / 255.0}


class VisualQualityMetricTests(unittest.TestCase):
    def test_metric_interval(self):
        row = {"sample": "x", "metric_start_frame": 60, "metric_length": 360}
        self.assertEqual(MODULE.metric_interval(row, 480), (60, 360))
        with self.assertRaises(ValueError):
            MODULE.metric_interval(row, 400)

    def test_frame_step_and_no_reference_metric(self):
        reference = np.zeros((6, 4, 4, 3), dtype=np.uint8)
        generated = reference.copy()
        generated[2] = 10
        row = {"sample": "subject1/a", "subject": "subject1", "mode": "preserve"}
        with patch.object(MODULE, "load_aligned_pair",
                          return_value=(reference, generated, 30.0, 12)):
            record, _, _ = MODULE.evaluate_pair(
                row, FakeRunner(), ("psnr_db", "maniqa"), 2, 2)
        self.assertEqual(record["evaluated_frame_count"], 3)
        self.assertEqual(record["metric_start_frame"], 12)
        self.assertAlmostEqual(record["psnr_db"], (100 + 90 + 100) / 3)
        self.assertAlmostEqual(record["maniqa"], (1 + (1 - 10 / 255) + 1) / 3)
        self.assertIsNone(record["ssim"])

    def test_fvd_clips_include_last_interval(self):
        self.assertEqual(MODULE.fvd_clip_starts(36, 16, 16), [0, 16, 20])
        self.assertEqual(MODULE.fvd_clip_starts(15, 16, 16), [])

    def test_frechet_distance_is_zero_for_identical_features(self):
        features = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]])
        self.assertAlmostEqual(MODULE.frechet_distance(features, features), 0.0,
                               places=8)

    def test_subject_equal_aggregation(self):
        rows = []
        for sample, subject, value in (("a", "s1", 1.0), ("b", "s1", 1.0),
                                       ("c", "s2", 0.0)):
            row = {"sample": sample, "subject": subject, "mode": "preserve"}
            for metric in MODULE.FRAME_METRICS:
                row[metric] = value
            rows.append(row)
        _, groups = MODULE.aggregate_frame_metrics(rows)
        self.assertAlmostEqual(groups[0]["ssim"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
