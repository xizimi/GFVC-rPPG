import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "evaluate_pyvhr_bvp.py"
SPEC = importlib.util.spec_from_file_location("evaluate_pyvhr_bvp", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PyVHRBVPMetricTests(unittest.TestCase):
    def test_identical_and_inverted_waveforms(self):
        target = np.sin(np.linspace(0, 8 * np.pi, 150))
        same, reason = MODULE.waveform_metrics(target, target, 30)
        inverse, reason_inverse = MODULE.waveform_metrics(-target, target, 30)
        self.assertIsNone(reason)
        self.assertIsNone(reason_inverse)
        self.assertAlmostEqual(same["pcc"], 1.0, places=6)
        self.assertAlmostEqual(same["z_mae"], 0.0, places=6)
        self.assertAlmostEqual(inverse["pcc"], -1.0, places=6)
        self.assertAlmostEqual(inverse["abs_pcc"], 1.0, places=6)

    def test_mcc_reports_positive_lag_for_delayed_prediction(self):
        target = np.sin(np.linspace(0, 12 * np.pi, 150))
        delayed = np.concatenate((np.zeros(4), target[:-4]))
        metrics, reason = MODULE.waveform_metrics(delayed, target, 30, 0.2)
        self.assertIsNone(reason)
        self.assertEqual(metrics["mcc_lag_frames"], 4)
        self.assertAlmostEqual(metrics["mcc_lag_seconds"], 4 / 30)

    def test_invalid_signals_return_reasons(self):
        target = np.arange(150, dtype=float)
        metrics, reason = MODULE.waveform_metrics(np.ones(150), target, 30)
        self.assertIsNone(metrics)
        self.assertEqual(reason, "constant_signal")
        metrics, reason = MODULE.waveform_metrics(np.full(150, np.nan), target, 30)
        self.assertIsNone(metrics)
        self.assertEqual(reason, "non_finite_signal")

    def test_150_frame_window_aligns_without_off_by_one(self):
        target = np.sin(np.linspace(0, 8 * np.pi, 150))
        row = {"sample": "synthetic", "mode": "preserve", "fps": 30.0,
               "input_video": "fake.avi", "generated_video": "fake.avi"}
        with tempfile.TemporaryDirectory() as directory:
            signal_path = Path(directory) / "target.npy"
            np.save(signal_path, target)
            row["target_signal"] = str(signal_path)
            args = SimpleNamespace(patch_size=30, min_hz=0.7, max_hz=3.0,
                                   max_lag_seconds=0.2)
            with patch.object(MODULE, "video_info", return_value=(30.0, 150)), \
                 patch.object(MODULE, "run_pyvhr", return_value=([target[None]], np.array([2.5]))):
                records = MODULE.evaluate_video(row, "input", "cpu_POS", args)
        self.assertEqual(records[0]["target_start_frame"], 0)
        self.assertEqual(records[0]["target_length"], 150)
        self.assertEqual(records[0]["status"], "ok")

    def test_missing_rewrite_input_reference_is_explicit(self):
        row = {"sample": "synthetic", "mode": "rewrite", "fps": 30.0,
               "input_video": "fake.avi", "generated_video": "fake.avi",
               "target_signal": "target.npy"}
        args = SimpleNamespace(patch_size=30, min_hz=0.7, max_hz=3.0,
                               max_lag_seconds=0.2)
        records = MODULE.evaluate_video(row, "input", "cpu_POS", args)
        self.assertEqual(records[0]["status"], "missing_original_signal_for_rewrite_input")
        self.assertIsNone(records[0]["median_pcc"])

    def test_all_invalid_patches_are_reported_as_null(self):
        target = np.sin(np.linspace(0, 8 * np.pi, 150))
        row = {"sample": "synthetic", "mode": "preserve", "fps": 30.0,
               "input_video": "fake.avi", "generated_video": "fake.avi"}
        with tempfile.TemporaryDirectory() as directory:
            signal_path = Path(directory) / "target.npy"
            np.save(signal_path, target)
            row["target_signal"] = str(signal_path)
            args = SimpleNamespace(patch_size=30, min_hz=0.7, max_hz=3.0,
                                   max_lag_seconds=0.2)
            with patch.object(MODULE, "video_info", return_value=(30.0, 150)), \
                 patch.object(MODULE, "run_pyvhr", return_value=([np.ones((2, 150))], np.array([2.5]))):
                records = MODULE.evaluate_video(row, "input", "cpu_POS", args)
        self.assertEqual(records[0]["status"], "no_valid_patches")
        self.assertEqual(records[0]["valid_patch_count"], 0)
        self.assertIsNone(records[0]["median_pcc"])


if __name__ == "__main__":
    unittest.main()
