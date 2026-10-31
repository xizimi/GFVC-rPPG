import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "evaluate_pyvhr_bvp.py"
SPEC = importlib.util.spec_from_file_location("evaluate_pyvhr_bvp", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PyVHRBVPMetricTests(unittest.TestCase):
    def test_identical_and_inverted_waveforms(self):
        target = np.sin(np.linspace(0, 8 * np.pi, 150))
        same, reason = MODULE.waveform_metrics(target, target)
        inverse, reason_inverse = MODULE.waveform_metrics(-target, target)
        self.assertIsNone(reason)
        self.assertIsNone(reason_inverse)
        self.assertAlmostEqual(same["pcc"], 1.0, places=6)
        self.assertAlmostEqual(same["mae_z"], 0.0, places=6)
        self.assertAlmostEqual(same["rmse_z"], 0.0, places=6)
        self.assertAlmostEqual(same["dtw_normalized"], 0.0, places=6)
        self.assertAlmostEqual(inverse["pcc"], -1.0, places=6)
        self.assertGreater(inverse["mae_z"], 0.0)
        self.assertGreater(inverse["dtw_normalized"], 0.0)

    def test_dtw_reports_distance_for_delayed_prediction(self):
        target = np.sin(np.linspace(0, 12 * np.pi, 150))
        delayed = np.concatenate((np.zeros(4), target[:-4]))
        metrics, reason = MODULE.waveform_metrics(delayed, target)
        self.assertIsNone(reason)
        self.assertGreater(metrics["dtw_distance"], 0.0)
        self.assertGreater(metrics["dtw_path_length"], 150)
        self.assertLess(metrics["pcc"], 1.0)

    def test_dtw_uses_square_sum_then_square_root(self):
        distance, path_length = MODULE.dtw_distance([0.0, 0.0], [3.0, 4.0])
        self.assertEqual(path_length, 2)
        self.assertAlmostEqual(distance, 5.0)
        self.assertAlmostEqual(distance / np.sqrt(path_length), np.sqrt(12.5))

    def test_patch_median_is_reference_independent(self):
        target = np.sin(np.linspace(0, 8 * np.pi, 150))
        window = np.stack([target, 2.0 * target + 5.0, np.ones_like(target)])
        aggregated, valid, total, failures = MODULE.median_patch_bvp(window, 150)
        self.assertEqual((valid, total), (2, 3))
        self.assertEqual(failures["constant_signal"], 1)
        self.assertAlmostEqual(float(np.corrcoef(aggregated, target)[0, 1]), 1.0,
                               places=6)

    def test_official_pyvhr_error_mapping(self):
        errors = ModuleType("pyVHR.utils.errors")
        errors.getErrors = lambda *args: tuple(np.array([value]) for value in
                                                (1, 2, 3, 0.4, 0.5, 6))
        errors.printErrors = lambda *args: None
        errors.displayErrors = lambda *args: None
        pyvhr = ModuleType("pyVHR")
        utils = ModuleType("pyVHR.utils")
        with patch.dict(sys.modules, {"pyVHR": pyvhr, "pyVHR.utils": utils,
                                      "pyVHR.utils.errors": errors}):
            result = MODULE.official_pyvhr_errors(
                [np.ones((1, 10)), np.ones((1, 10))], 30,
                [70, 72], [71, 73], [2, 3], [2, 3])
        self.assertEqual(result["hr_mae_bpm"], 2.0)
        self.assertEqual(result["snr_db"], 6.0)
        self.assertEqual(result["hr_pcc"], 0.4)

    def test_invalid_signals_return_reasons(self):
        target = np.arange(150, dtype=float)
        metrics, reason = MODULE.waveform_metrics(np.ones(150), target)
        self.assertIsNone(metrics)
        self.assertEqual(reason, "constant_signal")
        metrics, reason = MODULE.waveform_metrics(np.full(150, np.nan), target)
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
                                   bpm_window_seconds=2.0, display_errors=False)
            with patch.object(MODULE, "video_info", return_value=(30.0, 150)), \
                 patch.object(MODULE, "run_pyvhr", side_effect=[
                     ([target[None]], np.array([2.5]), np.array([72.0])),
                     ([target[None], target[None]], np.array([1.5, 3.5]),
                      np.array([72.0, 72.0]))]), \
                 patch.object(MODULE, "official_pyvhr_errors", return_value={
                     "hr_rmse_bpm": 0.0, "hr_mae_bpm": 0.0,
                     "hr_max_error_bpm": 0.0, "hr_pcc": 1.0, "hr_ccc": 1.0,
                     "snr_db": 10.0, "predicted_hr_bpm": 72.0,
                     "target_hr_bpm": 72.0}):
                records = MODULE.evaluate_video(row, "input", "cpu_POS", args)
        self.assertEqual(records[0]["target_start_frame"], 0)
        self.assertEqual(records[0]["target_length"], 150)
        self.assertEqual(records[0]["status"], "ok")

    def test_missing_rewrite_input_reference_is_explicit(self):
        row = {"sample": "synthetic", "mode": "rewrite", "fps": 30.0,
               "input_video": "fake.avi", "generated_video": "fake.avi",
               "target_signal": "target.npy"}
        args = SimpleNamespace(patch_size=30, min_hz=0.7, max_hz=3.0,
                               bpm_window_seconds=2.0, display_errors=False)
        records = MODULE.evaluate_video(row, "input", "cpu_POS", args)
        self.assertEqual(records[0]["status"], "missing_original_signal_for_rewrite_input")
        self.assertIsNone(records[0]["pcc"])

    def test_all_invalid_patches_are_reported_as_null(self):
        target = np.sin(np.linspace(0, 8 * np.pi, 150))
        row = {"sample": "synthetic", "mode": "preserve", "fps": 30.0,
               "input_video": "fake.avi", "generated_video": "fake.avi"}
        with tempfile.TemporaryDirectory() as directory:
            signal_path = Path(directory) / "target.npy"
            np.save(signal_path, target)
            row["target_signal"] = str(signal_path)
            args = SimpleNamespace(patch_size=30, min_hz=0.7, max_hz=3.0,
                                   bpm_window_seconds=2.0, display_errors=False)
            with patch.object(MODULE, "video_info", return_value=(30.0, 150)), \
                 patch.object(MODULE, "run_pyvhr", side_effect=[
                     ([np.ones((2, 150))], np.array([2.5]), np.array([72.0])),
                     ([target[None], target[None]], np.array([1.5, 3.5]),
                      np.array([72.0, 72.0]))]), \
                 patch.object(MODULE, "official_pyvhr_errors", return_value={
                     "hr_rmse_bpm": 0.0, "hr_mae_bpm": 0.0,
                     "hr_max_error_bpm": 0.0, "hr_pcc": 1.0, "hr_ccc": 1.0,
                     "snr_db": 10.0, "predicted_hr_bpm": 72.0,
                     "target_hr_bpm": 72.0}):
                records = MODULE.evaluate_video(row, "input", "cpu_POS", args)
        self.assertEqual(records[0]["status"], "no_valid_patches")
        self.assertEqual(records[0]["valid_patch_count"], 0)
        self.assertIsNone(records[0]["pcc"])

    def test_summary_weights_subjects_equally(self):
        records = []
        for sample, subject, pcc in (("a0", "subject1", 1.0),
                                     ("a1", "subject1", 1.0),
                                     ("b0", "subject2", 0.0)):
            row = {"sample": sample, "subject": subject, "mode": "preserve",
                   "video_kind": "generated", "method": "cpu_CHROM",
                   "status": "ok"}
            for metric in MODULE.METRICS + MODULE.DIAGNOSTICS:
                row[metric] = pcc
            records.append(row)
        result = MODULE.summary(records)
        group = result["groups"][0]
        self.assertEqual(group["subject_count"], 2)
        self.assertEqual(group["sample_count"], 3)
        self.assertAlmostEqual(group["pcc"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
