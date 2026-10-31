import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "validate_ubfc_chrom_dataset.py"
SPEC = importlib.util.spec_from_file_location("validate_ubfc_chrom_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ValidateUBFCChromDatasetTests(unittest.TestCase):
    def make_sample(self, root, split, frames, pcc):
        directory = root / split / ("subject_" + split)
        directory.mkdir(parents=True, exist_ok=True)
        base = directory / "clip_0000"
        reference = np.sin(np.linspace(0, 4 * np.pi, frames, endpoint=False))
        raw_noise = np.cos(np.linspace(0, 7 * np.pi, frames, endpoint=False))
        reference = (reference - reference.mean()) / reference.std()
        raw_noise -= raw_noise.mean()
        raw_noise -= np.mean(raw_noise * reference) * reference
        noise = raw_noise / raw_noise.std()
        waveform = (pcc * reference + np.sqrt(1 - pcc ** 2) * noise).astype(np.float32)
        np.save(str(base) + ".face.npy",
                np.zeros((frames, 256, 256, 3), dtype=np.uint8), allow_pickle=False)
        np.save(str(base) + ".skin.npy",
                np.ones((frames, 256, 256), dtype=np.float32), allow_pickle=False)
        np.save(str(base) + ".sig.npy", {
            "rppg_signal": waveform, "fps": 29.5,
            "reference_rppg_signal": reference.astype(np.float32),
            "label_source": "pyVHR_cpu_CHROM_context480_central360",
            "label_reference_pearson": pcc,
            "label_selection_threshold": 0.75,
            "label_selection_rule": "signed_zero_lag_pearson_gt",
        }, allow_pickle=True)

    def test_accepts_150_train_and_360_test_with_true_fps_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sample(root, "train", 10, 0.8)
            self.make_sample(root, "val", 10, 0.85)
            self.make_sample(root, "test", 12, 0.9)
            expected = {"train": 10, "val": 10, "test": 12}
            report = MODULE.validate(root, expected=expected)
            self.assertTrue(report["summary"]["valid"], report["errors"])
            self.assertAlmostEqual(
                report["summary"]["mean_duration_seconds"]["test"],
                12 / 29.5)

    def test_rejects_equal_threshold_and_wrong_test_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sample(root, "train", 10, 0.8)
            self.make_sample(root, "val", 10, 0.85)
            self.make_sample(root, "test", 10, 0.75)
            report = MODULE.validate(
                root, expected={"train": 10, "val": 10, "test": 12})
            self.assertFalse(report["summary"]["valid"])
            joined = "\n".join(report["errors"])
            self.assertIn("pcc_not_strictly_above_threshold", joined)
            self.assertIn("expected_12_frames_got_10", joined)


if __name__ == "__main__":
    unittest.main()
