import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "curate_ubfc_chrom.py"
SPEC = importlib.util.spec_from_file_location("curate_ubfc_chrom", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CurateUBFCChromTests(unittest.TestCase):
    def test_signed_zero_lag_pearson(self):
        target = np.sin(np.linspace(0, 8 * np.pi, 150))
        self.assertAlmostEqual(MODULE.pearson(target, target), 1.0, places=6)
        self.assertAlmostEqual(MODULE.pearson(-target, target), -1.0, places=6)

    def test_patch_median_does_not_use_reference_or_flip_polarity(self):
        target = np.sin(np.linspace(0, 8 * np.pi, 150))
        window = np.stack((target, target * 2, np.ones_like(target)))
        chrom, valid, total = MODULE.median_patch_waveform(window, 150)
        self.assertEqual((valid, total), (2, 3))
        self.assertAlmostEqual(MODULE.pearson(chrom, target), 1.0, places=6)

    def test_acceptance_and_export_preserve_mask_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            subject = root / "subject1"
            subject.mkdir(parents=True)
            face_path = subject / "clip_0000.face.npy"
            sig_path = subject / "clip_0000.sig.npy"
            mask_path = subject / "clip_0000.skin.npy"
            frames = np.zeros((8, 256, 256, 3), dtype=np.uint8)
            target = np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            np.save(face_path, frames, allow_pickle=False)
            np.save(mask_path, np.ones((8, 256, 256), dtype=np.uint8),
                    allow_pickle=False)
            np.save(sig_path, {"rppg_signal": target, "hr": np.arange(8),
                               "time": np.arange(8) / 30, "fps": 30.0},
                    allow_pickle=True)
            original_mask = mask_path.read_bytes()
            args = SimpleNamespace(threshold=0.7, default_fps=30.0,
                                   patch_size=30, min_hz=0.7, max_hz=3.0,
                                   cuda=False, output_root=output,
                                   copy_files=True, example_plot=None,
                                   example_written=False)
            with patch.object(MODULE, "extract_chrom",
                              return_value=(target.astype(np.float32), 4, 5)):
                row = MODULE.evaluate_one(face_path, root, args)
            self.assertTrue(row["accepted"], row)
            self.assertEqual(row["status"], "accepted")
            exported_mask = output / "subject1" / "clip_0000.skin.npy"
            self.assertEqual(exported_mask.read_bytes(), original_mask)
            exported = np.load(output / "subject1" / "clip_0000.sig.npy",
                               allow_pickle=True).item()
            np.testing.assert_allclose(exported["rppg_signal"], target)
            np.testing.assert_array_equal(exported["hr"], np.arange(8))
            self.assertEqual(exported["label_source"],
                             "pyVHR_cpu_CHROM_patch_median")
            self.assertAlmostEqual(exported["label_reference_pearson"], 1.0)
            self.assertEqual(exported["label_selection_threshold"], 0.7)
            self.assertEqual(exported["label_selection_rule"],
                             "signed_zero_lag_pearson_gt")
            np.testing.assert_allclose(exported["reference_rppg_signal"], target)

    def test_low_or_inverse_correlation_is_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            subject = root / "subject1"
            subject.mkdir(parents=True)
            face_path = subject / "clip_0000.face.npy"
            target = np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            np.save(face_path, np.zeros((8, 256, 256, 3), dtype=np.uint8))
            np.save(subject / "clip_0000.skin.npy",
                    np.ones((8, 256, 256), dtype=np.uint8))
            np.save(subject / "clip_0000.sig.npy",
                    {"rppg_signal": target, "fps": 30.0}, allow_pickle=True)
            args = SimpleNamespace(threshold=0.7, default_fps=30.0,
                                   patch_size=30, min_hz=0.7, max_hz=3.0,
                                   cuda=False, output_root=Path(directory) / "out",
                                   copy_files=True, example_plot=None,
                                   example_written=False)
            with patch.object(MODULE, "extract_chrom",
                              return_value=(-target.astype(np.float32), 2, 2)):
                row = MODULE.evaluate_one(face_path, root, args)
            self.assertFalse(row["accepted"])
            self.assertEqual(row["status"], "below_threshold")
            self.assertFalse(args.output_root.exists())

    def test_export_failure_is_not_counted_as_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            subject = root / "subject1"
            subject.mkdir(parents=True)
            face_path = subject / "clip_0000.face.npy"
            target = np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            np.save(face_path, np.zeros((8, 256, 256, 3), dtype=np.uint8))
            np.save(subject / "clip_0000.skin.npy",
                    np.ones((8, 256, 256), dtype=np.uint8))
            np.save(subject / "clip_0000.sig.npy",
                    {"rppg_signal": target, "fps": 30.0}, allow_pickle=True)
            args = SimpleNamespace(threshold=0.7, default_fps=30.0,
                                   patch_size=30, min_hz=0.7, max_hz=3.0,
                                   cuda=False, output_root=Path(directory) / "out",
                                   copy_files=True, example_plot=None,
                                   example_written=False)
            with patch.object(MODULE, "extract_chrom",
                              return_value=(target.astype(np.float32), 2, 2)), \
                 patch.object(MODULE, "export_sample", side_effect=OSError("disk full")):
                row = MODULE.evaluate_one(face_path, root, args)
            self.assertTrue(row["selection_passed"])
            self.assertFalse(row["accepted"])
            self.assertEqual(row["status"], "export_error")
            self.assertEqual(row["export_status"], "error", row)

    def test_default_threshold_is_strictly_greater_than_point_75(self):
        args = MODULE.build_parser().parse_args([
            "--input-root", "input", "--report", "report.json"])
        self.assertEqual(args.threshold, 0.75)

    def test_partial_export_is_rolled_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            subject = root / "subject1"
            subject.mkdir(parents=True)
            face = subject / "clip_0000.face.npy"
            mask = subject / "clip_0000.skin.npy"
            np.save(face, np.zeros((8, 2, 2, 3), dtype=np.uint8))
            np.save(mask, np.ones((8, 2, 2), dtype=np.uint8))
            original = MODULE.link_or_copy
            calls = [0]

            def fail_second(source, destination, copy_files):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("mask write failed")
                return original(source, destination, copy_files)

            with patch.object(MODULE, "link_or_copy", side_effect=fail_second):
                with self.assertRaises(OSError):
                    MODULE.export_sample(
                        face, subject / "clip_0000.sig.npy",
                        {"rppg_signal": np.arange(8), "fps": 30.0},
                        np.arange(8), 1.0, 0.75, root, output, True)
            self.assertFalse(any(output.rglob("*.npy")))


if __name__ == "__main__":
    unittest.main()
