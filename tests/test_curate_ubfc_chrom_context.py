import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "curate_ubfc_chrom_context.py"
SPEC = importlib.util.spec_from_file_location("curate_ubfc_chrom_context", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CurateUBFCChromContextTests(unittest.TestCase):
    def test_final_window_layout(self):
        self.assertEqual(MODULE.window_specs("train"),
                         [(0, 150), (75, 225), (150, 300)])
        self.assertEqual(MODULE.window_specs("val"),
                         [(0, 150), (75, 225), (150, 300)])
        self.assertEqual(MODULE.window_specs("test"), [(0, 360)])

    def test_export_uses_identical_face_mask_and_signal_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "candidates"
            source_dir = root / "train" / "subject1"
            source_dir.mkdir(parents=True)
            face_path = source_dir / "clip_0000.face.npy"
            mask_path = source_dir / "clip_0000.skin.npy"
            face = np.arange(20 * 2 * 2 * 3, dtype=np.uint16).reshape(20, 2, 2, 3)
            mask = np.arange(20 * 2 * 2, dtype=np.uint8).reshape(20, 2, 2)
            np.save(face_path, face)
            np.save(mask_path, mask)
            output_face = Path(directory) / "out" / "train" / "subject1" / "clip_w00.face.npy"
            reference = np.arange(6, dtype=float)
            chrom = reference + 0.5
            MODULE.export_window(
                face_path, mask_path,
                {"rppg_signal": np.arange(20), "hr": np.arange(20) + 60,
                 "time": np.arange(20) / 30, "fps": 30.0},
                chrom, reference, 5, output_face, root, 0.9, 0.75, 0.8)
            np.testing.assert_array_equal(np.load(output_face), face[5:11])
            np.testing.assert_array_equal(
                np.load(str(output_face).replace(".face.npy", ".skin.npy")),
                mask[5:11])
            signal = np.load(
                str(output_face).replace(".face.npy", ".sig.npy"),
                allow_pickle=True).item()
            np.testing.assert_array_equal(signal["rppg_signal"], chrom)
            np.testing.assert_array_equal(signal["reference_rppg_signal"], reference)
            np.testing.assert_array_equal(signal["hr"], np.arange(20)[5:11] + 60)
            self.assertEqual(signal["source_start_frame"], 5)
            self.assertEqual(signal["label_source"],
                             "pyVHR_cpu_CHROM_context480_central360")


if __name__ == "__main__":
    unittest.main()
