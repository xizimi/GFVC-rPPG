import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "tools" / "preprocess_ubfc2.py"
SPEC = importlib.util.spec_from_file_location("preprocess_ubfc2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"cv2": SimpleNamespace()}):
    SPEC.loader.exec_module(MODULE)


class PreprocessSplitWindowTests(unittest.TestCase):
    def test_windowing_keeps_only_complete_windows(self):
        self.assertEqual(MODULE.make_windows(359, 360, 360), [])
        self.assertEqual(MODULE.make_windows(720, 360, 360),
                         [(0, 360), (360, 720)])
        self.assertEqual(MODULE.make_windows(1000, 480, 360),
                         [(0, 480), (360, 840)])

    def test_split_map_rejects_identity_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps({"train": ["subject1"],
                                        "val": ["subject1"], "test": []}))
            with self.assertRaises(ValueError):
                MODULE.load_split_map(str(path))

    def test_overwrite_removes_only_known_window_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / "subject1"
            subject.mkdir()
            for name in ("clip_0000.face.npy", "clip_0000.sig.npy",
                         "clip_0000.skin.npy", "meta.json"):
                (subject / name).write_bytes(b"old")
            MODULE.prepare_subject_output(subject, overwrite=True)
            self.assertEqual(list(subject.iterdir()), [])
            (subject / "notes.txt").write_text("keep")
            with self.assertRaises(RuntimeError):
                MODULE.prepare_subject_output(subject, overwrite=True)
            self.assertTrue((subject / "notes.txt").exists())


if __name__ == "__main__":
    unittest.main()
