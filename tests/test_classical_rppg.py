import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from evaluate_classical_rppg import (  # noqa: E402
    pyvhr_chrom, pyvhr_green, pyvhr_pos,
)


class TestClassicalRPPG(unittest.TestCase):
    def test_pyvhr_cpu_methods_have_expected_length(self):
        fps = 30.0
        time = np.arange(150, dtype=np.float64) / fps
        pulse = np.sin(2.0 * np.pi * 1.2 * time)  # 72 BPM
        rgb = np.stack([100.0 + 0.8 * pulse, 120.0 + pulse, 90.0 + 0.5 * pulse])
        for method in (pyvhr_pos, pyvhr_chrom, pyvhr_green):
            values = method(rgb, fps) if method is pyvhr_pos else method(rgb)
            self.assertEqual(values.shape, (150,))
            self.assertTrue(np.isfinite(values).all())


if __name__ == '__main__':
    unittest.main()
