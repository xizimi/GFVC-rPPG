import os
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from modules.rppg import RePPGTeacherAdapter, SkinRPPGTeacher, physiological_losses
from rppg_gfvc_dataset import GFVCRPPGWindowDataset
from train_gfvc_rppg import teacher_preflight


class _FakeHourglass(nn.Module):
    def forward(self, frame, *_):
        value = frame[:, :1].mean(dim=(2, 3), keepdim=True)
        return frame, frame, value.repeat(1, 10, 1, 1)


class _FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_conv = nn.Identity()
        self.pre_bn = nn.Identity()
        self.HG3 = _FakeHourglass()
        self.rppg_ffn = nn.Linear(10, 1, bias=False)
        nn.init.constant_(self.rppg_ffn.weight, 0.1)


class RPPGPipelineTests(unittest.TestCase):
    def test_skin_teacher_has_full_waveform_gradient(self):
        teacher = SkinRPPGTeacher(channels=8, temporal_channels=32, gru_hidden=16)
        video = torch.rand(1, 9, 3, 32, 32, requires_grad=True)
        waveform = teacher(video)
        self.assertEqual(tuple(waveform.shape), (1, 9))
        waveform.square().mean().backward()
        self.assertTrue(torch.isfinite(video.grad).all())

    def test_teacher_preflight_averages_all_windows_per_subject(self):
        class Dataset:
            samples = [
                ('subject1_clip0.face.npy', None, None, None),
                ('subject1_clip1.face.npy', None, None, None),
                ('subject2_clip0.face.npy', None, None, None),
            ]

            def __getitem__(self, index):
                video = torch.zeros(3, 3, 2, 2)
                video[:, 0] = torch.tensor([0., 1., 2.])[:, None, None]
                target = torch.tensor([0., 1., 2.])
                if index == 1:
                    target = -target
                return {'driving_video': video,
                        'skin_masks': torch.ones(3, 1, 2, 2),
                        'original_rppg': target}

        class Teacher(nn.Module):
            def forward(self, video):
                return video[:, :, 0].mean(dim=(2, 3))

        score = teacher_preflight(Teacher(), Dataset(), torch.device('cpu'), 2)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_teacher_keeps_batch_rows_separate(self):
        adapter = RePPGTeacherAdapter.__new__(RePPGTeacherAdapter)
        nn.Module.__init__(adapter)
        adapter.teacher = _FakeTeacher()
        video = torch.zeros(2, 5, 3, 4, 4)
        video[0, :, 2] = torch.tensor([0., 1., 2., 3., 4.])[:, None, None]
        video[1, :, 2] = torch.tensor([4., 3., 2., 1., 0.])[:, None, None]
        output = adapter(video)
        self.assertTrue(torch.allclose(output[0], torch.linspace(0, 1, 5)))
        self.assertTrue(torch.allclose(output[1], torch.linspace(1, 0, 5)))

    def test_losses_support_mixed_fps_and_missing_hr(self):
        pred = torch.randn(2, 150, requires_grad=True)
        target = torch.randn(2, 150)
        losses = physiological_losses(pred, target, torch.tensor([23.0, 30.0]),
                                     target_bpm=torch.tensor([72.0, float('nan')]))
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_rewrite_waveform_uses_driving_timebase(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, fps, length in [('a', 20.0, 60), ('b', 30.0, 100)]:
                base = os.path.join(directory, name)
                np.save(base + '.face.npy', np.zeros((length, 8, 8, 3), dtype=np.uint8))
                np.save(base + '.skin.npy', np.ones((length, 8, 8), dtype=np.float32))
                times = np.arange(length) / fps
                np.save(base + '.sig.npy', {
                    'rppg_signal': np.sin(2 * np.pi * times),
                    'hr': np.full(length, 60.0),
                    'time': times,
                    'fps': fps,
                })
            dataset = GFVCRPPGWindowDataset(
                directory, sequence_length=60, rewrite_probability=1.0,
                signal_preprocess='none', deterministic=True,
            )
            sample = dataset[0]
            waveform = sample['target_rppg'].numpy()
            frequencies = np.fft.rfftfreq(len(waveform), d=1 / 20.0)
            peak = frequencies[np.abs(np.fft.rfft(waveform)).argmax()]
            self.assertAlmostEqual(peak, 1.0, places=5)
            self.assertEqual(int(sample['mode']), 1)


if __name__ == '__main__':
    unittest.main()
