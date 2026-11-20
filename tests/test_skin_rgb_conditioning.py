import unittest
import torch
from modules.rppg import TemporalRPPGEncoder
from modules.physiology_motion import MotionAlignedPhysioModulator, transfer_bvp_to_skin_rgb
from modules.skin_rgb import skin_rgb_means, normalize_skin_rgb, skin_rgb_statistics, center_skin_rgb


class SkinRGBTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)

    def test_mask_area_and_empty_mask(self):
        video = torch.zeros(1, 4, 3, 2, 2)
        video[:, :, :, 0, 0] = torch.tensor([.2, .4, .6])
        mask = torch.zeros(1, 4, 1, 2, 2)
        mask[:, :, :, 0, 0] = 1
        expected = torch.tensor([.2, .4, .6]).expand(1, 4, 3)
        torch.testing.assert_close(skin_rgb_means(video, mask), expected)
        with self.assertRaises(ValueError):
            skin_rgb_means(video, torch.zeros_like(mask))
        self.assertTrue(torch.isfinite(normalize_skin_rgb(expected)).all())

    def test_rgb_channels_gradient_and_transfer(self):
        class Model(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.physiology_input_mode = 'skin_rgb' if channels == 3 else 'bvp'
                self.physiology_encoder = TemporalRPPGEncoder(8, 8, 1, input_channels=channels)
                self.physiology_modulator = MotionAlignedPhysioModulator(
                    4, 3, 8, 8, 2, 8, value_channels=channels)

        old, model = Model(1), Model(3)
        report = transfer_bvp_to_skin_rgb(model, old.state_dict())
        self.assertEqual(len(report['initialized']), 5)
        for name in report['copied']:
            self.assertTrue(torch.equal(model.state_dict()[name], old.state_dict()[name]))
        rgb = torch.rand(2, 9, 3, requires_grad=True)
        encoded = model.physiology_encoder(rgb)
        self.assertEqual(encoded.shape, (2, 9, 8))
        mod = model.physiology_modulator
        features = torch.randn(2, 8, 5, 5)
        basis = mod.make_basis(torch.randn(2, 4, 3, 5, 5))
        output = mod(features, basis, encoded[:, 3], center_skin_rgb(rgb)[:, 3],
                     torch.ones(2, 1, 5, 5), aligned=False)
        output['feature_delta'].square().mean().backward()
        self.assertTrue(torch.isfinite(rgb.grad).all())
        self.assertTrue((rgb.grad.abs().sum(dim=(0, 1)) > 0).all())
        self.assertGreater(float(mod.value_amplitude.weight.grad.abs().sum()), 0)
        with self.assertRaises(RuntimeError):
            transfer_bvp_to_skin_rgb(model, model.state_dict())

    def test_baseline_invariance_and_amplitude_response(self):
        encoder = TemporalRPPGEncoder(8, 8, 1, input_channels=3)
        time = torch.arange(120) / 30
        pulse = torch.sin(2 * torch.pi * 1.2 * time)[None, :, None]
        baseline = torch.tensor([.6, .4, .3])[None, None]
        amplitude = torch.tensor([.01, .03, .015])[None, None]
        rgb = baseline + pulse * amplitude
        scaled = baseline + pulse * amplitude * 2
        shifted = rgb + torch.tensor([.1, -.1, .2])[None, None]
        for other in (scaled, shifted):
            torch.testing.assert_close(normalize_skin_rgb(rgb), normalize_skin_rgb(other),
                                       atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(encoder(rgb), encoder(shifted), atol=2e-5, rtol=2e-5)
        self.assertFalse(torch.allclose(encoder(rgb), encoder(scaled), atol=1e-6, rtol=1e-6))
        mean, deviation = skin_rgb_statistics(rgb)
        torch.testing.assert_close(normalize_skin_rgb(rgb) * deviation + mean, rgb)
        # Isolate the statistics path from any floating-point differences in z-score.
        statistics = deviation.detach().requires_grad_()
        encoder.rgb_statistics(statistics).square().mean().backward()
        self.assertTrue((statistics.grad.abs() > 0).all())
        mod = MotionAlignedPhysioModulator(4, 3, 8, 8, 2, 8, value_channels=3)
        self.assertIsNone(mod.value_amplitude.bias)
        features = torch.randn(1, 8, 5, 5)
        basis = mod.make_basis(torch.randn(1, 4, 3, 5, 5))
        visibility = torch.ones(1, 1, 5, 5)

        def modulation(values):
            return mod(features, basis, encoder(values)[:, 3], center_skin_rgb(values)[:, 3],
                       visibility, aligned=False)

        original = modulation(rgb)
        shifted_result = modulation(shifted)
        torch.testing.assert_close(original['coefficients'], shifted_result['coefficients'],
                                   atol=1e-7, rtol=1e-3)
        torch.testing.assert_close(original['feature_delta'], shifted_result['feature_delta'],
                                   atol=1e-9, rtol=1e-3)
        self.assertGreater(float((original['coefficients'] - modulation(scaled)['coefficients']).abs().max()), 1e-7)
        constant = baseline.expand(1, 120, 3)
        zero = modulation(constant)['feature_delta']
        torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-9, rtol=0)


if __name__ == '__main__':
    unittest.main()
