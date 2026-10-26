import os
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import sys


class RPPGEncoder(nn.Module):
    """
    Extracts rPPG signal from a sequence of face frames.
    Input:  (B, T, 3, H, W) normalized frames
    Output: (B, T) rPPG signal
    """

    def __init__(self, feat_dim=64, hidden_dim=128):
        super(RPPGEncoder, self).__init__()

        # Per-frame spatial feature extractor
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, feat_dim, kernel_size=7, stride=2, padding=3),   # H/2
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim * 2, kernel_size=5, stride=2, padding=2),  # H/4
            nn.BatchNorm2d(feat_dim * 2),
            nn.ReLU(),
            nn.Conv2d(feat_dim * 2, feat_dim * 4, kernel_size=5, stride=2, padding=2),  # H/8
            nn.BatchNorm2d(feat_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (B, feat_dim*4, 1, 1)
        )

        # Temporal modeling across frames
        self.temporal_gru = nn.GRU(
            input_size=feat_dim * 4,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Predict scalar rPPG per frame
        self.rppg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        # Reshape to process all frames at once
        frames = x.view(B * T, C, H, W)
        features = self.frame_encoder(frames)      # (B*T, D, 1, 1)
        features = features.view(B, T, -1)          # (B, T, D)
        temporal, _ = self.temporal_gru(features)   # (B, T, hidden*2)
        rppg = self.rppg_head(temporal)             # (B, T, 1)
        rppg = rppg.squeeze(-1)                     # (B, T)
        return rppg


class _TemporalResidualBlock(nn.Module):
    """Dilated, length-preserving temporal residual block."""

    def __init__(self, channels, dilation):
        super().__init__()
        padding = dilation * 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding,
                      dilation=dilation, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, value):
        return F.silu(value + self.layers(value))


class SkinRPPGTeacher(nn.Module):
    """Differentiable rPPG teacher trained directly on masked face videos.

    The input is RGB video in ``[B, T, 3, H, W]`` where non-skin pixels have
    already been set to zero by the caller. GroupNorm keeps this model stable
    with the small batches required by 150-frame training windows. It returns
    an unnormalised waveform in ``[B, T]``; losses standardise it as needed.
    """

    # cuDNN only implements GRU input gradients in training mode.  This
    # teacher deliberately uses GroupNorm and no dropout, so enabling training
    # mode changes neither its learned statistics nor its deterministic output;
    # its parameters remain frozen by GFVCRPPGSequenceModel.
    requires_train_mode_for_input_grad = True

    def __init__(self, channels=32, temporal_channels=128, gru_hidden=96):
        super().__init__()
        def spatial_block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, stride=stride,
                          padding=2, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(inplace=True),
            )

        self.spatial = nn.Sequential(
            spatial_block(3, channels, 2),
            spatial_block(channels, channels * 2, 2),
            spatial_block(channels * 2, channels * 4, 2),
            spatial_block(channels * 4, channels * 4, 2),
        )
        feature_channels = channels * 4
        self.attention = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.temporal_input = nn.Sequential(
            nn.Conv1d(feature_channels * 2, temporal_channels, 1, bias=False),
            nn.GroupNorm(8, temporal_channels),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            _TemporalResidualBlock(temporal_channels, 1),
            _TemporalResidualBlock(temporal_channels, 2),
            _TemporalResidualBlock(temporal_channels, 4),
            _TemporalResidualBlock(temporal_channels, 8),
        )
        self.gru = nn.GRU(temporal_channels, gru_hidden, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(gru_hidden, 1),
        )

    def forward(self, video):
        if video.dim() != 5:
            raise ValueError('video must have shape [B, T, 3, H, W]')
        batch, time, channels, height, width = video.shape
        if channels != 3:
            raise ValueError('SkinRPPGTeacher expects RGB input')
        features = self.spatial(video.reshape(batch * time, channels, height, width))
        _, feature_channels, feature_height, feature_width = features.shape
        attention = self.attention(features).reshape(batch * time, 1, -1)
        attention = torch.softmax(attention, dim=-1)
        flattened = features.reshape(batch * time, feature_channels, -1)
        attended = (flattened * attention).sum(dim=-1)
        average = flattened.mean(dim=-1)
        spatial_features = torch.cat([attended, average], dim=1)
        temporal = spatial_features.view(batch, time, -1).transpose(1, 2)
        temporal = self.temporal(self.temporal_input(temporal)).transpose(1, 2)
        temporal, _ = self.gru(temporal)
        return self.head(temporal).squeeze(-1)


class RPPGDecoder(nn.Module):
    """
    Decodes target rPPG signal into a feature map for injection into the generator.
    Input:  (B,) scalar rPPG value for a single frame (or (B, 1))
    Output: (B, C, H, W) feature map
    """

    def __init__(self, out_channels=256, out_size=64, hidden_dim=512):
        super(RPPGDecoder, self).__init__()
        self.out_size = out_size

        self.fc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels * (out_size // 16) * (out_size // 16)),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),  # 4 -> 8
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),  # 8 -> 16
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),  # 16 -> 32
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),  # 32 -> 64
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, rppg_value):
        """
        rppg_value: (B,) or (B, 1)
        """
        if rppg_value.dim() == 1:
            rppg_value = rppg_value.unsqueeze(-1)      # (B, 1)
        B = rppg_value.shape[0]
        x = self.fc(rppg_value)                        # (B, C*4*4)
        s = self.out_size // 16
        x = x.view(B, -1, s, s)                        # (B, C, 4, 4)
        x = self.spatial(x)                            # (B, C, 64, 64)
        return x


class RPPGFusion(nn.Module):
    """
    Fuses rPPG decoded features with generator intermediate features.
    Simple residual addition with learnable scale.
    """

    def __init__(self, channels):
        super(RPPGFusion, self).__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, generator_feat, rppg_feat):
        return generator_feat + self.scale * rppg_feat


class TemporalRPPGEncoder(nn.Module):
    """Encode a complete waveform into one conditioning vector per frame.

    The module is deliberately fully-convolutional in time so training and
    inference may use different window lengths. Input and output are
    ``(batch, time)`` and ``(batch, time, embedding_dim)`` respectively.
    """

    def __init__(self, embedding_dim=64, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.input = nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3)
        blocks = []
        for dilation in [2 ** i for i in range(num_blocks)]:
            padding = dilation * 2
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5,
                          padding=padding, dilation=dilation),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Conv1d(hidden_dim, embedding_dim, kernel_size=1)

    def forward(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError("waveform must have shape (B, T)")
        waveform = waveform.float()
        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        waveform = waveform / (waveform.std(dim=-1, keepdim=True) + 1e-6)
        x = F.silu(self.input(waveform.unsqueeze(1)))
        for block in self.blocks:
            x = x + block(x)
        return self.output(x).transpose(1, 2)


class SkinPhysiologyResidual(nn.Module):
    """Produce a bounded RGB residual conditioned on the target waveform.

    The last layer is zero initialized, so adding this module to a pretrained
    GFVC generator initially leaves its output bit-for-bit unchanged (up to the
    final clamp). Skin localization is learned from the masked training losses;
    no mask is required in the receiver's transmitted input.
    """

    def __init__(self, feature_channels, condition_dim=64, hidden_channels=64,
                 max_delta=4.0 / 255.0):
        super().__init__()
        self.max_delta = float(max_delta)
        self.affine = nn.Linear(condition_dim, feature_channels * 2)
        self.residual = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, feature, condition):
        if condition.dim() != 2:
            raise ValueError("condition must have shape (B, D)")
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        conditioned = feature * (1.0 + gamma) + beta
        delta = self.max_delta * torch.tanh(self.residual(conditioned))

        return delta


def negative_pearson_loss(pred, target, eps=1e-8):
    """One minus Pearson correlation, averaged over the batch."""
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    numerator = (pred * target).sum(dim=-1)
    denominator = torch.sqrt((pred.square().sum(dim=-1) + eps) *
                             (target.square().sum(dim=-1) + eps))
    return (1.0 - numerator / denominator).mean()


def normalized_power_spectrum(signal, fps, low_hz=0.7, high_hz=3.0,
                              eps=1e-8):
    signal = signal - signal.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(signal, dim=-1)
    power = spectrum.real.square() + spectrum.imag.square()
    frequencies = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / float(fps)).to(signal.device)
    valid = (frequencies >= low_hz) & (frequencies <= high_hz)
    power = power[..., valid]
    return power / (power.sum(dim=-1, keepdim=True) + eps), frequencies[valid]


def physiological_losses(pred, target, fps, target_bpm=None, eps=1e-8):
    """Stable time/frequency losses for complete-waveform supervision."""
    if pred.shape != target.shape:
        raise ValueError("predicted and target waveforms must have equal shape")
    pred_norm = (pred - pred.mean(-1, keepdim=True)) / (pred.std(-1, keepdim=True) + eps)
    target_norm = (target - target.mean(-1, keepdim=True)) / (target.std(-1, keepdim=True) + eps)
    fps_values = torch.as_tensor(fps).flatten().tolist()
    if len(fps_values) == 1:
        fps_values *= pred.shape[0]
    if len(fps_values) != pred.shape[0]:
        raise ValueError('fps must be scalar or contain one value per sample')
    if target_bpm is not None:
        target_bpm = torch.as_tensor(target_bpm, device=pred.device,
                                     dtype=pred.dtype).flatten()
        if target_bpm.numel() != pred.shape[0]:
            raise ValueError('target_bpm must contain one value per sample')

    spectral_terms, kl_terms, bpm_terms = [], [], []
    for index, sample_fps in enumerate(fps_values):
        pred_psd, frequencies = normalized_power_spectrum(
            pred[index:index + 1], sample_fps, eps=eps
        )
        target_psd, _ = normalized_power_spectrum(
            target[index:index + 1], sample_fps, eps=eps
        )
        spectral_terms.append(F.l1_loss(pred_psd, target_psd))
        # A small probability floor avoids excessive KL from nearly empty FFT bins.
        pred_kl = pred_psd.clamp_min(1e-4)
        target_kl = target_psd.clamp_min(1e-4)
        pred_kl = pred_kl / pred_kl.sum(dim=-1, keepdim=True)
        target_kl = target_kl / target_kl.sum(dim=-1, keepdim=True)
        kl_pt = (pred_kl * (pred_kl.log() - target_kl.log())).sum(-1)
        kl_tp = (target_kl * (target_kl.log() - pred_kl.log())).sum(-1)
        kl_terms.append(0.5 * (kl_pt + kl_tp).mean())
        # Soft frequency selection is differentiable; the target is measured HR.
        pred_hr = 60.0 * (
            torch.softmax(pred_psd / 0.05, dim=-1) * frequencies
        ).sum(-1)
        if target_bpm is None or not torch.isfinite(target_bpm[index]):
            target_hr = 60.0 * (
                torch.softmax(target_psd / 0.05, dim=-1) * frequencies
            ).sum(-1)
        else:
            target_hr = target_bpm[index:index + 1]
        bpm_terms.append(F.l1_loss(pred_hr, target_hr))
    return {
        "pearson": negative_pearson_loss(pred, target, eps),
        "waveform": F.smooth_l1_loss(pred_norm, target_norm),
        "spectral": torch.stack(spectral_terms).mean(),
        "kl": torch.stack(kl_terms).mean(),
        "bpm": torch.stack(bpm_terms).mean(),
    }


class RePPGTeacherAdapter(nn.Module):
    """Frozen adapter around the pretrained rePPG Step-2 Hourglass teacher.

    The teacher is used only as a differentiable measurement network: gradients
    flow through its input back to GFVC, but its parameters are never updated.
    Input is RGB ``[B,T,3,H,W]`` and output is ``[B,T]``. The reference
    dataloader passes its OpenCV-stored BGR arrays to the teacher unchanged.
    """
    def __init__(self, project_root, checkpoint, base_filter=16, gray=True,
                 checkpoint_frames=False):
        super().__init__()
        model_path = os.path.join(project_root, 'model.py')
        spec = importlib.util.spec_from_file_location('reppg_reference_model', model_path)
        if spec is None or spec.loader is None:
            raise ImportError('Cannot import rePPG reference model from ' + model_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.teacher = module.HourglassNetStep2(baseFilter=base_filter, gray=gray)
        state = torch.load(checkpoint, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.teacher.load_state_dict(state, strict=True)
        self.checkpoint_frames = bool(checkpoint_frames)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode=True):
        # Keep BatchNorm/dropout deterministic even when the parent enters train().
        super().train(False)
        self.teacher.eval()
        return self

    def _frame_features(self, frame):
        frame = self.teacher.pre_conv(frame)
        frame = F.relu(self.teacher.pre_bn(frame))
        _, _, rppg = self.teacher.HG3(frame, None, None, 0, 0)
        return rppg[:, :, 0, 0]

    def forward(self, video):
        if video.dim() != 5:
            raise ValueError('teacher input must be [B,T,3,H,W]')
        video = video[:, :, [2, 1, 0]]  # GFVC RGB -> reference teacher BGR
        # The reference get_rppg_only writes rppg[0] into every batch row and
        # divides by a zero range for constant predictions. Keep its learned
        # layers while fixing both issues here, without changing the checkpoint.
        features = []
        for frame_index in range(video.shape[1]):
            frame = video[:, frame_index]
            if getattr(self, 'checkpoint_frames', False) and torch.is_grad_enabled():
                features.append(checkpoint(self._frame_features, frame,
                                           use_reentrant=False))
            else:
                features.append(self._frame_features(frame))
        values = self.teacher.rppg_ffn(torch.stack(features, dim=1)).squeeze(-1)
        minimum = values.amin(dim=-1, keepdim=True)
        amplitude = values.amax(dim=-1, keepdim=True) - minimum
        return (values - minimum) / amplitude.clamp_min(1e-6)
