import os
import torch
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
    final clamp). The caller supplies a soft skin mask; non-skin pixels cannot
    be modified by this branch.
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

    def forward(self, feature, condition, skin_mask):
        if condition.dim() != 2:
            raise ValueError("condition must have shape (B, D)")
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        conditioned = feature * (1.0 + gamma) + beta
        delta = self.max_delta * torch.tanh(self.residual(conditioned))

        if skin_mask is None:
            raise ValueError("skin_mask is required when target_rppg is used")
        if skin_mask.dim() == 3:
            skin_mask = skin_mask.unsqueeze(1)
        if skin_mask.shape[1] != 1:
            skin_mask = skin_mask.mean(dim=1, keepdim=True)
        if skin_mask.shape[-2:] != delta.shape[-2:]:
            skin_mask = F.interpolate(skin_mask.float(), size=delta.shape[-2:],
                                      mode="bilinear", align_corners=False)
        return delta * skin_mask.clamp(0.0, 1.0)


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


def physiological_losses(pred, target, fps, eps=1e-8):
    """Stable time/frequency losses for complete-waveform supervision."""
    if pred.shape != target.shape:
        raise ValueError("predicted and target waveforms must have equal shape")
    pred_norm = (pred - pred.mean(-1, keepdim=True)) / (pred.std(-1, keepdim=True) + eps)
    target_norm = (target - target.mean(-1, keepdim=True)) / (target.std(-1, keepdim=True) + eps)
    pred_psd, frequencies = normalized_power_spectrum(pred, fps, eps=eps)
    target_psd, _ = normalized_power_spectrum(target, fps, eps=eps)
    spectral = F.l1_loss(pred_psd, target_psd)
    kl_pt = (pred_psd * ((pred_psd + eps).log() - (target_psd + eps).log())).sum(-1)
    kl_tp = (target_psd * ((target_psd + eps).log() - (pred_psd + eps).log())).sum(-1)
    symmetric_kl = 0.5 * (kl_pt + kl_tp).mean()
    # Differentiable heart-rate estimate used only for training.
    temperature = 0.05
    pred_bpm = 60.0 * (torch.softmax(pred_psd / temperature, dim=-1) * frequencies).sum(-1)
    target_bpm = 60.0 * (torch.softmax(target_psd / temperature, dim=-1) * frequencies).sum(-1)
    return {
        "pearson": negative_pearson_loss(pred, target, eps),
        "waveform": F.smooth_l1_loss(pred_norm, target_norm),
        "spectral": spectral,
        "kl": symmetric_kl,
        "bpm": F.l1_loss(pred_bpm, target_bpm),
    }


class RePPGTeacherAdapter(nn.Module):
    """Frozen adapter around the pretrained rePPG Step-2 Hourglass teacher.

    The teacher is used only as a differentiable measurement network: gradients
    flow through its input back to GFVC, but its parameters are never updated.
    Input is RGB ``[B,T,3,H,W]`` and output is ``[B,T]``.
    """
    def __init__(self, project_root, checkpoint, base_filter=16, gray=True):
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
        self.teacher.load_state_dict(state, strict=False)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode=True):
        # Keep BatchNorm/dropout deterministic even when the parent enters train().
        super().train(False)
        self.teacher.eval()
        return self

    def forward(self, video):
        if video.dim() != 5:
            raise ValueError('teacher input must be [B,T,3,H,W]')
        return self.teacher.get_rppg_only(video)
