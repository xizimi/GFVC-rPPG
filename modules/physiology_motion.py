"""Low-rank reference-feature modulation sharing the existing GFVC 3-D warp.

The basis is a learned modulation basis, NOT a physical blood-flow estimate.
No driving image or ground-truth semantic mask is an input to this module.
"""
import math
import torch
from torch import nn


FUSION_MODES = ('rgb_residual', 'feature_unaligned', 'feature_motion_aligned', 'none')


class MotionAlignedPhysioModulator(nn.Module):
    def __init__(self, reference_channels=32, depth=16, feature_channels=256,
                 condition_dim=64, rank=2, hidden_channels=16,
                 max_feature_delta=0.1, initial_strength=0.1):
        super().__init__()
        if rank < 1 or depth < 1 or not 0 < initial_strength < 1 or max_feature_delta <= 0:
            raise ValueError('Invalid rank/depth/modulation strength')
        self.rank, self.depth = int(rank), int(depth)
        self.feature_channels = int(feature_channels)
        self.max_feature_delta = float(max_feature_delta)
        self.basis_encoder = nn.Sequential(
            nn.Conv3d(reference_channels, hidden_channels, 1), nn.SiLU(),
            nn.Conv3d(hidden_channels, rank, 3, padding=1), nn.Tanh())
        # Flatten precisely in (rank,depth) order, as the GFVC (channel,depth)
        # projection does. Grouped weights learn depth reduction per basis.
        self.depth_projection = nn.Conv2d(rank * depth, rank, 1, groups=rank)
        nn.init.constant_(self.depth_projection.weight, 1. / depth)
        nn.init.zeros_(self.depth_projection.bias)
        self.skin_gate = nn.Sequential(nn.Conv2d(feature_channels, hidden_channels, 1),
            nn.SiLU(), nn.Conv2d(hidden_channels, 1, 1))
        self.modulation_head = nn.Sequential(nn.Linear(condition_dim + 1, 64),
            nn.SiLU(), nn.Linear(64, rank * feature_channels))
        nn.init.normal_(self.modulation_head[-1].weight, std=0.02)
        nn.init.zeros_(self.modulation_head[-1].bias)
        # Nonzero initialization keeps gradients open to ALL new submodules.
        self.strength_logit = nn.Parameter(torch.tensor(math.log(initial_strength / (1 - initial_strength))))

    def make_basis(self, reference_volume):
        if reference_volume.ndim != 5 or reference_volume.shape[2] != self.depth:
            raise ValueError('reference must be [B,C,D,H,W] with configured depth')
        return self.basis_encoder(reference_volume)

    def align_basis(self, basis, deformation, deform_input, aligned):
        if aligned:
            if deformation is None:
                raise ValueError('Motion-aligned mode requires the existing GFVC deformation')
            # Intentionally reuse the SAME function/grid as appearance; do not
            # create, estimate, detach, or reinterpret another motion field.
            return deform_input(basis, deformation)
        return basis

    def forward(self, features, reference_basis, condition, physiology_value,
                occlusion_map, deformation=None, deform_input=None, aligned=True):
        b, c, h, w = features.shape
        if c != self.feature_channels or condition.ndim != 2 or condition.shape[0] != b:
            raise ValueError('Invalid generated features or temporal condition')
        value = physiology_value.reshape(b, 1)
        basis = self.align_basis(reference_basis, deformation, deform_input, aligned)
        if basis.shape != (b, self.rank, self.depth, h, w):
            raise ValueError('Warped basis must match GFVC feature spatial dimensions')
        response = torch.tanh(self.depth_projection(basis.reshape(b, self.rank * self.depth, h, w)))
        skin = torch.sigmoid(self.skin_gate(features))
        visibility = torch.ones_like(skin) if occlusion_map is None else occlusion_map
        if visibility.shape != skin.shape:
            raise ValueError('Use the same resized 2-D occlusion map as GFVC appearance')
        gate = skin * visibility.clamp(0., 1.)
        coefficients = torch.tanh(self.modulation_head(torch.cat((condition, value), 1)))
        coefficients = coefficients.reshape(b, self.rank, c) * torch.tanh(value).unsqueeze(2)
        modulation = torch.einsum('bkhw,bkc->bchw', response, coefficients) / self.rank
        delta = self.max_feature_delta * torch.sigmoid(self.strength_logit) * gate * torch.tanh(modulation)
        return dict(feature_delta=delta, skin_mask=skin, gate=gate,
                    response_maps=response, aligned_basis=basis,
                    coefficients=coefficients)


def load_generator_compatible(generator, state, declared_mode=None, initialize_new=False):
    """Strict schema checks before loading; only explicit transfer adds modules."""
    if declared_mode is not None and declared_mode != getattr(generator, 'physiology_fusion_mode', 'rgb_residual'):
        raise RuntimeError('Checkpoint fusion mode mismatch: %s' % declared_mode)
    current = generator.state_dict()
    unexpected = sorted(set(state) - set(current))
    missing = sorted(set(current) - set(state))
    prefixes = ('physiology_encoder.', 'physiology_residual.', 'rppg_fusion.', 'physiology_modulator.')
    original_gfvc = not any(k.startswith(prefixes) for k in state)
    allowed = prefixes if original_gfvc else (('physiology_modulator.',) if initialize_new else ())
    bad = [k for k in missing if not k.startswith(allowed)]
    mismatched = [k for k in state if k in current and state[k].shape != current[k].shape]
    if unexpected or bad or mismatched:
        raise RuntimeError('Checkpoint mismatch: missing=%s unexpected=%s shapes=%s' % (bad, unexpected, mismatched))
    # Merge only the enumerated, intentionally fresh keys; strict=True still
    # validates every tensor, including all original pretrained GFVC parameters.
    merged = dict(current)
    merged.update(state)
    generator.load_state_dict(merged, strict=True)
    return dict(original_GFVC=original_gfvc, fresh_keys=missing, unexpected_keys=[])
