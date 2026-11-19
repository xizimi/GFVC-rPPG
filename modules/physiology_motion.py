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
                 max_feature_delta=0.1, initial_strength=0.1, value_channels=1):
        super().__init__()
        if rank < 1 or depth < 1 or not 0 < initial_strength < 1 or max_feature_delta <= 0:
            raise ValueError('Invalid rank/depth/modulation strength')
        self.value_channels = value_channels
        self.value_amplitude = nn.Linear(3, 1) if value_channels == 3 else None
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
        self.modulation_head = nn.Sequential(nn.Linear(condition_dim + value_channels, 64),
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
        value = physiology_value.reshape(b, self.value_channels)
        amplitude = self.value_amplitude(value) if self.value_amplitude is not None else value
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
        coefficients = coefficients.reshape(b, self.rank, c) * torch.tanh(amplitude).unsqueeze(2)
        modulation = torch.einsum('bkhw,bkc->bchw', response, coefficients) / self.rank
        delta = self.max_feature_delta * torch.sigmoid(self.strength_logit) * gate * torch.tanh(modulation)
        return dict(feature_delta=delta, skin_mask=skin, gate=gate,
                    response_maps=response, aligned_basis=basis,
                    coefficients=coefficients)


def transfer_bvp_to_skin_rgb(generator, state):
    """Explicitly initialize RGB conditioning from a trained scalar-BVP model."""
    if getattr(generator, 'physiology_input_mode', None) != 'skin_rgb':
        raise ValueError('Transfer requires a skin_rgb generator')
    current = generator.state_dict()
    changed = {'physiology_encoder.input.weight',
               'physiology_modulator.modulation_head.0.weight'}
    fresh = {'physiology_modulator.value_amplitude.weight',
             'physiology_modulator.value_amplitude.bias'}
    if set(state) != set(current) - fresh:
        raise RuntimeError('Transfer source must have the exact scalar-BVP model schema')
    expected_shapes = {
        'physiology_encoder.input.weight': (current['physiology_encoder.input.weight'].shape[0], 1, 7),
        'physiology_modulator.modulation_head.0.weight':
            (current['physiology_modulator.modulation_head.0.weight'].shape[0],
             current['physiology_modulator.modulation_head.0.weight'].shape[1] - 2),
    }
    for name, value in state.items():
        expected = expected_shapes.get(name, current[name].shape)
        if tuple(value.shape) != tuple(expected):
            raise RuntimeError('Unexpected transfer tensor shape: ' + name)
    merged = dict(current)
    for name, value in state.items():
        if name not in changed:
            merged[name] = value
    generator.load_state_dict(merged, strict=True)
    return dict(copied=sorted(set(state) - changed), initialized=sorted(changed | fresh))


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
