"""Skin-area RGB conditioning shared by training and video inference."""
import torch


def skin_rgb_means(video, masks):
    """Return [B,T,3] means over skin pixels of RGB video in [0,1]."""
    if video.ndim != 5 or video.shape[2] != 3:
        raise ValueError('video must have shape [B,T,3,H,W]')
    if masks.shape != video.shape[:2] + (1,) + video.shape[-2:]:
        raise ValueError('masks must have shape [B,T,1,H,W] matching video')
    if not torch.isfinite(video).all() or not torch.isfinite(masks).all():
        raise ValueError('video and masks must be finite')
    if (video < 0).any() or (video > 1).any() or (masks < 0).any() or (masks > 1).any():
        raise ValueError('video and masks must be in [0,1]')
    area = masks.sum(dim=(-2, -1))
    if (area <= 0).any():
        raise ValueError('Empty skin mask in conditioning window')
    return (video * masks).sum(dim=(-2, -1)) / area


def skin_rgb_statistics(values):
    """Return the physical-scale per-window channel means and deviations."""
    if values.ndim != 3 or values.shape[-1] != 3 or values.shape[1] < 2:
        raise ValueError('skin RGB must have shape [B,T,3] with T >= 2')
    if not torch.isfinite(values).all():
        raise ValueError('skin RGB must be finite')
    values = values.float()
    return values.mean(dim=1, keepdim=True), values.std(
        dim=1, keepdim=True, unbiased=False)


def normalize_skin_rgb(values):
    mean, deviation = skin_rgb_statistics(values)
    return (values.float() - mean) / deviation.clamp_min(1e-6)
