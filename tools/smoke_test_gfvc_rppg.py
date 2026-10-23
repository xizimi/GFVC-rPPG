"""One-window forward/backward check; does not update model weights."""

import argparse

import torch
import yaml
from torch.utils.data._utils.collate import default_collate

from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import HEEstimator, KPDetector
from modules.model import GFVCRPPGSequenceModel
from modules.rppg import RePPGTeacherAdapter, physiological_losses
from rppg_gfvc_dataset import GFVCRPPGWindowDataset
from train_gfvc_rppg import load_component_checkpoint, set_stage2_trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/train')
    parser.add_argument('--gfvc_checkpoint', required=True)
    parser.add_argument('--rppg_checkpoint', required=True)
    parser.add_argument('--reppg_root', default='/opt/data/private/project/rePPG-Rewriting-Photoplethysmography-Signal-to-Video-master')
    parser.add_argument('--config', default='config/vox-256.yaml')
    parser.add_argument('--mode', choices=['preserve', 'rewrite'], default='rewrite')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    common = config['model_params']['common_params']
    generator = OcclusionAwareSPADEGenerator(
        **config['model_params']['generator_params'], **common
    )
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'], **common)
    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'], **common)
    load_component_checkpoint(args.gfvc_checkpoint, generator, kp_detector, he_estimator)
    teacher = RePPGTeacherAdapter(args.reppg_root, args.rppg_checkpoint)
    trainable = set_stage2_trainable(generator, kp_detector, he_estimator)
    model = GFVCRPPGSequenceModel(
        kp_detector, he_estimator, generator, teacher,
        loss_weights=config['train_params'].get('physiology_loss_weights'),
        estimate_jacobian=common['estimate_jacobian'],
    ).cuda().train()
    dataset = GFVCRPPGWindowDataset(
        args.data_root, rewrite_probability=float(args.mode == 'rewrite'),
        deterministic=True,
    )
    batch = default_collate([dataset[0]])
    batch = {
        key: value.cuda() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    losses, output = model(batch)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite loss')
    total.backward()
    gradients = {
        name: float(parameter.grad.norm())
        for name, parameter in generator.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    if not gradients or max(gradients.values()) <= 0:
        raise RuntimeError('Physiology modules received no gradient')
    print('mode=%s fps=%.3f target_bpm=%.3f' %
          (args.mode, float(batch['fps'][0]), float(batch['target_bpm'][0])))
    print('losses:', {name: round(float(value), 6) for name, value in losses.items()})
    print('total=%.6f output=%s gradient_max=%.6f' %
          (float(total), tuple(output['prediction'].shape), max(gradients.values())))
    with torch.no_grad():
        real_waveform = teacher(batch['driving_video'] * batch['skin_masks'])
        real_losses = physiological_losses(
            real_waveform, batch['original_rppg'], batch['fps'],
        )
    print('teacher_on_real_ubfc:', {
        name: round(float(real_losses[name]), 6)
        for name in ('pearson', 'spectral', 'bpm')
    })
    print('peak_gpu_gib=%.3f' % (torch.cuda.max_memory_allocated() / 2 ** 30))


if __name__ == '__main__':
    main()
