"""Stage-2 training for the unified G(keyframe, motion, target_rPPG) model."""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator
from modules.model import GFVCRPPGSequenceModel
from modules.rppg import RPPGEncoder, RePPGTeacherAdapter
from rppg_gfvc_dataset import GFVCRPPGWindowDataset


def load_component_checkpoint(path, generator, kp_detector, he_estimator):
    checkpoint = torch.load(path, map_location='cpu')
    for key, module in [('generator', generator), ('kp_detector', kp_detector),
                        ('he_estimator', he_estimator)]:
        if key not in checkpoint:
            raise KeyError('%s is missing component %s' % (path, key))
        missing, unexpected = module.load_state_dict(checkpoint[key], strict=False)
        # New physiology layers are intentionally absent from legacy checkpoints.
        allowed_new = ('physiology', 'rppg_fusion')
        unexpected = [name for name in unexpected if not name.startswith(allowed_new)]
        missing = [name for name in missing if not name.startswith(allowed_new)]
        if missing or unexpected:
            raise RuntimeError(
                '%s checkpoint architecture mismatch: missing=%s unexpected=%s' %
                (key, missing, unexpected)
            )


def set_stage2_trainable(generator, kp_detector, he_estimator):
    for module in (generator, kp_detector, he_estimator):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    trainable = []
    for name, parameter in generator.named_parameters():
        if name.startswith('physiology_encoder.') or name.startswith('physiology_residual.'):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError('No physiology parameters were selected for training')
    return trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vox-256.yaml')
    parser.add_argument('--generator', choices=['spade', 'original'], default='spade')
    parser.add_argument('--data_root', default='/opt/data/private/datasets/UBFC2_fv2v/train')
    parser.add_argument('--gfvc_checkpoint', required=True)
    parser.add_argument('--rppg_checkpoint', required=True)
    parser.add_argument('--rppg_backend', choices=['reppg', 'native'], default='reppg')
    parser.add_argument('--reppg_root', default='/opt/data/private/project/rePPG-Rewriting-Photoplethysmography-Signal-to-Video-master')
    parser.add_argument('--output', default='checkpoint/gfvc_rppg_stage2.pth.tar')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sequence_length', type=int, default=150)
    parser.add_argument('--rewrite_probability', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--require_semantic_masks', action='store_true')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    common = config['model_params']['common_params']
    generator_class = (OcclusionAwareSPADEGenerator if args.generator == 'spade'
                       else OcclusionAwareGenerator)
    generator = generator_class(
        **config['model_params']['generator_params'], **common
    )
    kp_detector = KPDetector(
        **config['model_params']['kp_detector_params'], **common
    )
    he_estimator = HEEstimator(
        **config['model_params']['he_estimator_params'], **common
    )
    load_component_checkpoint(
        args.gfvc_checkpoint, generator, kp_detector, he_estimator
    )

    if args.rppg_backend == 'reppg':
        rppg_extractor = RePPGTeacherAdapter(args.reppg_root, args.rppg_checkpoint)
    else:
        rppg_extractor = RPPGEncoder()
        rppg_state = torch.load(args.rppg_checkpoint, map_location='cpu')
        if isinstance(rppg_state, dict) and 'state_dict' in rppg_state:
            rppg_state = rppg_state['state_dict']
        rppg_extractor.load_state_dict(rppg_state)

    trainable = set_stage2_trainable(generator, kp_detector, he_estimator)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = GFVCRPPGSequenceModel(
        kp_detector, he_estimator, generator, rppg_extractor,
        loss_weights=config['train_params'].get('physiology_loss_weights'),
        estimate_jacobian=common['estimate_jacobian'],
    ).to(device)
    optimizer = torch.optim.Adam(trainable, lr=args.lr, betas=(0.5, 0.999))

    dataset = GFVCRPPGWindowDataset(
        args.data_root,
        sequence_length=args.sequence_length,
        rewrite_probability=args.rewrite_probability,
        require_semantic_masks=args.require_semantic_masks,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = {}
        progress = tqdm(loader, desc='stage2 %d/%d' % (epoch + 1, args.epochs))
        for batch in progress:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            losses, _ = model(batch, epoch=epoch, total_epochs=args.epochs)
            total = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for name, value in losses.items():
                running[name] = 0.95 * running.get(name, value.item()) + 0.05 * value.item()
            progress.set_postfix(total='%.4f' % total.item(), **{
                name: '%.3f' % value for name, value in running.items()
            })

        torch.save({
            'epoch': epoch + 1,
            'generator': generator.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
        }, args.output)


if __name__ == '__main__':
    main()
