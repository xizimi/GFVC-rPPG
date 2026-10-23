"""Stage-2 training for the unified G(keyframe, motion, target_rPPG) model."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator
from modules.model import GFVCRPPGSequenceModel
from modules.rppg import RPPGEncoder, RePPGTeacherAdapter, negative_pearson_loss
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


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    totals = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        losses, output = model(batch)
        values = {name: value.item() for name, value in losses.items()}
        values['total'] = sum(values.values())
        values['video_l1'] = F.l1_loss(
            output['prediction'], batch['driving_video']
        ).item()
        values['baseline_l1'] = F.l1_loss(
            output['prediction_base'], batch['driving_video']
        ).item()
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
        if count >= max_batches:
            break
    if count == 0:
        raise RuntimeError('Validation loader has no batches')
    return {name: value / count for name, value in totals.items()}


def save_checkpoint(path, state):
    temporary = path + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def teacher_preflight(teacher, dataset, device, subject_count):
    """Measure teacher/cPPG correlation on distinct held-out subjects."""
    seen, correlations = set(), []
    for index, (face_path, _, _, _) in enumerate(dataset.samples):
        subject = os.path.basename(face_path).split('_clip')[0]
        if subject in seen:
            continue
        sample = dataset[index]
        video = (sample['driving_video'] * sample['skin_masks']).unsqueeze(0).to(device)
        target = sample['original_rppg'].unsqueeze(0).to(device)
        extracted = teacher(video)
        correlation = 1.0 - float(negative_pearson_loss(extracted, target))
        correlations.append(correlation)
        seen.add(subject)
        print('teacher preflight %s: Pearson %.3f' % (subject, correlation), flush=True)
        if len(correlations) >= subject_count:
            break
    if not correlations:
        raise RuntimeError('Teacher preflight found no validation subjects')
    return sum(correlations) / len(correlations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/vox-256.yaml')
    parser.add_argument('--generator', choices=['spade', 'original'], default='spade')
    parser.add_argument('--data_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/train')
    parser.add_argument('--val_data_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/val')
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
    parser.add_argument('--signal_preprocess', choices=['reppg', 'none'], default='reppg')
    parser.add_argument('--max_val_batches', type=int, default=4)
    parser.add_argument('--teacher_preflight_subjects', type=int, default=5)
    parser.add_argument('--min_teacher_pearson', type=float, default=0.2)
    parser.add_argument('--skip_teacher_preflight', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--require_semantic_masks', action='store_true')
    parser.add_argument('--allow_fallback_masks', action='store_true')
    args = parser.parse_args()
    if args.max_val_batches < 1:
        parser.error('--max_val_batches must be at least 1')
    if args.teacher_preflight_subjects < 1:
        parser.error('--teacher_preflight_subjects must be at least 1')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    start_epoch, best_val = 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu')
        generator.load_state_dict(saved['generator'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch = int(saved['epoch'])
        best_val = float(saved.get('best_val', float('inf')))

    require_masks = args.require_semantic_masks or not args.allow_fallback_masks

    dataset = GFVCRPPGWindowDataset(
        args.data_root,
        sequence_length=args.sequence_length,
        rewrite_probability=args.rewrite_probability,
        require_semantic_masks=require_masks,
        signal_preprocess=args.signal_preprocess,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    validation_loaders = {}
    preserve_validation = None
    for mode, probability in [('preserve', 0.0), ('rewrite', 1.0)]:
        validation = GFVCRPPGWindowDataset(
            args.val_data_root,
            sequence_length=args.sequence_length,
            rewrite_probability=probability,
            require_semantic_masks=require_masks,
            signal_preprocess=args.signal_preprocess,
            deterministic=True,
            seed=args.seed,
        )
        validation_loaders[mode] = DataLoader(
            validation, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        if mode == 'preserve':
            preserve_validation = validation
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if args.rppg_backend == 'reppg' and not args.skip_teacher_preflight:
        mean_correlation = teacher_preflight(
            rppg_extractor, preserve_validation, device,
            args.teacher_preflight_subjects,
        )
        print('teacher preflight mean Pearson: %.3f' % mean_correlation, flush=True)
        if mean_correlation < args.min_teacher_pearson:
            raise RuntimeError(
                'Frozen rePPG teacher does not track held-out UBFC cPPG '
                '(mean Pearson %.3f < %.3f). Validate or adapt the teacher '
                'before stage-2 training. --skip_teacher_preflight is only '
                'for deliberate diagnostic experiments.' %
                (mean_correlation, args.min_teacher_pearson)
            )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {}
        progress = tqdm(loader, desc='stage2 %d/%d' % (epoch + 1, args.epochs))
        for batch in progress:
            batch = move_batch(batch, device)
            losses, _ = model(batch, epoch=epoch, total_epochs=args.epochs)
            total = sum(losses.values())
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite training loss at epoch %d' % (epoch + 1))
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            for name, value in losses.items():
                running[name] = 0.95 * running.get(name, value.item()) + 0.05 * value.item()
            progress.set_postfix(total='%.4f' % total.item(), **{
                name: '%.3f' % value for name, value in running.items()
            })

        validation_metrics = {
            mode: evaluate(model, val_loader, device, args.max_val_batches)
            for mode, val_loader in validation_loaders.items()
        }
        val_score = 0.5 * (
            validation_metrics['preserve']['total'] +
            validation_metrics['rewrite']['total']
        )
        if not np.isfinite(val_score):
            raise FloatingPointError('Non-finite validation loss at epoch %d' % (epoch + 1))
        improved = val_score < best_val
        best_val = min(best_val, val_score)
        print('epoch %d validation: %s' % (epoch + 1, validation_metrics), flush=True)
        state = {
            'epoch': epoch + 1,
            'generator': generator.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'best_val': best_val,
            'validation': validation_metrics,
            'args': vars(args),
        }
        save_checkpoint(args.output, state)
        if improved:
            save_checkpoint(args.output + '.best', state)


if __name__ == '__main__':
    main()
