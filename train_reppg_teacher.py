"""Adapt the rePPG Hourglass measurement network to real UBFC skin video.

This produces a plain HourglassNetStep2 state_dict compatible with
RePPGTeacherAdapter and train_gfvc_rppg.py.  It does not train the GFVC model.
"""

import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from modules.rppg import RePPGTeacherAdapter, physiological_losses
from rppg_gfvc_dataset import GFVCRPPGWindowDataset


def subject_name(face_path):
    return os.path.basename(face_path).split('_clip')[0]


def subject_subset(dataset, limit):
    if limit <= 0:
        return dataset
    groups = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        groups[subject_name(sample[0])].append(index)
    indices = []
    for group in groups.values():
        positions = np.linspace(0, len(group) - 1, min(limit, len(group)))
        indices.extend(group[int(round(position))] for position in positions)
    return Subset(dataset, sorted(set(indices)))


def random_subject_subset(dataset, limit, seed):
    if limit <= 0:
        return dataset
    groups = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        groups[subject_name(sample[0])].append(index)
    rng = np.random.RandomState(seed)
    indices = []
    for group in groups.values():
        chosen = rng.choice(group, size=min(limit, len(group)), replace=False)
        indices.extend(chosen.tolist())
    return Subset(dataset, sorted(indices))


def set_trainable(adapter, phase):
    for parameter in adapter.teacher.parameters():
        parameter.requires_grad_(False)
    selected = [adapter.teacher.rppg_ffn]
    if phase == 'light':
        light = adapter.teacher.light
        selected += [light.conv1, light.conv2, light.conv3,
                     light.predict_relu1, light.predict_FC]
    elif phase == 'full':
        selected += [adapter.teacher.pre_conv, adapter.teacher.pre_bn,
                     adapter.teacher.HG3]
    for module in selected:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in adapter.parameters()
            if parameter.requires_grad]


def calculate(adapter, batch, device):
    video = batch['driving_video'].to(device, non_blocking=True)
    masks = batch['skin_masks'].to(device, non_blocking=True)
    target = batch['original_rppg'].to(device, non_blocking=True)
    fps = batch['fps'].to(device, non_blocking=True)
    bpm = batch['target_bpm'].to(device, non_blocking=True)
    # Masks are constants. Gradients flow through the video's skin RGB values.
    prediction = adapter(video * masks).float()
    losses = physiological_losses(prediction, target, fps, target_bpm=bpm)
    total = (losses['pearson'] + 0.25 * losses['waveform'] +
             0.5 * losses['spectral'] + 0.01 * losses['kl'] +
             0.001 * losses['bpm'])
    return total, losses


def validate(adapter, loader, device):
    adapter.eval()
    by_subject = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            total, losses = calculate(adapter, batch, device)
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite teacher validation loss')
            for index, subject in enumerate(batch['subject']):
                # Validation loader uses batch size one, so this also records
                # a per-window value before subject-balanced aggregation.
                by_subject[subject].append(1.0 - float(losses['pearson']))
    correlations = {subject: float(np.mean(values))
                    for subject, values in by_subject.items()}
    return float(np.mean(list(correlations.values()))), correlations


class NamedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        base = self.dataset.dataset if isinstance(self.dataset, Subset) else self.dataset
        source_index = self.dataset.indices[index] if isinstance(self.dataset, Subset) else index
        sample['subject'] = subject_name(base.samples[source_index][0])
        return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/train')
    parser.add_argument('--val_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/val')
    parser.add_argument('--reppg_root', default='/opt/data/private/project/rePPG-Rewriting-Photoplethysmography-Signal-to-Video-master')
    parser.add_argument('--initial_checkpoint', required=True)
    parser.add_argument('--output', default='checkpoint/reppg_teacher_ubfc.pt')
    parser.add_argument('--phase', choices=['head', 'light', 'full'], default='head')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--val_windows_per_subject', type=int, default=0,
                        help='0 validates every window (recommended); positive values are diagnostic only')
    parser.add_argument('--max_train_windows', type=int, default=0,
                        help='Diagnostic only; 0 uses every training window')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='Diagnostic only; 0 uses every training window')
    parser.add_argument('--train_windows_per_subject', type=int, default=0,
                        help='Draw new balanced windows each epoch; 0 uses all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--amp', action='store_true',
                        help='Use CUDA mixed precision after a full-precision smoke test')
    args = parser.parse_args()
    if args.epochs < 1 or args.accumulate < 1:
        parser.error('epochs and accumulate must be positive')
    if args.max_train_windows > 0 and args.train_windows_per_subject > 0:
        parser.error('max_train_windows and train_windows_per_subject conflict')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = GFVCRPPGWindowDataset(
        args.train_root, rewrite_probability=0.0, require_semantic_masks=True)
    val_data = GFVCRPPGWindowDataset(
        args.val_root, rewrite_probability=0.0, require_semantic_masks=True,
        deterministic=True, seed=args.seed)
    if args.max_train_windows > 0:
        train_data = Subset(train_data, list(range(min(args.max_train_windows,
                                                    len(train_data)))))
    val_data = subject_subset(val_data, args.val_windows_per_subject)
    val_loader = DataLoader(NamedDataset(val_data), batch_size=1,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == 'cuda')

    adapter = RePPGTeacherAdapter(
        args.reppg_root, args.initial_checkpoint,
        checkpoint_frames=args.phase == 'full').to(device)
    parameters = set_trainable(adapter, args.phase)
    if args.phase in ('light', 'full'):
        head_parameters = list(adapter.teacher.rppg_ffn.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        light_parameters = [parameter for parameter in parameters
                            if id(parameter) not in head_ids]
        optimizer_groups = [
            {'params': head_parameters, 'lr': args.lr},
            {'params': light_parameters, 'lr': args.lr * 0.1},
        ]
    else:
        optimizer_groups = [{'params': parameters, 'lr': args.lr}]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    print('phase=%s train=%d val=%d trainable=%d device=%s' %
          (args.phase, len(train_data), len(val_data),
           sum(p.numel() for p in parameters), device), flush=True)

    best_correlation, baseline_subjects = validate(adapter, val_loader, device)
    torch.save(adapter.teacher.state_dict(), args.output)
    print('initial_val_subject_mean_pearson=%.3f subjects=%s' %
          (best_correlation, baseline_subjects), flush=True)
    bad_epochs = 0
    for epoch in range(args.epochs):
        started = time.time()
        epoch_data = random_subject_subset(
            train_data, args.train_windows_per_subject, args.seed + epoch)
        train_loader = DataLoader(
            NamedDataset(epoch_data), batch_size=1, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        print('epoch=%d selected_train_windows=%d' %
              (epoch + 1, len(epoch_data)), flush=True)
        adapter.eval()  # Freeze reference BatchNorm statistics throughout.
        optimizer.zero_grad(set_to_none=True)
        running, steps = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            with torch.cuda.amp.autocast(enabled=use_amp):
                total, losses = calculate(adapter, batch, device)
            if not torch.isfinite(total):
                print('loss_components=%s' %
                      {name: float(value.detach()) for name, value in losses.items()},
                      flush=True)
                raise FloatingPointError('Non-finite teacher loss at epoch %d step %d' %
                                         (epoch + 1, step))
            scaler.scale(total / args.accumulate).backward()
            if step % args.accumulate == 0 or step == len(train_loader) or (
                    args.max_steps > 0 and step >= args.max_steps):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(total.detach())
            steps += 1
            if step == 1 or step % 20 == 0:
                print('epoch=%d step=%d/%d loss=%.4f pearson=%.3f' %
                      (epoch + 1, step, len(train_loader),
                       running / steps, 1.0 - float(losses['pearson'])),
                      flush=True)
            if args.max_steps > 0 and step >= args.max_steps:
                break
        correlation, subjects = validate(adapter, val_loader, device)
        print('epoch=%d train_loss=%.4f val_subject_mean_pearson=%.3f subjects=%s' %
              (epoch + 1, running / max(steps, 1), correlation, subjects),
              flush=True)
        print('epoch=%d elapsed_seconds=%.1f cuda_peak_mib=%.0f' %
              (epoch + 1, time.time() - started,
               torch.cuda.max_memory_allocated() / 1048576
               if device.type == 'cuda' else 0), flush=True)
        state = adapter.teacher.state_dict()
        torch.save(state, args.output + '.latest')
        if correlation > best_correlation + 1e-4:
            best_correlation = correlation
            bad_epochs = 0
            torch.save(state, args.output)
            print('saved_best=%s mean_pearson=%.3f' %
                  (args.output, best_correlation), flush=True)
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print('early_stop=no validation improvement for %d epochs' %
                      bad_epochs, flush=True)
                break


if __name__ == '__main__':
    main()
