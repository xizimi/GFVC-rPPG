"""Train a standalone, differentiable rPPG measurement teacher on UBFC.

The teacher sees only real RGB video masked with the precomputed MediaPipe skin
ROI and predicts its complete contact-PPG waveform.  It is intentionally
separate from GFVC and must pass validation before it is used as a frozen loss
network for physiological video generation.
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.rppg import SkinRPPGTeacher, physiological_losses, normalized_power_spectrum
from rppg_gfvc_dataset import GFVCRPPGWindowDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_from_name(name):
    return os.path.basename(name).split('_clip')[0]


def peak_bpm(waveform, fps):
    spectrum, frequencies = normalized_power_spectrum(waveform.unsqueeze(0), fps)
    return float(frequencies[int(spectrum[0].argmax())] * 60.0)


def batch_loss(model, batch, device, weights):
    video = batch['driving_video'].to(device, non_blocking=True)
    masks = batch['skin_masks'].to(device, non_blocking=True)
    target = batch['original_rppg'].to(device, non_blocking=True)
    fps = batch['fps'].to(device, non_blocking=True)
    bpm = batch['target_bpm'].to(device, non_blocking=True)
    prediction = model(video * masks)
    pieces = physiological_losses(prediction, target, fps, target_bpm=bpm)
    total = sum(weights[name] * pieces[name] for name in weights)
    return total, pieces, prediction, target


@torch.no_grad()
def validate(model, loader, device, weights):
    model.eval()
    subjects = defaultdict(lambda: {'corr': [], 'hr': [], 'loss': []})
    for batch in loader:
        total, pieces, prediction, target = batch_loss(model, batch, device, weights)
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite validation loss')
        for index, name in enumerate(batch['name']):
            subject = subject_from_name(name)
            corr = 1.0 - float(physiological_losses(
                prediction[index:index + 1], target[index:index + 1],
                batch['fps'][index:index + 1])['pearson'])
            fps = float(batch['fps'][index])
            target_bpm = float(batch['target_bpm'][index])
            if not np.isfinite(target_bpm):
                target_bpm = peak_bpm(target[index], fps)
            subjects[subject]['corr'].append(corr)
            subjects[subject]['hr'].append(abs(peak_bpm(prediction[index], fps) - target_bpm))
            subjects[subject]['loss'].append(float(total))
    report = {
        subject: {key: float(np.mean(values)) for key, values in metrics.items()}
        for subject, metrics in subjects.items()
    }
    mean_corr = float(np.mean([item['corr'] for item in report.values()]))
    mean_hr = float(np.mean([item['hr'] for item in report.values()]))
    mean_loss = float(np.mean([item['loss'] for item in report.values()]))
    return mean_corr, mean_hr, mean_loss, report


def checkpoint(path, model, optimizer, epoch, metrics, args):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'args': vars(args),
    }
    temporary = path + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/train')
    parser.add_argument('--val_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/val')
    parser.add_argument('--output', default='checkpoint/skin_rppg_teacher.pt')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch_size must be positive')
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    weights = {'pearson': 1.0, 'waveform': 0.25, 'spectral': 0.5,
               'kl': 0.01, 'bpm': 0.002}

    train_set = GFVCRPPGWindowDataset(
        args.train_root, sequence_length=150, rewrite_probability=0.0,
        require_semantic_masks=True, signal_preprocess='reppg')
    val_set = GFVCRPPGWindowDataset(
        args.val_root, sequence_length=150, rewrite_probability=0.0,
        require_semantic_masks=True, signal_preprocess='reppg',
        deterministic=True, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                              drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    model = SkinRPPGTeacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    print('train_windows=%d val_windows=%d device=%s parameters=%d' %
          (len(train_set), len(val_set), device,
           sum(parameter.numel() for parameter in model.parameters())), flush=True)

    best_corr, stale = -float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, count = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            total, pieces, _, _ = batch_loss(model, batch, device, weights)
            if not torch.isfinite(total):
                raise FloatingPointError('Non-finite loss at epoch %d step %d: %s' %
                                         (epoch, step, {key: float(value) for key, value in pieces.items()}))
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(total.detach())
            count += 1
            if step == 1 or step % 50 == 0:
                print('epoch=%d step=%d/%d train_loss=%.4f pearson=%.3f' %
                      (epoch, step, len(train_loader), running / count,
                       1.0 - float(pieces['pearson'])), flush=True)
        scheduler.step()
        corr, hr, val_loss, detail = validate(model, val_loader, device, weights)
        metrics = {'pearson': corr, 'hr_mae': hr, 'loss': val_loss, 'subjects': detail}
        checkpoint(args.output + '.latest', model, optimizer, epoch, metrics, args)
        print('epoch=%d validation pearson=%.3f hr_mae=%.2f loss=%.4f subjects=%s' %
              (epoch, corr, hr, val_loss,
               {name: round(value['corr'], 3) for name, value in detail.items()}), flush=True)
        if corr > best_corr + 1e-4:
            best_corr, stale = corr, 0
            checkpoint(args.output, model, optimizer, epoch, metrics, args)
            print('saved_best=%s pearson=%.3f' % (args.output, corr), flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print('early_stop: no Pearson improvement for %d epochs' % stale,
                      flush=True)
                break


if __name__ == '__main__':
    main()
