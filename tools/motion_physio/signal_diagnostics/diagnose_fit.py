"""Compare a short overfit probe with and without a fixed amplitude gain."""
import argparse
import math
from pathlib import Path
import time

import numpy as np
import torch

from diagnose_signal import load_models, correlation, records, train
from data import sample
from modules.model import GFVCRPPGSequenceModel
from train_gfvc_rppg import set_stage2_trainable


def grad_norm(gradients):
    return float(torch.stack([grad.detach().square().sum() for grad in gradients if grad is not None]).sum().sqrt())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--gain', type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.gain) or args.gain <= 0:
        parser.error('--gain must be positive and finite')
    train.seed()
    generator, detector, estimator, teacher, common = load_models(args.checkpoint)
    params = set_stage2_trainable(generator, detector, estimator, 'physiology')
    generator.checkpoint_decoder = True
    config = torch.load(args.checkpoint, map_location='cpu')['config']
    model = GFVCRPPGSequenceModel(detector, estimator, generator, teacher,
        loss_weights=config['train_params']['physiology_loss_weights'],
        estimate_jacobian=common['estimate_jacobian']).cuda().train()
    optimizer = torch.optim.Adam(params, lr=1e-4, betas=(.5, .999))
    rows = records('train')
    chosen = [next(row for row in rows if row['dataset'] == key) for key in ('UBFC', 'PURE', 'LGI-PPGI')]
    batches = []
    for row in chosen:
        batch = sample(row, torch.device('cuda'), np.random.RandomState(42))
        # Keep multiple pulse cycles while keeping this probe much shorter than an epoch.
        for key in ('driving_video', 'skin_masks', 'target_rppg', 'original_rppg'):
            batch[key] = batch[key][:, :90]
        batches.append(batch)
    tracked = [generator.physiology_modulator.value_amplitude.weight,
               generator.physiology_modulator.modulation_head[0].weight,
               generator.physiology_encoder.input.weight]
    gains = generator.physiology_modulator.value_amplitude.register_forward_hook(
        lambda module, inputs, output: output * args.gain)
    records_out = []
    began = time.monotonic()
    for step in range(15):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        losses, output = model(batch, epoch=1, total_epochs=30)
        physiology = sum(value for key, value in losses.items() if key.startswith('rppg_'))
        regularization = sum(value for key, value in losses.items() if not key.startswith('rppg_'))
        combined = physiology + regularization
        result = dict(step=step, sample=batch['sample_id'], gain=args.gain,
                      pcc=correlation(output['rppg_extracted'].detach().cpu(), batch['target_rppg'].cpu()),
                      total=float(combined.detach()), losses={key: float(value.detach()) for key, value in losses.items()})
        if step < len(batches):
            for name, loss in (('physiology', physiology), ('regularization', regularization)):
                gradient = torch.autograd.grad(loss, tracked, allow_unused=True, retain_graph=True)
                result[name+'_grad_norm'] = grad_norm(gradient)
        if step < 12:
            combined.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
        records_out.append(result)
        train.save(args.output, records_out)
        print('FIT', step, result['sample'], result['pcc'], result['total'], time.monotonic()-began, flush=True)
        del losses, output, physiology, regularization, combined
    gains.remove()


if __name__ == '__main__':
    main()
