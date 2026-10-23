"""Read-only calibration of the frozen rePPG teacher on real skin-masked video."""

import argparse
import os

import torch
from torch.utils.data._utils.collate import default_collate

from modules.rppg import RePPGTeacherAdapter, physiological_losses
from rppg_gfvc_dataset import GFVCRPPGWindowDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/val')
    parser.add_argument('--reppg_root', default='/opt/data/private/project/rePPG-Rewriting-Photoplethysmography-Signal-to-Video-master')
    parser.add_argument('--rppg_checkpoint', required=True)
    parser.add_argument('--subjects', type=int, default=5)
    args = parser.parse_args()

    dataset = GFVCRPPGWindowDataset(
        args.data_root, rewrite_probability=0.0, deterministic=True,
    )
    teacher = RePPGTeacherAdapter(args.reppg_root, args.rppg_checkpoint).cuda().eval()
    selected, seen = [], set()
    for index, (face_path, _, _, _) in enumerate(dataset.samples):
        subject = os.path.basename(face_path).split('_clip')[0]
        if subject not in seen:
            selected.append((subject, index))
            seen.add(subject)
        if len(selected) >= args.subjects:
            break

    measurements = {'bgr': [], 'rgb': []}
    with torch.no_grad():
        for subject, index in selected:
            batch = default_collate([dataset[index]])
            video = (batch['driving_video'] * batch['skin_masks']).cuda()
            target = batch['original_rppg'].cuda()
            fps = batch['fps'].cuda()
            result = {}
            for color in ('bgr', 'rgb'):
                # The adapter converts RGB to BGR; reversing first tests the
                # opposite convention without altering the checkpoint.
                model_input = video if color == 'bgr' else video[:, :, [2, 1, 0]]
                extracted = teacher(model_input)
                losses = physiological_losses(extracted, target, fps)
                correlation = 1.0 - float(losses['pearson'])
                result[color] = correlation
                measurements[color].append(correlation)
            print('%s fps=%.2f cPPG_HR=%.1f corr_Bgr=%.3f corr_Rgb=%.3f' %
                  (subject, float(fps[0]), float(batch['target_bpm'][0]),
                   result['bgr'], result['rgb']), flush=True)
    for color, values in measurements.items():
        print('%s mean_pearson=%.3f' % (color, sum(values) / len(values)))


if __name__ == '__main__':
    main()
