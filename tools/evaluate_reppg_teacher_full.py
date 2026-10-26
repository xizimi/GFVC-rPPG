"""Subject-balanced evaluation of a rePPG Hourglass checkpoint on UBFC."""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.rppg import RePPGTeacherAdapter, normalized_power_spectrum
from rppg_gfvc_dataset import GFVCRPPGWindowDataset, _prepare_waveform


def correlation(prediction, target):
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    return float((prediction * target).sum() /
                 (prediction.norm() * target.norm()).clamp_min(1e-8))


def peak_bpm(waveform, fps):
    spectrum, frequencies = normalized_power_spectrum(waveform, fps)
    index = spectrum.argmax(dim=-1)
    return float(frequencies[index].squeeze() * 60.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/val')
    parser.add_argument('--reppg_root', default='/opt/data/private/project/rePPG-Rewriting-Photoplethysmography-Signal-to-Video-master')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--max_windows_per_subject', type=int, default=0)
    args = parser.parse_args()
    dataset = GFVCRPPGWindowDataset(
        args.data_root, rewrite_probability=0.0,
        deterministic=True, require_semantic_masks=True)
    teacher = RePPGTeacherAdapter(args.reppg_root, args.checkpoint).cuda().eval()
    by_subject = defaultdict(list)
    count = defaultdict(int)
    with torch.no_grad():
        for index in range(len(dataset)):
            subject = os.path.basename(dataset.samples[index][0]).split('_clip')[0]
            if args.max_windows_per_subject and count[subject] >= args.max_windows_per_subject:
                continue
            sample = dataset[index]
            green = sample['driving_video'][:, 1:2]
            skin = sample['skin_masks']
            green = (green * skin).sum(dim=(1, 2, 3)) / skin.sum(
                dim=(1, 2, 3)).clamp_min(1.0)
            video = (sample['driving_video'] * sample['skin_masks']).unsqueeze(0).cuda()
            target = sample['original_rppg'].cuda()
            output = teacher(video)[0]
            fps = float(sample['fps'])
            target_bpm = float(sample['target_bpm'])
            if not np.isfinite(target_bpm):
                target_bpm = peak_bpm(target.unsqueeze(0), fps)
            estimate_bpm = peak_bpm(output.unsqueeze(0), fps)
            # Reflected green intensity usually has the opposite sign from
            # blood-volume cPPG; fix the sign before seeing evaluation scores.
            green_waveform, _ = _prepare_waveform(
                green.cpu().numpy(), fps, 'reppg')
            green_corr = correlation(
                -torch.as_tensor(green_waveform, device=target.device), target)
            by_subject[subject].append((correlation(output, target),
                                        abs(estimate_bpm - target_bpm), green_corr))
            count[subject] += 1
    subject_scores = []
    for subject in sorted(by_subject):
        values = np.asarray(by_subject[subject])
        score = values.mean(axis=0)
        subject_scores.append(score)
        print('%s windows=%d pearson=%.3f hr_mae=%.2f neg_green_pearson=%.3f' %
              (subject, len(values), score[0], score[1], score[2]), flush=True)
    scores = np.asarray(subject_scores).mean(axis=0)
    print('subjects=%d windows=%d mean_pearson=%.3f subject_mean_hr_mae=%.2f '
          'neg_green_mean_pearson=%.3f' %
          (len(by_subject), sum(count.values()), scores[0], scores[1], scores[2]),
          flush=True)


if __name__ == '__main__':
    main()
