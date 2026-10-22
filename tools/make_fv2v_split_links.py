#!/usr/bin/env python3
import argparse
import json
import os


def replace_link(target, link):
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(target, link)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', default='/opt/data/private/datasets/UBFC2_processed')
    parser.add_argument('--splits', default='splits.json')
    parser.add_argument('--out', default='/opt/data/private/datasets/UBFC2_fv2v')
    args = parser.parse_args()
    with open(args.splits, encoding='utf-8') as handle:
        splits = json.load(handle)

    for split in ('train', 'val', 'test'):
        split_dir = os.path.join(args.out, split)
        os.makedirs(split_dir, exist_ok=True)
        pairs = 0
        for subject in splits[split]:
            meta_path = os.path.join(args.processed, subject, 'meta.json')
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, encoding='utf-8') as handle:
                meta = json.load(handle)
            for window in meta['windows']:
                face_name = window['clip']
                signal_name = window['signal']
                face_target = os.path.join(args.processed, subject, face_name)
                signal_target = os.path.join(args.processed, subject, signal_name)
                if not os.path.exists(face_target) or not os.path.exists(signal_target):
                    continue
                replace_link(face_target, os.path.join(split_dir, '%s_%s' % (subject, face_name)))
                replace_link(signal_target, os.path.join(split_dir, '%s_%s' % (subject, signal_name)))
                skin_name = face_name.replace('.face.npy', '.skin.npy')
                skin_target = os.path.join(args.processed, subject, skin_name)
                if os.path.exists(skin_target):
                    replace_link(skin_target, os.path.join(split_dir, '%s_%s' % (subject, skin_name)))
                pairs += 1
        print(split, 'pairs', pairs)


if __name__ == '__main__':
    main()
