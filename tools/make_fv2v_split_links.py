#!/usr/bin/env python3
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed",
        default="/opt/data/private/datasets/UBFC2_processed",
    )
    parser.add_argument(
        "--splits",
        default="/opt/data/private/project/One-Shot_Free-View_Neural_Talking_Head_Synthesis-main/splits.json",
    )
    parser.add_argument(
        "--out",
        default="/opt/data/private/datasets/UBFC2_fv2v",
    )
    args = parser.parse_args()

    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    for split in ("train", "val", "test"):
        split_dir = os.path.join(args.out, split)
        os.makedirs(split_dir, exist_ok=True)
        for subject in splits[split]:
            meta_path = os.path.join(args.processed, subject, "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for window in meta["windows"]:
                clip = os.path.join(args.processed, subject, window["clip"])
                link_name = f"{subject}_{window['clip']}"
                # clip name in meta may be .face.npy from raw preprocessing
                assert link_name.endswith(".face.npy"), f"unexpected clip extension: {link_name}"
                link = os.path.join(split_dir, link_name)
                if os.path.lexists(link):
                    os.remove(link)
                os.symlink(clip, link)

    for split in ("train", "val", "test"):
        split_dir = os.path.join(args.out, split)
        print(split, len(os.listdir(split_dir)))


if __name__ == "__main__":
    main()
