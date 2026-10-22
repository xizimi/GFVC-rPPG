#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/opt/data/private/datasets/UBFC2")
    parser.add_argument("--out", default="splits.json")
    parser.add_argument("--n_val", type=int, default=5)
    parser.add_argument("--n_test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    subjects = sorted(
        os.path.basename(p)
        for p in glob(os.path.join(args.root, "subject*"))
        if os.path.isdir(p)
    )
    if len(subjects) <= args.n_val + args.n_test:
        raise ValueError("Not enough subjects for the requested split")

    start = args.seed % len(subjects)
    rotated = subjects[start:] + subjects[:start]
    val = rotated[: args.n_val]
    test = rotated[args.n_val : args.n_val + args.n_test]
    train = rotated[args.n_val + args.n_test :]

    split = {
        "seed": args.seed,
        "n_subjects": len(subjects),
        "train": train,
        "val": val,
        "test": test,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(
        f"train={len(train)} val={len(val)} test={len(test)} "
        f"total={len(subjects)} out={args.out}"
    )


if __name__ == "__main__":
    main()
