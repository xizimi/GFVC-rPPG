"""Render a fixed set of GFVC-rPPG windows for independent pyVHR evaluation.

Streams are FFV1 AVI: every decoded BGR frame is byte-identical to the uint8
model input/output submitted to the writer.  The accompanying BVP files are
frame-aligned references used by ``evaluate_pyvhr_bvp.py``.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rppg_gfvc_dataset import GFVCRPPGWindowDataset
from render_stage2_example import build_model
from train_gfvc_rppg import move_batch


def write_ffv1(path, frames, fps):
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), fps,
                             (width, height))
    if not writer.isOpened():
        raise RuntimeError("FFV1 writer unavailable; no lossy fallback is used")
    for frame in frames:
        # Model tensors carry the dataset's BGR channel order. This conversion
        # makes OpenCV's decoded BGR bytes exactly match the source tensor.
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def as_uint8(video):
    return (video.permute(0, 2, 3, 1).detach().cpu().numpy().clip(0, 1) * 255.0
            ).round().astype(np.uint8)


@torch.no_grad()
def render_mode(model, args, device, mode, indices):
    dataset = GFVCRPPGWindowDataset(
        args.split_root, sequence_length=150,
        rewrite_probability=1.0 if mode == "rewrite" else 0.0,
        require_semantic_masks=True, signal_preprocess="reppg",
        deterministic=True, seed=0,
    )
    rows = []
    for index in indices:
        if not 0 <= index < len(dataset):
            raise IndexError("Dataset index out of range: %d" % index)
        raw = dataset[index]
        batch = move_batch(next(iter(DataLoader([raw], batch_size=1))), device)
        _, output = model(batch, epoch=args.checkpoint_epoch - 1,
                          total_epochs=args.total_epochs)
        name = raw["name"].replace(".face.npy", "")
        prefix = "%s_%04d_%s" % (mode, index, name)
        input_path = os.path.join(args.output_dir, prefix + "_input.avi")
        generated_path = os.path.join(args.output_dir, prefix + "_generated.avi")
        target_path = os.path.join(args.output_dir, prefix + "_target_bvp.npy")
        original_path = os.path.join(args.output_dir, prefix + "_original_bvp.npy")
        write_ffv1(input_path, as_uint8(batch["driving_video"][0]),
                   float(batch["fps"][0]))
        write_ffv1(generated_path, as_uint8(output["prediction"][0]),
                   float(batch["fps"][0]))
        np.save(target_path, batch["target_rppg"][0].detach().cpu().numpy())
        np.save(original_path, batch["original_rppg"][0].detach().cpu().numpy())
        rows.append({
            "sample": "%04d_%s" % (index, name), "mode": mode,
            "input_video": input_path, "generated_video": generated_path,
            "target_signal": target_path, "original_signal": original_path,
            "fps": float(batch["fps"][0]),
        })
        print("%s %d/%d: %s" % (mode, len(rows), len(indices), name), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--indices", required=True,
                        help="Comma-separated, fixed dataset indices")
    parser.add_argument("--modes", nargs="+", default=("preserve", "rewrite"),
                        choices=("preserve", "rewrite"))
    parser.add_argument("--split-root", default="/opt/data/private/datasets/UBFC2_fixed256_fv2v/test")
    parser.add_argument("--teacher-checkpoint", default="checkpoint/skin_rppg_teacher.pt")
    parser.add_argument("--gfvc-checkpoint", default="checkpoint/00000189-checkpoint.pth.tar")
    parser.add_argument("--config", default="config/vox-256.yaml")
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",") if value]
    if not indices or len(indices) != len(set(indices)):
        parser.error("indices must be nonempty and unique")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, saved = build_model(args, device)
    args.checkpoint_epoch = int(saved["epoch"])
    rows = []
    for mode in args.modes:
        rows.extend(render_mode(model, args, device, mode, indices))
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print("manifest: %s (%d rows)" % (args.manifest, len(rows)))


if __name__ == "__main__":
    main()
