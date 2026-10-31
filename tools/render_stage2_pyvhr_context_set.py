"""Render 480-frame context clips for fair central-360 pyVHR evaluation.

The CHROM-PCC075 labels were built by extracting CHROM from 480 frames and
retaining the central 360.  This tool preserves that temporal context at model
input and pyVHR evaluation time, while declaring the central 360-frame metric
interval in the resulting manifest.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from render_stage2_pyvhr_set import as_uint8, write_ffv1
from render_stage2_example import build_model
from rppg_gfvc_dataset import GFVCRPPGWindowDataset
from train_gfvc_rppg import move_batch


def sample_map(dataset):
    return {os.path.relpath(path, dataset.root_dir).replace(os.sep, "/"): index
            for index, (path, _, _, _) in enumerate(dataset.samples)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-root", default="/opt/data/private/datasets/UBFC2_CHROM_PCC075/test")
    parser.add_argument("--context-root", default="/opt/data/private/datasets/UBFC2_PCC075_candidates/test")
    parser.add_argument("--context-signal-manifest", required=True,
                        help="JSON mapping sample_id to precomputed 480-frame CHROM .npy")
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--context-frames", type=int, default=60)
    parser.add_argument("--target-frames", type=int, default=360)
    parser.add_argument("--teacher-checkpoint", default="checkpoint/skin_rppg_teacher.pt")
    parser.add_argument("--gfvc-checkpoint", default="checkpoint/00000189-checkpoint.pth.tar")
    parser.add_argument("--config", default="config/vox-256.yaml")
    parser.add_argument("--total-epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rewrite-seed", type=int, default=None,
                        help="If set, inject a reproducibly shuffled, non-self test BVP into each source video.")
    args = parser.parse_args()
    if args.context_frames < 0 or args.target_frames < 3:
        parser.error("Invalid context or target frame count")
    indices = [int(value) for value in args.indices.split(",") if value]
    if not indices or len(indices) != len(set(indices)):
        parser.error("indices must be nonempty and unique")
    total_frames = args.target_frames + 2 * args.context_frames
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, saved = build_model(args, device)
    target_dataset = GFVCRPPGWindowDataset(
        args.target_root, sequence_length=args.target_frames,
        rewrite_probability=0.0, require_semantic_masks=True,
        signal_preprocess="none", deterministic=True, seed=0)
    context_dataset = GFVCRPPGWindowDataset(
        args.context_root, sequence_length=total_frames,
        rewrite_probability=0.0, require_semantic_masks=True,
        signal_preprocess="none", deterministic=True, seed=0)
    context_indices = sample_map(context_dataset)
    with open(args.context_signal_manifest, encoding="utf-8") as handle:
        context_signals = json.load(handle)
    selected_ids = [target_dataset[index]["sample_id"] for index in indices]
    donor_for = {}
    if args.rewrite_seed is not None:
        if len(selected_ids) < 2:
            raise ValueError("Rewrite evaluation needs at least two selected samples")
        rng = np.random.RandomState(args.rewrite_seed)
        donor_ids = list(selected_ids)
        # A derangement prevents the preserve case from leaking into rewrite.
        for _ in range(100):
            rng.shuffle(donor_ids)
            if all(source != donor for source, donor in zip(selected_ids, donor_ids)):
                break
        else:
            donor_ids = selected_ids[1:] + selected_ids[:1]
        donor_for = dict(zip(selected_ids, donor_ids))
    rows = []
    for count, index in enumerate(indices, 1):
        if not 0 <= index < len(target_dataset):
            raise IndexError("Target index out of range: %d" % index)
        target_raw = target_dataset[index]
        sample_id = target_raw["sample_id"]
        if sample_id not in context_indices:
            raise FileNotFoundError("No 480-frame context clip for %s" % sample_id)
        context_raw = context_dataset[context_indices[sample_id]]
        target_sig = target_dataset.samples[index][1]
        signal_data = np.load(target_sig, allow_pickle=True).item()
        fps = float(signal_data["fps"])
        if sample_id not in context_signals:
            raise FileNotFoundError("No precomputed context signal for %s" % sample_id)
        source_chrom = np.load(context_signals[sample_id], allow_pickle=False)
        if len(source_chrom) != total_frames:
            raise ValueError("Context CHROM length mismatch for %s" % sample_id)
        saved_core = np.asarray(signal_data["rppg_signal"], dtype=np.float32)
        core = source_chrom[args.context_frames:args.context_frames + args.target_frames]
        replay_pcc = float(np.corrcoef(core, saved_core)[0, 1])
        if replay_pcc < 0.99999:
            raise RuntimeError("CHROM label replay mismatch %.8f for %s" %
                               (replay_pcc, sample_id))
        donor_id = donor_for.get(sample_id, sample_id)
        full_chrom = np.load(context_signals[donor_id], allow_pickle=False)
        if len(full_chrom) != total_frames:
            raise ValueError("Donor context CHROM length mismatch for %s" % donor_id)
        batch = move_batch(next(iter(DataLoader([context_raw], batch_size=1))), device)
        condition = torch.from_numpy(full_chrom.astype(np.float32)).unsqueeze(0).to(device)
        batch["target_rppg"] = condition
        batch["original_rppg"] = condition.clone()
        _, output = model(batch, epoch=int(saved["epoch"]) - 1,
                          total_epochs=args.total_epochs)
        name = target_raw["name"].replace(".face.npy", "")
        prefix = "%04d_%s" % (index, name)
        input_path = os.path.join(args.output_dir, prefix + "_input_context480.avi")
        generated_path = os.path.join(args.output_dir, prefix + "_generated_context480.avi")
        target_path = os.path.join(args.output_dir, prefix + "_chrom_context480.npy")
        original_path = os.path.join(args.output_dir, prefix + "_source_chrom_context480.npy")
        write_ffv1(input_path, as_uint8(batch["driving_video"][0]), fps)
        write_ffv1(generated_path, as_uint8(output["prediction"][0]), fps)
        np.save(target_path, full_chrom.astype(np.float32))
        np.save(original_path, source_chrom.astype(np.float32))
        rows.append({
            "sample": "%04d_%s" % (index, name),
            "subject": sample_id.split("/", 1)[0],
            "mode": "rewrite" if args.rewrite_seed is not None else "preserve",
            "input_video": input_path, "generated_video": generated_path,
            "target_signal": target_path, "original_signal": original_path,
            "fps": fps, "metric_start_frame": args.context_frames,
            "metric_length": args.target_frames,
            "label_replay_pcc": replay_pcc,
            "source_sample_id": sample_id, "donor_sample_id": donor_id,
        })
        print("%d/%d: %s replay PCC %.8f" %
              (count, len(indices), sample_id, replay_pcc), flush=True)
    with open(args.manifest, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print("manifest: %s (%d rows)" % (args.manifest, len(rows)))


if __name__ == "__main__":
    main()
