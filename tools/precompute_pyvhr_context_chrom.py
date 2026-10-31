"""Precompute 480-frame CHROM waveforms used by context-aware model rendering."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from curate_ubfc_chrom import extract_chrom


def valid_target_samples(root):
    rows = []
    for face in sorted(root.rglob("*.face.npy")):
        signal = Path(str(face).replace(".face.npy", ".sig.npy"))
        skin = Path(str(face).replace(".face.npy", ".skin.npy"))
        if not signal.exists() or not skin.exists():
            continue
        frames = np.load(face, mmap_mode="r").shape[0]
        data = np.load(signal, allow_pickle=True).item()
        if frames >= 360 and len(data["rppg_signal"]) >= 360:
            rows.append((face, signal))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--context-frames", type=int, default=60)
    parser.add_argument("--target-frames", type=int, default=360)
    args = parser.parse_args()
    rows = valid_target_samples(args.target_root)
    indices = [int(value) for value in args.indices.split(",") if value]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    expected = args.target_frames + 2 * args.context_frames
    for count, index in enumerate(indices, 1):
        target_face, target_sig = rows[index]
        sample_id = str(target_face.relative_to(args.target_root)).replace(os.sep, "/")
        context_face = args.context_root / sample_id
        if not context_face.exists():
            raise FileNotFoundError("Missing context face: %s" % context_face)
        context_frames = np.load(context_face, mmap_mode="r").shape[0]
        if context_frames != expected:
            raise ValueError("Expected %d context frames, got %d: %s" %
                             (expected, context_frames, context_face))
        data = np.load(target_sig, allow_pickle=True).item()
        chrom, valid, patches = extract_chrom(context_face, float(data["fps"]),
                                               30, 0.7, 3.0, True)
        target = np.asarray(data["rppg_signal"], dtype=np.float64)
        core = chrom[args.context_frames:args.context_frames + args.target_frames]
        replay_pcc = float(np.corrcoef(core, target)[0, 1])
        if replay_pcc < 0.99999:
            raise RuntimeError("Label replay mismatch %.8f: %s" % (replay_pcc, sample_id))
        output = args.output_dir / (sample_id.replace("/", "__").replace(".face.npy", ".npy"))
        np.save(output, chrom.astype(np.float32))
        mapping[sample_id] = str(output)
        print("%d/%d %s PCC %.8f (%d/%d patches)" %
              (count, len(indices), sample_id, replay_pcc, valid, patches), flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
