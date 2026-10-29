"""Plot pyVHR POS/CHROM BVPs from a generated video against its target BVP.

The figure is qualitative only.  POS and CHROM preserve their native polarity;
they are not sign-flipped or time-shifted to improve their apparent agreement.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pyvhr_bvp import (load_signal, patch_waveforms, reference_for,
                                run_pyvhr, video_info, zscore)


def median_patch_bvp(window, length):
    """Equal-weight robust aggregation of native-polarity patch BVPs."""
    patches = patch_waveforms(window, length)
    valid = []
    for patch in patches:
        try:
            valid.append(zscore(patch))
        except ValueError:
            continue
    if not valid:
        raise RuntimeError("No nonconstant pyVHR patch BVPs")
    return np.median(np.stack(valid), axis=0), len(valid), len(patches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--mode", choices=("preserve", "rewrite"), required=True)
    parser.add_argument("--video-kind", choices=("input", "generated"),
                        default="generated")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=30)
    parser.add_argument("--min-hz", type=float, default=0.7)
    parser.add_argument("--max-hz", type=float, default=3.0)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    row = next((item for item in rows if item["sample"] == args.sample
                and item["mode"] == args.mode), None)
    if row is None:
        raise ValueError("Sample/mode not present in manifest")
    signal_path, unavailable_reason = reference_for(row, args.video_kind)
    if unavailable_reason:
        raise ValueError(unavailable_reason)
    video = Path(row[args.video_kind + "_video"])
    target = load_signal(signal_path)
    fps, frames = video_info(video)
    if frames != len(target):
        raise ValueError("Generated-video/target length mismatch")
    if abs(fps - float(row["fps"])) > 0.05:
        raise ValueError("Generated-video/manifest FPS mismatch")

    seconds = np.arange(frames) / fps
    target_z = zscore(target)
    curves = {}
    for method in ("cpu_POS", "cpu_CHROM"):
        bvps, centers = run_pyvhr(video, method, args.patch_size, args.min_hz,
                                  args.max_hz, frames / fps)
        if len(bvps) != 1 or len(centers) != 1:
            raise RuntimeError("Expected one full-clip pyVHR BVP window")
        bvp, valid_count, patch_count = median_patch_bvp(bvps[0], frames)
        curves[method] = (bvp, valid_count, patch_count)

    figure, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, dpi=170)
    for axis, method, color, label in (
            (axes[0], "cpu_POS", "#0072b2", "POS"),
            (axes[1], "cpu_CHROM", "#d55e00", "CHROM")):
        bvp, valid_count, patch_count = curves[method]
        axis.plot(seconds, target_z, color="#222222", linewidth=1.7,
                  label="Injected target BVP")
        axis.plot(seconds, bvp, color=color, linewidth=1.35,
                  label="pyVHR %s (patch median)" % label)
        axis.set_ylabel("z-scored amplitude")
        axis.set_title("%s: %d/%d valid convex-hull patches; native polarity preserved" %
                       (label, valid_count, patch_count))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    reference_label = ("Injected target BVP" if args.mode == "rewrite"
                       and args.video_kind == "generated" else "Ground-truth input BVP")
    for axis in axes:
        axis.get_legend().get_texts()[0].set_text(reference_label)
    figure.suptitle("%s video vs reference BVP | %s | %s" %
                    (args.video_kind.capitalize(), args.sample,
                     args.mode.capitalize()), y=0.995)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
