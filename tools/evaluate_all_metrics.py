"""Run physiological and visual evaluation once and combine their summaries."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent


def run(command, pythonpath=None):
    print("running: %s" % " ".join(str(value) for value in command), flush=True)
    environment = os.environ.copy()
    if pythonpath:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (str(pythonpath) if not existing else
                                      str(pythonpath) + os.pathsep + existing)
    subprocess.run([str(value) for value in command], check=True, env=environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pyvhr-python", type=Path, default=Path(sys.executable),
                        help="Python executable containing pyVHR dependencies")
    parser.add_argument("--pyvhr-pythonpath", type=Path,
                        help="Optional pyVHR source root prepended to PYTHONPATH")
    parser.add_argument("--visual-python", type=Path, default=Path(sys.executable),
                        help="Python executable containing torch and pyiqa")
    parser.add_argument("--methods", nargs="+", choices=("cpu_POS", "cpu_CHROM"),
                        default=("cpu_POS", "cpu_CHROM"))
    parser.add_argument("--video-kinds", nargs="+", choices=("input", "generated"),
                        default=("input", "generated"))
    parser.add_argument("--patch-size", type=int, default=30)
    parser.add_argument("--min-hz", type=float, default=0.7)
    parser.add_argument("--max-hz", type=float, default=3.0)
    parser.add_argument("--bpm-window-seconds", type=float, default=5.0)
    parser.add_argument("--visual-batch-size", type=int, default=16)
    parser.add_argument("--maniqa-batch-size", type=int, default=2)
    parser.add_argument("--pyiqa-weights-dir", type=Path,
                        help="Offline DISTS/LPIPS/MANIQA checkpoint directory")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--skip-fvd", action="store_true")
    parser.add_argument("--fvd-i3d-path", type=Path)
    parser.add_argument("--fvd-clip-frames", type=int, default=16)
    parser.add_argument("--fvd-clip-stride", type=int, default=16)
    parser.add_argument("--fvd-batch-size", type=int, default=4)
    parser.add_argument("--save-frame-scores", action="store_true")
    args = parser.parse_args()
    if not args.skip_fvd and args.fvd_i3d_path is None:
        parser.error("--fvd-i3d-path is required unless --skip-fvd is used")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bvp_detail = args.output_dir / "bvp_metrics_detail.json"
    bvp_summary = args.output_dir / "bvp_metrics_summary.json"
    visual_output = args.output_dir / "visual_metrics.json"
    combined_output = args.output_dir / "all_metrics.json"

    bvp_command = [
        args.pyvhr_python, TOOLS_DIR / "evaluate_pyvhr_bvp.py",
        "--manifest", args.manifest, "--output", bvp_detail,
        "--summary-output", bvp_summary, "--patch-size", args.patch_size,
        "--min-hz", args.min_hz, "--max-hz", args.max_hz,
        "--bpm-window-seconds", args.bpm_window_seconds,
        "--methods", *args.methods, "--video-kinds", *args.video_kinds,
    ]
    run(bvp_command, args.pyvhr_pythonpath)

    visual_command = [
        args.visual_python, TOOLS_DIR / "evaluate_visual_quality.py",
        "--manifest", args.manifest, "--output", visual_output,
        "--device", args.device, "--batch-size", args.visual_batch_size,
        "--maniqa-batch-size", args.maniqa_batch_size,
        "--frame-step", args.frame_step,
        "--fvd-clip-frames", args.fvd_clip_frames,
        "--fvd-clip-stride", args.fvd_clip_stride,
        "--fvd-batch-size", args.fvd_batch_size,
    ]
    if args.skip_fvd:
        visual_command.append("--skip-fvd")
    else:
        visual_command.extend(("--fvd-i3d-path", args.fvd_i3d_path))
    if args.save_frame_scores:
        visual_command.append("--save-frame-scores")
    if args.pyiqa_weights_dir is not None:
        visual_command.extend(("--pyiqa-weights-dir", args.pyiqa_weights_dir))
    run(visual_command)

    combined = {
        "manifest": str(args.manifest),
        "executables": {"pyvhr_python": str(args.pyvhr_python),
                        "visual_python": str(args.visual_python)},
        "physiological": json.loads(bvp_summary.read_text(encoding="utf-8")),
        "visual": json.loads(visual_output.read_text(encoding="utf-8")),
        "detail_files": {
            "physiological_windows": str(bvp_detail),
            "physiological_summary": str(bvp_summary),
            "visual": str(visual_output),
        },
    }
    combined_output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("combined metrics: %s" % combined_output)


if __name__ == "__main__":
    main()
