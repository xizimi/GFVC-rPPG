"""Curate processed UBFC clips with pyVHR CHROM waveform agreement.

Input samples follow ``preprocess_ubfc2.py``: ``*.face.npy`` contains BGR
uint8 frames and the matching ``*.sig.npy`` is a dictionary whose
``rppg_signal`` is frame-aligned ground truth.  For every clip this script:

1. losslessly renders the frames to a temporary FFV1 video;
2. extracts a full-clip CHROM BVP with the actual pyVHR Pipeline;
3. robustly combines valid patch waveforms by their pointwise median;
4. computes zero-lag, signed Pearson correlation against the original BVP.

No lag search, polarity flip, or reference-guided patch selection is used.
When ``--output-root`` is supplied, accepted face/mask files are copied (or
hard-linked) unchanged and a new signal file is written with CHROM as
``rppg_signal``.  Without it, the command is evaluation-only.
"""

import argparse
import csv
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def zscore(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("non_finite_signal")
    std = float(values.std())
    if std < 1e-8:
        raise ValueError("constant_signal")
    return (values - values.mean()) / std


def pearson(prediction, target):
    prediction = zscore(prediction)
    target = zscore(target)
    if len(prediction) != len(target) or len(target) < 3:
        raise ValueError("length_mismatch")
    return float(np.mean(prediction * target))


def patch_waveforms(window, expected_length):
    """Normalize version-dependent pyVHR output to [patch, time]."""
    array = np.asarray(window, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    axes = [axis for axis, size in enumerate(array.shape)
            if size == expected_length]
    if not axes:
        raise ValueError("CHROM output has no %d-sample time axis: %s" %
                         (expected_length, array.shape))
    array = np.moveaxis(array, axes[-1], -1)
    return array.reshape(-1, expected_length)


def median_patch_waveform(window, expected_length):
    """Fuse patches without consulting the reference waveform."""
    patches = patch_waveforms(window, expected_length)
    valid = []
    for patch in patches:
        try:
            valid.append(zscore(patch))
        except ValueError:
            pass
    if not valid:
        raise ValueError("no_valid_chrom_patches")
    waveform = np.median(np.stack(valid), axis=0)
    # Store a consistently scaled label; this does not change PCC.
    waveform = zscore(waveform).astype(np.float32)
    return waveform, len(valid), len(patches)


def write_ffv1(path, frames, fps):
    import cv2

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("face array must have shape [T,H,W,3]")
    if frames.dtype != np.uint8:
        raise ValueError("face array must be uint8")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"),
                             float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV FFV1 writer is unavailable")
    for frame in frames:
        writer.write(frame)  # preprocessing arrays are already BGR
    writer.release()


def extract_chrom(face_path, fps, patch_size, min_hz, max_hz, cuda):
    from pyVHR.analysis.pipeline import Pipeline

    frames = np.load(face_path, mmap_mode="r", allow_pickle=False)
    if len(frames) < 3:
        raise ValueError("clip_too_short")
    Pipeline.minHz = float(min_hz)
    Pipeline.maxHz = float(max_hz)
    with tempfile.TemporaryDirectory(prefix="ubfc_chrom_") as directory:
        video_path = Path(directory) / "clip.avi"
        write_ffv1(video_path, frames, fps)
        bvps, centers, _ = Pipeline().run_on_video(
            str(video_path), winsize=len(frames) / float(fps),
            roi_method="convexhull", roi_approach="patches",
            method="cpu_CHROM", estimate="clustering",
            patch_size=int(patch_size), RGB_LOW_HIGH_TH=(5, 230),
            Skin_LOW_HIGH_TH=(5, 230), pre_filt=True, post_filt=True,
            cuda=bool(cuda), verb=False,
        )
    if len(bvps) != 1 or len(centers) != 1:
        raise ValueError("expected_one_full_clip_window_got_%d" % len(bvps))
    return median_patch_waveform(bvps[0], len(frames))


def load_signal(path):
    value = np.load(path, allow_pickle=True)
    if not (isinstance(value, np.ndarray) and value.shape == ()
            and isinstance(value.item(), dict)):
        raise ValueError("signal file must contain a dictionary")
    signal_dict = value.item()
    if "rppg_signal" not in signal_dict:
        raise ValueError("signal dictionary has no rppg_signal")
    return signal_dict


def link_or_copy(source, destination, copy_files):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        with open(source, "rb") as source_handle, open(destination, "xb") as output_handle:
            shutil.copyfileobj(source_handle, output_handle)
        return "copied"
    try:
        os.link(source, destination)
        return "hardlinked"
    except OSError:
        if destination.exists():
            raise
        with open(source, "rb") as source_handle, open(destination, "xb") as output_handle:
            shutil.copyfileobj(source_handle, output_handle)
        return "copied"


def export_sample(face_path, sig_path, signal_dict, chrom, correlation,
                  threshold, input_root, output_root, copy_files):
    relative_face = face_path.relative_to(input_root)
    output_face = output_root / relative_face
    output_sig = output_root / sig_path.relative_to(input_root)
    mask_path = Path(str(face_path).replace(".face.npy", ".skin.npy"))
    if not mask_path.exists():
        raise FileNotFoundError("missing skin mask: %s" % mask_path)
    output_mask = output_root / mask_path.relative_to(input_root)
    finals = (output_face, output_mask, output_sig)
    if any(path.exists() for path in finals):
        raise FileExistsError("output exists; use a new/empty output root: %s" %
                              output_face.parent)
    output_face.parent.mkdir(parents=True, exist_ok=True)
    new_signal = dict(signal_dict)
    new_signal["reference_rppg_signal"] = np.asarray(
        signal_dict["rppg_signal"], dtype=np.float64).copy()
    new_signal["rppg_signal"] = np.asarray(chrom, dtype=np.float32)
    new_signal["label_source"] = "pyVHR_cpu_CHROM_patch_median"
    new_signal["label_reference_pearson"] = float(correlation)
    new_signal["label_selection_threshold"] = float(threshold)
    new_signal["label_selection_rule"] = "signed_zero_lag_pearson_gt"
    signal_bytes = io.BytesIO()
    np.save(signal_bytes, new_signal, allow_pickle=True)
    installed = []
    try:
        transfer = link_or_copy(face_path, output_face, copy_files)
        installed.append(output_face)
        link_or_copy(mask_path, output_mask, copy_files)
        installed.append(output_mask)
        with output_sig.open("xb") as handle:
            installed.append(output_sig)
            handle.write(signal_bytes.getvalue())
        return transfer
    except Exception:
        for path in installed:
            if path.exists():
                path.unlink()
        raise


def plot_example(path, target, chrom, fps, correlation, sample):
    """Save one real accepted label/CHROM overlay for manual inspection."""
    import matplotlib.pyplot as plt

    target = zscore(target)
    chrom = zscore(chrom)
    seconds = np.arange(len(target), dtype=np.float64) / float(fps)
    figure, axis = plt.subplots(figsize=(12, 4.8), dpi=170)
    axis.plot(seconds, target, color="#222222", linewidth=1.7,
              label="Original UBFC BVP label")
    axis.plot(seconds, chrom, color="#0072B2", linewidth=1.45,
              label="pyVHR CHROM (patch median)")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Z-scored amplitude")
    axis.set_title("%s | signed zero-lag PCC = %.4f" %
                   (sample, correlation))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), bbox_inches="tight")
    plt.close(figure)


def evaluate_one(face_path, input_root, args):
    sig_path = Path(str(face_path).replace(".face.npy", ".sig.npy"))
    row = {"sample": face_path.relative_to(input_root).as_posix(),
           "face": str(face_path), "signal": str(sig_path),
           "threshold": float(args.threshold), "selection_passed": False,
           "accepted": False, "export_status": "not_requested",
           "status": "error"}
    try:
        if not sig_path.exists():
            raise FileNotFoundError("missing_signal_file")
        signal_dict = load_signal(sig_path)
        target = np.asarray(signal_dict["rppg_signal"], dtype=np.float64).reshape(-1)
        frames = np.load(face_path, mmap_mode="r", allow_pickle=False)
        frame_count = len(frames)
        mask_path = Path(str(face_path).replace(".face.npy", ".skin.npy"))
        if not mask_path.exists():
            raise FileNotFoundError("missing_skin_mask")
        if frames.shape[1:3] != (256, 256):
            raise ValueError("expected_256x256_frames_got_%s" %
                             (frames.shape[1:3],))
        if frame_count != len(target):
            raise ValueError("frame_signal_length_mismatch_%d_%d" %
                             (frame_count, len(target)))
        relative_parts = face_path.relative_to(input_root).parts
        expected_by_split = {
            "train": getattr(args, "expected_train_frames", 150),
            "val": getattr(args, "expected_val_frames", 150),
            "test": getattr(args, "expected_test_frames", 360),
        }
        if relative_parts and relative_parts[0] in expected_by_split:
            expected = expected_by_split[relative_parts[0]]
            if frame_count != expected:
                raise ValueError("%s_split_requires_%d_frames_got_%d" %
                                 (relative_parts[0], expected, frame_count))
        # Close the mmap before atomic hard-link renames (required on Windows).
        del frames
        fps = float(signal_dict.get("fps", args.default_fps))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("invalid_fps")
        chrom, valid_patches, patch_count = extract_chrom(
            face_path, fps, args.patch_size, args.min_hz, args.max_hz,
            args.cuda)
        correlation = pearson(chrom, target)
        accepted = correlation > args.threshold
        row.update({"fps": fps, "frames": frame_count,
                    "pearson": correlation, "abs_pearson": abs(correlation),
                    "valid_patch_count": valid_patches,
                    "patch_count": patch_count,
                    "selection_passed": accepted, "accepted": accepted,
                    "status": "accepted" if accepted else "below_threshold"})
        if accepted and args.output_root is not None:
            row["export_status"] = "writing"
            row["face_transfer"] = export_sample(
                face_path, sig_path, signal_dict, chrom, correlation,
                args.threshold, input_root, args.output_root, args.copy_files)
            row["export_status"] = "ok"
        if accepted and args.example_plot is not None and not args.example_written:
            try:
                plot_example(args.example_plot, target, chrom, fps, correlation,
                             row["sample"])
                row["example_plot"] = str(args.example_plot)
                args.example_written = True
            except Exception as plot_error:
                row["plot_error"] = "%s: %s" % (type(plot_error).__name__, plot_error)
    except Exception as error:
        if row.get("selection_passed"):
            row["accepted"] = False
            row["status"] = "export_error"
            row["export_status"] = "error"
        row["reason"] = "%s: %s" % (type(error).__name__, error)
    return row


def write_reports(rows, json_path, csv_path, args):
    valid = [row for row in rows if row.get("pearson") is not None]
    selected = [row for row in rows if row.get("selection_passed")]
    accepted = [row for row in rows if row["accepted"]]
    payload = {
        "configuration": {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root) if args.output_root else None,
            "threshold": args.threshold, "criterion": "signed_pearson_gt",
            "method": "pyVHR cpu_CHROM", "aggregation": "patch_zscore_median",
            "polarity_flip": False, "lag_alignment": False,
            "patch_size": args.patch_size, "min_hz": args.min_hz,
            "max_hz": args.max_hz,
        },
        "summary": {"discovered": len(rows), "evaluated": len(valid),
                    "selected": len(selected), "accepted": len(accepted),
                    "rejected_or_error": len(rows) - len(accepted),
                    "mean_pearson": (float(np.mean([r["pearson"] for r in valid]))
                                     if valid else None)},
        "samples": rows,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True,
                        help="JSON report path; CSV is written beside it")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="If set, materialize accepted clips here")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Accept when signed zero-lag Pearson > value")
    parser.add_argument("--patch-size", type=int, default=30)
    parser.add_argument("--min-hz", type=float, default=0.7)
    parser.add_argument("--max-hz", type=float, default=3.0)
    parser.add_argument("--default-fps", type=float, default=30.0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--copy-files", action="store_true",
                        help="Copy faces/masks instead of trying hard links")
    parser.add_argument("--example-plot", type=Path, default=None,
                        help="Save label/CHROM overlay for first accepted clip")
    parser.add_argument("--expected-train-frames", type=int, default=150)
    parser.add_argument("--expected-val-frames", type=int, default=150)
    parser.add_argument("--expected-test-frames", type=int, default=360)
    return parser


def main():
    args = build_parser().parse_args()
    args.example_written = False
    args.input_root = args.input_root.resolve()
    if args.output_root is not None:
        args.output_root = args.output_root.resolve()
    if not args.input_root.is_dir():
        raise ValueError("input root is not a directory: %s" % args.input_root)
    if not -1.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [-1, 1]")
    if args.patch_size <= 0 or args.min_hz <= 0 or args.max_hz <= args.min_hz:
        raise ValueError("invalid patch size or frequency band")
    faces = sorted(args.input_root.rglob("*.face.npy"))
    if not faces:
        raise ValueError("no *.face.npy clips found under %s" % args.input_root)
    rows = []
    for index, face_path in enumerate(faces, 1):
        row = evaluate_one(face_path, args.input_root, args)
        rows.append(row)
        print("[%d/%d] %s: %s%s" %
              (index, len(faces), row["sample"], row["status"],
               (" PCC=%.6f" % row["pearson"]
                if row.get("pearson") is not None else "")), flush=True)
    csv_path = args.report.with_suffix(".csv")
    write_reports(rows, args.report, csv_path, args)
    print("report: %s\ncsv: %s" % (args.report, csv_path))


if __name__ == "__main__":
    main()
