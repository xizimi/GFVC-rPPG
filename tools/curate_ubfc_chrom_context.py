"""Build final UBFC CHROM labels from 60+360+60-frame context blocks.

CHROM sees all 480 frames. Only the central 360 frames are eligible data.
Test uses the complete central window; train/validation use 150-frame windows
with a 75-frame step. Every final window is independently required to have a
signed, zero-lag PCC strictly greater than the threshold.
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from curate_ubfc_chrom import extract_chrom, load_signal, pearson


def window_specs(split, target_frames=360, train_frames=150, train_step=75):
    if split == "test":
        return [(0, target_frames)]
    if split not in ("train", "val"):
        raise ValueError("unknown split: %s" % split)
    result, start = [], 0
    while start + train_frames <= target_frames:
        result.append((start, start + train_frames))
        start += train_step
    return result


def npy_bytes(value, allow_pickle=False):
    output = io.BytesIO()
    np.save(output, value, allow_pickle=allow_pickle)
    return output.getvalue()


def write_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except Exception:
        if path.exists():
            path.unlink()
        raise


def export_window(face_path, mask_path, signal_data, chrom, reference,
                  source_start, output_face, input_root, pcc, threshold,
                  block_pcc):
    output_mask = Path(str(output_face).replace(".face.npy", ".skin.npy"))
    output_sig = Path(str(output_face).replace(".face.npy", ".sig.npy"))
    outputs = (output_face, output_mask, output_sig)
    if any(path.exists() for path in outputs):
        raise FileExistsError("final output already exists: %s" % output_face)
    length = len(chrom)
    face = np.load(face_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    end = source_start + length
    if end > len(face) or end > len(mask):
        raise ValueError("source slice lies outside face/mask context")
    new_signal = {}
    for key, value in signal_data.items():
        if key in ("rppg_signal", "target_offset", "target_frames"):
            continue
        array = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else None
        if array is not None and array.ndim > 0 and len(array) == len(face):
            new_signal[key] = np.asarray(array[source_start:end]).copy()
        else:
            new_signal[key] = value
    new_signal.update({
        "rppg_signal": np.asarray(chrom, dtype=np.float32),
        "reference_rppg_signal": np.asarray(reference, dtype=np.float64),
        "label_source": "pyVHR_cpu_CHROM_context480_central360",
        "label_reference_pearson": float(pcc),
        "label_selection_threshold": float(threshold),
        "label_selection_rule": "signed_zero_lag_pearson_gt",
        "context_block_pearson": float(block_pcc),
        "source_context_sample": face_path.relative_to(input_root).as_posix(),
        "source_start_frame": int(source_start),
        "target_offset": 0,
        "target_frames": int(length),
    })
    payloads = (
        npy_bytes(np.asarray(face[source_start:end]), allow_pickle=False),
        npy_bytes(np.asarray(mask[source_start:end]), allow_pickle=False),
        npy_bytes(new_signal, allow_pickle=True),
    )
    installed = []
    try:
        for path, payload in zip(outputs, payloads):
            write_exclusive(path, payload)
            installed.append(path)
    except Exception:
        for path in installed:
            if path.exists():
                path.unlink()
        raise


def output_name(face_path, split, index):
    stem = face_path.name[:-len(".face.npy")]
    suffix = "" if split == "test" else "_w%02d" % index
    return stem + suffix + ".face.npy"


def evaluate_block(face_path, input_root, args):
    relative = face_path.relative_to(input_root)
    split = relative.parts[0] if relative.parts else "unknown"
    block = {"sample": relative.as_posix(), "split": split,
             "status": "error", "windows": []}
    try:
        if split not in ("train", "val", "test"):
            raise ValueError("context sample is not below train/val/test")
        signal_path = Path(str(face_path).replace(".face.npy", ".sig.npy"))
        mask_path = Path(str(face_path).replace(".face.npy", ".skin.npy"))
        if not signal_path.exists() or not mask_path.exists():
            raise FileNotFoundError("missing context signal or mask")
        data = load_signal(signal_path)
        face = np.load(face_path, mmap_mode="r", allow_pickle=False)
        mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        reference_full = np.asarray(data["rppg_signal"], dtype=np.float64).reshape(-1)
        expected = args.target_frames + 2 * args.context_frames
        if len(face) != expected or len(mask) != expected or len(reference_full) != expected:
            raise ValueError("context requires %d aligned frames" % expected)
        if face.shape[1:] != (256, 256, 3) or mask.shape[1:3] != (256, 256):
            raise ValueError("invalid context face or mask shape")
        if int(data.get("target_offset", -1)) != args.context_frames:
            raise ValueError("wrong target_offset metadata")
        if int(data.get("target_frames", -1)) != args.target_frames:
            raise ValueError("wrong target_frames metadata")
        fps = float(data["fps"])
        del face, mask
        chrom_full, valid_patches, patch_count = extract_chrom(
            face_path, fps, args.patch_size, args.min_hz, args.max_hz, args.cuda)
        center_start = args.context_frames
        center_end = center_start + args.target_frames
        block_chrom = chrom_full[center_start:center_end]
        block_reference = reference_full[center_start:center_end]
        block_pcc = pearson(block_chrom, block_reference)
        block.update({"fps": fps, "context_frames": expected,
                      "target_frames": args.target_frames,
                      "block_pearson": block_pcc,
                      "valid_patch_count": valid_patches,
                      "patch_count": patch_count, "status": "ok"})
        specs = window_specs(split, args.target_frames,
                             args.train_frames, args.train_step)
        for index, (start, end) in enumerate(specs):
            chrom = block_chrom[start:end]
            reference = block_reference[start:end]
            pcc = pearson(chrom, reference)
            selected = pcc > args.threshold
            row = {"sample": relative.as_posix(), "split": split,
                   "window": index, "target_start": start, "target_end": end,
                   "frames": end - start, "fps": fps,
                   "duration_seconds": (end - start) / fps,
                   "pearson": pcc, "threshold": args.threshold,
                   "selection_passed": selected, "accepted": selected,
                   "status": "accepted" if selected else "below_threshold",
                   "export_status": "not_selected" if not selected else "writing"}
            if selected:
                output_face = (args.output_root / relative.parent /
                               output_name(face_path, split, index))
                try:
                    export_window(
                        face_path, mask_path, data, chrom, reference,
                        center_start + start, output_face, input_root,
                        pcc, args.threshold, block_pcc)
                    row["output_face"] = str(output_face)
                    row["export_status"] = "ok"
                except Exception as error:
                    row.update({"accepted": False, "status": "export_error",
                                "export_status": "error",
                                "reason": "%s: %s" %
                                (type(error).__name__, error)})
            block["windows"].append(row)
    except Exception as error:
        block["reason"] = "%s: %s" % (type(error).__name__, error)
    return block


def write_reports(blocks, report_path, args):
    windows = [row for block in blocks for row in block["windows"]]
    payload = {
        "configuration": {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "threshold": args.threshold,
            "context_frames_each_side": args.context_frames,
            "target_frames": args.target_frames,
            "train_frames": args.train_frames,
            "train_step": args.train_step,
            "criterion": "signed_zero_lag_pearson_gt",
            "polarity_flip": False, "lag_alignment": False,
        },
        "summary": {
            "block_count": len(blocks),
            "valid_block_count": sum(block["status"] == "ok" for block in blocks),
            "window_count": len(windows),
            "selected_count": sum(row["selection_passed"] for row in windows),
            "accepted_count": sum(row["accepted"] for row in windows),
            "export_error_count": sum(row["status"] == "export_error" for row in windows),
        },
        "blocks": blocks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = report_path.with_suffix(".csv")
    fields = sorted(set().union(*(row.keys() for row in windows))) if windows else []
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(windows)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--context-frames", type=int, default=60)
    parser.add_argument("--target-frames", type=int, default=360)
    parser.add_argument("--train-frames", type=int, default=150)
    parser.add_argument("--train-step", type=int, default=75)
    parser.add_argument("--patch-size", type=int, default=30)
    parser.add_argument("--min-hz", type=float, default=0.7)
    parser.add_argument("--max-hz", type=float, default=3.0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    if not args.input_root.is_dir():
        parser.error("input root does not exist")
    if not -1 <= args.threshold <= 1:
        parser.error("threshold must be in [-1,1]")
    if any(value <= 0 for value in (args.context_frames, args.target_frames,
                                    args.train_frames, args.train_step,
                                    args.patch_size)):
        parser.error("frame counts, steps, and patch size must be positive")
    faces = sorted(args.input_root.rglob("*.face.npy"))
    if not faces:
        parser.error("no context face arrays found")
    blocks = []
    for index, face_path in enumerate(faces, 1):
        block = evaluate_block(face_path, args.input_root, args)
        blocks.append(block)
        accepted = sum(row.get("accepted", False) for row in block["windows"])
        print("[%d/%d] %s status=%s accepted=%d" %
              (index, len(faces), block["sample"], block["status"], accepted),
              flush=True)
    csv_path = write_reports(blocks, args.report, args)
    print("report: %s\ncsv: %s" % (args.report, csv_path))


if __name__ == "__main__":
    main()
