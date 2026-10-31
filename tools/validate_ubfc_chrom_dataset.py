"""Validate the split-aware UBFC CHROM dataset without modifying it."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left) != len(right) or len(left) < 3:
        raise ValueError("length_mismatch")
    left = (left - left.mean()) / left.std()
    right = (right - right.mean()) / right.std()
    return float(np.mean(left * right))


def validate(root, threshold=0.75, expected=None, candidate_root=None):
    expected = expected or {"train": 150, "val": 150, "test": 360}
    errors, rows = [], []
    counts = Counter()
    durations = defaultdict(list)
    subject_splits = defaultdict(set)
    for face_path in sorted(root.rglob("*.face.npy")):
        relative = face_path.relative_to(root)
        split = relative.parts[0] if relative.parts else "unknown"
        sig_path = Path(str(face_path).replace(".face.npy", ".sig.npy"))
        mask_path = Path(str(face_path).replace(".face.npy", ".skin.npy"))
        sample_errors = []
        face = None
        mask = None
        if split not in expected:
            sample_errors.append("unknown_split")
        if not sig_path.exists():
            sample_errors.append("missing_signal")
        if not mask_path.exists():
            sample_errors.append("missing_mask")
        try:
            face = np.load(face_path, mmap_mode="r", allow_pickle=False)
            frames = int(face.shape[0])
            if face.ndim != 4 or face.shape[1:] != (256, 256, 3):
                sample_errors.append("invalid_face_shape_%s" % (face.shape,))
            if face.dtype != np.uint8:
                sample_errors.append("invalid_face_dtype_%s" % face.dtype)
        except Exception as error:
            frames = None
            sample_errors.append("invalid_face:%s" % error)
        data = None
        if sig_path.exists():
            try:
                data = np.load(sig_path, allow_pickle=True).item()
                signal = np.asarray(data["rppg_signal"]).reshape(-1)
                reference = np.asarray(data["reference_rppg_signal"]).reshape(-1)
                fps = float(data["fps"])
                pcc = float(data["label_reference_pearson"])
                if frames is not None and len(signal) != frames:
                    sample_errors.append("face_signal_length_mismatch")
                if not np.isfinite(signal).all() or signal.std() < 1e-8:
                    sample_errors.append("invalid_chrom_signal")
                valid_signal = (np.isfinite(signal).all() and signal.std() >= 1e-8)
                valid_reference = (len(reference) == len(signal) and
                                   np.isfinite(reference).all() and
                                   reference.std() >= 1e-8)
                if not valid_reference:
                    sample_errors.append("invalid_reference_signal")
                elif valid_signal:
                    recomputed = pearson(signal, reference)
                    if abs(recomputed - pcc) > 1e-5:
                        sample_errors.append("stored_pcc_mismatch_%.8f_%.8f" %
                                             (pcc, recomputed))
                    if not recomputed > threshold:
                        sample_errors.append("recomputed_pcc_not_strictly_above_threshold")
                if not np.isfinite(fps) or fps <= 0:
                    sample_errors.append("invalid_fps")
                if not pcc > threshold:
                    sample_errors.append("pcc_not_strictly_above_threshold")
                if data.get("label_source") != "pyVHR_cpu_CHROM_context480_central360":
                    sample_errors.append("wrong_label_source")
                if data.get("label_selection_rule") != "signed_zero_lag_pearson_gt":
                    sample_errors.append("wrong_selection_rule")
                if float(data.get("label_selection_threshold", np.nan)) != threshold:
                    sample_errors.append("wrong_selection_threshold")
                for name in ("time", "hr"):
                    if name in data and len(np.asarray(data[name]).reshape(-1)) != len(signal):
                        sample_errors.append("%s_length_mismatch" % name)
            except Exception as error:
                sample_errors.append("invalid_signal:%s" % error)
        if mask_path.exists() and frames is not None:
            try:
                mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
                mask_frames = int(mask.shape[0])
                if mask_frames != frames:
                    sample_errors.append("face_mask_length_mismatch")
                if mask.ndim not in (3, 4) or mask.shape[1:3] != (256, 256):
                    sample_errors.append("invalid_mask_shape_%s" % (mask.shape,))
                if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
                    sample_errors.append("invalid_mask_values")
            except Exception as error:
                sample_errors.append("invalid_mask:%s" % error)
        if split in expected and frames is not None and frames != expected[split]:
            sample_errors.append("expected_%d_frames_got_%d" %
                                 (expected[split], frames))
        if data is not None and frames is not None:
            fps_value = float(data.get("fps", np.nan))
            if np.isfinite(fps_value) and fps_value > 0:
                durations[split].append(frames / fps_value)
        counts[split] += 1
        subject = (relative.parts[1] if len(relative.parts) >= 3
                   else relative.name.split("_clip")[0])
        subject_splits[subject].add(split)
        if candidate_root is not None:
            source_sample = data.get("source_context_sample") if data else None
            source_start = int(data.get("source_start_frame", -1)) if data else -1
            candidate_face = candidate_root / source_sample if source_sample else None
            candidate_mask = (Path(str(candidate_face).replace(".face.npy", ".skin.npy"))
                              if candidate_face else None)
            if (candidate_face is None or not candidate_face.exists() or
                    source_start < 0 or frames is None):
                sample_errors.append("invalid_candidate_source_metadata")
            else:
                candidate_face_values = np.load(candidate_face, mmap_mode="r",
                                                allow_pickle=False)
                expected_face = np.asarray(
                    candidate_face_values[source_start:source_start + frames])
                if face is None or not np.array_equal(np.asarray(face), expected_face):
                    sample_errors.append("face_differs_from_candidate_slice")
            if (candidate_mask is None or not candidate_mask.exists() or
                    source_start < 0 or frames is None or not mask_path.exists()):
                sample_errors.append("invalid_candidate_mask_source")
            else:
                candidate_mask_values = np.load(candidate_mask, mmap_mode="r",
                                                allow_pickle=False)
                expected_mask = np.asarray(
                    candidate_mask_values[source_start:source_start + frames])
                if mask is None or not np.array_equal(np.asarray(mask), expected_mask):
                    sample_errors.append("mask_differs_from_candidate_slice")
        row = {"sample": relative.as_posix(), "split": split,
               "frames": frames, "errors": sample_errors}
        rows.append(row)
        errors.extend("%s: %s" % (relative.as_posix(), item)
                      for item in sample_errors)
    for split in expected:
        if counts[split] == 0:
            errors.append("split_has_no_samples: %s" % split)
    for subject, splits in subject_splits.items():
        if len(splits) > 1:
            errors.append("subject_in_multiple_splits: %s %s" %
                          (subject, sorted(splits)))
    face_bases = {str(path)[:-len(".face.npy")] for path in root.rglob("*.face.npy")}
    for suffix in (".sig.npy", ".skin.npy"):
        for path in root.rglob("*" + suffix):
            if str(path)[:-len(suffix)] not in face_bases:
                errors.append("orphan_file: %s" % path.relative_to(root).as_posix())
    summary = {
        "root": str(root), "threshold": threshold,
        "expected_frames": expected, "sample_counts": dict(counts),
        "mean_duration_seconds": {
            split: float(np.mean(values)) for split, values in durations.items()
            if values
        },
        "error_count": len(errors), "valid": bool(rows) and not errors,
    }
    return {"summary": summary, "errors": errors, "samples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-root", type=Path, default=None,
                        help="Optionally verify exported face/mask bytes against candidates")
    args = parser.parse_args()
    candidate_root = args.candidate_root.resolve() if args.candidate_root else None
    report = validate(args.root.resolve(), args.threshold,
                      candidate_root=candidate_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if not report["summary"]["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
