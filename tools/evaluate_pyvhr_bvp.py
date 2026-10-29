"""Evaluate frame-aligned BVP waveforms extracted by the actual pyVHR pipeline.

The manifest is a JSON list. Every row requires ``sample``, ``mode``
(``preserve`` or ``rewrite``), ``input_video``, ``generated_video``,
``target_signal`` and ``fps``. ``original_signal`` is optional and is used for
the Input side of rewrite experiments. Signals are one-dimensional ``.npy``
arrays with one value per video frame.

The script reports patch-median BVP metrics. It never selects a patch by its
correlation with the reference and never shifts a signal for PCC or Z-MAE.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = ("pcc", "abs_pcc", "z_mae", "mcc_abs", "abs_mcc_lag_frames",
           "abs_mcc_lag_seconds")


def zscore(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("non_finite_signal")
    std = values.std()
    if std < 1e-8:
        raise ValueError("constant_signal")
    return (values - values.mean()) / std


def waveform_metrics(prediction, target, fps, max_lag_seconds=0.2):
    """Return waveform metrics without changing the zero-lag main metrics.

    Positive MCC lag means that prediction lags target: correlation is computed
    between ``prediction[lag:]`` and ``target[:-lag]``. MCC searches absolute
    correlation in the bounded range, while retaining the original sign.
    """
    try:
        prediction = zscore(prediction)
        target = zscore(target)
    except ValueError as error:
        return None, str(error)
    if len(prediction) != len(target) or len(target) < 3:
        return None, "length_mismatch"

    pcc = float(np.mean(prediction * target))
    max_lag = min(int(round(max_lag_seconds * fps)), len(target) - 2)
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            predicted_overlap, target_overlap = prediction[lag:], target[:-lag]
        elif lag < 0:
            predicted_overlap, target_overlap = prediction[:lag], target[-lag:]
        else:
            predicted_overlap, target_overlap = prediction, target
        predicted_std = predicted_overlap.std()
        target_std = target_overlap.std()
        if predicted_std < 1e-8 or target_std < 1e-8:
            continue
        correlation = float(np.mean(
            (predicted_overlap - predicted_overlap.mean()) / predicted_std
            * (target_overlap - target_overlap.mean()) / target_std))
        candidates.append((abs(correlation), -abs(lag), correlation, lag))
    if not candidates:
        return None, "constant_overlap"
    _, _, mcc_signed, mcc_lag = max(candidates)
    return {
        "pcc": pcc,
        "abs_pcc": abs(pcc),
        "z_mae": float(np.mean(np.abs(prediction - target))),
        "mcc_abs": abs(mcc_signed),
        "mcc_signed": mcc_signed,
        "mcc_lag_frames": int(mcc_lag),
        "mcc_lag_seconds": float(mcc_lag / fps),
        "abs_mcc_lag_frames": abs(int(mcc_lag)),
        "abs_mcc_lag_seconds": abs(float(mcc_lag / fps)),
    }, None


def load_signal(path):
    signal = np.load(path, allow_pickle=True)
    if isinstance(signal, np.ndarray) and signal.dtype == object and signal.shape == ():
        signal = signal.item()
    if isinstance(signal, dict):
        signal = signal["rppg_signal"]
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if not np.isfinite(signal).all():
        raise ValueError("Target BVP has NaN or Inf: %s" % path)
    return signal


def video_info(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Cannot open video: %s" % path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not np.isfinite(fps) or fps <= 0 or frames <= 0:
        raise ValueError("Invalid FPS or frame count: %s" % path)
    return fps, frames


def patch_waveforms(window, reference_length):
    """Normalize pyVHR's version-dependent BVP window shape to [patch, time]."""
    array = np.asarray(window, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    time_axes = [axis for axis, size in enumerate(array.shape) if size == reference_length]
    if not time_axes:
        raise ValueError("pyVHR BVP shape has no reference-length time axis: %s" %
                         (array.shape,))
    array = np.moveaxis(array, time_axes[-1], -1)
    return array.reshape(-1, array.shape[-1])


def run_pyvhr(video, method, patch_size, min_hz, max_hz, window_seconds):
    """Run pyVHR and return BVP windows and their center times.

    Current pyVHR Pipeline always estimates BPM after BVP extraction. The BPM
    return is deliberately discarded: it does not fuse the patch BVPs used here.
    """
    from pyVHR.analysis.pipeline import Pipeline

    Pipeline.minHz = min_hz
    Pipeline.maxHz = max_hz
    bvps, times_es, _ = Pipeline().run_on_video(
        str(video), winsize=window_seconds, roi_method="convexhull", roi_approach="patches",
        method=method, estimate="clustering", patch_size=patch_size,
        RGB_LOW_HIGH_TH=(5, 230), Skin_LOW_HIGH_TH=(5, 230),
        pre_filt=(method == "cpu_CHROM"), post_filt=True,
        cuda=True, verb=False,
    )
    return bvps, np.asarray(times_es, dtype=np.float64).reshape(-1)


def reference_for(row, video_kind):
    if row["mode"] == "preserve" or video_kind == "generated":
        return row["target_signal"], None
    if "original_signal" not in row:
        return None, "missing_original_signal_for_rewrite_input"
    return row["original_signal"], None


def empty_record(row, video_kind, method, reason):
    record = {"sample": row["sample"], "mode": row["mode"],
              "video_kind": video_kind, "method": method, "window": None,
              "center_seconds": None, "patch_count": 0,
              "valid_patch_count": 0, "status": reason}
    for metric in METRICS:
        record["median_" + metric] = None
    return record


def evaluate_video(row, video_kind, method, args):
    signal_path, unavailable_reason = reference_for(row, video_kind)
    if unavailable_reason:
        return [empty_record(row, video_kind, method, unavailable_reason)]
    target = load_signal(signal_path)
    video = Path(row[video_kind + "_video"])
    fps, frames = video_info(video)
    if abs(fps - float(row["fps"])) > 0.05:
        raise ValueError("Manifest/video FPS mismatch: %s" % video)
    if frames != len(target):
        raise ValueError("Video/BVP frame mismatch: %s (%d vs %d)" %
                         (video, frames, len(target)))
    # Use every frame in the rendered clip. This protects alignment when its
    # container reports a non-integer FPS (for example, 150 / 29.372 = 5.106 s).
    window_seconds = frames / fps

    bvps, times_es = run_pyvhr(video, method, args.patch_size,
                                args.min_hz, args.max_hz, window_seconds)
    if len(bvps) != len(times_es) or not len(bvps):
        return [empty_record(row, video_kind, method, "pyvhr_returned_no_windows")]
    records = []
    for window_index, (bvp_window, center) in enumerate(zip(bvps, times_es)):
        # pyVHR uses the complete clip window. Its actual BVP length must match
        # the clip frame count; no target samples are discarded.
        expected_length = frames
        expected_start = int(round(center * fps - expected_length / 2.0))
        if expected_start < 0 or expected_start + expected_length > len(target):
            raise ValueError("pyVHR window outside target BVP at %.6f s" % center)
        reference = target[expected_start:expected_start + expected_length]
        patches = patch_waveforms(bvp_window, expected_length)
        valid, failures = [], Counter()
        for patch in patches:
            metrics, reason = waveform_metrics(patch, reference, fps,
                                                args.max_lag_seconds)
            if metrics is None:
                failures[reason] += 1
            else:
                valid.append(metrics)
        record = {"sample": row["sample"], "mode": row["mode"],
                  "video_kind": video_kind, "method": method,
                  "window": window_index, "center_seconds": float(center),
                  "target_start_frame": expected_start,
                  "target_length": expected_length, "video_frame_count": frames,
                  "patch_count": int(len(patches)),
                  "valid_patch_count": int(len(valid)),
                  "invalid_patch_reasons": dict(failures),
                  "status": "ok" if valid else "no_valid_patches"}
        for metric in METRICS:
            record["median_" + metric] = (
                float(np.median([item[metric] for item in valid])) if valid else None)
        records.append(record)
    return records


def summary(records):
    """Aggregate first within a sample, then across samples with equal weight."""
    per_sample = defaultdict(list)
    for record in records:
        key = (record["sample"], record["mode"], record["video_kind"], record["method"])
        per_sample[key].append(record)
    sample_rows = []
    for key, windows in per_sample.items():
        row = {"sample": key[0], "mode": key[1], "video_kind": key[2], "method": key[3],
               "window_count": len(windows),
               "status": "ok" if any(w["status"] == "ok" for w in windows) else windows[0]["status"]}
        for metric in METRICS:
            values = [w["median_" + metric] for w in windows if w["median_" + metric] is not None]
            row["median_" + metric] = float(np.median(values)) if values else None
        sample_rows.append(row)

    groups = []
    grouped = defaultdict(list)
    for row in sample_rows:
        grouped[(row["mode"], row["video_kind"], row["method"])].append(row)
    for (mode, kind, method), rows in sorted(grouped.items()):
        item = {"mode": mode, "video_kind": kind, "method": method,
                "sample_count": len(rows),
                "valid_sample_count": sum(r["status"] == "ok" for r in rows),
                "invalid_sample_count": sum(r["status"] != "ok" for r in rows)}
        for metric in METRICS:
            values = [r["median_" + metric] for r in rows if r["median_" + metric] is not None]
            item[metric] = ({"mean": float(np.mean(values)), "std": float(np.std(values)),
                             "median": float(np.median(values)), "count": len(values)}
                            if values else None)
        groups.append(item)

    deltas = []
    preserve = defaultdict(dict)
    for row in sample_rows:
        if row["mode"] == "preserve":
            preserve[(row["sample"], row["method"])][row["video_kind"]] = row
    for (sample, method), pair in preserve.items():
        if "input" not in pair or "generated" not in pair:
            continue
        delta = {"sample": sample, "method": method}
        for metric in METRICS:
            before, after = pair["input"]["median_" + metric], pair["generated"]["median_" + metric]
            delta["generated_minus_input_" + metric] = None if before is None or after is None else after - before
        deltas.append(delta)
    return {"sample_level": sample_rows, "groups": groups,
            "preserve_generated_minus_input": deltas}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Detailed per-window JSON output")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--patch-size", type=int, default=30)
    parser.add_argument("--min-hz", type=float, default=0.7)
    parser.add_argument("--max-hz", type=float, default=3.0)
    parser.add_argument("--max-lag-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if args.patch_size <= 0 or args.min_hz <= 0 or args.max_hz <= args.min_hz:
        parser.error("Invalid patch size or physiological frequency band")
    if args.max_lag_seconds < 0:
        parser.error("max-lag-seconds must be nonnegative")
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list")
    details = []
    for row in rows:
        if row.get("mode") not in ("preserve", "rewrite"):
            raise ValueError("mode must be preserve or rewrite")
        for kind in ("input", "generated"):
            for method in ("cpu_POS", "cpu_CHROM"):
                details.extend(evaluate_video(row, kind, method, args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(details, indent=2), encoding="utf-8")
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary(details), indent=2), encoding="utf-8")
    print("detail: %s\nsummary: %s" % (args.output, summary_path))


if __name__ == "__main__":
    main()
