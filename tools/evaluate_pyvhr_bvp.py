"""Evaluate frame-aligned BVP waveforms extracted by the actual pyVHR pipeline.

The manifest is a JSON list. Every row requires ``sample``, ``mode``
(``preserve`` or ``rewrite``), ``input_video``, ``generated_video``,
``target_signal`` and ``fps``. ``original_signal`` is optional and is used for
the Input side of rewrite experiments. Signals are one-dimensional ``.npy``
arrays with one value per video frame.

Each pyVHR patch waveform is standardized independently and the pointwise
median is used as the single extracted BVP. The script reports signed zero-lag
PCC, DTW, spectral heart-rate MAE, and SNR. It never selects a patch by its
agreement with the reference, shifts a signal, or flips polarity.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = ("pcc", "mae_z", "rmse_z", "dtw_distance", "dtw_normalized", "hr_mae_bpm", "snr_db")
DIAGNOSTICS = ("dtw_path_length", "predicted_hr_bpm", "target_hr_bpm",
               "hr_rmse_bpm", "hr_max_error_bpm", "hr_pcc", "hr_ccc")


def subject_for(row):
    if row.get("subject"):
        return str(row["subject"])
    match = re.search(r"subject\d+", str(row.get("sample", "")))
    return match.group(0) if match else str(row["sample"])


def zscore(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("non_finite_signal")
    std = values.std()
    if std < 1e-8:
        raise ValueError("constant_signal")
    return (values - values.mean()) / std


def dtw_distance(left, right):
    """Exact Euclidean DTW: min sqrt(sum squared error along the path)."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    previous_cost = np.full(len(right) + 1, np.inf, dtype=np.float64)
    previous_steps = np.zeros(len(right) + 1, dtype=np.int32)
    previous_cost[0] = 0.0
    for left_value in left:
        current_cost = np.full(len(right) + 1, np.inf, dtype=np.float64)
        current_steps = np.zeros(len(right) + 1, dtype=np.int32)
        for column, right_value in enumerate(right, 1):
            candidates = (
                (previous_cost[column - 1], previous_steps[column - 1]),
                (previous_cost[column], previous_steps[column]),
                (current_cost[column - 1], current_steps[column - 1]),
            )
            best_cost, best_steps = min(candidates, key=lambda item: (item[0], item[1]))
            current_cost[column] = best_cost + (left_value - right_value) ** 2
            current_steps[column] = best_steps + 1
        previous_cost, previous_steps = current_cost, current_steps
    return float(np.sqrt(previous_cost[-1])), int(previous_steps[-1])


def power_spectrum(values, fps, min_hz, max_hz):
    """Return a Hann-periodogram restricted to the physiological band."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    nfft = max(4096, 1 << int(np.ceil(np.log2(len(values)))))
    windowed = (values - values.mean()) * np.hanning(len(values))
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fps)
    power = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
    keep = (frequencies >= min_hz) & (frequencies <= max_hz)
    if not np.any(keep) or not np.isfinite(power[keep]).all():
        raise ValueError("invalid_power_spectrum")
    return frequencies[keep], power[keep]


def estimate_hr_bpm(values, fps, min_hz, max_hz):
    frequencies, power = power_spectrum(values, fps, min_hz, max_hz)
    return float(frequencies[int(np.argmax(power))] * 60.0)


def waveform_metrics(prediction, target):
    """Compute frame-aligned waveform metrics after independent z-scoring."""
    try:
        prediction = zscore(prediction)
        target = zscore(target)
    except ValueError as error:
        return None, str(error)
    if len(prediction) != len(target) or len(target) < 3:
        return None, "length_mismatch"

    try:
        distance, path_length = dtw_distance(prediction, target)
    except ValueError as error:
        return None, str(error)
    return {
        "pcc": float(np.mean(prediction * target)),
        # POS/CHROM amplitudes are arbitrary, so only standardized waveform
        # MAE/RMSE are comparable across videos and extraction methods.
        "mae_z": float(np.mean(np.abs(prediction - target))),
        "rmse_z": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "dtw_distance": distance,
        "dtw_normalized": float(distance / np.sqrt(path_length)),
        "dtw_path_length": path_length,
    }, None


def scalar_metric(value):
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else None


def official_pyvhr_errors(bvps, fps, bpm_es, bpm_gt, times_es, times_gt,
                          display=False):
    """Call pyVHR's unmodified getErrors/printErrors/displayErrors interface."""
    from pyVHR.utils.errors import displayErrors, getErrors, printErrors

    bpm_es = np.asarray(bpm_es, dtype=np.float64).reshape(-1)
    bpm_gt = np.asarray(bpm_gt, dtype=np.float64).reshape(-1)
    times_es = np.asarray(times_es, dtype=np.float64).reshape(-1)
    times_gt = np.asarray(times_gt, dtype=np.float64).reshape(-1)
    if len(bpm_es) < 2:
        raise ValueError("pyvhr_getErrors_requires_at_least_two_bpm_windows")
    if len(bpm_es) != len(times_es) or len(bpm_gt) != len(times_gt):
        raise ValueError("pyvhr_bpm_time_length_mismatch")
    values = getErrors(bvps, fps, bpm_es, bpm_gt, times_es, times_gt)
    printErrors(*values)
    if display:
        displayErrors(bpm_es, bpm_gt, times_es, times_gt)
    rmse, mae, maximum, pcc, ccc, snr = values
    return {
        "hr_rmse_bpm": scalar_metric(rmse),
        "hr_mae_bpm": scalar_metric(mae),
        "hr_max_error_bpm": scalar_metric(maximum),
        "hr_pcc": scalar_metric(pcc),
        "hr_ccc": scalar_metric(ccc),
        "snr_db": scalar_metric(snr),
        "predicted_hr_bpm": float(np.mean(bpm_es)),
        "target_hr_bpm": float(np.mean(bpm_gt)),
    }


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


def median_patch_bvp(window, reference_length):
    """Build one BVP without using the reference to choose or weight patches."""
    patches = patch_waveforms(window, reference_length)
    valid = []
    failures = Counter()
    for patch in patches:
        try:
            valid.append(zscore(patch))
        except ValueError as error:
            failures[str(error)] += 1
    if not valid:
        return None, 0, len(patches), failures
    try:
        aggregated = zscore(np.median(np.stack(valid, axis=0), axis=0))
    except ValueError as error:
        failures["aggregate_" + str(error)] += 1
        return None, len(valid), len(patches), failures
    return aggregated, len(valid), len(patches), failures


def run_pyvhr(video, method, patch_size, min_hz, max_hz, window_seconds):
    """Run pyVHR and return its native BVP, time, and BPM outputs."""
    from pyVHR.analysis.pipeline import Pipeline

    # This pyVHR revision uses winsize in Python slice indices inside
    # BPM_clustering. Preserve non-integer full-clip durations, but pass exact
    # integer-second BPM windows as int rather than (for example) 5.0.
    rounded_window = int(round(window_seconds))
    pyvhr_window = (rounded_window if abs(window_seconds - rounded_window) < 1e-9
                    else window_seconds)
    Pipeline.minHz = min_hz
    Pipeline.maxHz = max_hz
    bvps, times_es, bpm_es = Pipeline().run_on_video(
        str(video), winsize=pyvhr_window, roi_method="convexhull", roi_approach="patches",
        method=method, estimate="clustering", patch_size=patch_size,
        RGB_LOW_HIGH_TH=(5, 230), Skin_LOW_HIGH_TH=(5, 230),
        pre_filt=(method == "cpu_CHROM"), post_filt=True,
        cuda=True, verb=False,
    )
    return (bvps, np.asarray(times_es, dtype=np.float64).reshape(-1),
            np.asarray(bpm_es, dtype=np.float64).reshape(-1))


def target_window_bpm(signal, fps, center_seconds, window_seconds,
                      min_hz, max_hz):
    start = int(round((center_seconds - window_seconds / 2.0) * fps))
    end = int(round((center_seconds + window_seconds / 2.0) * fps))
    segment = np.asarray(signal[max(0, start):min(len(signal), end)],
                         dtype=np.float64)
    if len(segment) < 3 or not np.isfinite(segment).all():
        raise ValueError("invalid_target_bpm_window")
    return estimate_hr_bpm(segment, fps, min_hz, max_hz)


def reference_for(row, video_kind):
    if row["mode"] == "preserve" or video_kind == "generated":
        return row["target_signal"], None
    if "original_signal" not in row:
        return None, "missing_original_signal_for_rewrite_input"
    return row["original_signal"], None


def empty_record(row, video_kind, method, reason):
    record = {"sample": row["sample"], "subject": subject_for(row),
              "mode": row["mode"],
              "video_kind": video_kind, "method": method, "window": None,
              "center_seconds": None, "patch_count": 0,
              "valid_patch_count": 0, "status": reason}
    for metric in METRICS + DIAGNOSTICS:
        record[metric] = None
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
    metric_start = int(row.get("metric_start_frame", 0))
    metric_length = int(row.get("metric_length", len(target) - metric_start))
    if metric_start < 0 or metric_length < 3 or metric_start + metric_length > len(target):
        raise ValueError("Invalid metric interval in manifest: %s" % video)
    # Use every frame in the rendered clip. This protects alignment when its
    # container reports a non-integer FPS (for example, 150 / 29.372 = 5.106 s).
    window_seconds = frames / fps

    bvps, times_es, _ = run_pyvhr(video, method, args.patch_size,
                                  args.min_hz, args.max_hz, window_seconds)
    if len(bvps) != len(times_es) or not len(bvps):
        return [empty_record(row, video_kind, method, "pyvhr_returned_no_windows")]

    bpm_bvps, bpm_times, bpm_es = run_pyvhr(
        video, method, args.patch_size, args.min_hz, args.max_hz,
        args.bpm_window_seconds)
    if not len(bpm_bvps) or len(bpm_bvps) != len(bpm_times) or len(bpm_es) != len(bpm_times):
        raise ValueError("pyVHR returned inconsistent BPM windows: %s" % video)
    bpm_gt = np.asarray([
        target_window_bpm(target, fps, center, args.bpm_window_seconds,
                          args.min_hz, args.max_hz)
        for center in bpm_times
    ], dtype=np.float64)
    official = official_pyvhr_errors(
        bpm_bvps, fps, bpm_es, bpm_gt, bpm_times, bpm_times.copy(),
        display=args.display_errors)
    records = []
    for window_index, (bvp_window, center) in enumerate(zip(bvps, times_es)):
        # pyVHR uses the complete clip window.  Context frames may be present to
        # make temporal filtering match label construction; metrics can then be
        # computed only on the declared central target interval.
        expected_length = frames
        expected_start = int(round(center * fps - expected_length / 2.0))
        if expected_start < 0 or expected_start + expected_length > len(target):
            raise ValueError("pyVHR window outside target BVP at %.6f s" % center)
        full_reference = target[expected_start:expected_start + expected_length]
        bvp, valid_count, patch_count, failures = median_patch_bvp(
            bvp_window, expected_length)
        reference = full_reference[metric_start:metric_start + metric_length]
        extracted = (bvp[metric_start:metric_start + metric_length]
                     if bvp is not None else None)
        metrics, reason = (waveform_metrics(extracted, reference) if extracted is not None else
            (None, "no_valid_patches"))
        if metrics is not None:
            metrics.update(official)
        if reason and bvp is not None:
            failures[reason] += 1
        record = {"sample": row["sample"], "subject": subject_for(row),
                  "mode": row["mode"],
                  "video_kind": video_kind, "method": method,
                  "window": window_index, "center_seconds": float(center),
                  "target_start_frame": expected_start + metric_start,
                  "target_length": metric_length, "video_frame_count": frames,
                  "patch_count": int(patch_count),
                  "valid_patch_count": int(valid_count),
                  "invalid_patch_reasons": dict(failures),
                  "aggregation": "patch_zscore_pointwise_median",
                  "status": "ok" if metrics is not None else reason}
        for metric in METRICS + DIAGNOSTICS:
            record[metric] = metrics[metric] if metrics is not None else None
        records.append(record)
    return records


def summary(records):
    """Aggregate windows to samples, samples to subjects, then subjects equally."""
    per_sample = defaultdict(list)
    for record in records:
        key = (record["sample"], record["subject"], record["mode"],
               record["video_kind"], record["method"])
        per_sample[key].append(record)
    sample_rows = []
    for key, windows in per_sample.items():
        row = {"sample": key[0], "subject": key[1], "mode": key[2],
               "video_kind": key[3], "method": key[4],
               "window_count": len(windows),
               "status": "ok" if any(w["status"] == "ok" for w in windows) else windows[0]["status"]}
        for metric in METRICS + DIAGNOSTICS:
            values = [w[metric] for w in windows if w[metric] is not None]
            row[metric] = float(np.median(values)) if values else None
        sample_rows.append(row)

    subject_rows = []
    by_subject = defaultdict(list)
    for row in sample_rows:
        by_subject[(row["subject"], row["mode"], row["video_kind"],
                    row["method"])].append(row)
    for key, rows in sorted(by_subject.items()):
        item = {"subject": key[0], "mode": key[1], "video_kind": key[2],
                "method": key[3], "sample_count": len(rows),
                "valid_sample_count": sum(r["status"] == "ok" for r in rows),
                "status": "ok" if any(r["status"] == "ok" for r in rows)
                else rows[0]["status"]}
        for metric in METRICS + DIAGNOSTICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            item[metric] = float(np.mean(values)) if values else None
        subject_rows.append(item)

    groups = []
    grouped = defaultdict(list)
    for row in subject_rows:
        grouped[(row["mode"], row["video_kind"], row["method"])].append(row)
    for (mode, kind, method), rows in sorted(grouped.items()):
        item = {"mode": mode, "video_kind": kind, "method": method,
                "subject_count": len(rows),
                "valid_subject_count": sum(r["status"] == "ok" for r in rows),
                "invalid_subject_count": sum(r["status"] != "ok" for r in rows),
                "sample_count": sum(r["sample_count"] for r in rows)}
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
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
            before, after = pair["input"][metric], pair["generated"][metric]
            delta["generated_minus_input_" + metric] = None if before is None or after is None else after - before
        deltas.append(delta)
    return {"metric_definitions": {
                "pcc": "signed zero-lag Pearson correlation; higher is better",
                "mae_z": "frame-aligned MAE after independently z-scoring predicted and target BVP; lower is better",
                "rmse_z": "frame-aligned RMSE after independently z-scoring predicted and target BVP; lower is better",
                "dtw_distance": "minimum square root of summed squared z-amplitude error along the DTW path",
                "dtw_normalized": "square root of mean squared error along the DTW path; lower is better",
                "hr_mae_bpm": "pyVHR getErrors MAE(bpmES, bpmGT); lower is better",
                "snr_db": "pyVHR getErrors/get_SNR de Haan BVP SNR; higher is better",
            },
            "patch_aggregation": "independent patch z-score then pointwise median; no reference-guided selection",
            "alignment": "frame aligned; no lag shift and no polarity flip",
            "group_aggregation": "window median -> sample; sample mean -> subject; equal-weight subject statistics",
            "sample_level": sample_rows, "subject_level": subject_rows,
            "groups": groups,
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
    parser.add_argument("--video-kinds", nargs="+", choices=("input", "generated"),
                        default=("input", "generated"))
    parser.add_argument("--methods", nargs="+", choices=("cpu_POS", "cpu_CHROM"),
                        default=("cpu_POS", "cpu_CHROM"))
    parser.add_argument("--bpm-window-seconds", type=float, default=5.0,
                        help="pyVHR BPM window used by getErrors; test clips need at least two windows")
    parser.add_argument("--display-errors", action="store_true",
                        help="Call pyVHR displayErrors in addition to printErrors")
    args = parser.parse_args()
    if args.patch_size <= 0 or args.min_hz <= 0 or args.max_hz <= args.min_hz:
        parser.error("Invalid patch size or physiological frequency band")
    if args.bpm_window_seconds <= 0:
        parser.error("bpm-window-seconds must be positive")
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list")
    details = []
    for row in rows:
        if row.get("mode") not in ("preserve", "rewrite"):
            raise ValueError("mode must be preserve or rewrite")
        for kind in args.video_kinds:
            for method in args.methods:
                details.extend(evaluate_video(row, kind, method, args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(details, indent=2), encoding="utf-8")
    summary_path = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary(details), indent=2), encoding="utf-8")
    print("detail: %s\nsummary: %s" % (args.output, summary_path))


if __name__ == "__main__":
    main()
