"""Evaluate lossless input/generated videos with the actual pyVHR Pipeline.

Run this script in the pyVHR environment, with PYTHONPATH pointing to both
the GFVC project and the pyVHR repository. It does not render videos or load
the training teacher. Each manifest row describes an already rendered,
continuous video and its aligned target BVP samples.

Manifest JSON example:
[
  {"sample": "subject14_clip_0000", "mode": "preserve",
   "input_video": "/absolute/input.avi",
   "generated_video": "/absolute/generated.avi",
   "target_signal": "/absolute/target.npy", "fps": 30.0}
]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.utils.errors import getErrors, printErrors


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Cannot open video: %s" % path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if not np.isfinite(fps) or fps <= 0 or frames <= 0:
        raise ValueError("Invalid FPS or frame count: %s" % path)
    return fps, frames


def target_window_bpm(signal, fps, center, width):
    start = int(round((center - width / 2.0) * fps))
    end = int(round((center + width / 2.0) * fps))
    segment = np.asarray(signal[max(0, start):min(len(signal), end)], dtype=float)
    if len(segment) < 2 or not np.isfinite(segment).all():
        raise ValueError("Invalid target window at %.3f seconds" % center)
    segment = segment - segment.mean()
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
    frequencies = np.fft.rfftfreq(len(segment), 1.0 / fps)
    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    return float(frequencies[valid][np.argmax(spectrum[valid])] * 60.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wsize", type=float, default=10.0)
    parser.add_argument("--patch-size", type=int, default=30)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.wsize <= 0 or args.patch_size <= 0:
        parser.error("wsize and patch-size must be positive")
    if args.plot:
        from pyVHR.plot.visualize import displayErrors

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Manifest must be a nonempty JSON list")
    results = []
    for row in manifest:
        signal = np.load(row["target_signal"], allow_pickle=False).reshape(-1)
        target_fps = float(row["fps"])
        if target_fps <= 0 or not np.isfinite(signal).all():
            raise ValueError("Invalid target waveform/FPS: %s" % row["sample"])
        for video_kind in ("input", "generated"):
            video = Path(row[video_kind + "_video"])
            fps, frames = video_info(video)
            if abs(fps - target_fps) > 0.05 or len(signal) != frames:
                raise ValueError("Video/BVP frame alignment mismatch: %s" % video)
            if frames / fps < args.wsize:
                raise ValueError("Video shorter than pyVHR window: %s" % video)
            for method in ("cpu_POS", "cpu_CHROM"):
                # POS already performs temporal normalization/overlap-add.
                # Use pyVHR's default pre-filter setting and one post-filter.
                pre_filt = method == "cpu_CHROM"
                pipe = Pipeline()
                bvps, timesES, bpmES = pipe.run_on_video(
                    str(video), winsize=args.wsize,
                    roi_method="convexhull", roi_approach="patches",
                    method=method, estimate="clustering",
                    patch_size=args.patch_size,
                    RGB_LOW_HIGH_TH=(5, 230), Skin_LOW_HIGH_TH=(5, 230),
                    pre_filt=pre_filt, post_filt=True, cuda=True, verb=True,
                )
                timesES = np.asarray(timesES, dtype=float).reshape(-1)
                bpmES = np.asarray(bpmES, dtype=float).reshape(-1)
                if not len(timesES) or len(timesES) != len(bpmES):
                    raise RuntimeError("pyVHR returned no aligned BPM windows")
                bpmGT = np.asarray([target_window_bpm(signal, fps, t, args.wsize)
                                    for t in timesES])
                # getErrors expects a ground-truth BPM time series; its inputs
                # here share exactly the pyVHR window centers.
                timesGT = timesES.copy()
                metrics = getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT)
                print("%s %s %s" % (row["sample"], video_kind, method))
                printErrors(*metrics)
                if args.plot:
                    displayErrors(bpmES, bpmGT, timesES, timesGT)
                results.append({
                    "sample": row["sample"], "mode": row["mode"],
                    "video_kind": video_kind, "video": str(video),
                    "method": method, "fps": fps, "window_seconds": args.wsize,
                    "pre_filt": pre_filt, "post_filt": True,
                    "timesES": timesES.tolist(), "bpmES": bpmES.tolist(),
                    "bpmGT": bpmGT.tolist(),
                    "metrics": {name: np.asarray(value).tolist() for name, value in zip(
                        ("RMSE", "MAE", "MAX", "PCC", "CCC", "SNR"), metrics)},
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
