#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np


def parse_ground_truth(path):
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"Unexpected ground truth shape: {arr.shape}")
    rppg, hr, time = arr[0], arr[1], arr[2]
    return rppg, hr, time


def detect_face_bbox(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    # prefer the largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return int(x), int(y), int(w), int(h)


def fixed_crop_box(bbox, frame_shape, size=256):
    """Return one fixed, in-bounds ``size x size`` crop around the face.

    The crop is computed once from the detected face centre and reused for every
    frame.  In particular, frames are *not* resized: this preserves a common
    pixel coordinate system for motion and rPPG colour changes.
    """
    fh, fw = frame_shape[:2]
    if fh < size or fw < size:
        raise ValueError(
            'Frame %dx%d is smaller than fixed crop %dx%d; refusing to resize' %
            (fw, fh, size, size)
        )
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    left = int(round(cx - size / 2.0))
    top = int(round(cy - size / 2.0))
    left = max(0, min(left, fw - size))
    top = max(0, min(top, fh - size))
    return left, top, left + size, top + size


def crop_fixed(frame, crop_box):
    left, top, right, bottom = crop_box
    crop = frame[top:bottom, left:right]
    if crop.shape[:2] != (bottom - top, right - left):
        raise RuntimeError('Fixed crop unexpectedly falls outside the frame')
    return crop


def find_face_bbox(video_path, detector, probes=(0, 0.25, 0.5, 0.75, 0.9)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for p in probes:
        idx = min(total - 1, int(total * p))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = detect_face_bbox(frame, detector)
        if bbox is not None:
            cap.release()
            return bbox
    cap.release()
    return None


def make_windows(n_frames, window_size, step_size):
    windows = []
    start = 0
    while start + window_size <= n_frames:
        windows.append((start, start + window_size))
        start += step_size
    if not windows and n_frames >= window_size // 2:
        # keep one shorter final clip so tiny subjects are not lost
        windows.append((0, n_frames))
    return windows


def process_subject(
    subject_dir,
    out_dir,
    detector,
    window_size,
    step_size,
    size,
    overwrite=False,
):
    video_path = os.path.join(subject_dir, "vid.avi")
    gt_path = os.path.join(subject_dir, "ground_truth.txt")
    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        return None

    rppg, hr, time = parse_ground_truth(gt_path)
    n_signal = len(rppg)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_signal, n_frames)

    bbox = find_face_bbox(video_path, detector)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, reference_frame = cap.read()
    if not ok:
        cap.release()
        return None
    if bbox is None:
        # safe fallback: central crop
        h, w = reference_frame.shape[:2]
        side = min(h, w)
        bbox = ((w - side) // 2, (h - side) // 2, side, side)
    crop_box = fixed_crop_box(bbox, reference_frame.shape, size=size)

    subject_name = os.path.basename(subject_dir)
    subject_out = os.path.join(out_dir, subject_name)
    os.makedirs(subject_out, exist_ok=True)
    meta_path = os.path.join(subject_out, "meta.json")
    if not overwrite and os.path.exists(meta_path):
        cap.release()
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    window_bounds = make_windows(n, window_size, step_size)
    if not window_bounds:
        cap.release()
        return None

    windows = []
    for wi, (start, end) in enumerate(window_bounds):
        clip_name = f"clip_{wi:04d}.face.npy"
        sig_name = f"clip_{wi:04d}.sig.npy"
        clip_path = os.path.join(subject_out, clip_name)
        sig_path = os.path.join(subject_out, sig_name)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        written = 0
        for _ in range(end - start):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(crop_fixed(frame, crop_box))
            written += 1

        if not frames:
            continue

        face = np.stack(frames, axis=0)
        np.save(clip_path, face, allow_pickle=False)

        window_sig = {
            "rppg_signal": rppg[start : start + written],
            "hr": hr[start : start + written],
            "time": time[start : start + written],
            "fps": float(fps),
        }
        np.save(sig_path, window_sig, allow_pickle=True)
        windows.append(
            {
                "clip": clip_name,
                "signal": sig_name,
                "start": int(start),
                "end": int(start + written),
                "frames": int(written),
                "fps": float(fps),
            }
        )

    cap.release()
    if not windows:
        return None

    meta = {
        "subject": subject_name,
        "video": os.path.basename(video_path),
        "ground_truth": os.path.basename(gt_path),
        "fps": float(fps),
        "n_frames": int(n),
        "n_signal": int(n_signal),
        "n_windows": int(len(windows)),
        "window_size": int(window_size),
        "step_size": int(step_size),
        "crop_size": int(size),
        "bbox": [int(v) for v in bbox],
        "fixed_crop_box_xyxy": [int(v) for v in crop_box],
        "windows": windows,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/opt/data/private/datasets/UBFC2")
    parser.add_argument("--out", default="/opt/data/private/datasets/UBFC2_processed")
    parser.add_argument("--subjects", default="*", help="subject name or glob")
    parser.add_argument("--window_size", type=int, default=150)
    parser.add_argument("--step_size", type=int, default=75)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    subject_dirs = sorted(glob(os.path.join(args.root, args.subjects)))
    for subject_dir in subject_dirs:
        if not os.path.isdir(subject_dir):
            continue
        print(f"Processing {subject_dir}")
        meta = process_subject(
            subject_dir,
            args.out,
            detector,
            args.window_size,
            args.step_size,
            args.size,
            overwrite=args.overwrite,
        )
        if meta is None:
            print("  skipped")
        else:
            print(f"  windows: {meta['n_windows']}")


if __name__ == "__main__":
    main()
