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


def expand_to_square(bbox, frame_shape, margin=1.25):
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * margin
    left = int(round(cx - side / 2))
    right = int(round(cx + side / 2))
    top = int(round(cy - side / 2))
    bot = int(round(cy + side / 2))

    fh, fw = frame_shape[:2]
    # keep aspect ratio while clamping inside the frame
    side = min(right - left, bot - top)
    left = max(0, min(left, fw - side))
    top = max(0, min(top, fh - side))
    right = left + side
    bot = top + side
    return left, top, right, bot


def crop_face(frame, bbox, size=256):
    left, top, right, bot = expand_to_square(bbox, frame.shape)
    crop = frame[top:bot, left:right]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


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
    if bbox is None:
        # safe fallback: central crop
        ok, frame = cap.read()
        h, w = frame.shape[:2]
        side = min(h, w)
        bbox = ((w - side) // 2, (h - side) // 2, side, side)

    subject_name = os.path.basename(subject_dir)
    subject_out = os.path.join(out_dir, subject_name)
    os.makedirs(subject_out, exist_ok=True)

    # store raw cropped frames once per subject; windowing happens later
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(crop_face(frame, bbox, size=size))
    cap.release()

    if not frames:
        return None

    face = np.stack(frames, axis=0)
    sig = {
        "rppg_signal": rppg[: len(frames)],
        "hr": hr[: len(frames)],
        "time": time[: len(frames)],
    }
    np.save(os.path.join(subject_out, "face.npy"), face, allow_pickle=False)
    np.save(os.path.join(subject_out, "sig.npy"), sig, allow_pickle=True)

    cap.release()

    meta = {
        "subject": subject_name,
        "video": os.path.basename(video_path),
        "ground_truth": os.path.basename(gt_path),
        "fps": float(fps),
        "n_frames": int(len(frames)),
        "n_signal": int(len(frames)),
        "window_size": int(window_size),
        "step_size": int(step_size),
        "size": int(size),
        "bbox": [int(v) for v in bbox],
    }
    with open(os.path.join(subject_out, "meta.json"), "w", encoding="utf-8") as f:
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
        )
        if meta is None:
            print("  skipped")
        else:
            print(f"  frames: {meta['n_frames']}")


if __name__ == "__main__":
    main()

