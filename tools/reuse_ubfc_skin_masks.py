#!/usr/bin/env python3
"""Reuse frame-aligned skin masks from an existing windowed UBFC dataset.

Frames are joined by their saved timestamps, not by clip number.  Every reused
mask is accepted only after the corresponding old and new face frames compare
equal pixel-for-pixel.  When an old frame occurs in overlapping windows, the
occurrence closest to the centre of its source window is preferred.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def clip_number(path: Path) -> int:
    stem = path.name.split(".", 1)[0]
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot parse clip number: {path}") from error


def load_signal(path: Path) -> dict:
    value = np.load(path, allow_pickle=True)
    if not isinstance(value, np.ndarray) or value.shape != ():
        raise ValueError(f"Expected scalar dict array: {path}")
    result = value.item()
    if not isinstance(result, dict) or "time" not in result:
        raise ValueError(f"Missing time metadata: {path}")
    return result


def build_old_index(subject_dir: Path, old_step: int) -> dict[int, tuple[Path, Path, int, float]]:
    choices: dict[int, list[tuple[float, Path, Path, int, float]]] = defaultdict(list)
    for sig_path in sorted(subject_dir.glob("*.sig.npy")):
        stem = sig_path.name[: -len(".sig.npy")]
        face_path = subject_dir / f"{stem}.face.npy"
        mask_path = subject_dir / f"{stem}.skin.npy"
        if not face_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Incomplete old sample: {sig_path}")
        signal = load_signal(sig_path)
        times = np.asarray(signal["time"]).reshape(-1)
        face = np.load(face_path, mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")
        if len(times) != len(face) or len(times) != len(mask):
            raise ValueError(f"Old sample length mismatch: {sig_path}")
        centre = (len(times) - 1) / 2.0
        source_start = clip_number(sig_path) * old_step
        for index, time_value in enumerate(times):
            choices[source_start + index].append(
                (abs(index - centre), face_path, mask_path, index, float(time_value))
            )
    return {
        key: min(options, key=lambda option: option[0])[1:]
        for key, options in choices.items()
    }


def reuse_subject(
    old_subject: Path,
    new_subject: Path,
    overwrite: bool,
    old_step: int,
    new_step: int,
) -> tuple[int, int]:
    old_index = build_old_index(old_subject, old_step)
    written = 0
    frames_written = 0
    for sig_path in sorted(new_subject.glob("*.sig.npy")):
        stem = sig_path.name[: -len(".sig.npy")]
        face_path = new_subject / f"{stem}.face.npy"
        output_path = new_subject / f"{stem}.skin.npy"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite: {output_path}")
        signal = load_signal(sig_path)
        times = np.asarray(signal["time"]).reshape(-1)
        new_face = np.load(face_path, mmap_mode="r")
        if len(times) != len(new_face):
            raise ValueError(f"New sample length mismatch: {sig_path}")

        source_start = clip_number(sig_path) * new_step
        references = []
        missing = []
        for frame_index, time_value in enumerate(times):
            reference = old_index.get(source_start + frame_index)
            if reference is None:
                missing.append((frame_index, float(time_value)))
            else:
                references.append(reference)
        if missing:
            raise KeyError(f"{sig_path}: {len(missing)} timestamps absent; first={missing[0]}")

        face_cache: dict[Path, np.ndarray] = {}
        mask_cache: dict[Path, np.ndarray] = {}
        output = None
        for frame_index, (old_face_path, old_mask_path, old_index_value, old_time) in enumerate(references):
            if abs(float(times[frame_index]) - old_time) > 1e-6:
                raise ValueError(
                    f"Time mismatch: {sig_path}[{frame_index}]={times[frame_index]} != "
                    f"{old_time}"
                )
            if old_face_path not in face_cache:
                face_cache[old_face_path] = np.load(old_face_path, mmap_mode="r")
            old_face = face_cache[old_face_path]
            if not np.array_equal(new_face[frame_index], old_face[old_index_value]):
                raise ValueError(
                    f"Face mismatch: {face_path}[{frame_index}] != "
                    f"{old_face_path}[{old_index_value}]"
                )
            if old_mask_path not in mask_cache:
                mask_cache[old_mask_path] = np.load(old_mask_path, mmap_mode="r")
            old_mask = mask_cache[old_mask_path]
            mask_frame = old_mask[old_index_value]
            if output is None:
                output = np.empty((len(times), *mask_frame.shape), dtype=mask_frame.dtype)
            output[frame_index] = mask_frame
        if output is None:
            raise ValueError(f"Empty sample: {sig_path}")
        np.save(output_path, output)
        written += 1
        frames_written += len(output)
    return written, frames_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-step", type=int, default=75)
    parser.add_argument("--new-step", type=int, default=360)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    total_samples = 0
    total_frames = 0
    for split_dir in sorted(path for path in args.new_root.iterdir() if path.is_dir()):
        for new_subject in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            old_subject = args.old_root / new_subject.name
            if not old_subject.is_dir():
                raise FileNotFoundError(f"Old subject missing: {old_subject}")
            samples, frames = reuse_subject(
                old_subject,
                new_subject,
                args.overwrite,
                args.old_step,
                args.new_step,
            )
            total_samples += samples
            total_frames += frames
            print(f"{split_dir.name}/{new_subject.name}: samples={samples} frames={frames}")
    print(f"done samples={total_samples} frames={total_frames}")


if __name__ == "__main__":
    main()
