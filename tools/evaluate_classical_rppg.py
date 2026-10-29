"""Independent POS/CHROM/GREEN rPPG evaluation for generated GFVC videos.

This utility intentionally does not use ``SkinRPPGTeacher``.  It detects a new
MediaPipe convex-hull skin ROI on every rendered frame, computes its mean RGB
trace, and applies the CPU POS/CHROM/GREEN formulas used by pyVHR.  This makes
it suitable for testing whether a signal remains observable by a conventional
rPPG method rather than merely by the neural teacher used during training.

The input may be a recovered-only video or a side-by-side render; use
``--crop-right-half`` for the latter.  It expects the target ``.sig.npy``
created by this project's UBFC preprocessing, or ``--target-npy`` containing a
one-dimensional signal.
"""
import argparse
import json
from pathlib import Path

import numpy as np


LOW_HZ, HIGH_HZ = 0.7, 3.0


def landmark_ids(edge_set):
    return np.asarray(sorted({index for edge in edge_set for index in edge}), np.int32)


def skin_mask(rgb, face_mesh, all_face_ids, eye_ids, lip_ids):
    import cv2
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    height, width = rgb.shape[:2]
    points = np.asarray([(int(p.x * width), int(p.y * height))
                         for p in result.multi_face_landmarks[0].landmark], np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points[all_face_ids]), 1)
    for indices in eye_ids + [lip_ids]:
        cv2.fillConvexPoly(mask, cv2.convexHull(points[indices]), 0)
    return mask.astype(np.float32)


def pyvhr_chrom(rgb):
    """pyVHR ``cpu_CHROM`` formula; rgb is [3, T]."""
    x = 3.0 * rgb[0] - 2.0 * rgb[1]
    y = 1.5 * rgb[0] + rgb[1] - 1.5 * rgb[2]
    return x - (np.std(x) / max(np.std(y), 1e-8)) * y


def pyvhr_green(rgb):
    """pyVHR ``cpu_GREEN`` formula; rgb is [3, T]."""
    return rgb[1].copy()


def pyvhr_pos(rgb, fps):
    """pyVHR ``cpu_POS`` overlap-add formula; rgb is [3, T]."""
    frames = rgb.shape[1]
    window = max(2, int(1.6 * fps))
    projection = np.asarray([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    output = np.zeros(frames, dtype=np.float64)
    for n in range(frames):
        m = n - window
        if m >= 0:
            normalized = rgb[:, m:n] / np.maximum(np.mean(rgb[:, m:n], axis=1, keepdims=True), 1e-8)
            s = projection @ normalized
            alpha = np.std(s[0]) / max(np.std(s[1]), 1e-8)
            h = s[0] + alpha * s[1]
            h -= np.mean(h)
            output[m:n] += h
    return output


def bandpass_and_standardize(values, fps):
    from scipy import signal as scipy_signal
    values = np.asarray(values, dtype=np.float64)
    nyquist_ok = fps > 2.1 * HIGH_HZ
    if not nyquist_ok:
        raise ValueError('FPS %.3f is too low for %.1f Hz band-pass' % (fps, HIGH_HZ))
    sos = scipy_signal.butter(5, [LOW_HZ, HIGH_HZ], btype='bandpass', fs=fps, output='sos')
    # sosfiltfilt requires roughly 30 samples; generated clips are 150 frames.
    values = scipy_signal.sosfiltfilt(sos, values)
    return (values - values.mean()) / max(values.std(), 1e-8)


def prepare_reference(values, fps):
    """Same contact-PPG preprocessing used by ``GFVCRPPGWindowDataset``."""
    values = np.asarray(values, dtype=np.float64)
    spread = np.ptp(values)
    if spread < 1e-8:
        return np.zeros_like(values)
    values = (values - values.min()) / spread
    width = min(30, len(values))
    kernel = np.ones(width, dtype=np.float64)
    local_count = np.convolve(np.ones(len(values)), kernel, mode='same')
    local_mean = np.convolve(values, kernel, mode='same') / local_count
    values = (values - local_mean) / np.maximum(local_mean, 1e-6)
    return bandpass_and_standardize(values, fps)


def peak_bpm(values, fps):
    freqs = np.fft.rfftfreq(len(values), 1.0 / fps)
    power = np.abs(np.fft.rfft(values - np.mean(values))) ** 2
    valid = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    return float(freqs[valid][np.argmax(power[valid])] * 60.0)


def load_frames(path, crop_right_half):
    import cv2
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if crop_right_half:
            bgr = bgr[:, bgr.shape[1] // 2:]
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError('No decodable frames in ' + str(path))
    if fps <= 0:
        raise RuntimeError('Invalid FPS in ' + str(path))
    return np.stack(frames), fps


def extract_mean_rgb(frames):
    import mediapipe as mp
    face_module = mp.solutions.face_mesh
    all_face_ids = np.arange(468, dtype=np.int32)
    eye_ids = [landmark_ids(face_module.FACEMESH_LEFT_EYE),
               landmark_ids(face_module.FACEMESH_RIGHT_EYE)]
    lip_ids = landmark_ids(face_module.FACEMESH_LIPS)
    mesh = face_module.FaceMesh(static_image_mode=False, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
    traces, previous_mask, misses = [], None, 0
    try:
        for frame in frames:
            current = skin_mask(frame, mesh, all_face_ids, eye_ids, lip_ids)
            if current is None:
                misses += 1
                if previous_mask is None:
                    raise RuntimeError('Face Mesh failed on first frame')
                current = previous_mask
            previous_mask = current
            denominator = max(float(current.sum()), 1.0)
            traces.append((frame.astype(np.float64) * current[..., None]).sum(axis=(0, 1)) / denominator)
    finally:
        mesh.close()
    return np.asarray(traces, dtype=np.float64).T, misses


def load_target(path, expected_length, fps):
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == () and isinstance(loaded.item(), dict):
        data = loaded.item()
        values = data['rppg_signal']
    else:
        values = np.asarray(loaded)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) != expected_length:
        old_time = np.linspace(0.0, 1.0, len(values))
        new_time = np.linspace(0.0, 1.0, expected_length)
        values = np.interp(new_time, old_time, values)
    return prepare_reference(values, fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--target-npy', required=True, type=Path)
    parser.add_argument('--crop-right-half', action='store_true')
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()

    frames, fps = load_frames(args.video, args.crop_right_half)
    mean_rgb, missed = extract_mean_rgb(frames)
    target = load_target(args.target_npy, mean_rgb.shape[1], fps)
    methods = {'POS': pyvhr_pos(mean_rgb, fps), 'CHROM': pyvhr_chrom(mean_rgb),
               'GREEN': pyvhr_green(mean_rgb)}
    report = {'video': str(args.video), 'fps': fps, 'frames': int(mean_rgb.shape[1]),
              'face_mesh_misses_reused': int(missed), 'methods': {}}
    for name, values in methods.items():
        predicted = bandpass_and_standardize(values, fps)
        corr = float(np.corrcoef(predicted, target)[0, 1])
        report['methods'][name] = {
            'pearson_signed': corr,
            'pearson_sign_invariant': abs(corr),
            'peak_bpm': peak_bpm(predicted, fps),
            'target_peak_bpm': peak_bpm(target, fps),
            'peak_bpm_abs_error': abs(peak_bpm(predicted, fps) - peak_bpm(target, fps)),
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
