import glob
import os

import cv2
import numpy as np
import torch
from scipy import signal as scipy_signal
from torch.utils.data import Dataset


def _normalise_waveform(waveform):
    waveform = waveform.astype(np.float32)
    mean = float(waveform.mean())
    waveform = waveform - mean
    scale = waveform.std()
    if scale < 1e-6:
        return np.zeros_like(waveform), mean, 1.0
    return waveform / scale, mean, float(scale)


def _prepare_waveform(waveform, fps, mode):
    """Match the cPPG preprocessing used to train the rePPG teacher."""
    waveform = np.asarray(waveform, dtype=np.float64)
    if mode == 'reppg':
        value_range = np.ptp(waveform)
        if value_range < 1e-8:
            return np.zeros_like(waveform, dtype=np.float32), 1.0
        waveform = (waveform - waveform.min()) / value_range
        width = min(30, len(waveform))
        kernel = np.ones(width, dtype=np.float64)
        local_count = np.convolve(np.ones(len(waveform)), kernel, mode='same')
        local_mean = np.convolve(waveform, kernel, mode='same') / local_count
        waveform = (waveform - local_mean) / np.maximum(local_mean, 1e-6)
        sos = scipy_signal.butter(5, [0.7, 3.0], btype='bandpass', fs=fps,
                                  output='sos')
        waveform = scipy_signal.sosfiltfilt(sos, waveform)
    elif mode != 'none':
        raise ValueError('Unsupported signal preprocessing: %s' % mode)
    prepared, _, scale = _normalise_waveform(waveform)
    return prepared, scale


def _robust_bpm(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) & (values >= 42.0) & (values <= 180.0)
    if valid.sum() < 0.7 * len(values):
        return np.nan
    return float(np.median(values[valid]))


def estimate_skin_masks(rgb_video):
    """Conservative fallback mask for already-cropped, frontal face clips.

    Production experiments should use precomputed semantic masks named
    ``*.skin.npy``. This fallback lets the pipeline be validated without making
    the unsafe assumption that a square face crop is itself a skin ROI.
    """
    time, height, width, _ = rgb_video.shape
    yy, xx = np.ogrid[:height, :width]
    ellipse = (((xx - width * 0.5) / (width * 0.38)) ** 2 +
               ((yy - height * 0.48) / (height * 0.46)) ** 2) <= 1.0
    masks = []
    kernel = np.ones((5, 5), np.uint8)
    for frame in rgb_video:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        skin = ((cr >= 133) & (cr <= 180) & (cb >= 70) & (cb <= 135) & ellipse)
        skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
        skin = cv2.GaussianBlur(skin.astype(np.float32), (11, 11), 0)
        masks.append(np.clip(skin, 0.0, 1.0))
    return np.stack(masks, axis=0)


class GFVCRPPGWindowDataset(Dataset):
    """Paired GFVC video windows and contact-PPG waveforms.

    Supports both nested processed datasets and the flat symlink split created
    by ``tools/make_fv2v_split_links.py``.
    """

    def __init__(self, root_dir, sequence_length=150, rewrite_probability=0.3,
                 require_semantic_masks=True, signal_preprocess='reppg',
                 deterministic=False, seed=0):
        self.root_dir = root_dir
        self.sequence_length = int(sequence_length)
        self.rewrite_probability = float(rewrite_probability)
        self.require_semantic_masks = bool(require_semantic_masks)
        self.signal_preprocess = signal_preprocess
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        if not 0.0 <= self.rewrite_probability <= 1.0:
            raise ValueError('rewrite_probability must be between 0 and 1')
        candidates = glob.glob(os.path.join(root_dir, '**', '*.face.npy'), recursive=True)
        self.samples = []
        for face_path in sorted(candidates):
            signal_path = face_path.replace('.face.npy', '.sig.npy')
            if not os.path.exists(signal_path):
                continue
            try:
                frames = np.load(face_path, mmap_mode='r').shape[0]
                signal_data = np.load(signal_path, allow_pickle=True).item()
                signal = signal_data['rppg_signal']
            except (KeyError, ValueError, OSError):
                continue
            usable = min(frames, len(signal))
            signal_fps = self._signal_fps(signal_data)
            times = signal_data.get('time')
            duration = (float(times[usable - 1]) - float(times[0])
                        if times is not None else (usable - 1) / signal_fps)
            mask_path = face_path.replace('.face.npy', '.skin.npy')
            if self.require_semantic_masks:
                if not os.path.exists(mask_path):
                    raise RuntimeError('Missing semantic skin mask: %s' % mask_path)
                mask_shape = np.load(mask_path, mmap_mode='r').shape
                if len(mask_shape) not in (3, 4) or mask_shape[0] < usable:
                    raise RuntimeError('Invalid semantic skin mask shape: %s %s' %
                                       (mask_path, mask_shape))
            if usable >= self.sequence_length:
                self.samples.append((face_path, signal_path, usable, duration))
        if not self.samples:
            raise RuntimeError('No valid .face.npy/.sig.npy windows found in %s' % root_dir)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _signal_fps(data):
        fps = data.get('fps', 30.0)
        if isinstance(fps, (list, tuple, np.ndarray)):
            fps = fps[0] if len(fps) else 30.0
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 6.0:
            raise ValueError('Invalid signal FPS: %s' % fps)
        return fps

    def _load_waveform(self, sample_index, start, length):
        signal_path = self.samples[sample_index][1]
        data = np.load(signal_path, allow_pickle=True).item()
        fps = self._signal_fps(data)
        waveform, scale = _prepare_waveform(
            data['rppg_signal'][start:start + length], fps, self.signal_preprocess
        )
        bpm = _robust_bpm(data['hr'][start:start + length]) if 'hr' in data else np.nan
        return waveform, fps, scale, bpm

    def _load_rewrite_waveform(self, sample_index, length, output_fps, rng):
        data = np.load(self.samples[sample_index][1], allow_pickle=True).item()
        input_fps = self._signal_fps(data)
        raw = np.asarray(data['rppg_signal'], dtype=np.float64)
        times = np.asarray(data.get('time', np.arange(len(raw)) / input_fps),
                           dtype=np.float64)
        if len(times) != len(raw) or not np.all(np.isfinite(times)):
            raise ValueError('Invalid PPG timestamps: %s' % self.samples[sample_index][1])
        # Repeated timestamps occur in UBFC; interpolation needs strictly increasing x.
        times, unique_indices = np.unique(times, return_index=True)
        raw = raw[unique_indices]
        if len(times) < 2 or times[-1] - times[0] < (length - 1) / output_fps:
            raise ValueError('Rewrite PPG window is shorter than the driving video')
        valid_starts = np.flatnonzero(times <= times[-1] - (length - 1) / output_fps)
        start = int(rng.choice(valid_starts))
        output_times = times[start] + np.arange(length) / output_fps
        waveform = np.interp(output_times, times, raw)
        waveform, scale = _prepare_waveform(waveform, output_fps,
                                            self.signal_preprocess)
        if 'hr' in data:
            hr = np.asarray(data['hr'], dtype=np.float64)[unique_indices]
            bpm = _robust_bpm(np.interp(output_times, times, hr))
        else:
            bpm = np.nan
        return waveform, scale, bpm

    def __getitem__(self, index):
        face_path, _, usable, _ = self.samples[index]
        rng = np.random.RandomState(self.seed + index) if self.deterministic else np.random
        max_start = usable - self.sequence_length
        start = rng.randint(max_start + 1) if max_start > 0 else 0
        # OpenCV preprocessing stored BGR; convert to RGB for the GFVC model.
        bgr = np.asarray(np.load(face_path, mmap_mode='r')[
            start:start + self.sequence_length
        ])
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        original, fps, original_scale, original_bpm = self._load_waveform(
            index, start, self.sequence_length
        )

        rewrite = rng.random() < self.rewrite_probability and len(self.samples) > 1
        if rewrite:
            required_duration = (self.sequence_length - 1) / fps
            eligible = [i for i, sample in enumerate(self.samples)
                        if i != index and sample[3] >= required_duration]
            if not eligible:
                raise RuntimeError('No rewrite PPG clip long enough for %.3f s' %
                                   required_duration)
            target_index = int(rng.choice(eligible))
            target, target_scale, target_bpm = self._load_rewrite_waveform(
                target_index, self.sequence_length, fps, rng
            )
            mode = 1
        else:
            target, target_scale, target_bpm, mode = (
                original.copy(), original_scale, original_bpm, 0
            )

        mask_path = face_path.replace('.face.npy', '.skin.npy')
        if os.path.exists(mask_path):
            masks = np.asarray(np.load(mask_path, mmap_mode='r')[
                start:start + self.sequence_length
            ], dtype=np.float32)
            if masks.ndim == 4:
                masks = masks[..., 0]
        elif self.require_semantic_masks:
            raise RuntimeError('Missing semantic skin mask: %s' % mask_path)
        else:
            masks = estimate_skin_masks(rgb)

        video = torch.from_numpy(rgb.copy()).float().permute(0, 3, 1, 2) / 255.0
        masks = torch.from_numpy(masks.copy()).float().unsqueeze(1)
        if not torch.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
            raise ValueError('Invalid skin mask values: %s' % mask_path)
        if masks.shape[0] != self.sequence_length:
            raise ValueError('Skin mask/video length mismatch: %s' % mask_path)
        return {
            'source': video[0],
            'driving_video': video,
            'skin_masks': masks,
            'target_rppg': torch.from_numpy(target.copy()).float(),
            'original_rppg': torch.from_numpy(original.copy()).float(),
            'fps': torch.tensor(fps, dtype=torch.float32),
            'target_bpm': torch.tensor(target_bpm, dtype=torch.float32),
            'waveform_scale': torch.tensor(target_scale, dtype=torch.float32),
            'mode': torch.tensor(mode, dtype=torch.long),  # 0 preserve, 1 rewrite
            'name': os.path.basename(face_path),
        }
