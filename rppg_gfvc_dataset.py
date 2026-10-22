import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalise_waveform(waveform):
    waveform = waveform.astype(np.float32)
    mean = float(waveform.mean())
    waveform = waveform - mean
    scale = waveform.std()
    if scale < 1e-6:
        return np.zeros_like(waveform), mean, 1.0
    return waveform / scale, mean, float(scale)


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
                 require_semantic_masks=False):
        self.root_dir = root_dir
        self.sequence_length = int(sequence_length)
        self.rewrite_probability = float(rewrite_probability)
        self.require_semantic_masks = bool(require_semantic_masks)
        candidates = glob.glob(os.path.join(root_dir, '**', '*.face.npy'), recursive=True)
        self.samples = []
        for face_path in sorted(candidates):
            signal_path = face_path.replace('.face.npy', '.sig.npy')
            if not os.path.exists(signal_path):
                continue
            try:
                frames = np.load(face_path, mmap_mode='r').shape[0]
                signal = np.load(signal_path, allow_pickle=True).item()['rppg_signal']
            except (KeyError, ValueError, OSError):
                continue
            usable = min(frames, len(signal))
            if usable >= self.sequence_length:
                self.samples.append((face_path, signal_path, usable))
        if not self.samples:
            raise RuntimeError('No valid .face.npy/.sig.npy windows found in %s' % root_dir)

    def __len__(self):
        return len(self.samples)

    def _load_waveform(self, sample_index, start, length):
        signal_path = self.samples[sample_index][1]
        data = np.load(signal_path, allow_pickle=True).item()
        waveform = data['rppg_signal'][start:start + length]
        waveform, _, scale = _normalise_waveform(waveform)
        fps = data.get('fps', 30.0)
        if isinstance(fps, (list, tuple, np.ndarray)):
            fps = fps[0] if len(fps) else 30.0
        return waveform, float(fps), scale

    def __getitem__(self, index):
        face_path, _, usable = self.samples[index]
        max_start = usable - self.sequence_length
        start = np.random.randint(max_start + 1) if max_start > 0 else 0
        # OpenCV preprocessing stored BGR; convert to RGB for the GFVC model.
        bgr = np.asarray(np.load(face_path, mmap_mode='r')[
            start:start + self.sequence_length
        ])
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        original, fps, original_scale = self._load_waveform(
            index, start, self.sequence_length
        )

        rewrite = np.random.random() < self.rewrite_probability and len(self.samples) > 1
        if rewrite:
            target_index = np.random.randint(len(self.samples) - 1)
            if target_index >= index:
                target_index += 1
            target_usable = self.samples[target_index][2]
            target_start_max = target_usable - self.sequence_length
            target_start = np.random.randint(target_start_max + 1) if target_start_max > 0 else 0
            target, _, target_scale = self._load_waveform(
                target_index, target_start, self.sequence_length
            )
            mode = 1
        else:
            target, target_scale, mode = original.copy(), original_scale, 0

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
        return {
            'source': video[0],
            'driving_video': video,
            'skin_masks': masks,
            'target_rppg': torch.from_numpy(target.copy()).float(),
            'original_rppg': torch.from_numpy(original.copy()).float(),
            'fps': torch.tensor(fps, dtype=torch.float32),
            'waveform_scale': torch.tensor(target_scale, dtype=torch.float32),
            'mode': torch.tensor(mode, dtype=torch.long),  # 0 preserve, 1 rewrite
            'name': os.path.basename(face_path),
        }
