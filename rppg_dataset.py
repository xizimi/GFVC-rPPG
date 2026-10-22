import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from rppg_gfvc_dataset import estimate_skin_masks


def detrend_signal(arr, wsize):
    norm = np.convolve(np.ones(len(arr)), np.ones(wsize), mode="same")
    mean = np.convolve(arr, np.ones(wsize), mode="same") / norm
    return (arr - mean) / (mean + 1e-15)


def filter_bandpass(arr, srate, band):
    nyq = srate / 2.0
    b, a = signal.butter(5, [band[0] / nyq, band[1] / nyq], "bandpass")
    return signal.filtfilt(b, a, arr)


class UBFC2RPPGDataset(Dataset):
    """
    Reads processed .face.npy + .sig.npy pairs.
    Returns short sequences for rPPG encoder training.
    """

    def __init__(self, root_dir, seq_len=64, step_size=32, is_train=True):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.step_size = step_size
        self.is_train = is_train

        self.clips = []
        face_paths = sorted(glob.glob(
            os.path.join(root_dir, "**", "*.face.npy"), recursive=True
        ))
        for face_path in face_paths:
            sig_path = face_path.replace(".face.npy", ".sig.npy")
            if not os.path.exists(sig_path):
                continue
            n_frames = np.load(face_path, mmap_mode="r").shape[0]
            signal_len = len(np.load(sig_path, allow_pickle=True).item()["rppg_signal"])
            n_frames = min(n_frames, signal_len)
            if n_frames >= seq_len:
                n_windows = (n_frames - seq_len) // step_size + 1
                for i in range(n_windows):
                    self.clips.append((face_path, sig_path, i * step_size))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        face_path, sig_path, start = self.clips[idx]
        end = start + self.seq_len

        face = np.load(face_path, mmap_mode="r")[start:end]  # (T, H, W, 3)
        sig = np.load(sig_path, allow_pickle=True).item()

        rppg = sig["rppg_signal"][start:end].astype(np.float32)  # (T,)
        fps = sig.get("fps", 30.0)
        if isinstance(fps, (list, np.ndarray)):
            fps = fps[0] if len(fps) > 0 else 30.0

        # Detrend and bandpass filter
        try:
            rppg = detrend_signal(rppg, min(30, len(rppg)))
            rppg = filter_bandpass(rppg, fps, (0.7, 3.0))
        except Exception:
            pass

        # Zero-mean/unit-variance preserves the complete waveform and avoids
        # window-dependent [0,1] amplitude distortion.
        rppg = rppg - rppg.mean()
        rstd = rppg.std()
        rppg = rppg / rstd if rstd > 1e-8 else np.zeros_like(rppg)

        # Preprocessing used OpenCV (BGR); models consume RGB. The extractor is
        # trained only on skin pixels to prevent background/identity shortcuts.
        face_rgb = np.ascontiguousarray(np.asarray(face)[..., ::-1])
        mask_path = face_path.replace('.face.npy', '.skin.npy')
        if os.path.exists(mask_path):
            masks = np.asarray(np.load(mask_path, mmap_mode='r')[start:end], dtype=np.float32)
            if masks.ndim == 4:
                masks = masks[..., 0]
        else:
            masks = estimate_skin_masks(face_rgb)
        face = torch.tensor(face_rgb, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        masks = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)
        face = face * masks
        rppg = torch.tensor(rppg, dtype=torch.float32)             # (T,)

        return {"frames": face, "skin_masks": masks, "rppg": rppg}


def get_ubfc2_dataloader(root_dir, batch_size=4, seq_len=64, step_size=32, shuffle=True, num_workers=4):
    dataset = UBFC2RPPGDataset(root_dir, seq_len=seq_len, step_size=step_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
