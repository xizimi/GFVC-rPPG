import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy import signal

from modules.rppg import RPPGEncoder, physiological_losses
from rppg_dataset import UBFC2RPPGDataset, detrend_signal, filter_bandpass


def pearson_loss(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    num = (pred * target).sum(dim=-1)
    den = pred.norm(dim=-1) * target.norm(dim=-1) + 1e-8
    return 1.0 - (num / den).mean()


def compute_hr(waveforms, fps):
    """Estimate heart rate (BPM) from signal."""
    B, T = waveforms.shape
    hr_list = []
    for i in range(B):
        sig = waveforms[i].detach().cpu().numpy()
        freqs, psd = signal.periodogram(sig, fs=fps)
        valid = (freqs >= 0.7) & (freqs <= 3.0)
        if valid.sum() == 0:
            hr_list.append(0.0)
            continue
        peak_freq = freqs[valid][np.argmax(psd[valid])]
        hr_list.append(peak_freq * 60.0)
    return np.array(hr_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/opt/data/private/datasets/UBFC2_processed")
    parser.add_argument("--save_path", default="checkpoint/rppg_encoder.pth.tar")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--step_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = RPPGEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    # If split directories exist, use them; otherwise use root
    if os.path.isdir(train_dir):
        train_set = UBFC2RPPGDataset(train_dir, seq_len=args.seq_len, step_size=args.step_size)
        val_set = UBFC2RPPGDataset(val_dir, seq_len=args.seq_len, step_size=args.step_size) if os.path.isdir(val_dir) else None
    else:
        train_set = UBFC2RPPGDataset(args.data_root, seq_len=args.seq_len, step_size=args.step_size)
        val_set = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_set else None

    print(f"Train samples: {len(train_set)}")
    if val_set:
        print(f"Val samples: {len(val_set)}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            frames = batch["frames"].to(device)     # (B, T, 3, H, W)
            target = batch["rppg"].to(device)       # (B, T)

            pred = model(frames)                    # (B, T)
            components = physiological_losses(pred, target, fps=30.0)
            progress = float(epoch) / max(float(args.num_epochs), 1.0)
            alpha = 0.1 * (0.5 ** progress)
            beta = 1.0 * (5.0 ** progress)
            loss = (alpha * components["pearson"] +
                    0.5 * components["waveform"] +
                    beta * (components["spectral"] + 0.1 * components["kl"]) +
                    0.01 * components["bpm"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        msg = f"Epoch {epoch+1}: train_loss={avg_loss:.4f}"

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_hr_err = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    frames = batch["frames"].to(device)
                    target = batch["rppg"].to(device)
                    pred = model(frames)
                    components = physiological_losses(pred, target, fps=30.0)
                    val_loss += (components["pearson"] + components["waveform"] +
                                 components["spectral"] + 0.1 * components["kl"]).item()
                    hr_pred = compute_hr(pred, fps=30.0)
                    hr_tgt = compute_hr(target, fps=30.0)
                    val_hr_err += np.abs(hr_pred - hr_tgt).mean()
                    n_val += 1
            val_loss /= max(n_val, 1)
            val_hr_err /= max(n_val, 1)
            msg += f" val_loss={val_loss:.4f} val_hr_err={val_hr_err:.1f}bpm"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.save_path)
        else:
            torch.save(model.state_dict(), args.save_path)

        print(msg)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
