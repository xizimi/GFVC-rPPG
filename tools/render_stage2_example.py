"""Render one held-out GFVC-rPPG example for qualitative inspection.

Produces a lossless FFV1 AVI (input and recovered frames side by side), an
animated GIF preview, and a BVP comparison plot.  This script is read-only with
respect to model checkpoints and dataset files.
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import HEEstimator, KPDetector
from modules.model import GFVCRPPGSequenceModel
from modules.rppg import SkinRPPGTeacher, physiological_losses
from rppg_gfvc_dataset import GFVCRPPGWindowDataset
from train_gfvc_rppg import load_component_checkpoint, move_batch


def standardize(values):
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / max(values.std(), 1e-6)


def peak_bpm(waveform, fps):
    signal = standardize(waveform)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal)) ** 2
    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    return float(frequencies[valid][np.argmax(power[valid])] * 60.0)


def build_model(args, device):
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    common = config["model_params"]["common_params"]
    generator = OcclusionAwareSPADEGenerator(
        **config["model_params"]["generator_params"], **common
    )
    kp_detector = KPDetector(
        **config["model_params"]["kp_detector_params"], **common
    )
    he_estimator = HEEstimator(
        **config["model_params"]["he_estimator_params"], **common
    )
    load_component_checkpoint(args.gfvc_checkpoint, generator, kp_detector,
                              he_estimator)
    saved = torch.load(args.checkpoint, map_location="cpu")
    generator.load_state_dict(saved["generator"], strict=True)
    teacher = SkinRPPGTeacher()
    teacher.load_state_dict(torch.load(args.teacher_checkpoint,
                                       map_location="cpu")["model"])
    return GFVCRPPGSequenceModel(
        kp_detector, he_estimator, generator, teacher,
        loss_weights=config["train_params"]["physiology_loss_weights"],
        estimate_jacobian=common["estimate_jacobian"],
    ).to(device).eval(), saved


def panel(left, right, title_left, title_right):
    height, width, _ = left.shape
    result = np.concatenate([left, right], axis=1)
    cv2.rectangle(result, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.rectangle(result, (width, 0), (2 * width, 30), (0, 0, 0), -1)
    cv2.putText(result, title_left, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, title_right, (width + 8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_root", default="/opt/data/private/datasets/UBFC2_fixed256_fv2v/test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", choices=("preserve", "rewrite"), default="preserve")
    parser.add_argument("--teacher_checkpoint", default="checkpoint/skin_rppg_teacher.pt")
    parser.add_argument("--gfvc_checkpoint", default="checkpoint/00000189-checkpoint.pth.tar")
    parser.add_argument("--config", default="config/vox-256.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, saved = build_model(args, device)
    dataset = GFVCRPPGWindowDataset(
        args.split_root, sequence_length=150,
        rewrite_probability=1.0 if args.mode == "rewrite" else 0.0,
        require_semantic_masks=True, signal_preprocess="reppg",
        deterministic=True, seed=0,
    )
    if not 0 <= args.index < len(dataset):
        raise IndexError("index must be in [0, %d)" % len(dataset))
    raw = dataset[args.index]
    batch = move_batch(next(iter(DataLoader([raw], batch_size=1))), device)
    with torch.no_grad():
        _, output = model(batch, epoch=int(saved["epoch"]) - 1, total_epochs=30)
    input_video = (batch["driving_video"][0].permute(0, 2, 3, 1).cpu().numpy()
                   .clip(0, 1) * 255).round().astype(np.uint8)
    recovered_video = (output["prediction"][0].permute(0, 2, 3, 1).cpu().numpy()
                       .clip(0, 1) * 255).round().astype(np.uint8)
    target = batch["target_rppg"][0].cpu().numpy()
    original = batch["original_rppg"][0].cpu().numpy()
    extracted = output["rppg_extracted"][0].cpu().numpy()
    fps = float(batch["fps"][0])
    correlation = float(np.corrcoef(standardize(target), standardize(extracted))[0, 1])
    physiology = physiological_losses(
        output["rppg_extracted"], batch["target_rppg"], batch["fps"],
        target_bpm=batch["target_bpm"],
    )
    prefix = "%s_%s_%03d" % (args.mode, raw["name"].replace(".face.npy", ""), args.index)
    avi_path = os.path.join(args.output_dir, prefix + "_input_vs_recovered.avi")
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"FFV1"), fps,
                             (input_video.shape[2] * 2, input_video.shape[1]))
    if not writer.isOpened():
        raise RuntimeError("FFV1 writer unavailable; no lossy fallback is used")
    previews = []
    for i, (left, right) in enumerate(zip(input_video, recovered_video)):
        frame = panel(left, right, "Input video", "Recovered video")
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i % 3 == 0:
            previews.append(Image.fromarray(frame))
    writer.release()
    gif_path = os.path.join(args.output_dir, prefix + "_input_vs_recovered.gif")
    previews[0].save(gif_path, save_all=True, append_images=previews[1:],
                     duration=int(3000.0 / fps), loop=0)

    seconds = np.arange(len(target)) / fps
    plt.figure(figsize=(12, 4.2), dpi=160)
    plt.plot(seconds, standardize(original), color="#7f7f7f", linewidth=1.4,
             label="Input cPPG/BVP")
    if args.mode == "rewrite":
        plt.plot(seconds, standardize(target), color="#d55e00", linewidth=1.7,
                 label="Injected target BVP")
    plt.plot(seconds, standardize(extracted), color="#0072b2", linewidth=1.5,
             linestyle="--", label="BVP extracted from recovered video")
    plt.xlabel("Time (s)")
    plt.ylabel("Standardized amplitude")
    plt.title("%s | target-vs-recovered Pearson %.3f | target BPM %.1f, recovered BPM %.1f" % (
        args.mode.capitalize(), correlation, peak_bpm(target, fps), peak_bpm(extracted, fps)))
    plt.grid(alpha=0.22)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, prefix + "_bvp_comparison.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    report = {
        "sample": raw["name"], "mode": args.mode, "checkpoint_epoch": int(saved["epoch"]),
        "fps": fps, "target_vs_extracted_pearson": correlation,
        "target_bpm": peak_bpm(target, fps), "extracted_bpm": peak_bpm(extracted, fps),
        "bpm_mae": float(physiology["bpm"]), "avi": avi_path,
        "gif": gif_path, "plot": plot_path,
    }
    with open(os.path.join(args.output_dir, prefix + "_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
