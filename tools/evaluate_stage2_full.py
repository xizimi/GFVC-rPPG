"""Evaluate the selected GFVC-rPPG checkpoint on every fixed UBFC window.

This reports unweighted physiological metrics so that checkpoint-selection losses
are not mistaken for Pearson correlation or BPM error.  It never writes model
weights and keeps preserve/rewrite targets separate.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# When invoked as ``python tools/evaluate_stage2_full.py``, Python otherwise
# places ``tools/`` rather than the repository root on sys.path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import HEEstimator, KPDetector
from modules.model import GFVCRPPGSequenceModel
from modules.rppg import (SkinRPPGTeacher, normalized_power_spectrum,
                          physiological_losses)
from rppg_gfvc_dataset import GFVCRPPGWindowDataset
from train_gfvc_rppg import load_component_checkpoint, move_batch


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
    teacher_state = torch.load(args.teacher_checkpoint, map_location="cpu")
    teacher.load_state_dict(teacher_state["model"])
    model = GFVCRPPGSequenceModel(
        kp_detector, he_estimator, generator, teacher,
        loss_weights=config["train_params"]["physiology_loss_weights"],
        estimate_jacobian=common["estimate_jacobian"],
    ).to(device).eval()
    return model, saved


def hard_peak_bpm(waveform, fps):
    """Non-differentiable spectral-peak BPM for evaluation only."""
    power, frequencies = normalized_power_spectrum(waveform.unsqueeze(0), fps)
    return float(frequencies[int(power[0].argmax())] * 60.0)


@torch.no_grad()
def evaluate(model, dataset, device, total_epochs, checkpoint_epoch):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                        pin_memory=device.type == "cuda")
    values = {key: [] for key in (
        "pearson", "waveform", "spectral", "kl",
        "bpm_mae_peak_target", "bpm_mae_peak_hr_metadata",
        "video_l1", "baseline_l1", "delta_l1", "outside_skin_delta_l1",
    )}
    for batch in loader:
        batch = move_batch(batch, device)
        _, output = model(batch, epoch=checkpoint_epoch - 1,
                          total_epochs=total_epochs)
        physiology = physiological_losses(
            output["rppg_extracted"], batch["target_rppg"], batch["fps"],
            target_bpm=batch["target_bpm"],
        )
        values["pearson"].append(1.0 - float(physiology["pearson"]))
        values["waveform"].append(float(physiology["waveform"]))
        values["spectral"].append(float(physiology["spectral"]))
        values["kl"].append(float(physiology["kl"]))
        fps = float(batch["fps"][0])
        recovered_bpm = hard_peak_bpm(output["rppg_extracted"][0], fps)
        target_bpm = hard_peak_bpm(batch["target_rppg"][0], fps)
        values["bpm_mae_peak_target"].append(abs(recovered_bpm - target_bpm))
        metadata_bpm = float(batch["target_bpm"][0])
        if np.isfinite(metadata_bpm):
            values["bpm_mae_peak_hr_metadata"].append(
                abs(recovered_bpm - metadata_bpm)
            )
        prediction = output["prediction"]
        baseline = output["prediction_base"]
        delta = output["physiology_delta"]
        masks = batch["skin_masks"]
        values["video_l1"].append(float(F.l1_loss(prediction, batch["driving_video"])))
        values["baseline_l1"].append(float(F.l1_loss(baseline, batch["driving_video"])))
        values["delta_l1"].append(float(delta.abs().mean()))
        values["outside_skin_delta_l1"].append(
            float((delta.abs() * (1.0 - masks)).mean())
        )
    return {
        key: (float(np.mean(metric)) if metric else None)
        for key, metric in values.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_checkpoint", default="checkpoint/skin_rppg_teacher.pt")
    parser.add_argument("--gfvc_checkpoint", default="checkpoint/00000189-checkpoint.pth.tar")
    parser.add_argument("--config", default="config/vox-256.yaml")
    parser.add_argument("--val_root", default="/opt/data/private/datasets/UBFC2_fixed256_fv2v/val")
    parser.add_argument("--test_root", default="/opt/data/private/datasets/UBFC2_fixed256_fv2v/test")
    parser.add_argument("--sequence_length", type=int, default=150)
    parser.add_argument("--total_epochs", type=int, default=30)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, saved = build_model(args, device)
    report = {"checkpoint_epoch": int(saved["epoch"]), "best_val": float(saved["best_val"])}
    for split, root in (("val", args.val_root), ("test", args.test_root)):
        for mode, probability in (("preserve", 0.0), ("rewrite", 1.0)):
            dataset = GFVCRPPGWindowDataset(
                root, sequence_length=args.sequence_length,
                rewrite_probability=probability, require_semantic_masks=True,
                signal_preprocess="reppg", deterministic=True, seed=0,
            )
            metrics = evaluate(model, dataset, device, args.total_epochs,
                               int(saved["epoch"]))
            report[f"{split}_{mode}"] = metrics
            print(f"{split}_{mode}: {metrics}", flush=True)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
