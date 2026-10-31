"""Evaluate frame fidelity and video-distribution quality for GFVC outputs.

The input is the same JSON manifest used by ``evaluate_pyvhr_bvp.py``.  Each
row must contain ``sample``, ``mode``, ``input_video`` and ``generated_video``.
The two videos must be frame aligned.  If ``metric_start_frame`` and
``metric_length`` are present, only that interval is evaluated.

Frame metrics are computed with IQA-PyTorch (``pyiqa``): PSNR, SSIM, DISTS and
LPIPS compare each generated RGB frame to the corresponding input frame;
MANIQA is no-reference and therefore receives only the generated frame.  FVD
uses the StyleGAN-V TorchScript port of the official TensorFlow-Hub I3D model.
It is a distribution metric and is deliberately reported only for collections
of real/generated clips, never as a per-video score.
"""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


FRAME_METRICS = ("psnr_db", "ssim", "dists", "lpips", "maniqa")
METRIC_MODELS = {
    "psnr_db": "psnr",
    # pyiqa's `ssim` is luminance-only; `ssimc` is the RGB/color variant.
    "ssim": "ssimc",
    "dists": "dists",
    "lpips": "lpips",
    "maniqa": "maniqa",
}
WEIGHT_FILENAMES = {
    "dists": "DISTS_weights-f5e65c96.pth",
    "lpips": "LPIPS_v0.1_alex-df73285e.pth",
    "maniqa": "ckpt_koniq10k.pt",
}


def subject_for(row):
    if row.get("subject"):
        return str(row["subject"])
    match = re.search(r"subject\d+", str(row.get("sample", "")))
    return match.group(0) if match else str(row["sample"])


def read_video_rgb(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("Cannot open video: %s" % path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid or empty video: %s" % path)
    return np.stack(frames), fps


def metric_interval(row, frame_count):
    start = int(row.get("metric_start_frame", 0))
    length = int(row.get("metric_length", frame_count - start))
    if start < 0 or length < 1 or start + length > frame_count:
        raise ValueError("Invalid metric interval for %s" % row["sample"])
    return start, length


def load_aligned_pair(row):
    reference, reference_fps = read_video_rgb(Path(row["input_video"]))
    generated, generated_fps = read_video_rgb(Path(row["generated_video"]))
    if reference.shape != generated.shape:
        raise ValueError("Input/generated shape mismatch for %s: %s vs %s" %
                         (row["sample"], reference.shape, generated.shape))
    if abs(reference_fps - generated_fps) > 0.05:
        raise ValueError("Input/generated FPS mismatch for %s" % row["sample"])
    if "fps" in row and abs(reference_fps - float(row["fps"])) > 0.05:
        raise ValueError("Manifest/video FPS mismatch for %s" % row["sample"])
    start, length = metric_interval(row, len(reference))
    return (reference[start:start + length], generated[start:start + length],
            reference_fps, start)


class PyiqaFrameMetrics:
    """Lazy IQA-PyTorch wrapper with explicit RGB [0, 1] tensor inputs."""

    def __init__(self, names, device, weights_dir=None, maniqa_batch_size=2):
        import pyiqa
        import torch

        self.torch = torch
        self.pyiqa_version = getattr(pyiqa, "__version__", "unknown")
        self.torch_version = getattr(torch, "__version__", "unknown")
        self.device = torch.device(device)
        self.models = {}
        self.weight_sha256 = {}
        self.maniqa_batch_size = int(maniqa_batch_size)
        for name in names:
            model_options = {}
            weight_path = None
            if weights_dir is not None and name in WEIGHT_FILENAMES:
                weight_path = Path(weights_dir) / WEIGHT_FILENAMES[name]
                if not weight_path.is_file():
                    raise FileNotFoundError("Missing pyiqa weight: %s" % weight_path)
                if name != "maniqa":
                    model_options["pretrained_model_path"] = str(weight_path)
                self.weight_sha256[name] = file_sha256(weight_path)
            try:
                if name == "maniqa" and weight_path is not None:
                    # The MANIQA checkpoint contains the complete ViT. Avoid a
                    # redundant online timm initialization before loading it.
                    # pyiqa's explicit-path branch expects a `params` wrapper,
                    # while its own Koniq file is a direct state dict. Redirect
                    # only the default Koniq URL so the correct branch is used.
                    import timm
                    from pyiqa.archs import maniqa_arch

                    original_create_model = timm.create_model
                    original_koniq_url = maniqa_arch.default_model_urls["koniq"]

                    def create_uninitialized_model(model_name, *args, **kwargs):
                        kwargs["pretrained"] = False
                        return original_create_model(model_name, *args, **kwargs)

                    timm.create_model = create_uninitialized_model
                    maniqa_arch.default_model_urls["koniq"] = str(weight_path)
                    try:
                        model = pyiqa.create_metric(
                            METRIC_MODELS[name], device=self.device,
                            **model_options)
                    finally:
                        timm.create_model = original_create_model
                        maniqa_arch.default_model_urls["koniq"] = original_koniq_url
                else:
                    model = pyiqa.create_metric(
                        METRIC_MODELS[name], device=self.device,
                        **model_options)
                self.models[name] = model.eval()
            except Exception as error:
                raise RuntimeError("Cannot initialize pyiqa metric %s: %s" %
                                   (name, error)) from error

    def _tensor(self, frames):
        return self.torch.from_numpy(
            np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
        ).to(self.device, dtype=self.torch.float32).div_(255.0)

    def score_batch(self, reference, generated):
        reference_tensor = self._tensor(reference)
        generated_tensor = self._tensor(generated)
        result = {}
        with self.torch.no_grad():
            for name, model in self.models.items():
                metric_batch_size = (self.maniqa_batch_size if name == "maniqa"
                                     else len(generated_tensor))
                pieces = []
                for start in range(0, len(generated_tensor), metric_batch_size):
                    end = start + metric_batch_size
                    values = (model(generated_tensor[start:end]) if name == "maniqa"
                              else model(generated_tensor[start:end],
                                         reference_tensor[start:end]))
                    pieces.append(values.detach().float().cpu().numpy().reshape(-1))
                values = np.concatenate(pieces)
                if len(values) != len(generated):
                    raise ValueError("%s returned %d values for a %d-frame batch" %
                                     (name, len(values), len(generated)))
                if not np.isfinite(values).all():
                    raise ValueError("%s returned NaN or Inf" % name)
                result[name] = values.astype(np.float64)
        return result


def evaluate_pair(row, runner, metric_names, batch_size, frame_step,
                  save_frame_scores=False):
    reference, generated, fps, interval_start = load_aligned_pair(row)
    selected = np.arange(0, len(reference), frame_step, dtype=np.int64)
    scores = {name: [] for name in metric_names}
    for offset in range(0, len(selected), batch_size):
        indices = selected[offset:offset + batch_size]
        batch_scores = runner.score_batch(reference[indices], generated[indices])
        for name in metric_names:
            scores[name].extend(float(value) for value in batch_scores[name])
    record = {
        "sample": row["sample"], "subject": subject_for(row),
        "mode": row["mode"], "fps": fps,
        "metric_start_frame": interval_start,
        "metric_length": len(reference), "frame_step": frame_step,
        "evaluated_frame_count": len(selected), "status": "ok",
    }
    for name in FRAME_METRICS:
        values = scores.get(name)
        record[name] = float(np.mean(values)) if values else None
        record[name + "_std_frames"] = float(np.std(values)) if values else None
    if save_frame_scores:
        record["frame_indices"] = (selected + interval_start).tolist()
        record["frame_scores"] = scores
    return record, reference, generated


def fvd_clip_starts(frame_count, clip_frames, stride):
    if frame_count < clip_frames:
        return []
    starts = list(range(0, frame_count - clip_frames + 1, stride))
    last = frame_count - clip_frames
    if starts[-1] != last:
        starts.append(last)
    return starts


class I3DFeatureExtractor:
    """StyleGAN-V I3D feature extractor compatible with official FVD."""

    def __init__(self, model_path, device, resolution=224):
        import torch

        self.torch = torch
        self.functional = torch.nn.functional
        self.device = torch.device(device)
        self.resolution = int(resolution)
        self.model_sha256 = file_sha256(model_path)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def _preprocess(self, clips):
        # uint8 [B,T,H,W,C] -> float [B,C,T,224,224] in [-1,1].
        tensor = self.torch.from_numpy(
            np.ascontiguousarray(clips.transpose(0, 4, 1, 2, 3))
        ).to(self.device, dtype=self.torch.float32).div_(255.0)
        _, _, _, height, width = tensor.shape
        scale = self.resolution / min(height, width)
        target_height = int(math.ceil(height * scale))
        target_width = int(math.ceil(width * scale))
        batch, channels, times = tensor.shape[:3]
        images = tensor.permute(0, 2, 1, 3, 4).reshape(
            batch * times, channels, height, width)
        images = self.functional.interpolate(
            images, size=(target_height, target_width), mode="bilinear",
            align_corners=False)
        top = (target_height - self.resolution) // 2
        left = (target_width - self.resolution) // 2
        images = images[:, :, top:top + self.resolution,
                        left:left + self.resolution]
        tensor = images.reshape(batch, times, channels, self.resolution,
                                self.resolution).permute(0, 2, 1, 3, 4)
        return tensor.mul_(2.0).sub_(1.0).contiguous()

    def extract(self, clips, batch_size):
        features = []
        with self.torch.no_grad():
            for offset in range(0, len(clips), batch_size):
                batch = self._preprocess(clips[offset:offset + batch_size])
                try:
                    output = self.model(x=batch, rescale=False, resize=False,
                                        return_features=True)
                except TypeError:
                    output = self.model(batch, rescale=False, resize=False,
                                        return_features=True)
                output = output.detach().float().cpu().numpy()
                features.append(output.reshape(len(batch), -1))
        return np.concatenate(features, axis=0)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def symmetric_matrix_sqrt(matrix):
    """Return the PSD square root of a real symmetric matrix."""
    matrix = (np.asarray(matrix, dtype=np.float64) +
              np.asarray(matrix, dtype=np.float64).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)).dot(eigenvectors.T)


def frechet_distance(real_features, generated_features):
    real = np.asarray(real_features, dtype=np.float64)
    generated = np.asarray(generated_features, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("FVD features must be [sample, feature] with equal width")
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("FVD requires at least two real and two generated clips")
    real_mean, generated_mean = real.mean(axis=0), generated.mean(axis=0)
    real_cov = np.cov(real, rowvar=False)
    generated_cov = np.cov(generated, rowvar=False)
    # Tr(sqrt(C_r C_g)) equals Tr(sqrt(sqrt(C_r) C_g sqrt(C_r))).
    # The latter matrix is symmetric PSD, so eigh avoids complex sqrtm noise.
    real_cov_sqrt = symmetric_matrix_sqrt(real_cov)
    covariance_product_sqrt = symmetric_matrix_sqrt(
        real_cov_sqrt.dot(generated_cov).dot(real_cov_sqrt))
    distance = (np.square(real_mean - generated_mean).sum() +
                np.trace(real_cov + generated_cov) -
                2.0 * np.trace(covariance_product_sqrt))
    return float(max(distance, 0.0))


def extract_fvd_pair(extractor, reference, generated, clip_frames, stride,
                     batch_size):
    starts = fvd_clip_starts(len(reference), clip_frames, stride)
    if not starts:
        raise ValueError("Metric interval has fewer frames than one FVD clip")
    reference_clips = np.stack([
        reference[start:start + clip_frames] for start in starts
    ])
    reference_features = extractor.extract(reference_clips, batch_size)
    del reference_clips
    generated_clips = np.stack([
        generated[start:start + clip_frames] for start in starts
    ])
    generated_features = extractor.extract(generated_clips, batch_size)
    return reference_features, generated_features, len(starts)


def aggregate_frame_metrics(records):
    by_subject = defaultdict(list)
    for row in records:
        by_subject[(row["subject"], row["mode"])].append(row)
    subject_rows = []
    for (subject, mode), rows in sorted(by_subject.items()):
        item = {"subject": subject, "mode": mode, "sample_count": len(rows)}
        for metric in FRAME_METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            item[metric] = float(np.mean(values)) if values else None
        subject_rows.append(item)
    groups = []
    by_mode = defaultdict(list)
    for row in subject_rows:
        by_mode[row["mode"]].append(row)
    for mode, rows in sorted(by_mode.items()):
        item = {"mode": mode, "subject_count": len(rows),
                "sample_count": sum(row["sample_count"] for row in rows)}
        for metric in FRAME_METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            item[metric] = ({"mean": float(np.mean(values)),
                             "std": float(np.std(values)),
                             "median": float(np.median(values)),
                             "count": len(values)} if values else None)
        groups.append(item)
    return subject_rows, groups


def metric_definitions():
    return {
        "psnr_db": "frame-aligned RGB PSNR in dB; higher is better",
        "ssim": "frame-aligned RGB structural similarity; higher is better",
        "dists": "frame-aligned Deep Image Structure and Texture Similarity distance; lower is better",
        "lpips": "frame-aligned Learned Perceptual Image Patch Similarity distance; lower is better",
        "maniqa": "MANIQA no-reference score of generated frames only; higher is better",
        "fvd": "I3D feature-distribution Frechet Video Distance over clip sets; lower is better",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metrics", nargs="+", choices=FRAME_METRICS,
                        default=FRAME_METRICS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--maniqa-batch-size", type=int, default=2,
                        help="Internal MANIQA batch; kept small due multi-crop inference")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Evaluate every Nth aligned frame (default: every frame)")
    parser.add_argument("--save-frame-scores", action="store_true")
    parser.add_argument("--pyiqa-weights-dir", type=Path,
                        help="Optional offline DISTS/LPIPS/MANIQA weight directory")
    parser.add_argument("--skip-fvd", action="store_true")
    parser.add_argument("--fvd-i3d-path", type=Path,
                        help="StyleGAN-V i3d_torchscript.pt; required unless --skip-fvd")
    parser.add_argument("--fvd-clip-frames", type=int, default=16)
    parser.add_argument("--fvd-clip-stride", type=int, default=16)
    parser.add_argument("--fvd-batch-size", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.maniqa_batch_size, args.frame_step, args.fvd_clip_frames,
           args.fvd_clip_stride, args.fvd_batch_size) <= 0:
        parser.error("Batch sizes, frame steps and FVD clip sizes must be positive")
    if not args.skip_fvd and args.fvd_i3d_path is None:
        parser.error("--fvd-i3d-path is required unless --skip-fvd is used")
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list")
    runner = PyiqaFrameMetrics(args.metrics, args.device, args.pyiqa_weights_dir,
                               args.maniqa_batch_size)
    extractor = (None if args.skip_fvd else
                 I3DFeatureExtractor(args.fvd_i3d_path, args.device))
    fvd_features = defaultdict(lambda: {"reference": [], "generated": [],
                                        "clip_count": 0})
    details = []
    for index, row in enumerate(rows, 1):
        if row.get("mode") not in ("preserve", "rewrite"):
            raise ValueError("mode must be preserve or rewrite")
        record, reference, generated = evaluate_pair(
            row, runner, args.metrics, args.batch_size, args.frame_step,
            args.save_frame_scores)
        if not args.skip_fvd:
            real_features, generated_features, clip_count = extract_fvd_pair(
                extractor, reference, generated, args.fvd_clip_frames,
                args.fvd_clip_stride, args.fvd_batch_size)
            fvd_features[row["mode"]]["reference"].append(real_features)
            fvd_features[row["mode"]]["generated"].append(generated_features)
            fvd_features[row["mode"]]["clip_count"] += clip_count
            record["fvd_clip_count"] = clip_count
        details.append(record)
        print("%d/%d %s" % (index, len(rows), row["sample"]), flush=True)
    subject_rows, groups = aggregate_frame_metrics(details)
    fvd_results = []
    if not args.skip_fvd:
        for mode, features in sorted(fvd_features.items()):
            real_features = np.concatenate(features["reference"], axis=0)
            generated_features = np.concatenate(features["generated"], axis=0)
            fvd_results.append({
                "mode": mode, "fvd": frechet_distance(real_features, generated_features),
                "clip_count": features["clip_count"],
                "clip_frames": args.fvd_clip_frames,
                "clip_stride": args.fvd_clip_stride,
                "feature_dimension": int(real_features.shape[1]),
                "implementation": "StyleGAN-V TorchScript port of official TF-Hub I3D",
                "i3d_sha256": extractor.model_sha256,
            })
    result = {
        "metric_definitions": metric_definitions(),
        "frame_metric_implementation": {
            "library": "IQA-PyTorch (pyiqa)",
            "pyiqa_version": runner.pyiqa_version,
            "torch_version": runner.torch_version,
            "input": "frame-aligned RGB tensors in [0,1]",
            "models": {name: METRIC_MODELS[name] for name in args.metrics},
            "weight_sha256": runner.weight_sha256,
        },
        "frame_aggregation": "frame mean -> sample; sample mean -> subject; equal-weight subject statistics",
        "alignment": "same-index input/generated frames; manifest metric interval only",
        "sample_level": details, "subject_level": subject_rows,
        "groups": groups, "fvd_groups": fvd_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("visual metrics: %s" % args.output)


if __name__ == "__main__":
    main()
