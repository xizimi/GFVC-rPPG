# Visual and physiological evaluation

The single entry point is `tools/evaluate_all_metrics.py`. It evaluates POS and
CHROM BVP/HR metrics, then evaluates frame fidelity and video quality without
repeating the visual models for each rPPG method.

Install the visual-only dependencies in the evaluation environment:

```bash
pip install -r requirements-visual-eval.txt
```

FVD needs the StyleGAN-V `i3d_torchscript.pt` model, a TorchScript port of the
TensorFlow-Hub I3D network used by the original FVD implementation. The path is
always explicit; the evaluator never silently downloads or changes weights.

Example:

```bash
python tools/evaluate_all_metrics.py \
  --manifest /path/to/manifest.json \
  --output-dir /path/to/metrics \
  --device cuda:0 \
  --pyvhr-python /opt/conda/envs/pyvhr/bin/python \
  --pyvhr-pythonpath /opt/data/private/project/pyVHR-master \
  --visual-python /opt/data/private/envs/gfvc-visual-eval/bin/python \
  --pyiqa-weights-dir /opt/data/private/models/pyiqa \
  --methods cpu_POS cpu_CHROM \
  --video-kinds input generated \
  --bpm-window-seconds 5 \
  --fvd-i3d-path /path/to/i3d_torchscript.pt
```

Outputs:

- `bvp_metrics_detail.json`: per-window physiological results.
- `bvp_metrics_summary.json`: sample/subject/group physiological results.
- `visual_metrics.json`: PSNR, SSIM, DISTS, LPIPS, MANIQA and FVD.
- `all_metrics.json`: combined report.

PSNR, color-SSIM (`ssimc`), DISTS and LPIPS compare frame-aligned
input/generated RGB frames. The JSON records the pyiqa/PyTorch versions and
the SHA-256 digest of the I3D checkpoint used for reproducibility.
MANIQA evaluates generated frames without a reference. Frame scores are first
averaged per sample, then per subject, and subjects are equally weighted in the
reported group statistics. FVD is a set metric: by default, each declared
manifest interval is divided into 16-frame clips and real/generated I3D feature
distributions are compared separately for `preserve` and `rewrite` modes.
