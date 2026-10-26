# GFVC-rPPG experiment status

## Data and protocol

- Server environment: RTX 4090 with `/opt/conda/envs/fv2v/bin/python`.
- Dataset: uncompressed fixed 256x256 UBFC clips, 150 frames per window.
- Subject-disjoint split: 778/120/128 windows from 32/5/5 train/validation/test subjects.
- MediaPipe convex-hull skin masks are used only for teacher measurement and loss; they are not a receiver input or transmitted payload.

## SkinRPPGTeacher

The earlier rePPG Hourglass checkpoint did not transfer to UBFC reliably, so it
was not used for stage-2 training. A differentiable `SkinRPPGTeacher` was
trained on real masked UBFC videos and frozen after validation.

Best teacher checkpoint: `checkpoint/skin_rppg_teacher.pt` (epoch 8).

| Split | Pearson | HR MAE |
| --- | ---: | ---: |
| Validation | 0.887 | 4.25 BPM |
| Held-out test | 0.903 | 5.61 BPM |

## GFVC conditional generation

The stage-2 model freezes the GFVC motion/generation backbone and the teacher.
Only `TemporalRPPGEncoder` and `SkinPhysiologyResidual` are optimized. The
teacher requires training mode for cuDNN GRU input gradients, but its parameters
remain frozen; it uses GroupNorm and no dropout.

Thirty stage-2 epochs completed. The selected checkpoint is
`checkpoint/gfvc_rppg_stage2_skin.pth.tar.best` (epoch 26). Complete-window
evaluation uses all validation and held-out test windows:

| Split | Mode | Pearson | HR MAE |
| --- | --- | ---: | ---: |
| Validation | Preserve | 0.981 | 3.96 BPM |
| Validation | Rewrite | 0.986 | 4.22 BPM |
| Held-out test | Preserve | 0.986 | 5.07 BPM |
| Held-out test | Rewrite | 0.987 | 5.63 BPM |

The mean generated-video L1 is effectively unchanged from the frozen GFVC
baseline. These physiological results are measured by the frozen teacher;
independent POS/CHROM/GREEN or independent-network evaluation is still required
before making a claim of real-world physiological fidelity.
