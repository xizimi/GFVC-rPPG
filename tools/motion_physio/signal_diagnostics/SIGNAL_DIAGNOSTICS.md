# Diagnose skin RGB conditioning

These probes use the CHROM-supervised experiment with the frozen teacher from
the BVP epoch-28 run. They read the experiment's `train.py`, `data.py`,
`run/code_snapshot`, checkpoints, frame arrays, masks, CHROM labels, and
contact PPG. The ROI and waveform probes also use the experiment's
`evaluate_pyvhr_validation.py` and pyVHR environment. Point `JOB` to the
experiment directory on the machine that holds those inputs.

```bash
JOB=/path/to/experiment
TOOLS=/path/to/GFVC-rPPG/tools/motion_physio/signal_diagnostics
OUT="$JOB/signal_diagnostics"
CKPT="$JOB/run/branch/epoch_0002.pth.tar"
source /path/to/pyvhr_env.sh
export NUMBA_NUM_THREADS=4
export PYTHONPATH="$JOB:${PYTHONPATH:-}"

python "$TOOLS/diagnose_signal.py" roi "$OUT"
python "$TOOLS/diagnose_signal.py" model "$OUT" --checkpoint "$CKPT"
python "$TOOLS/diagnose_signal.py" extract "$OUT"
python "$TOOLS/diagnose_fit.py" "$CKPT" "$OUT/fit_1.json" --gain 1
python "$TOOLS/diagnose_fit.py" "$CKPT" "$OUT/fit_100.json" --gain 100
```

The ROI probe compares independent RGB mean calculations and runs CHROM on
the saved skin mask, a compact forehead-and-cheek ROI, and the original
pyVHR convex hull. The model probe holds motion fixed and compares disabled
modulation, RGB conditioning, temporally shifted conditioning, amplified RGB
conditioning, and the BVP checkpoint. It records changes in floating-point
RGB and after rounding to eight bits. The extraction probe runs CHROM on the
diagnostic videos and compares against the saved BVP and contact PPG.

The fit probe performs 12 optimizer steps and measures the physiological
and regularization gradients on the selected conditioning parameters. Its
gain applies only to the amplitude projection. The gains in these commands
are experimental controls. The probes save results in the output directory,
including intermediate video arrays for the extraction probe.
