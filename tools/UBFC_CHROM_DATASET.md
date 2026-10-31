# UBFC CHROM dataset workflow

The dataset standard is subject-disjoint, uses signed zero-lag Pearson
correlation, and never shifts or polarity-flips CHROM to improve selection.

## 1. Create split-aware fixed crops

```bash
python tools/preprocess_ubfc2.py \
  --root /opt/data/private/datasets/UBFC2 \
  --out /opt/data/private/datasets/UBFC2_PCC075_candidates \
  --splits splits.json \
  --chrom_context_mode \
  --context_frames 60 --target_frames 360 \
  --size 256
```

This produces non-overlapping target blocks with 60 frames of context on each
side: 60 + 360 + 60 = 480 frames. Context blocks advance by 360 frames.
Durations and later metrics use each video's true FPS; 30 FPS is not assumed.

## 2. Reuse the existing masks

```bash
python tools/reuse_ubfc_skin_masks.py \
  --old-root /opt/data/private/datasets/UBFC2_fixed256_processed \
  --new-root /opt/data/private/datasets/UBFC2_PCC075_candidates \
  --old-step 75 --new-step 360 \
  --overwrite
```

The reuse step maps both datasets back to the original source-frame index. It
also requires every matched old/new 256x256 face frame to be pixel-identical
and requires the saved timestamps to agree before copying a mask. For source
frames present in two overlapping old windows, the occurrence closest to the
old window centre is used.

## 3. Extract CHROM, select, and write labels

Run in the pyVHR environment:

```bash
PYTHONPATH=/opt/data/private/project/pyVHR-master \
/opt/conda/envs/pyvhr/bin/python tools/curate_ubfc_chrom_context.py \
  --input-root /opt/data/private/datasets/UBFC2_PCC075_candidates \
  --output-root /opt/data/private/datasets/UBFC2_CHROM_PCC075 \
  --report reports/ubfc_chrom_pcc075.json \
  --threshold 0.75 --cuda
```

CHROM sees all 480 frames, then only its central 360 frames are retained. Test
uses the full central window. Train/validation create 150-frame windows at
starts 0, 75, and 150 within the central region. Every final window must have
signed zero-lag `PCC > 0.75`. Face and mask arrays use exactly the same slices;
the signal dictionary retains FPS/time/HR, stores CHROM in `rppg_signal`, and
keeps the original BVP in `reference_rppg_signal` for independent validation.

## 4. Validate before training

```bash
python tools/validate_ubfc_chrom_dataset.py \
  --root /opt/data/private/datasets/UBFC2_CHROM_PCC075 \
  --candidate-root /opt/data/private/datasets/UBFC2_PCC075_candidates \
  --threshold 0.75 \
  --output reports/ubfc_chrom_validation.json
```

## 5. Evaluate the model

Training keeps `--sequence_length 150 --signal_preprocess none`. Full evaluation uses 150 frames for
validation and 360 frames for test, while BPM, spectra, and durations use each
sample's stored FPS:

```bash
python tools/evaluate_stage2_full.py \
  --checkpoint checkpoint/gfvc_rppg_stage2.pth.tar \
  --val_sequence_length 150 --test_sequence_length 360 \
  --output reports/stage2_full.json
```
