# Skin RGB temporal conditioning

This option extends the motion-aligned physiology model used for the epoch-28
experiment. Set `physiology_params.input_mode: skin_rgb` and
`fusion_mode: feature_motion_aligned` in the SPADE generator configuration.
The training entry point exposes the equivalent flags
`--physiology_input_mode skin_rgb --physiology_fusion_mode feature_motion_aligned`.
Existing configurations default to `bvp` for compatibility with saved checkpoints.

For every driving frame, `skin_rgb_means(video, masks)` calculates three skin-area
means as `sum(mask * channel) / sum(mask)`. Inputs are RGB floats in `[0,1]` with
shape `[B,T,3,H,W]` and masks `[B,T,1,H,W]` in `[0,1]`. Empty masks raise a
descriptive error. The sequence has shape `[B,T,3]` and preserves frame order.
Each channel is centered and divided by its population standard deviation over
the actual conditioning window, with a `1e-6` floor. Constant channels become zero.
The encoder also projects all three original means and all three original standard
deviations through a learned `Linear(6,64)` and adds the result to every temporal
feature. These statistics retain the original color baseline and channel amplitudes.
Training and inference use the same preprocessing without filtering or resampling.

The temporal encoder takes three channels and returns `[B,T,64]` features.
The modulation head concatenates the temporal feature with the current
raw RGB vector in `[0,1]`. A learned `Linear(3,1)` predicts its amplitude before
`tanh`. This per-frame path receives unstandardized skin-area means.
The reference basis, existing motion warp, skin/visibility gating and GFVC decoder
continue to operate through the same feature modulation path.

`GFVCRPPGSequenceModel` computes conditioning from `driving_video` and `skin_masks`.
Its scalar `target_rppg` remains the supervision and metric reference, including
when that field stores a prepared CHROM label. Set `--rewrite_probability 0` for
this experiment. RGB conditioning uses the driving clip's signal, so substituted
scalar donor labels would mismatch its input. Validation uses the preserve set.

For inference, compute the complete window's means with `skin_rgb_means`, then use
`generator.forward_skin_rgb(source, kp_driving, kp_source, rgb_means, frame_index)`.
The encoder therefore needs the RGB sequence and its preprocessing mask on the
sender side. The decoder receives RGB conditions and the existing motion inputs.

To initialize from epoch 28, use the native training entry point with
`--contact_clip_root`, the original base/teacher arguments, and
`--physiology_transfer_checkpoint /path/to/epoch_0028.pth.tar`.
Select `--training_stage physiology` for the first RGB fine-tuning run.
The transfer copies every shape-compatible generator tensor, plus the checkpoint's
keypoint detector and head estimator when present. For native generator-only
checkpoints, it verifies the original base-checkpoint SHA-256 and retains those
frozen motion components from that base. It initializes the new three-channel encoder
input, the widened modulation-head input, the RGB statistics projection and the
amplitude projection afresh.
It checks the exact source schema and logs copied/initialized tensor names and
checkpoint provenance. The optimizer starts fresh. Use `--resume` for a subsequent
RGB checkpoint. BVP-to-RGB initialization requires fine-tuning before evaluation. The earlier RGB
prototype checkpoint lacks the statistics projection; strict resume detects this
schema difference. All scalar-BVP checkpoint tensors retain their original schema.

Regression checks run with
`python -m unittest discover -s tests -p 'test_motion_physio.py'` and
`python -m unittest discover -s tests -p 'test_skin_rgb_conditioning.py'`.
