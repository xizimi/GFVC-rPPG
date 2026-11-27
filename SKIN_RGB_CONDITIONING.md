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
The encoder projects the three original channel standard deviations through a
learned `Linear(3,64)` and adds the result to every temporal feature. The physiology
condition retains channel amplitudes and is invariant to constant per-channel RGB
offsets, up to floating-point precision. Average RGB stays outside this condition.
Training and inference use the same preprocessing without filtering or resampling.

The temporal encoder takes three channels and returns `[B,T,64]` features.
The modulation head concatenates the temporal feature with the current
centered RGB vector `rgb(t) - mean(rgb)`. A learned `Linear(3,1,bias=False)`
predicts its signed amplitude before `tanh`. Centering uses the actual input window
and retains the original numerical amplitude and channel amplitude ratios. The
modulation head receives this centered vector as well. A constant RGB window gives
zero direct amplitude drive. Training and inference share `center_skin_rgb`.
The reference basis, existing motion warp, skin/visibility gating and GFVC decoder
continue to operate through the same feature modulation path.

The Epoch 29 + 10 PS-FV2V run uses `physiology_params.rgb_amplitude_gain: 100.0`.
This gain scales the signed RGB amplitude projection before `tanh`. For inference
with that checkpoint, construct the generator from the checkpoint's stored
`config` to recover the experiment value. The default gain is `1.0` and applies
to the RGB projection only.

`GFVCRPPGSequenceModel` computes conditioning from `driving_video` and `skin_masks`.
Its scalar `target_rppg` remains the supervision and metric reference. The current
experiment uses aligned contact PPG labels. This network change does not convert
existing label files; the loader must supply the intended PPG source. Set `--rewrite_probability 0` for
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
prototype checkpoints have different statistics projections or an amplitude bias;
strict resume detects these schema differences. All scalar-BVP checkpoint tensors retain their original schema.

Regression checks run with
`python -m unittest discover -s tests -p 'test_motion_physio.py'` and
`python -m unittest discover -s tests -p 'test_skin_rgb_conditioning.py'`.

Baseline-invariance tests hold the source features, motion and visibility fixed.
Appearance features can still depend on the source image. Time-varying illumination
and motion artifacts remain part of the measured RGB sequence.
