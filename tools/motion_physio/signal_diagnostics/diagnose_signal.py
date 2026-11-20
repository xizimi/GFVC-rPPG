"""Read-only checkpoint diagnostics. All results stay in the supplied output directory."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

import train
from data import records

JOB = Path(train.__file__).resolve().parent
RUN = JOB / 'run'
sys.path[:0] = [str(RUN / 'code_snapshot'), str(train.PROJECT)]
from modules.skin_rgb import center_skin_rgb, skin_rgb_means


def correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if min(a.std(), b.std()) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def selected():
    rows = records('val')
    for dataset in ('UBFC', 'PURE', 'LGI-PPGI'):
        group = [row for row in rows if row['dataset'] == dataset]
        if dataset == 'LGI-PPGI':
            for activity in sorted({row['activity'] for row in group}):
                clips = [row for row in group if row['activity'] == activity]
                yield clips[0]
                if len(clips) > 1:
                    yield clips[-1]
        else:
            by_subject = {}
            for row in group:
                by_subject.setdefault(row['subject'], []).append(row)
            subjects = list(by_subject.values())
            pool = [group[len(group)//2] for group in subjects]
            pool += [group[0] for group in subjects if len(group) > 1]
            yield from pool[:4]


def inputs(row, device):
    rgb = np.load(row['face'], mmap_mode='r')[..., ::-1]
    mask = np.load(row['mask'], mmap_mode='r')
    saved = np.load(row['supervision_signal'], allow_pickle=True).item()
    bvp = np.asarray(saved['rppg_signal'], dtype=np.float32)
    area = mask.sum(axis=(1, 2), dtype=np.float64)
    direct = np.stack([(frame.astype(np.float64) * weights[..., None]).sum((0, 1)) / count / 255
                       for frame, weights, count in zip(rgb, mask, area)])
    means = []
    for start in range(0, len(rgb), 30):
        video = torch.from_numpy(np.array(rgb[start:start+30])).float().permute(0, 3, 1, 2)[None].to(device)/255
        weights = torch.from_numpy(np.array(mask[start:start+30])).float()[None, :, None].to(device)
        means.append(skin_rgb_means(video, weights))
    model = torch.cat(means, dim=1)
    error = np.max(np.abs(direct - model[0].cpu().numpy()))
    assert error < 1e-6 and len(bvp) == len(rgb)
    return rgb, mask, saved, bvp, model, area, float(error)


def extract(video, fps, supplied_rgb=None):
    from pyVHR.analysis.pipeline import Pipeline
    from pyVHR.extraction.sig_processing import SignalProcessing
    from evaluate_pyvhr_validation import export_uncompressed

    with TemporaryDirectory(prefix='rgb_diagnostic_') as folder:
        path = Path(folder)/'sample.avi'
        export_uncompressed(video, fps, path)
        captured = []
        native = SignalProcessing.extract_holistic

        def roi(instance, filename):
            sequence = native(instance, filename) if supplied_rgb is None else supplied_rgb[:, None, :]
            captured.append(sequence)
            return sequence

        with patch.object(SignalProcessing, 'extract_holistic', roi):
            bvp, _, bpm = Pipeline().run_on_video(str(path), winsize=12, roi_method='convexhull',
                roi_approach='holistic', method='cpu_CHROM', estimate='holistic', patch_size=0,
                RGB_LOW_HIGH_TH=(5,230), Skin_LOW_HIGH_TH=(5,230),
                pre_filt=True, post_filt=True, cuda=True, verb=False)
        assert len(bvp) == len(bpm) == 1
    return np.asarray(bvp[0]).ravel(), float(np.median(np.asarray(bpm))), captured[0][:, 0]


def compact_roi(rgb, masks):
    import mediapipe as mp
    import cv2

    means, areas = [], []
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=.5,
                                        min_tracking_confidence=.5) as mesh:
        for image, mask in zip(rgb, masks):
            result = mesh.process(np.ascontiguousarray(image))
            if not result.multi_face_landmarks:
                raise ValueError('FaceMesh missing a frame in compact ROI')
            xy = np.array([(point.x * 256, point.y * 256)
                           for point in result.multi_face_landmarks[0].landmark])
            radius = round(np.linalg.norm(xy[33] - xy[263]) * .15)
            centers = [xy[[9, 10]].mean(0), xy[[50, 101, 205]].mean(0), xy[[280, 330, 425]].mean(0)]
            region = np.zeros((256, 256), dtype=np.uint8)
            for center in centers:
                cv2.circle(region, tuple(np.rint(center).astype(int)), radius, 1, -1)
            valid = (region > 0) & (mask > 0)
            areas.append(int(valid.sum()))
            means.append(image[valid].mean(0))
    result = np.asarray(means, dtype=np.float32)
    assert np.isfinite(result).all()
    return result, areas


def roi_check(rows, output):
    for row in rows:
        identity = row['sample_id']
        rgb, mask, saved, target, model, area, error = inputs(row, 'cuda')
        fps = float(saved['fps'])
        if 12*fps > len(rgb) + .1:
            fps = len(rgb)/12
        native, native_hr, native_rgb = extract(rgb, fps)
        compact, compact_area = compact_roi(rgb, mask)
        model_rgb = model[0].cpu().numpy()
        whole, whole_hr, _ = extract(rgb, fps, model_rgb*255)
        small, small_hr, _ = extract(rgb, fps, compact)
        assert all(len(seq) == len(target) for seq in (native, whole, small))
        report = dict(sample=identity, dataset=row['dataset'], activity=row['activity'],
                      rgb_max_error=error, original_fps=float(saved['fps']), analysis_fps=fps,
                      mask_area_median=float(np.median(area)), mask_area_cv=float(area.std()/area.mean()),
                      compact_area_median=float(np.median(compact_area)),
                      rgb_ac_std=model_rgb.std(0).tolist(),
                      native_chrom_pcc=correlation(native, target),
                      saved_mask_chrom_pcc=correlation(whole, target),
                      compact_chrom_pcc=correlation(small, target),
                      native_hr=native_hr, saved_mask_hr=whole_hr, compact_hr=small_hr)
        folder = output/identity
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder/'roi_traces.npz', rgb_saved=model_rgb, rgb_native=native_rgb/255,
                 rgb_compact=compact/255, native=native, saved_mask=whole, compact=small, target=target)
        train.save(folder/'roi.json', report)
        print('ROI', json.dumps(report), flush=True)


def load_models(path):
    from modules.generator import OcclusionAwareSPADEGenerator
    from modules.keypoint_detector import KPDetector, HEEstimator
    from modules.rppg import SkinRPPGTeacher

    state = torch.load(path, map_location='cpu')
    params = state['config']['model_params']
    common = params['common_params']
    generator = OcclusionAwareSPADEGenerator(**params['generator_params'], **common)
    generator.load_state_dict(state['generator'], strict=True)
    detector = KPDetector(**params['kp_detector_params'], **common)
    estimator = HEEstimator(**params['he_estimator_params'], **common)
    base = torch.load(train.BASE, map_location='cpu')
    detector.load_state_dict(base['kp_detector'], strict=True)
    estimator.load_state_dict(base['he_estimator'], strict=True)
    teacher = SkinRPPGTeacher()
    teacher.load_state_dict(torch.load(train.TEACHER_PARENT, map_location='cpu')['model'], strict=True)
    for module in (generator, detector, estimator, teacher):
        module.cuda().eval().requires_grad_(False)
    return generator, detector, estimator, teacher, common


@torch.no_grad()
def model_check(rows, output, checkpoint):
    from modules.model import keypoint_transformation

    gen, kp, he, teacher, common = load_models(checkpoint)
    old, _, _, _, _ = load_models(train.PARENT)
    for row in rows:
        rgb, mask, saved, bvp, sequence, _, _ = inputs(row, 'cuda')
        frame = torch.from_numpy(np.array(rgb)).cuda().permute(0, 3, 1, 2)[None].float()/255
        weights = torch.from_numpy(np.array(mask)).float().cuda()[None, :, None]
        label = torch.from_numpy(bvp).cuda()[None]
        label = (label-label.mean())/label.std(unbiased=False)
        shifts = round(len(rgb)*.17)
        centered = center_skin_rgb(sequence)
        amplitude = gen.physiology_modulator.value_amplitude(centered)
        gain = float((label.std(unbiased=False)/amplitude.std(unbiased=False).clamp_min(1e-8)).clamp(1, 1000))
        enc = gen.physiology_encoder(sequence)
        enc_old = old.physiology_encoder(label)
        options = {'off': (gen, None, None), 'rgb': (gen, enc, centered),
                   'shifted': (gen, enc.roll(shifts, 1), centered.roll(shifts, 1)),
                   'gain': (gen, enc, centered*gain), 'bvp28': (old, enc_old, label)}
        teacher_inputs = {name: [] for name in options}
        delta = {name: [] for name in options if name != 'off'}
        source = frame[:, 0]
        canonical = kp(source)
        transform = lambda image: keypoint_transformation(canonical, he(image), common['estimate_jacobian'])
        source_points = transform(source)
        reference = gen.prepare_reference(source)
        reference_old = old.prepare_reference(source)
        for t in range(len(rgb)):
            driving = transform(frame[:, t])
            context = gen.motion_context(source, driving, source_points, reference)
            context_old = {**context, 'reference_context': reference_old}
            base = gen.decode_context(context)['prediction']
            base_u8 = (base*255).clamp(0,255).round().to(torch.uint8)
            for name, (model, condition, value) in options.items():
                result = None if name == 'off' else model.decode_context(
                    context_old if name == 'bvp28' else context, condition[:, t], value[:, t],
                    prediction_base_override=base)
                prediction = base if result is None else result['prediction']
                u8 = (prediction*255).clamp(0,255).round().to(torch.uint8)
                teacher_inputs[name].append(u8[0].cpu().permute(1,2,0).numpy())
                if result is not None:
                    skin = weights[:, t].bool().expand_as(prediction)
                    delta[name].append([float(result['physiology_feature_norm'].mean()),
                                        float(result['physiology_gate'][weights[:, t].bool()].mean()),
                                        float((prediction-base).abs()[skin].mean()*255),
                                        float((u8[skin]!=base_u8[skin]).float().mean())])
        folder = output/row['sample_id']
        folder.mkdir(parents=True, exist_ok=True)
        report = dict(sample=row['sample_id'], checkpoint=str(checkpoint),
                      gain=gain, direct_rgb_amplitude_std=float(amplitude.std(unbiased=False)),
                      bvp_amplitude_std=float(label.std(unbiased=False)),
                      teacher_original_chrom_pcc=correlation(teacher(frame*weights).cpu(), bvp),
                      variants={})
        for name, images in teacher_inputs.items():
            video = np.stack(images)
            tensor = torch.from_numpy(video).cuda().permute(0,3,1,2)[None].float()/255
            wave = teacher(tensor*weights).cpu().numpy().ravel()
            item = dict(teacher_chrom_pcc=correlation(wave, bvp),
                        teacher_shifted_chrom_pcc=correlation(wave, np.roll(bvp, shifts)),
                        teacher_contact_ppg_pcc=correlation(wave, np.load(row['signal'])))
            if name in delta:
                item.update(dict(zip(('feature_delta_rms', 'skin_gate_mean', 'skin_delta_lsb',
                                      'changed_skin_channel_fraction'), np.mean(delta[name], axis=0).tolist())))
            np.save(folder/(name+'.npy'), video)
            report['variants'][name] = item
        train.save(folder/'model.json', report)
        print('MODEL', json.dumps(report), flush=True)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('phase', choices=['roi', 'model', 'extract'])
    args.add_argument('output', type=Path)
    args.add_argument('--checkpoint', type=Path)
    args.add_argument('--shard', type=int, default=0)
    args.add_argument('--shards', type=int, default=1)
    args = args.parse_args()
    if not 0 <= args.shard < args.shards:
        raise SystemExit('Expected 0 <= --shard < --shards')
    if args.phase == 'model' and args.checkpoint is None:
        raise SystemExit('The model phase requires --checkpoint')
    train.seed()
    all_rows = list(selected())
    first = [next(row for row in all_rows if row['dataset'] == d) for d in ('UBFC', 'PURE', 'LGI-PPGI')]
    if args.phase == 'roi':
        roi_check(all_rows[args.shard::args.shards], args.output)
    elif args.phase == 'model':
        model_check(first[args.shard::args.shards], args.output, args.checkpoint)
    else:
        for row in first[args.shard::args.shards]:
            folder = args.output/row['sample_id']
            bvp = np.load(row['supervision_signal'], allow_pickle=True).item()['rppg_signal']
            results = {}
            for name in ('off', 'rgb', 'shifted', 'gain', 'bvp28'):
                video = np.load(folder/(name+'.npy'), mmap_mode='r')
                wave, hr, _ = extract(video, len(video)/12)
                results[name] = dict(bpm=hr, saved_chrom_pcc=correlation(wave, bvp),
                    contact_ppg_pcc=correlation(wave, np.load(row['signal'])),
                    shifted_chrom_pcc=correlation(wave, np.roll(bvp, round(len(video)*.17))))
            train.save(folder/'generated_chrom.json', results)
            print('GENERATED_CHROM', row['sample_id'], json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
