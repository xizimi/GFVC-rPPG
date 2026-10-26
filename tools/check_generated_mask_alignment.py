"""Check whether driving-video skin masks align with GFVC baseline frames."""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator
from modules.model import keypoint_transformation
from rppg_gfvc_dataset import GFVCRPPGWindowDataset
from tools.generate_mediapipe_skin_masks import landmark_ids, make_skin_mask
from train_gfvc_rppg import load_component_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='config/vox-256.yaml')
    parser.add_argument('--data_root', default='/opt/data/private/datasets/UBFC2_fixed256_fv2v/val')
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--frames', default='0,75,149')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    common = config['model_params']['common_params']
    generator = OcclusionAwareSPADEGenerator(
        **config['model_params']['generator_params'], **common)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'], **common)
    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'], **common)
    load_component_checkpoint(args.checkpoint, generator, kp_detector,
                              he_estimator)
    generator = generator.cuda().eval()
    kp_detector = kp_detector.cuda().eval()
    he_estimator = he_estimator.cuda().eval()

    import mediapipe as mp
    mesh_module = mp.solutions.face_mesh
    all_face_ids = np.arange(468, dtype=np.int32)
    eye_ids = [landmark_ids(mesh_module.FACEMESH_LEFT_EYE),
               landmark_ids(mesh_module.FACEMESH_RIGHT_EYE)]
    lip_ids = landmark_ids(mesh_module.FACEMESH_LIPS)
    mesh = mesh_module.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True)
    dataset = GFVCRPPGWindowDataset(args.data_root, rewrite_probability=0.0,
                                    deterministic=True)
    sample = dataset[args.sample_index]
    source = sample['source'].unsqueeze(0).cuda()
    with torch.no_grad():
        canonical = kp_detector(source)
        source_he = he_estimator(source)
        kp_source = keypoint_transformation(canonical, source_he,
                                             common['estimate_jacobian'])
        for index in [int(value) for value in args.frames.split(',')]:
            driving = sample['driving_video'][index].unsqueeze(0).cuda()
            driving_he = he_estimator(driving)
            kp_driving = keypoint_transformation(
                canonical, driving_he, common['estimate_jacobian'])
            generated = generator(source, kp_driving=kp_driving,
                                  kp_source=kp_source)['prediction_base'][0]
            rgb = (generated.permute(1, 2, 0).cpu().numpy() * 255).clip(
                0, 255).astype(np.uint8)
            generated_mask = make_skin_mask(
                rgb, mesh, all_face_ids, eye_ids, lip_ids)
            original_mask = sample['skin_masks'][index, 0].numpy() > 0.5
            if generated_mask is None:
                print('frame=%d generated_face_detection=failed' % index)
                continue
            generated_mask = generated_mask > 0.5
            intersection = np.logical_and(original_mask, generated_mask).sum()
            union = np.logical_or(original_mask, generated_mask).sum()
            print('frame=%d mask_iou=%.3f driving_coverage=%.3f generated_coverage=%.3f' %
                  (index, intersection / max(union, 1),
                   original_mask.mean(), generated_mask.mean()), flush=True)
    mesh.close()


if __name__ == '__main__':
    main()
