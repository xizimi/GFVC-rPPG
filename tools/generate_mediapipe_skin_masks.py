"""Precompute pyVHR-style skin masks for fixed-crop ``*.face.npy`` clips.

The source clips remain unchanged. Each ``*.skin.npy`` is a [T,H,W] float mask
used only by the masked rePPG-loss branch during training.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def landmark_ids(edge_set):
    return np.asarray(sorted({index for edge in edge_set for index in edge}), np.int32)


def make_skin_mask(rgb, face_mesh, all_face_ids, eye_ids, lip_ids, feather=0):
    """pyVHR convex hull: all facial landmarks minus eyes and lips."""
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    height, width = rgb.shape[:2]
    landmarks = result.multi_face_landmarks[0].landmark
    points = np.asarray([
        (int(point.x * width), int(point.y * height)) for point in landmarks
    ], np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points[all_face_ids]), 1)
    for indices in eye_ids + [lip_ids]:
        cv2.fillConvexPoly(mask, cv2.convexHull(points[indices]), 0)
    if feather:
        kernel = 2 * feather + 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), 0)
    return mask.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--feather', type=int, default=0,
                        help='Gaussian edge radius in pixels; 0 keeps a binary mask.')
    parser.add_argument('--fail-on-missing-face', action='store_true')
    args = parser.parse_args()

    import mediapipe as mp
    face_module = mp.solutions.face_mesh
    all_face_ids = np.arange(468, dtype=np.int32)
    eye_ids = [
        landmark_ids(face_module.FACEMESH_LEFT_EYE),
        landmark_ids(face_module.FACEMESH_RIGHT_EYE),
    ]
    lip_ids = landmark_ids(face_module.FACEMESH_LIPS)
    face_mesh = face_module.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    paths = sorted(glob.glob(os.path.join(args.root, '**', '*.face.npy'), recursive=True))
    if not paths:
        raise RuntimeError('No .face.npy clips found below ' + args.root)
    for path in paths:
        output_path = path.replace('.face.npy', '.skin.npy')
        if os.path.exists(output_path) and not args.overwrite:
            continue
        bgr_video = np.load(path, mmap_mode='r')
        masks, previous, missed = [], None, 0
        for frame in bgr_video:
            # Preprocessing stores OpenCV BGR frames. Face Mesh requires RGB.
            rgb = np.ascontiguousarray(np.asarray(frame, np.uint8)[..., ::-1])
            mask = make_skin_mask(rgb, face_mesh, all_face_ids, eye_ids, lip_ids,
                                  feather=args.feather)
            if mask is None:
                missed += 1
                if args.fail_on_missing_face or previous is None:
                    raise RuntimeError('Face Mesh failed for ' + path)
                mask = previous.copy()
            previous = mask
            masks.append(mask)
        np.save(output_path, np.stack(masks).astype(np.float32), allow_pickle=False)
        print('%s frames=%d missed=%d coverage=%.4f' % (
            output_path, len(masks), missed, float(np.mean(masks))))
    face_mesh.close()


if __name__ == '__main__':
    main()
