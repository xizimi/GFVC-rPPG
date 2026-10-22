"""Generate conservative forehead/cheek skin masks for processed face clips."""
import argparse, glob, os
import cv2
import numpy as np

def fallback(h, w):
    yy, xx = np.ogrid[:h, :w]
    return ((((xx-w*.5)/(w*.38))**2 + ((yy-h*.48)/(h*.46))**2) <= 1).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    import mediapipe as mp
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5)
    oval = np.array(list({i for e in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in e}), np.int32)
    eyes = [np.array(list({i for e in edges for i in e}), np.int32) for edges in
             (mp.solutions.face_mesh.FACEMESH_LEFT_EYE, mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)]
    lips = np.array(list({i for e in mp.solutions.face_mesh.FACEMESH_LIPS for i in e}), np.int32)
    paths = glob.glob(os.path.join(args.root, '**', '*.face.npy'), recursive=True)
    for path in paths:
        out = path.replace('.face.npy', '.skin.npy')
        if os.path.exists(out) and not args.overwrite: continue
        video = np.load(path, mmap_mode='r')
        masks, previous = [], None
        for frame in video:
            rgb = np.asarray(frame, np.uint8)
            result = mesh.process(rgb)
            mask = None
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                pts = np.array([(int(p.x*rgb.shape[1]), int(p.y*rgb.shape[0])) for p in lm])
                mask = np.zeros(rgb.shape[:2], np.uint8)
                cv2.fillPoly(mask, [cv2.convexHull(pts[oval])], 1)
                for ids in eyes + [lips]: cv2.fillPoly(mask, [cv2.convexHull(pts[ids])], 0)
                mask = cv2.GaussianBlur(mask.astype(np.float32), (9,9), 0)
            if mask is None: mask = previous if previous is not None else fallback(rgb.shape[0], rgb.shape[1])
            previous = mask
            masks.append(np.clip(mask, 0, 1))
        np.save(out, np.stack(masks).astype(np.float32))
        print(out)
    mesh.close()
if __name__ == '__main__': main()
