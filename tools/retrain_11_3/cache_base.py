"""Cache detached float32 frozen-GFVC RGB for current UBFC train/val clips only."""
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from common import PROJECT,ROOT,RUN,datasets,seed_all,save,read,sha,event
from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector,HEEstimator
from modules.model import keypoint_transformation
from train_gfvc_rppg import load_component_checkpoint

CACHE=Path('/opt/data/private/datasets/GFVC-rPPG_generated/ubfc_11_3_frozen_base_float32_cache_loader_layout_v2')
ARCH=PROJECT/'checkpoint/gfvc_rppg_stage2_lowrank_v2.pth.tar.best'
BASE=PROJECT/'checkpoint/00000189-checkpoint.pth.tar'


def architecture():
    saved=torch.load(ARCH,map_location='cpu')
    cfg=saved['config'];del saved
    return cfg


def components(device):
    cfg=architecture();mp=cfg['model_params'];common=mp['common_params']
    generator=OcclusionAwareSPADEGenerator(**mp['generator_params'],**common)
    kp=KPDetector(**mp['kp_detector_params'],**common);he=HEEstimator(**mp['he_estimator_params'],**common)
    load_component_checkpoint(str(BASE),generator,kp,he)
    for model in (generator,kp,he):model.eval().requires_grad_(False).to(device)
    return cfg,generator,kp,he


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=CACHE)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    seed_all();train,val,provenance=datasets();device=torch.device('cuda:0')
    cfg,generator,kp,he=components(device)
    config=dict(provenance=provenance,base_checkpoint=str(BASE),base_checkpoint_sha256=sha(BASE),
        architecture_from_metadata_only=str(ARCH),old_physiology_weights_loaded=False,
        source_sha256={str(p):sha(p) for p in (PROJECT/'modules/generator.py',PROJECT/'modules/model.py',PROJECT/'modules/rppg.py',Path(__file__))},
        storage='unrounded float32 native RGB [T,3,256,256]; no quantization or codec',
        source='each clip own first frame, frozen eval backbone, original training keypoint transformation',
        frame_batch_size=1,smoke=args.smoke,
        tensor_layout='source NCHW contiguous; driving per-frame channels_last, matching indexed loader')
    if args.output.exists() and (args.output/'config.json').exists():assert read(args.output/'config.json')==config
    args.output.mkdir(parents=True,exist_ok=True);save(args.output/'config.json',config)
    rows=[dict(r,split='train') for r in train.rows]+[dict(r,split='val') for r in val.rows]
    if args.smoke:rows=rows[:1]
    manifest=read(args.output/'manifest.json') if (args.output/'manifest.json').exists() else {}
    for number,row in enumerate(rows,1):
        sid=row['sample_id'];a,b=row['start'],row['end'];n=b-a
        path=ROOT/row['path'];ref=np.load(path/'frames.npy',mmap_mode='r')
        source_hash=hashlib.sha256(np.ascontiguousarray(ref[a:b]).tobytes()).hexdigest()
        if sid in manifest:
            assert manifest[sid]['source_rgb_sha256']==source_hash
            assert (args.output/manifest[sid]['path']).exists();continue
        tag=hashlib.sha256(sid.encode()).hexdigest()[:20]
        folder=args.output/row['split'];folder.mkdir(exist_ok=True)
        destination=folder/(tag+'.npy');partial=folder/(tag+'.partial.npy')
        array=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(n,3,256,256))
        start=time.monotonic()
        with torch.no_grad():
            # Default collation makes source contiguous but retains channels_last
            # within each driving frame. Match both to retain the same CUDA kernels.
            video=torch.from_numpy(np.array(ref[a:b])).float().permute(0,3,1,2)/255.
            source=video[0:1].to(device).contiguous();canonical=kp(source)
            source_kp=keypoint_transformation(canonical,he(source),cfg['model_params']['common_params']['estimate_jacobian'])
            limit=min(n,4) if args.smoke else n
            for i in range(limit):
                driver=video[i:i+1].to(device).contiguous(memory_format=torch.channels_last)
                driver_kp=keypoint_transformation(canonical,he(driver),cfg['model_params']['common_params']['estimate_jacobian'])
                base=generator(source,kp_source=source_kp,kp_driving=driver_kp)['prediction_base']
                assert base.dtype==torch.float32 and torch.isfinite(base).all()
                array[i]=base[0].cpu().numpy()
        if args.smoke:
            array.flush();del array
            save(args.output/'smoke_report.json',dict(status='passed',frames=limit,seconds=time.monotonic()-start,
                 max_cuda_bytes=torch.cuda.max_memory_allocated(),shape=[n,3,256,256],not_a_complete_cache=True))
            event(read(args.output/'smoke_report.json'));return
        array.flush();del array;partial.replace(destination)
        manifest[sid]=dict(sample_id=sid,path=destination.relative_to(args.output).as_posix(),split=row['split'],
            source_path=row['path'],start=a,end=b,frames=n,source_rgb_sha256=source_hash,
            base_file_sha256=sha(destination),seconds=time.monotonic()-start)
        save(args.output/'manifest.json',manifest)
        event(dict(stage='cache',completed=len(manifest),total=len(rows),sample_id=sid,seconds=time.monotonic()-start))
        save(args.output/'progress.json',dict(status='caching',completed=len(manifest),total=len(rows)))
    assert len(manifest)==len(rows)
    save(args.output/'CACHE_COMPLETE.json',dict(status='passed',clips=len(manifest),frames=sum(x['frames'] for x in manifest.values()),
         logical_bytes=sum((args.output/x['path']).stat().st_size for x in manifest.values())))
    event(dict(stage='cache',status='complete',clips=len(manifest)))


if __name__=='__main__':main()
