"""Export actual learned basis/gates/features and image-space differences."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tools/retrain_11_3'))
from common import seed_all,save,sha
from tools.motion_physio.data import NativePhysioClips
from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector,HEEstimator
from modules.model import keypoint_transformation
from modules.physiology_motion import load_generator_compatible


def image(x):return Image.fromarray((np.clip(x,0,1)*255).round().astype(np.uint8).transpose(1,2,0))


def gray(x):return image(np.repeat(np.clip(x[None],0,1),3,axis=0))


def heat(x,lo,hi):
    v=np.clip((x-lo)/max(hi-lo,1e-12),0,1)
    return image(np.stack((v,1.-np.abs(2.*v-1.),1.-v)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();seed_all()
    saved=torch.load(args.checkpoint,map_location='cpu');run=saved['run_config'];cfg=saved['config']['model_params']
    common=cfg['common_params'];gen=OcclusionAwareSPADEGenerator(**cfg['generator_params'],**common)
    load_generator_compatible(gen,saved['generator'],declared_mode=run['fusion_mode']);gen.eval().cuda().requires_grad_(False)
    kp=KPDetector(**cfg['kp_detector_params'],**common);he=HEEstimator(**cfg['he_estimator_params'],**common)
    original=torch.load(run['base_checkpoint']['path'],map_location='cpu')
    kp.load_state_dict(original['kp_detector'],strict=True);he.load_state_dict(original['he_estimator'],strict=True)
    kp.eval().cuda().requires_grad_(False);he.eval().cuda().requires_grad_(False)
    sample=NativePhysioClips('val',0.,run['seed'])[0];index=int(sample['target_rppg'].abs().argmax())
    source=sample['source'].unsqueeze(0).cuda();driving=sample['driving_video'][index].unsqueeze(0).cuda()
    with torch.inference_mode():
        canonical=kp(source);ks=keypoint_transformation(canonical,he(source),common['estimate_jacobian'])
        kd=keypoint_transformation(canonical,he(driving),common['estimate_jacobian'])
        reference=gen.prepare_reference(source)
        out=gen(source,kd,ks,target_rppg=sample['target_rppg'].unsqueeze(0).cuda(),
                rppg_frame_index=index,reference_context=reference)
    arrays=dict(source_RGB=source[0].cpu().numpy(),driving_RGB=driving[0].cpu().numpy(),
        prediction_RGB=out['prediction'][0].cpu().numpy(),baseline_RGB=out['prediction_base'][0].cpu().numpy(),
        actual_RGB_delta=(out['prediction'][0]-out['prediction_base'][0]).cpu().numpy(),
        GT_skin_ROI_diagnostic_only=sample['skin_masks'][index].numpy(),
        predicted_skin=out['physiology_skin_mask'][0].cpu().numpy(),
        effective_gate=out.get('physiology_gate',out['physiology_skin_mask'])[0].cpu().numpy(),
        spatial_response=out['physiology_response_maps'][0].cpu().numpy(),
        deformation=out['deformation'][0].cpu().numpy())
    for key in ('physiology_feature_delta','physiology_basis_3d'):
        if key in out:arrays[key]=out[key][0].cpu().numpy()
    if 'reference_basis' in reference:arrays['reference_basis_3d']=reference['reference_basis'][0].cpu().numpy()
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'learned_physiology_maps.npz',**arrays)
    delta=np.abs(arrays['actual_RGB_delta']).mean(0);mask=arrays['GT_skin_ROI_diagnostic_only'][0]
    denom=lambda a:max(float(a.sum()),1e-8)
    report=dict(status='exported',checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        source_sha256=sha(Path(__file__)),sample_id=sample['sample_id'],frame=index,
        frame_selection='fixed first validation clip, maximum absolute target BVP sample',
        mode=run['fusion_mode'],rank=run['rank'],semantic_mask_input_to_generator=False,
        maps_are_learned_modulation_bases_not_physical_blood_flow=True,
        inside_skin_mean_abs_RGB_delta=float((delta*mask).sum()/denom(mask)),
        outside_skin_mean_abs_RGB_delta=float((delta*(1-mask)).sum()/denom(1-mask)),
        outside_fraction_of_total_abs_delta=float((delta*(1-mask)).sum()/denom(delta)),
        shape={k:list(v.shape) for k,v in arrays.items()},
        note='Single deterministic diagnostic frame; no claim that skin localization or final model quality has passed')
    save(args.output/'spatial_report.json',report)
    limit=max(float(np.percentile(delta,99)),1e-8)
    tiles=[('Reference RGB',image(arrays['source_RGB'])),('Unmodulated same-checkpoint RGB',image(arrays['baseline_RGB'])),
        ('Conditioned RGB',image(arrays['prediction_RGB'])),('GT skin ROI (diagnostic only)',gray(mask)),
        ('Predicted skin gate',gray(arrays['predicted_skin'][0])),('Effective gate',gray(arrays['effective_gate'][0])),
        ('Learned basis 1 [-1,1]',heat(arrays['spatial_response'][0],-1,1)),
        ('Learned basis 2 [-1,1]',heat(arrays['spatial_response'][1],-1,1)),
        ('Mean |RGB delta| 0..%.6f'%limit,heat(delta,0,limit))]
    canvas=Image.new('RGB',(768,840),(245,245,245));draw=ImageDraw.Draw(canvas)
    for i,(title,tile) in enumerate(tiles):
        x=(i%3)*256;y=(i//3)*280;canvas.paste(tile.resize((256,256)),(x,y+24));draw.text((x+3,y+6),title,fill=(0,0,0))
    canvas.save(args.output/'learned_physiology_maps.png')
    print({k:v for k,v in report.items() if k!='shape'},flush=True)


if __name__=='__main__':main()
