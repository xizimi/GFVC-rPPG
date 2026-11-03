"""Teacher-free, fixed-pose full12s BVP A/B/zero controls after a checkpoint."""
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
from modules.rppg import negative_pearson_loss


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();seed_all()
    saved=torch.load(args.checkpoint,map_location='cpu');run=saved['run_config'];cfg=saved['config']['model_params']
    common=cfg['common_params'];gen=OcclusionAwareSPADEGenerator(**cfg['generator_params'],**common)
    load_generator_compatible(gen,saved['generator'],declared_mode=run['fusion_mode']);gen.eval().cuda().requires_grad_(False)
    kp=KPDetector(**cfg['kp_detector_params'],**common);he=HEEstimator(**cfg['he_estimator_params'],**common)
    base=torch.load(run['base_checkpoint']['path'],map_location='cpu')
    kp.load_state_dict(base['kp_detector'],strict=True);he.load_state_dict(base['he_estimator'],strict=True)
    kp.eval().cuda().requires_grad_(False);he.eval().cuda().requires_grad_(False)
    preserve=NativePhysioClips('val',0.,run['seed'])[0];rewrite=NativePhysioClips('val',1.,run['seed'])[0]
    source=preserve['source'].unsqueeze(0).cuda();n=len(preserve['target_rppg']);middle=n//2
    driving=preserve['driving_video'][middle].unsqueeze(0).cuda()
    targets={name:wave.unsqueeze(0).cuda() for name,wave in [('A',preserve['target_rppg']),
        ('B',rewrite['target_rppg']),('zero',torch.zeros_like(preserve['target_rppg']))]}
    mask=preserve['skin_masks'][middle].unsqueeze(0).cuda()
    sequences={name:[] for name in targets};floats={name:[] for name in targets};quantized={name:[] for name in targets}
    with torch.inference_mode():
        canonical=kp(source);ks=keypoint_transformation(canonical,he(source),common['estimate_jacobian'])
        kd=keypoint_transformation(canonical,he(driving),common['estimate_jacobian'])
        ref=gen.prepare_reference(source);motion=gen.motion_context(source,kd,ks,ref)
        plain=gen.decoder(motion['features']);encoded={k:gen.physiology_encoder(v) for k,v in targets.items()}
        max_zero=0.;difference_sum=0.;difference_max=0.;zero_rgb_max=0.;saturation=0.;pixels=0
        for t in range(n):
            frame={}
            for name,target in targets.items():
                result=gen.decode_context(motion,encoded[name][:,t],target[:,t]);pred=result['prediction'];frame[name]=pred
                assert torch.isfinite(pred).all()
                q=pred.mul(255).round().clamp(0,255).to(torch.uint8).float()/255.
                denom=mask.sum().clamp_min(1e-8)
                floats[name].append(float((pred[:,1:2]*mask).sum()/denom))
                quantized[name].append(float((q[:,1:2]*mask).sum()/denom))
                if t in (0,middle,n-1):sequences[name].append(pred[0].cpu().numpy())
                saturation+=int(((pred<=.001)|(pred>=.999)).sum());pixels+=pred.numel()
            max_zero=max(max_zero,float((frame['zero']-plain).abs().max()))
            difference=frame['A']-frame['B'];difference_sum+=float(difference.abs().mean())
            difference_max=max(difference_max,float(difference.abs().max()))
            zero_rgb_max=max(zero_rgb_max,float((frame['A']-frame['zero']).abs().max()))
        assert max_zero<1e-6
    assert difference_max>1e-7,'Trained model ignores changed BVP'
    waveform={}
    for name,target in targets.items():
        a=torch.tensor(floats[name]).unsqueeze(0);b=torch.tensor(quantized[name]).unsqueeze(0);truth=target.cpu()
        waveform[name]=dict(target=truth[0].tolist(),green_float=floats[name],green_uint8=quantized[name],
            green_std_float=float(a.std()),green_std_uint8=float(b.std()),
            zero_lag_pearson_float=1.-float(negative_pearson_loss(a,truth)) if name!='zero' else None,
            zero_lag_pearson_uint8=1.-float(negative_pearson_loss(b,truth)) if name!='zero' else None)
    args.output.mkdir(parents=True,exist_ok=True);save(args.output/'control_waveforms.json',waveform)
    report=dict(status='passed',checkpoint=str(args.checkpoint),sha256=sha(args.checkpoint),
        fusion_mode=run['fusion_mode'],sample_id=preserve['sample_id'],frames=n,seconds=12,
        fixed_motion='same middle-frame motion used for every time point and condition',
        target_A='same held-out clip contact PPG',target_B=rewrite['conditioning_donor'],target_zero='all zeros',
        teacher_used=False,mask_used_as_generator_input=False,mask_used_for_diagnostics_only=True,
        maximum_zero_condition_deviation=max_zero,A_B_mean_abs_pixel_difference=difference_sum/n,
        A_B_max_abs_pixel_difference=difference_max,A_zero_max_abs_pixel_difference=zero_rgb_max,
        saturated_fraction=saturation/max(pixels,1),
        green_diagnostics={name:{k:v for k,v in x.items() if k not in ('target','green_float','green_uint8')} for name,x in waveform.items()},
        interpretation='Input responsiveness and independent waveform diagnostics; not evidence of superior final pyVHR or visual quality')
    save(args.output/'control_report.json',report)
    canvas=Image.new('RGB',(3*256,3*280),(245,245,245));draw=ImageDraw.Draw(canvas)
    for row,name in enumerate(('A','B','zero')):
        for col,frame in enumerate(sequences[name]):
            rgb=(np.clip(frame,0,1)*255).round().astype(np.uint8).transpose(1,2,0)
            canvas.paste(Image.fromarray(rgb),(col*256,row*280+24))
            draw.text((col*256+5,row*280+5),name+' t='+str((0,middle,n-1)[col]),fill=(0,0,0))
    canvas.save(args.output/'fixed_motion_BVP_controls.png')
    print({k:v for k,v in report.items() if k!='green_diagnostics'},flush=True)


if __name__=='__main__':main()
