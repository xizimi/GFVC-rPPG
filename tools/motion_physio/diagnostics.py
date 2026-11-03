"""Zero-lag waveform and image diagnostics independent of teacher outputs."""
import torch
from modules.rppg import negative_pearson_loss


@torch.no_grad()
def diagnose(output,batch):
    prediction=output['prediction'];base=output['prediction_base'];delta=prediction-base
    masks=batch['skin_masks'];target=batch['target_rppg'];eps=1e-8
    mean_rgb=(prediction*masks).sum((-2,-1))/masks.sum((-2,-1)).clamp_min(eps)
    base_rgb=(base*masks).sum((-2,-1))/masks.sum((-2,-1)).clamp_min(eps)
    green=mean_rgb[:,:,1];delta_green=mean_rgb[:,:,1]-base_rgb[:,:,1]
    mse=(prediction-base).square().mean();mse_driving=(prediction-batch['driving_video']).square().mean()
    outside=1.-masks
    result=dict(green_contact_pearson=1.-float(negative_pearson_loss(green,target)),
        delta_green_contact_pearson=1.-float(negative_pearson_loss(delta_green,target)),
        rgb_delta_l1=float(delta.abs().mean()),rgb_delta_max=float(delta.abs().max()),
        inside_skin_delta_l1=float((delta.abs()*masks).sum()/(3*masks.sum()).clamp_min(eps)),
        outside_skin_delta_l1=float((delta.abs()*outside).sum()/(3*outside.sum()).clamp_min(eps)),
        baseline_psnr_db=float(-10*torch.log10(mse.clamp_min(1e-12))),
        driving_psnr_db=float(-10*torch.log10(mse_driving.clamp_min(1e-12))),
        saturated_fraction=float(((prediction<=.001)|(prediction>=.999)).float().mean()),
        skin_gate_mean=float(output['physiology_skin_mask'].mean()))
    if 'physiology_feature_norm' in output:
        result['feature_delta_rms']=float(output['physiology_feature_norm'].mean())
        result['skin_and_occlusion_gate_mean']=float(output['physiology_gate'].mean())
    return result
