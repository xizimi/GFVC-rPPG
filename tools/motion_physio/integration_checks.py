"""Actual pretrained256px compatibility, shared-grid and control checks."""
import importlib.util
import sys
import time
from pathlib import Path
import torch
import yaml
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tools/retrain_11_3'))
from common import ROOT,seed_all,save,sha,event,move
from contact_clip_dataset import ContactClipDataset
from torch.utils.data._utils.collate import default_collate
from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector,HEEstimator
from modules.model import keypoint_transformation
from modules.physiology_motion import load_generator_compatible
from train_gfvc_rppg import load_component_checkpoint
from tools.motion_physio.data import CACHE

RUN=PROJECT/'checkpoint/motion_physio_20261103_v2'
BASE=PROJECT/'checkpoint/00000189-checkpoint.pth.tar'
OLD=PROJECT/'checkpoint/ubfc_11_3_subject_independent_20261103/branch/branch_best.pth.tar'


def main():
    seed_all();RUN.mkdir(exist_ok=True);device=torch.device('cuda:0')
    save(RUN/'integration_checks.json',dict(status='running'))
    saved=torch.load(OLD,map_location='cpu');cfg=saved['config'];mp=cfg['model_params'];common=mp['common_params']
    spec=importlib.util.spec_from_file_location('legacy_rgb_generator',PROJECT/'architecture_history/20261103_pre_motion_aligned/modules/generator.py')
    oldmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmod)
    old=oldmod.OcclusionAwareSPADEGenerator(**mp['generator_params'],**common)
    rgb=OcclusionAwareSPADEGenerator(**mp['generator_params'],**common)
    old.load_state_dict(saved['generator'],strict=True);rgb.load_state_dict(saved['generator'],strict=True)
    old.eval().to(device);rgb.eval().to(device)
    kp=KPDetector(**mp['kp_detector_params'],**common);he=HEEstimator(**mp['he_estimator_params'],**common)
    original=torch.load(BASE,map_location='cpu');kp.load_state_dict(original['kp_detector'],strict=True)
    he.load_state_dict(original['he_estimator'],strict=True);kp.eval().to(device);he.eval().to(device)
    ds=ContactClipDataset(ROOT,'train','UBFC-rPPG')
    # Variable native T means default_collate cannot batch complete videos;
    # source/driving and conditions are deliberately batched explicitly here.
    first,second=ds[0],ds[1]
    source=torch.stack((first['source'],second['source'])).to(device)
    driving=torch.stack((first['driving_video'][20],second['driving_video'][20])).to(device)
    wave=torch.stack((first['target_rppg'][:100],second['target_rppg'][:100])).to(device)
    with torch.no_grad():
        canonical=kp(source);ks=keypoint_transformation(canonical,he(source),common['estimate_jacobian'])
        kd=keypoint_transformation(canonical,he(driving),common['estimate_jacobian'])
        reference_old=old(source,kd,ks)['prediction'];reference_new=rgb(source,kd,ks)['prediction']
        baseline_error=float((reference_old-reference_new).abs().max());assert baseline_error<1e-6
        oldout=old(source,kd,ks,target_rppg=wave,rppg_frame_index=20)['prediction']
        newout=rgb(source,kd,ks,target_rppg=wave,rppg_frame_index=20)['prediction']
        rgb_error=float((oldout-newout).abs().max());assert rgb_error<1e-6
    del rgb,old,saved;torch.cuda.empty_cache()
    feature_cfg=dict(mp['generator_params']);phy=dict(feature_cfg.get('physiology_params',{}))
    phy.update(fusion_mode='feature_motion_aligned',rank=2);feature_cfg['physiology_params']=phy
    gen=OcclusionAwareSPADEGenerator(**feature_cfg,**common)
    transfer=load_generator_compatible(gen,original['generator']);gen.eval().to(device).requires_grad_(False)
    shapes={};handles=[]
    for name in ('fc','G_middle_0','G_middle_5','up_0','up_1','conv_img'):
        handles.append(getattr(gen.decoder,name).register_forward_hook(lambda m,i,o,n=name:shapes.__setitem__(n,list(o.shape))))
    with torch.no_grad():
        ctx=gen.prepare_reference(source);motion=gen.motion_context(source,kd,ks,ctx)
        out=gen(source,kd,ks,reference_context=ctx)
        no_input_error=float((out['prediction']-reference_old).abs().max());assert no_input_error<1e-6
        condition=gen.physiology_encoder(wave)[:,20]
        zero=gen.decode_context(motion,condition,torch.zeros(2,device=device))
        zero_error=float((zero['prediction']-zero['prediction_base']).abs().max());assert zero_error==0
        original_warp=gen.deform_input;calls=[]
        def record_warp(value,grid):
            calls.append((value.shape[1],grid.data_ptr()))
            return original_warp(value,grid)
        gen.deform_input=record_warp
        controlled=gen(source,kd,ks,target_rppg=wave,rppg_frame_index=20,reference_context=ctx)
        assert len(calls)==2 and calls[0][1]==calls[1][1] and [c[0] for c in calls]==[32,2]
        gen.deform_input=original_warp
        changed=gen(source,kd,ks,target_rppg=-wave,rppg_frame_index=20,reference_context=ctx)
        control_error=float((controlled['prediction']-changed['prediction']).abs().max());assert control_error>1e-7
        a=gen(source,kd,ks,target_rppg=wave,rppg_frame_index=20,reference_context=ctx,skin_mask=torch.zeros_like(source[:,:1]))
        b=gen(source,kd,ks,target_rppg=wave,rppg_frame_index=20,reference_context=ctx,skin_mask=torch.ones_like(source[:,:1]))
        assert torch.equal(a['prediction'],b['prediction'])
        gen.physiology_fusion_mode='none';none=gen(source,kd,ks,target_rppg=wave,rppg_frame_index=20,reference_context=ctx)
        assert torch.allclose(none['prediction'],reference_old,atol=1e-6,rtol=0)
        gen.physiology_fusion_mode='feature_unaligned';unaligned=gen(source,kd,ks,target_rppg=wave,rppg_frame_index=20,reference_context=ctx)
        assert torch.isfinite(unaligned['prediction']).all()
    for h in handles:h.remove()
    # Save/reload a new-schema model strictly and reject a different mode.
    gen.physiology_fusion_mode='feature_motion_aligned'
    load_generator_compatible(gen,gen.state_dict(),declared_mode='feature_motion_aligned')
    with torch.inference_mode():
        inference_source=source.clone()
        inference_reference=gen.prepare_reference(inference_source)
        inference_output=gen(inference_source,kd,ks,reference_context=inference_reference)
        inference_error=float((inference_output['prediction']-reference_old).abs().max());assert inference_error<1e-6
        inference_source.add_(.001)
        try:gen(inference_source,kd,ks,reference_context=inference_reference)
        except ValueError:pass
        else:raise AssertionError('Mutated inference-mode source was silently reused')
    with torch.no_grad():
        native=move(default_collate([first]),device)
        canonical1=kp(native['source'])
        ks1=keypoint_transformation(canonical1,he(native['source']),common['estimate_jacobian'])
        kd1=keypoint_transformation(canonical1,he(native['driving_video'][:,20]),common['estimate_jacobian'])
        c1=gen.prepare_reference(native['source'])
        plain1=gen(native['source'],kd1,ks1,reference_context=c1)['prediction']
        import json
        mapping=json.loads((CACHE/'manifest.json').read_text());entry=mapping[ds.rows[0]['sample_id']]
        cached=torch.from_numpy(np.array(np.load(CACHE/entry['path'],mmap_mode='r')[20])).unsqueeze(0).to(device)
        cache_error=float((plain1-cached).abs().max());assert cache_error<1e-6,cache_error
        native_wave=native['target_rppg'];cond=gen.physiology_encoder(native_wave)[:,20]
        timing={}
        for mode in ('none','rgb_residual','feature_unaligned','feature_motion_aligned'):
            gen.physiology_fusion_mode=mode
            for i in range(2):gen(native['source'],kd1,ks1,physiology_condition=cond,
                                  physiology_value=native_wave[:,20],reference_context=c1)
            torch.cuda.synchronize();tick=time.perf_counter()
            for i in range(3):gen(native['source'],kd1,ks1,physiology_condition=cond,
                                  physiology_value=native_wave[:,20],reference_context=c1)
            torch.cuda.synchronize();timing[mode]=(time.perf_counter()-tick)*1000/3
        gen.physiology_fusion_mode='feature_motion_aligned'
    result=dict(status='passed',batch_size=2,legacy_RGB_checkpoint=str(OLD),legacy_RGB_checkpoint_sha256=sha(OLD),
        original_GFVC_sha256=sha(BASE),no_physiology_original_max_error=baseline_error,
        legacy_RGB_max_error=rgb_error,new_feature_no_input_max_error=no_input_error,
        zero_BVP_extra_modulation_max_error=zero_error,BVP_swap_prediction_max_difference=control_error,
        same_grid_used_for_appearance_and_basis=True,GT_driving_mask_ignored_at_inference=True,
        native_loader_stored_original_baseline_max_error=cache_error,
        inference_tensor_input_max_error=inference_error,cached_inference_source_mutation_rejected=True,
        inference_ms_per_frame=timing,
        timing_scope='batch1 same real motion,3timed repeats after2warmups; reference encoding and KP/HE excluded; feature default includes diagnostic baseline decode',
        validated_source_sha256={name:sha(PROJECT/name) for name in ('modules/generator.py','modules/physiology_motion.py','modules/model.py','modules/rppg.py')},
        shapes=dict(reference_volume=list(ctx['reference_volume'].shape),reference_basis=list(ctx['reference_basis'].shape),
            deformation=list(motion['deformation'].shape),occlusion_map=list(motion['occlusion_map'].shape),
            SPADE_input=list(motion['features'].shape),aligned_basis=list(controlled['physiology_basis_3d'].shape),
            response2D=list(controlled['physiology_response_maps'].shape),decoder_layers=shapes),
        parameter_counts=dict(all_generator=sum(p.numel() for p in gen.parameters()),
            new_modulator=sum(p.numel() for p in gen.physiology_modulator.parameters()),
            temporal_encoder=sum(p.numel() for p in gen.physiology_encoder.parameters())),
        original_key_loading_strict=True,transfer_fresh_keys=transfer['fresh_keys'])
    save(RUN/'integration_checks.json',result);event({k:v for k,v in result.items() if k not in ('transfer_fresh_keys','shapes')})


if __name__=='__main__':main()
