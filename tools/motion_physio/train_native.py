"""Controlled full native5s training with checkpoint-by-checkpoint experiment cards."""
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'tools/retrain_11_3'))
from common import ROOT,save,read,sha,seed_all,checkpoint,event,datasets,move,runtime
from tools.motion_physio.data import NativePhysioClips,CACHE
from tools.motion_physio.diagnostics import diagnose
from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector,HEEstimator
from modules.model import GFVCRPPGSequenceModel
from modules.rppg import SkinRPPGTeacher,negative_pearson_loss
from modules.physiology_motion import load_generator_compatible


def build(args,config,device):
    from train_gfvc_rppg import load_component_checkpoint,set_stage2_trainable
    mp=config['model_params'];common=mp['common_params']
    gen=OcclusionAwareSPADEGenerator(**mp['generator_params'],**common)
    kp=KPDetector(**mp['kp_detector_params'],**common);he=HEEstimator(**mp['he_estimator_params'],**common)
    load_component_checkpoint(args.gfvc_checkpoint,gen,kp,he)
    gen.checkpoint_decoder=True
    params=set_stage2_trainable(gen,kp,he,args.training_stage)
    if not params:raise ValueError('none is a frozen inference baseline; no training is required')
    teacher=SkinRPPGTeacher();saved=torch.load(args.rppg_checkpoint,map_location='cpu')
    teacher.load_state_dict(saved['model'],strict=True);teacher.requires_grad_(False)
    model=GFVCRPPGSequenceModel(kp,he,gen,teacher,
        loss_weights=config['train_params'].get('physiology_loss_weights'),
        estimate_jacobian=common['estimate_jacobian']).to(device)
    groups=[dict(params=[p for n,p in gen.named_parameters() if p.requires_grad and not n.startswith('decoder.')],lr=args.lr)]
    decoder=[p for n,p in gen.named_parameters() if p.requires_grad and n.startswith('decoder.')]
    if decoder:groups.append(dict(params=decoder,lr=args.spade_lr))
    optimizer=torch.optim.Adam(groups,betas=(.5,.999))
    return model,optimizer,saved


def batches(dataset,shuffle,workers,seed):
    options=dict(batch_size=1,shuffle=shuffle,num_workers=workers,pin_memory=True,
                 generator=torch.Generator().manual_seed(seed))
    if workers:options.update(prefetch_factor=1)
    return DataLoader(dataset,**options)


def aggregate(details):
    keys=[k for k,v in details[0].items() if isinstance(v,(float,int))]
    return {k:float(np.mean([x[k] for x in details])) for k in keys}


@torch.no_grad()
def validate(model,ds,device,args,epoch,smoke=False):
    model.eval();details=[]
    for data in batches(ds,False,args.num_workers,args.seed):
        batch=move(data,device);losses,out=model(batch,epoch=epoch,total_epochs=args.epochs)
        total=sum(losses.values());assert torch.isfinite(total)
        record=dict(sample_id=data['sample_id'][0],total=float(total),
            teacher_waveform_pcc=1.-float(negative_pearson_loss(out['rppg_extracted'],batch['target_rppg'])),
            **diagnose(out,batch));details.append(record)
        if smoke:break
    assert len(details)==(1 if smoke else len(ds))
    return dict(mean=aggregate(details),clips=len(details),details=details)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint_card(out,file,state,run):
    detail=dict(schema='gfvc_motion_physio_checkpoint_v1',file=file.name,sha256=sha(file),
        experiment_id=run['experiment_id'],fusion_mode=run['fusion_mode'],rank=run['rank'],
        training_stage=run['training_stage'],epoch=state['epoch'],global_step=state['global_step'],
        validation_score=state['validation_score'],is_best_at_save=state['improved'],
        original_GFVC=run['base_checkpoint'],frozen_teacher=run['teacher_checkpoint'],
        initialization=run['initialization'],provenance=run['provenance'],
        supervision=run['supervision'],trainable_parameters=run['trainable_parameters'],
        frozen_parameters=run['frozen_parameters'],optimizer=run['optimizer'],
        validation=state['validation'],diagnostics_are_not_teacher_accuracy_proof=True,
        inference_inputs=['reference_RGB','existing_GFVC_motion','specified_BVP'],
        real_driving_mask_at_inference=False,test_used=False,LGI_used_for_training=False,
        run_config_sha256=sha(out/'run_config.json'),source_sha256=run['source_sha256'])
    save(file.with_suffix(file.suffix+'.json'),detail)
    text=f'''# {run['experiment_id']} / epoch {state['epoch']}

Weight file: {file.name}
SHA256: {detail['sha256']}
Fusion mode: {run['fusion_mode']}; rank {run['rank']}; stage {run['training_stage']}.
Global optimizer steps: {state['global_step']}; validation score {state['validation_score']:.8f}.
Best at save: {state['improved']}.

Initialization: {json.dumps(run['initialization'])}
Frozen GFVC: {run['base_checkpoint']['path']} ({run['base_checkpoint']['sha256']}).
Frozen teacher: {run['teacher_checkpoint']['path']} ({run['teacher_checkpoint']['sha256']}).
Data: UBFC Train30subjects717native5s clips, Validation6subjects28fixed12s clips.
LGI and Test videos never used for training/selection. No changes to processed data or splits.
Conditioning: same-clip contact PPG or a Train-contact-PPG donor, probability {run['rewrite_probability']}.
Preserve PPG stays aligned; rewrite uses donor timestamps on driving-relative timestamps, no phase search.
Semantic masks are training supervision and diagnostic ROIs only, never generator inputs.
Selection: equal mean of complete28preserve and28rewrite validation losses.
Teacher is an unchanged measurement network. Green-waveform and image diagnostics are independent checks.
Feature bases are learned modulation bases, not physically validated blood-flow maps.

Exact subject IDs, trainable/frozen parameter names, source hashes, optimizer,
normalization, full per-clip validation diagnostics and data hashes are in the JSON sidecar and run_config.json.
Resume using --resume {file}; mode/rank/data/teacher must match.
'''
    file.with_suffix(file.suffix+'.md').write_text(text)
    entries=read(out/'checkpoint_index.json') if (out/'checkpoint_index.json').exists() else []
    entries.append({k:detail[k] for k in ('file','sha256','fusion_mode','rank','training_stage','epoch','global_step','validation_score','is_best_at_save')})
    save(out/'checkpoint_index.json',entries)


def main_from_args(args,config):
    if args.generator!='spade' or args.rppg_backend!='skin' or args.signal_preprocess!='none':
        raise ValueError('Native controlled experiment requires spade, frozen skin teacher, and unfiltered contact labels')
    if Path(args.contact_clip_root).resolve()!=ROOT.resolve() or args.batch_size!=1:
        raise ValueError('Use the verified11-3-pro root and batch1 for native variable-length clips')
    if args.sequence_length!=150:raise ValueError('150 is the nominal30FPS training length; complete native5s indexes are retained')
    seed_all(args.seed);device=torch.device(args.device);assert device.type=='cuda'
    _,_,provenance=datasets();out=Path(args.output)
    cache_info=read(CACHE/'config.json')
    assert cache_info['provenance']==provenance,'Frozen-baseline data provenance mismatch'
    assert cache_info['base_checkpoint_sha256']==sha(args.gfvc_checkpoint),'Frozen-baseline GFVC checkpoint mismatch'
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError(str(out))
    out.mkdir(parents=True,exist_ok=True)
    model,optimizer,teacher_state=build(args,config,device);gen=model.generator
    assert teacher_state['training_subjects']==provenance['train_subjects']
    assert teacher_state['validation_subjects']==provenance['val_subjects']
    teacher_done=read(Path(args.rppg_checkpoint).parent/'TRAINING_COMPLETE.json')
    assert teacher_done['status']=='passed' and teacher_done['best_subject_mean_pearson']>=.2
    train=NativePhysioClips('train',args.rewrite_probability,args.seed)
    preserve=NativePhysioClips('val',0.,args.seed);rewrite=NativePhysioClips('val',1.,args.seed)
    trainable={n:p.numel() for n,p in gen.named_parameters() if p.requires_grad}
    frozen={n:p.numel() for n,p in gen.named_parameters() if not p.requires_grad}
    assert all(n.startswith(('physiology_encoder.','physiology_modulator.','physiology_residual.','decoder.up_1.')) for n in trainable)
    exempt=('physiology','rppg_fusion.')+(('decoder.up_1.',) if args.training_stage=='joint_up1' else ())
    original_hash={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in gen.state_dict().items() if not n.startswith(exempt)}
    teacher_hash={n:p.detach().clone() for n,p in model.rppg_extractor.named_parameters()}
    run=dict(experiment_id=out.parent.name+'/'+out.name,fusion_mode=args.physiology_fusion_mode,rank=args.physiology_rank,
        training_stage=args.training_stage,provenance=provenance,architecture_config=config,runtime=runtime(),
        command_args=vars(args),command_line=sys.argv,
        initialization=dict(kind='fresh physiology from original GFVC' if not args.resume else 'strict resume/declared parent',parent=args.resume),
        base_checkpoint=dict(path=str(Path(args.gfvc_checkpoint).resolve()),sha256=sha(args.gfvc_checkpoint)),
        teacher_checkpoint=dict(path=str(Path(args.rppg_checkpoint).resolve()),sha256=sha(args.rppg_checkpoint),validation=teacher_done),
        rewrite_probability=args.rewrite_probability,seed=args.seed,epochs=args.epochs,patience=args.patience,
        supervision='provided contact PPG; per-clip population zscore; donor Train-contact PPG only for deliberate rewriting',
        trainable_parameters=trainable,frozen_parameters=frozen,
        trainable_count=sum(trainable.values()),total_generator_count=sum(p.numel() for p in gen.parameters()),
        optimizer=dict(type='Adam',lr=args.lr,spade_lr=args.spade_lr,betas=[.5,.999],gradient_clip=1.),
        selection='0.5 mean complete28preserve +0.5 mean complete28rewrite12s validation total losses',
        feature_basis='learned modulation basis, not validated blood-flow distribution',
        no_GT_mask_at_inference=True,baseline_cache=str(CACHE),decoder_activation_recomputation=True,
        source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in (
            PROJECT/'modules/generator.py',PROJECT/'modules/physiology_motion.py',PROJECT/'modules/rppg.py',
            PROJECT/'modules/model.py',PROJECT/'modules/dense_motion.py',PROJECT/'modules/util.py',
            PROJECT/'train_gfvc_rppg.py',Path(__file__),
            Path(__file__).with_name('data.py'),Path(__file__).with_name('diagnostics.py'))})
    start=1;best=float('inf');stale=0;global_step=0;history=[]
    if args.resume:
        state=torch.load(args.resume,map_location='cpu');old=state['run_config']
        assert old['fusion_mode']==run['fusion_mode'] and old['rank']==run['rank']
        assert old['provenance']==provenance
        assert old['teacher_checkpoint']['sha256']==run['teacher_checkpoint']['sha256']
        assert old['base_checkpoint']['sha256']==run['base_checkpoint']['sha256']
        load_generator_compatible(gen,state['generator'],declared_mode=old['fusion_mode'])
        if old['training_stage']==args.training_stage:
            optimizer.load_state_dict(state['optimizer'])
            for v in optimizer.state.values():
                for k,x in v.items():
                    if torch.is_tensor(x):v[k]=x.to(device)
            start=state['epoch']+1;best=state['best_val'];stale=state['stale_epochs'];global_step=state['global_step']
            restore_rng(state['rng_state']);history=read(out/'history.json')
        else:
            assert args.training_stage=='joint_up1' and old['training_stage']=='physiology'
            run['initialization']=dict(kind='Stage2 from best frozen-decoder Stage1',parent=args.resume,sha256=sha(args.resume))
    if (out/'run_config.json').exists():
        old=read(out/'run_config.json')
        for key in ('fusion_mode','training_stage','provenance','source_sha256','epochs','rewrite_probability'):
            assert old[key]==run[key],key
        assert old['teacher_checkpoint']['sha256']==run['teacher_checkpoint']['sha256']
        run=old
    else:
        save(out/'run_config.json',run)
        for name,digest in run['source_sha256'].items():
            target=out/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(PROJECT/name,target);assert sha(target)==digest
    (out/'README.md').write_text('# '+run['experiment_id']+'\n\nMode: '+run['fusion_mode']+'\n\nStage: '+run['training_stage']+
        '\n\nEach epoch has immutable weights plus .json/.md sidecars. best/latest are symlinks. See checkpoint_index.json and run_config.json for provenance.\n')
    started_all=time.monotonic();parameters=[p for p in gen.parameters() if p.requires_grad]
    for epoch in range(start,args.epochs+1):
        train.set_epoch(epoch);model.train();running=[];began=time.monotonic()
        for step,data in enumerate(batches(train,True,args.num_workers,args.seed+epoch),1):
            tick=time.monotonic();batch=move(data,device);optimizer.zero_grad(set_to_none=True)
            losses,output=model(batch,epoch=epoch-1,total_epochs=args.epochs);loss=sum(losses.values())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-window loss')
            loss.backward();gradient_report={}
            for prefix in ('physiology_encoder.','physiology_modulator.basis_encoder.','physiology_modulator.depth_projection.',
                           'physiology_modulator.modulation_head.','physiology_modulator.skin_gate.','physiology_residual.','decoder.up_1.'):
                values=[p.grad for n,p in gen.named_parameters() if p.requires_grad and n.startswith(prefix) and p.grad is not None]
                if values:
                    assert all(torch.isfinite(v).all() for v in values),prefix
                    gradient_report[prefix]=sum(float(v.abs().sum()) for v in values)
            assert gradient_report and gradient_report['physiology_encoder.']>0
            if args.physiology_fusion_mode.startswith('feature_'):
                for k in ('basis_encoder','depth_projection','modulation_head','skin_gate'):
                    assert gradient_report['physiology_modulator.'+k+'.']>0,k
            assert all(p.grad is None for n,p in gen.named_parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();global_step+=1
            record=dict(loss=float(loss.detach()),seconds=time.monotonic()-tick,**diagnose(output,batch))
            running.append(record)
            if step==1 or step%10==0:
                save(out/'progress.json',dict(status='training',epoch=epoch,step=step,steps=len(train),global_step=global_step,
                    train_loss_mean=float(np.mean([v['loss'] for v in running])),last_step_seconds=record['seconds']))
                event(dict(epoch=epoch,step=step,loss=record['loss'],seconds=record['seconds'],gradient_l1=gradient_report,
                    max_cuda_bytes=torch.cuda.max_memory_allocated()))
            if args.smoke_steps and step>=args.smoke_steps:break
            del losses,output,loss,batch
        val={name:validate(model,ds,device,args,epoch-1,bool(args.smoke_steps)) for name,ds in [('preserve',preserve),('rewrite',rewrite)]}
        score=.5*(val['preserve']['mean']['total']+val['rewrite']['mean']['total']);assert np.isfinite(score)
        for n,v in gen.state_dict().items():
            if n in original_hash:assert hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()==original_hash[n],n
        assert all(torch.equal(p,teacher_hash[n]) for n,p in model.rppg_extractor.named_parameters())
        improved=score<best-1e-6
        if improved:best=score;stale=0
        else:stale+=1
        entry=dict(epoch=epoch,train=aggregate(running),validation=val,score=score,seconds=time.monotonic()-began)
        history.append(entry);save(out/'history.json',history)
        state=dict(schema='motion_physio_v1',epoch=epoch,global_step=global_step,generator=gen.state_dict(),optimizer=optimizer.state_dict(),
            config=config,args=vars(args),run_config=run,validation=val,validation_score=score,best_val=best,improved=improved,
            stale_epochs=stale,rng_state=rng_state())
        filename=out/('epoch_%04d.pth.tar'%epoch);checkpoint(filename,state);checkpoint_card(out,filename,state,run)
        for label,condition in [('latest',True),('best',improved)]:
            if condition:
                link=out/(label+'.pth.tar');temporary=out/(label+'.tmp')
                if temporary.is_symlink():temporary.unlink()
                temporary.symlink_to(filename.name);temporary.replace(link)
        save(out/'progress.json',dict(status='epoch_complete',epoch=epoch,global_step=global_step,best_val=best,stale_epochs=stale))
        if args.smoke_steps:
            save(out/'SMOKE_COMPLETE.json',dict(status='passed',full_native5s=True,steps=args.smoke_steps,
                full12s_validation_forward=True,validation_is_smoke_subset=True,all_new_submodule_gradients_nonzero=True,
                pretrained_backbone_unchanged=args.training_stage=='physiology',
                source_motion_and_frozen_decoder_layers_unchanged=True,
                spade_up1_intentionally_updated=args.training_stage=='joint_up1',
                teacher_unchanged=True,max_cuda_bytes=torch.cuda.max_memory_allocated(),
                gradient_l1=gradient_report,seconds=time.monotonic()-started_all))
            return
        if stale>=args.patience:break
    save(out/'TRAINING_COMPLETE.json',dict(status='complete',epochs=len(history),best_val=best,
        best_checkpoint=str(out/'best.pth.tar'),teacher_unchanged=True,backbone_unchanged_except_declared_up1=True,
        original_spade_all_frozen=args.training_stage=='physiology',spade_up1_updated=args.training_stage=='joint_up1',
        source_and_motion_backbone_unchanged=True,test_used=False,LGI_used_for_training=False))
