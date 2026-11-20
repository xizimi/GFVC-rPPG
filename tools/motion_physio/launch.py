"""Preflight every trainable mode, then start the authorized ablation queue."""
import fcntl
import os
import re
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tools/retrain_11_3'))
from common import read,save,sha
from tools.motion_physio.run_experiments import ROOT,PLAN,PY,execute,command


def smoke(mode,stage,parent=None):
    name,out,cmd=command(mode,stage,parent)
    name=name.replace('_stage1_','_stage1_smoke_v2_').replace('_stage2_up1_','_stage2_up1_smoke_v2_')
    out=ROOT/name;cmd[cmd.index('--output')+1]=out
    cmd[cmd.index('--num_workers')+1]='0';cmd+=['--smoke_steps','1']
    marker=out/'SMOKE_COMPLETE.json'
    if not marker.exists():execute(cmd,name)
    result=read(marker);assert result['status']=='passed'
    if stage=='joint_up1':assert result['gradient_l1']['decoder.up_1.']>0
    return dict(directory=str(out),result=result)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'launch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=ROOT/'launch_failure.json'
        if previous.exists():
            archive=ROOT/'preflight_archive';archive.mkdir(exist_ok=True)
            previous.replace(archive/('failure_'+sha(previous)[:16]+'.json'))
        save(ROOT/'launch_status.json',dict(status='preflight',pid=os.getpid()))
        try:
            execute([PY,'-m','unittest','discover','-s','tests','-p','test_motion_physio.py','-v'],'module_tests')
            execute([PY,'-m','unittest','discover','-s','tests','-p','test_rppg_pipeline.py','-v'],'legacy_tests')
            execute([PY,PROJECT/'tools/motion_physio/integration_checks.py'],'pretrained_integration_checks')
            aligned=ROOT/'feature_motion_aligned_stage1_smoke_v2_seed42_rank2'
            _,_,base_command=command('feature_motion_aligned')
            if '--resume' in base_command:
                index=base_command.index('--resume');del base_command[index:index+2]
            base_command[base_command.index('--output')+1]=aligned
            base_command[base_command.index('--num_workers')+1]='0'
            if not (aligned/'SMOKE_COMPLETE.json').exists():
                execute(base_command+['--smoke_steps','2'],'aligned_v2_full5s_smoke')
            if len(read(aligned/'checkpoint_index.json'))<2:
                execute(base_command+['--smoke_steps','1','--resume',aligned/'latest.pth.tar'],'aligned_v2_full5s_resume')
            prior=read(aligned/'run_config.json')
            assert len(read(aligned/'checkpoint_index.json'))>=2,'Full-window save/resume test has not passed'
            for path,digest in prior['source_sha256'].items():assert sha(PROJECT/path)==digest,path
            results=dict(aligned_save_resume=dict(directory=str(aligned),result=read(aligned/'SMOKE_COMPLETE.json')))
            for mode in ('feature_unaligned','rgb_residual'):
                results[mode]=smoke(mode,'physiology')
            results['joint_up1']=smoke('feature_motion_aligned','joint_up1',aligned/'best.pth.tar')
            execute([PY,PROJECT/'tools/motion_physio/inspect_checkpoint.py','--checkpoint',aligned/'best.pth.tar',
                     '--output',aligned/'control_diagnostics'],'smoke_independent_BVP_controls')
            report=dict(status='passed',integration=read(ROOT/'integration_checks.json'),
                smoke_modes=results,save_resume_verified=True,
                independent_controls=read(aligned/'control_diagnostics/control_report.json'),
                no_driving_mask_at_inference=True,
                skin_localization_on_full_trained_model='not yet established; inspect skin/outside diagnostics',
                unchanged_GFVC_motion_source=sha(PROJECT/'modules/dense_motion.py')==sha(PROJECT/'architecture_history/20261103_pre_motion_aligned/modules/dense_motion.py'),
                unchanged_util_source=sha(PROJECT/'modules/util.py')==sha(PROJECT/'architecture_history/20261103_pre_motion_aligned/modules/util.py'))
            for name in ('module_tests','legacy_tests'):
                text=(PLAN/('motion_physio_'+name+'.log')).read_text();matches=re.findall(r'Ran (\d+) tests',text)
                report[name]=dict(status='passed',count=int(matches[-1]))
            save(ROOT/'tests_report.json',report)
            save(ROOT/'launch_status.json',dict(status='formal_training_queue',pid=os.getpid()))
            execute([PY,PROJECT/'tools/motion_physio/run_experiments.py'],'controlled_training_queue')
            save(ROOT/'launch_status.json',dict(status='complete'))
        except Exception as exc:
            save(ROOT/'launch_failure.json',dict(status='failed',error=repr(exc)))
            raise


if __name__=='__main__':main()
