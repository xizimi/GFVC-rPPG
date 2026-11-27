"""Sequential, auditable controlled ablations; fail closed without touching data."""
import json
import os
import subprocess
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'tools/retrain_11_3'))
from common import save,read,sha

ROOT=PROJECT/'checkpoint/motion_physio_20261103_v2'
PLAN=Path('/opt/data/private/project/GFVC-PLAN')
PY='/opt/conda/envs/fv2v/bin/python'


def execute(command,name):
    save(ROOT/'experiment_progress.json',dict(status='running',stage=name,pid=os.getpid(),command=list(map(str,command))))
    with (PLAN/('motion_physio_'+name+'.log')).open('a') as f:
        subprocess.run(list(map(str,command)),cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,check=True)


def command(mode,stage='physiology',parent=None,smoke=False):
    suffix='stage1' if stage=='physiology' else 'stage2_up1'
    name=mode+'_'+suffix+'_seed42_rank2';out=ROOT/name
    args=[PY,PROJECT/'train_gfvc_rppg.py','--config',PROJECT/'config/vox-256-motion-physio.yaml',
        '--generator','spade','--gfvc_checkpoint',PROJECT/'checkpoint/00000189-checkpoint.pth.tar',
        '--rppg_backend','skin','--rppg_checkpoint',PROJECT/'checkpoint/ubfc_11_3_subject_independent_20261103/teacher/teacher_best.pt',
        '--contact_clip_root','/opt/data/private/datasets/GFVC-rPPG_processed/11-3-pro',
        '--sequence_length','150','--rewrite_probability','0.3','--require_semantic_masks','--signal_preprocess','none',
        '--physiology_fusion_mode',mode,'--physiology_rank','2','--training_stage',stage,
        '--epochs','30' if stage=='physiology' else '5','--patience','8','--seed','42',
        '--num_workers','1','--checkpoint_decoder','--lr','0.0001','--spade_lr','0.00001','--output',out]
    if (out/'latest.pth.tar').exists():args+=['--resume',out/'latest.pth.tar']
    elif parent is not None:args+=['--resume',parent]
    return name,out,args


def inspect(out,name):
    report=out/'control_diagnostics/control_report.json'
    if not report.exists():execute([PY,PROJECT/'tools/motion_physio/inspect_checkpoint.py',
        '--checkpoint',out/'best.pth.tar','--output',out/'control_diagnostics'],name+'_control_diagnostics')
    assert read(report)['status']=='passed'
    maps=out/'spatial_diagnostics/spatial_report.json'
    if not maps.exists():execute([PY,PROJECT/'tools/motion_physio/plot_checkpoint_maps.py',
        '--checkpoint',out/'best.pth.tar','--output',out/'spatial_diagnostics'],name+'_spatial_diagnostics')
    assert read(maps)['status']=='exported'


def main():
    ROOT.mkdir(exist_ok=True)
    assert read(ROOT/'integration_checks.json')['status']=='passed'
    smoke=ROOT/'feature_motion_aligned_stage1_smoke_v2_seed42_rank2'
    assert read(smoke/'SMOKE_COMPLETE.json')['status']=='passed'
    # The smoke resume test creates epoch2; epoch1-only output is insufficient.
    assert len(read(smoke/'checkpoint_index.json'))>=2
    save(ROOT/'experiment_plan.json',dict(status='authorized',pid=os.getpid(),seed=42,rank=2,
        UBFC_train_only=True,LGI_cross_dataset_only=True,unchanged_teacher=True,
        stages=['aligned frozen30epochs max','unaligned frozen30epochs max','matched RGB frozen30epochs max','aligned up1 joint5epochs max'],
        rewrite_probability=.3,data_protocol='717native5s Train;28fixed12s Val with preserve and Train-donor rewrite controls',
        no_test_access=True,controller_sha256=sha(Path(__file__)),
        stage2_selection='retain Stage1 unless teacher-PCC improves, baseline-PSNR drops<=0.5dB and outside-skin L1 increases<=0.001'))
    completed={}
    try:
        for mode in ('feature_motion_aligned','feature_unaligned','rgb_residual'):
            name,out,cmd=command(mode)
            if not (out/'TRAINING_COMPLETE.json').exists():execute(cmd,name)
            assert read(out/'TRAINING_COMPLETE.json')['status']=='complete'
            inspect(out,name);completed[mode]=str(out)
            save(ROOT/'completed_runs.json',completed)
        parent=Path(completed['feature_motion_aligned'])/'best.pth.tar'
        name,out,cmd=command('feature_motion_aligned','joint_up1',parent)
        if not (out/'TRAINING_COMPLETE.json').exists():execute(cmd,name)
        assert read(out/'TRAINING_COMPLETE.json')['status']=='complete';inspect(out,name)
        a=read(Path(str(parent.resolve())+'.json'))
        b=read(Path(str((out/'best.pth.tar').resolve())+'.json'))
        def means(x):
            return {k:sum(x['validation'][s]['mean'][k] for s in ('preserve','rewrite'))/2 for k in
                ('teacher_waveform_pcc','baseline_psnr_db','outside_skin_delta_l1','green_contact_pearson')}
        first,second=means(a),means(b)
        improved=(second['teacher_waveform_pcc']>first['teacher_waveform_pcc'] and
            second['baseline_psnr_db']>=first['baseline_psnr_db']-.5 and
            second['outside_skin_delta_l1']<=first['outside_skin_delta_l1']+.001)
        save(ROOT/'stage2_comparison.json',dict(stage1=first,stage2=second,choose_stage2=improved,
            selected_checkpoint=str(out/'best.pth.tar') if improved else str(parent),
            formal_pyVHR_not_evaluated=True,no_test_used=True))
        save(ROOT/'EXPERIMENTS_COMPLETE.json',dict(status='complete',runs=completed,stage2=str(out),
            selected_checkpoint=str(out/'best.pth.tar') if improved else str(parent)))
        save(ROOT/'experiment_progress.json',dict(status='complete'))
    except Exception as exc:
        save(ROOT/'experiment_failure.json',dict(status='failed',error=repr(exc),last_stage=read(ROOT/'experiment_progress.json')))
        raise


if __name__=='__main__':main()
