"""Shared provenance and fixed native-clip loading for subject-independent retraining."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT=Path('/opt/data/private/project/One-Shot_Free-View_Neural_Talking_Head_Synthesis-main')
ROOT=Path('/opt/data/private/datasets/GFVC-rPPG_processed/11-3-pro')
RUN=PROJECT/'checkpoint/ubfc_11_3_subject_independent_20261103'
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(ROOT/'code'))
from contact_clip_dataset import ContactClipDataset


def read(p):return json.loads(Path(p).read_text())


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()


def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def datasets():
    train=ContactClipDataset(ROOT,'train','UBFC-rPPG',normalize_ppg=True,seed=42)
    val=ContactClipDataset(ROOT,'val','UBFC-rPPG',normalize_ppg=True,seed=42)
    splits=read(ROOT/'metadata/splits.json')['datasets']['UBFC-rPPG']
    train_ids=sorted({r['subject'] for r in train.rows});val_ids=sorted({r['subject'] for r in val.rows})
    assert len(train)==717 and len(val)==28 and len(train_ids)==30 and len(val_ids)==6
    assert set(train_ids)==set(splits['train']) and set(val_ids)==set(splits['val'])
    assert not set(train_ids)&(set(val_ids)|set(splits['test']))
    assert all(r['target_duration_seconds']==5 for r in train.rows)
    assert all(r['target_duration_seconds']==12 for r in val.rows)
    assert all(r['sync_level']!='estimated_sync' for r in train.rows+val.rows)
    return train,val,dict(root=str(ROOT),dataset='UBFC-rPPG',seed=42,train_clips=len(train),val_clips=len(val),
        train_subjects=train_ids,val_subjects=val_ids,test_used=False,lgi_used_for_training=False,
        target='provided contact PPG; per-clip zscore, no temporal filtering/resampling/shifting or pseudo labels',
        timebase='native samples/timestamps retained; existing spectral losses use the clip effective mean FPS',
        hashes={name:sha(ROOT/'metadata'/name) for name in ('splits.json','train_clips.json','val_clips.json')})


def loader(ds,training,workers=1,seed=42):
    kwargs=dict(batch_size=1,shuffle=training,num_workers=workers,pin_memory=True,
                generator=torch.Generator().manual_seed(seed),drop_last=False)
    if workers:kwargs.update(prefetch_factor=1,persistent_workers=False)
    return DataLoader(ds,**kwargs)


def move(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def checkpoint(path,state):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(state,tmp);tmp.replace(path)


def event(value):print(json.dumps(value,allow_nan=False),flush=True)


def runtime():
    import scipy
    return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
                cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
                thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
