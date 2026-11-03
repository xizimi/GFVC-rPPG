"""Unchanged native clip data plus deterministic, split-safe BVP rewriting."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tools.retrain_11_3.common import ROOT, read
from tools.retrain_11_3.common import PROJECT
import sys
sys.path.insert(0,str(PROJECT/'tools/retrain_11_3'))
from cache_base import CACHE
from contact_clip_dataset import ContactClipDataset


class NativePhysioClips(Dataset):
    def __init__(self, split, rewrite_probability=0., seed=42):
        self.base=ContactClipDataset(ROOT,split,'UBFC-rPPG',normalize_ppg=True,seed=seed)
        self.rows=self.base.rows;self.probability=rewrite_probability;self.seed=seed;self.epoch=0
        if not 0 <= self.probability <= 1:raise ValueError('Invalid rewrite probability')
        assert read(CACHE/'CACHE_COMPLETE.json')['status']=='passed'
        self.mapping=read(CACHE/'manifest.json')
        assert all(r['sample_id'] in self.mapping for r in self.rows)
        # Donors are unions of contiguous VALID Train windows, never a held-out
        # video, excluded interval, cyclic signal, or pseudo-label source.
        train=ContactClipDataset(ROOT,'train','UBFC-rPPG')
        records={}
        for r in train.rows:records.setdefault(r['path'],[]).append(r)
        self.donors=[]
        for path,rows in sorted(records.items()):
            spans=[]
            for r in sorted(rows,key=lambda r:r['start']):
                if spans and r['start']<=spans[-1][1]:spans[-1][1]=max(spans[-1][1],r['end'])
                else:spans.append([r['start'],r['end']])
            times=np.load(ROOT/path/'timestamps.npy',mmap_mode='r')
            for a,b in spans:
                self.donors.append(dict(path=path,start=a,end=b,subject=rows[0]['subject'],
                    duration=float(times[b-1]-times[a])))
        self._verified=set()

    def set_epoch(self, epoch):self.epoch=epoch
    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        sample=self.base[index];row=self.rows[index];entry=self.mapping[row['sample_id']]
        assert entry['start']==row['start'] and entry['end']==row['end']
        if index not in self._verified:
            rgb=np.load(ROOT/row['path']/'frames.npy',mmap_mode='r')[row['start']:row['end']]
            assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()==entry['source_rgb_sha256']
            self._verified.add(index)
        sample['frozen_base_video']=torch.from_numpy(np.array(np.load(CACHE/entry['path'],mmap_mode='r')))
        assert sample['frozen_base_video'].shape==sample['driving_video'].shape
        rng=np.random.RandomState((self.seed+1000003*self.epoch+index) % (2**32-1))
        sample['mode']=torch.tensor(0);sample['conditioning_donor']='same_clip_contact_ppg'
        if rng.random() < self.probability:
            relative=sample['relative_timestamps'].numpy();duration=float(relative[-1])
            eligible=[d for d in self.donors if d['subject']!=row['subject'] and d['duration']>=duration]
            if not eligible:raise RuntimeError('No valid TRAIN donor long enough for the requested waveform')
            donor=eligible[rng.randint(len(eligible))];a,b=donor['start'],donor['end'];folder=ROOT/donor['path']
            times=np.asarray(np.load(folder/'timestamps.npy',mmap_mode='r')[a:b],dtype=np.float64)
            raw=np.asarray(np.load(folder/'ppg.npy',mmap_mode='r')[a:b],dtype=np.float64)
            times,unique=np.unique(times,return_index=True);raw=raw[unique]
            starts=np.flatnonzero(times<=times[-1]-duration)
            start=float(times[starts[rng.randint(len(starts))]])
            # A deliberately substituted input waveform, not a changed aligned
            # ground-truth label. Preserve original_rppg and original timestamps.
            target=np.interp(start+relative,times,raw)
            if not np.isfinite(target).all() or target.std()<1e-8:raise ValueError('Invalid contact-PPG donor')
            sample['target_rppg']=torch.from_numpy(((target-target.mean())/target.std()).astype(np.float32))
            sample['mode']=torch.tensor(1);sample['conditioning_donor']=donor['subject']+'/'+donor['path']
        return sample
