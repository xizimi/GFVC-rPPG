"""Read the 11-3-pro native RGB clip indexes, with no automatic re-splitting."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ContactClipDataset(Dataset):
    def __init__(self, root, split='train', dataset='UBFC-rPPG', estimated=False,
                 view='sequence', normalize_ppg=True, seed=42):
        if split not in ('train','val','test') or view not in ('sequence','pair','generation'):
            raise ValueError('Unknown split/view')
        if dataset=='LGI-PPGI' and not estimated:
            raise ValueError('LGI requires estimated=True; it has estimated synchronization')
        self.root=Path(root)
        prefix='estimated_' if estimated else ''
        self.rows=json.loads((self.root/'metadata'/(prefix+split+'_clips.json')).read_text())
        if dataset is not None:
            self.rows=[r for r in self.rows if r['dataset']==dataset]
        self.split,self.view,self.normalize_ppg,self.seed=split,view,normalize_ppg,seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];folder=self.root/row['path'];a,b=row['start'],row['end']
        frames=np.load(folder/'frames.npy',mmap_mode='r')
        result=dict(name=row['sample_id'],sample_id=row['sample_id'],subject=row['subject'],
                    sync_level=row['sync_level'])
        if self.view=='pair':
            rng=np.random if self.split=='train' else np.random.RandomState(self.seed+index)
            indices=np.sort(rng.choice(b-a,size=2,replace=False))+a
            rgb=np.array(frames[indices])
            tensor=torch.from_numpy(rgb).float().permute(0,3,1,2)/255.
            result.update(source=tensor[0],driving=tensor[1],source_frame_index=int(indices[0]),
                          driving_frame_index=int(indices[1]))
            return result
        rgb=torch.from_numpy(np.array(frames[a:b])).float().permute(0,3,1,2)/255.
        times=np.array(np.load(folder/'timestamps.npy',mmap_mode='r')[a:b])
        result.update(source=rgb[0],driving_video=rgb,
            timestamps=torch.from_numpy(times),relative_timestamps=torch.from_numpy(times-times[0]),
            fps=torch.tensor(row['effective_fps'],dtype=torch.float32),
            nominal_fps=torch.tensor(row['fps'],dtype=torch.float32),
            uniform_sampling=row['uniform_sampling'],target_duration_seconds=row['target_duration_seconds'])
        if self.view=='generation':
            # This branch never loads PPG/masks; pure GFVC receives RGB only.
            return result
        mask=np.array(np.load(folder/'skin_masks.npy',mmap_mode='r')[a:b])
        raw=np.array(np.load(folder/'ppg.npy',mmap_mode='r')[a:b],dtype=np.float64)
        if not np.isfinite(raw).all() or raw.std()<=1e-8:
            raise ValueError('Invalid contact PPG in indexed clip')
        mean,scale=float(raw.mean()),float(raw.std())
        signal=((raw-mean)/scale if self.normalize_ppg else raw).astype(np.float32)
        result.update(skin_masks=torch.from_numpy(mask).float().unsqueeze(1),
            ppg_contact=torch.from_numpy(raw),original_rppg=torch.from_numpy(signal.copy()),
            target_rppg=torch.from_numpy(signal.copy()),waveform_mean=torch.tensor(mean,dtype=torch.float64),
            waveform_scale=torch.tensor(scale,dtype=torch.float64),
            target_bpm=torch.tensor(float('nan')),mode=torch.tensor(0,dtype=torch.long))
        return result
