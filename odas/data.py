import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

def load_config(path):
    with open(path, encoding='utf-8') as f: return json.load(f)

def to_image(x, size=32):
    x=np.asarray(x,dtype=np.float32).squeeze()
    if x.shape==(12,10000): x=x.T
    if x.shape!=(10000,12): raise ValueError(f'Expected (10000,12), got {x.shape}')
    edges=np.linspace(0,10000,size+1).astype(int); x=np.stack([x[edges[i]:edges[i+1]].mean(0) for i in range(size)])
    old,new=np.linspace(0,1,x.shape[1]),np.linspace(0,1,size); x=np.stack([np.interp(new,old,row) for row in x])
    return ((x-x.min())/(x.max()-x.min()+1e-8)).astype(np.float32)

def read_index(root,split):
    root=Path(root); rows=[]
    with open(root/split/'label.txt',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line and not line.startswith('#'):
                rel,label=line.rsplit(maxsplit=1); rows.append((root/split/rel.lstrip('/\\'),int(label)))
    return rows

def validate_index(root, splits=('train','test'), limit=20):
    """Check that every path listed in label.txt exists before training starts."""
    missing=[]; counts={}
    for split in splits:
        rows=read_index(root,split); counts[split]=len(rows)
        missing.extend((split,str(path),label) for path,label in rows if not path.is_file())
    if missing:
        preview='\n'.join(f'  [{split}] {path} (label={label})' for split,path,label in missing[:limit])
        raise FileNotFoundError(f'Missing {len(missing)} of {sum(counts.values())} indexed .mat files.\n{preview}\n\nCheck data extraction/upload and label.txt before training.')
    return counts

class CaoDataset(Dataset):
    def __init__(self,root,split='train',indices=None,image_size=32):
        self.rows=read_index(root,split); self.rows=self.rows if indices is None else [self.rows[i] for i in indices]; self.image_size=image_size
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        path,label=self.rows[i]; mat=loadmat(path); key='data' if 'data' in mat else next(k for k in mat if not k.startswith('__'))
        return torch.from_numpy(to_image(mat[key],self.image_size)).unsqueeze(0),torch.tensor(label),str(path)

def make_open_world_indices(root,labeled_fraction=.5,seen=(0,1,2),seed=42):
    rows=read_index(root,'train'); rng=np.random.default_rng(seed); labeled=[]; unlabeled=[]
    for c in sorted({y for _,y in rows}):
        ids=np.array([i for i,(_,y) in enumerate(rows) if y==c]); rng.shuffle(ids); n=int(round(len(ids)*labeled_fraction)) if c in seen else 0
        if c in seen: labeled+=ids[:n].tolist(); unlabeled+=ids[n:].tolist()
        else: unlabeled+=ids.tolist()
    return labeled,unlabeled
