"""Validate Cao files and optionally materialize 32x32 .npy cache."""
import argparse,json
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odas.data import read_index,to_image
ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='Data'); ap.add_argument('--out',default='cache/cao_32'); ap.add_argument('--limit',type=int,default=0); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); rows=read_index(a.data_root,'train')+read_index(a.data_root,'test'); rows=rows[:a.limit] if a.limit else rows; n=0
for path,label in rows:
    mat=loadmat(path); key='data' if 'data' in mat else next(k for k in mat if not k.startswith('__')); np.save(out/f'{n:06d}.npy',to_image(mat[key])); n+=1
print(json.dumps({'files':n,'output':str(out)},ensure_ascii=False))
