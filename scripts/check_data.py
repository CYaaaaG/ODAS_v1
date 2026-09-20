import argparse
import sys
from pathlib import Path
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odas.data import validate_index

ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='Data'); ap.add_argument('--read',action='store_true',help='also open every MAT file and verify the data array'); a=ap.parse_args()
print(validate_index(a.data_root))
if a.read:
    bad=[]; total=0
    for p in Path(a.data_root).rglob('*.mat'):
        total += 1
        try:
            mat=loadmat(p); x=mat.get('data')
            if x is None or x.size == 0: bad.append((str(p),'missing or empty data variable'))
        except Exception as e: bad.append((str(p),repr(e)))
    if bad:
        print(f'Unreadable MAT files: {len(bad)} of {total}')
        for path,error in bad[:50]: print(f'  {path}: {error}')
        raise SystemExit(1)
    print(f'All {total} MAT files are readable.')
print('All indexed .mat files exist.')
