import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odas.data import validate_index

ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='Data'); a=ap.parse_args()
print(validate_index(a.data_root))
print('All indexed .mat files exist.')
