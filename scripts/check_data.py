import argparse
from odas.data import validate_index

ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='Data'); a=ap.parse_args()
print(validate_index(a.data_root))
print('All indexed .mat files exist.')
