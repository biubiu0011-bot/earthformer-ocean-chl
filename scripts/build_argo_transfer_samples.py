import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from earthformer_ocean_chl.argo.samples import write_transfer_samples
p=argparse.ArgumentParser(description='Validate and write prepared Argo transfer tensors.'); p.add_argument('--inputs-npy',required=True); p.add_argument('--targets-npy',required=True); p.add_argument('--output-root',required=True)
if __name__=='__main__':
 a=p.parse_args(); print(f'Wrote {write_transfer_samples(np.load(a.inputs_npy),np.load(a.targets_npy),a.output_root)} samples')
