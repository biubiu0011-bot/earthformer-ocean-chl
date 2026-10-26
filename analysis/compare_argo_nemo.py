"""Calculate reproducible metrics for matched Argo and NEMO values."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
def calculate_comparison_metrics(frame):
    clean=frame[['chla','nemo_chla']].dropna()
    if clean.empty: raise ValueError('matched table is empty')
    y=clean.chla.to_numpy(); p=clean.nemo_chla.to_numpy(); return {'rmse':float(np.sqrt(np.mean((p-y)**2))),'mae':float(np.mean(np.abs(p-y))),'pearson_r':float(np.corrcoef(y,p)[0,1]),'nse':float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2))}
p=argparse.ArgumentParser(description=__doc__); p.add_argument('--matched-csv',required=True); p.add_argument('--output-json',required=True)
if __name__=='__main__':
 a=p.parse_args(); out=Path(a.output_json); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(calculate_comparison_metrics(pd.read_csv(a.matched_csv)),indent=2))
