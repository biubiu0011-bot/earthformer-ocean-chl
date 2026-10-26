import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(description='Plot matched Argo versus NEMO chlorophyll values.'); p.add_argument('--matched-csv',required=True); p.add_argument('--output-png',required=True)
if __name__=='__main__':
 a=p.parse_args(); import matplotlib.pyplot as plt; d=pd.read_csv(a.matched_csv).dropna(subset=['chla','nemo_chla']); plt.scatter(d.chla,d.nemo_chla,s=4,alpha=.4); plt.xlabel('Argo Chl-a'); plt.ylabel('NEMO Chl-a'); Path(a.output_png).parent.mkdir(parents=True,exist_ok=True); plt.savefig(a.output_png,dpi=200)
