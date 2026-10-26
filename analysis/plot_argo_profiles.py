import argparse
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(description='Plot one processed BGC-Argo profile.'); p.add_argument('--profiles-csv',required=True); p.add_argument('--profile-id',required=True); p.add_argument('--output-png',required=True)
if __name__=='__main__':
 a=p.parse_args(); import matplotlib.pyplot as plt; d=pd.read_csv(a.profiles_csv); d=d[d.profile_id.astype(str)==a.profile_id]; plt.plot(d.chla,d.depth); plt.gca().invert_yaxis(); plt.xlabel('Chl-a'); plt.ylabel('Depth (m)'); Path(a.output_png).parent.mkdir(parents=True,exist_ok=True); plt.savefig(a.output_png,dpi=200)
