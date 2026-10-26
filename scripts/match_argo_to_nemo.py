import argparse, sys
from pathlib import Path
import pandas as pd, xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from earthformer_ocean_chl.argo.matching import match_profiles_to_nemo
p=argparse.ArgumentParser(description="Match processed BGC-Argo profiles to NEMO."); p.add_argument("--profiles-csv",required=True); p.add_argument("--nemo-path",required=True); p.add_argument("--variable",default="chl"); p.add_argument("--output-csv",required=True)
if __name__=="__main__":
 a=p.parse_args(); out=Path(a.output_csv); out.parent.mkdir(parents=True,exist_ok=True); match_profiles_to_nemo(pd.read_csv(a.profiles_csv),xr.open_dataset(a.nemo_path),a.variable,"2D",1.,5.).to_csv(out,index=False)
