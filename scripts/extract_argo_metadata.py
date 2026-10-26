import argparse, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from earthformer_ocean_chl.argo.matching import extract_float_metadata
p=argparse.ArgumentParser(description="Extract one metadata row per BGC-Argo float."); p.add_argument("--input-csv",required=True); p.add_argument("--output-csv",required=True)
if __name__=="__main__":
 a=p.parse_args(); out=Path(a.output_csv); out.parent.mkdir(parents=True,exist_ok=True); extract_float_metadata(pd.read_csv(a.input_csv)).to_csv(out,index=False)
