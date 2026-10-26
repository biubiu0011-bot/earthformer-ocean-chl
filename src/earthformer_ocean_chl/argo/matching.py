"""BGC-Argo to NEMO nearest-neighbour matching."""
from __future__ import annotations
import pandas as pd

def match_profiles_to_nemo(profiles, dataset, variable, time_tolerance, spatial_tolerance, depth_tolerance):
    if variable not in dataset: raise ValueError(f"Missing NEMO variable: {variable}")
    names = {"lat": "lat" if "lat" in dataset.coords else "latitude", "lon": "lon" if "lon" in dataset.coords else "longitude"}
    rows=[]
    for _, row in profiles.iterrows():
        selected=dataset[variable].sel(time=pd.Timestamp(row.time), method="nearest").sel({"depth": float(row.depth), names["lat"]: float(row.lat), names["lon"]: float(row.lon)}, method="nearest")
        value=float(selected.values)
        rows.append({**row.to_dict(), "nemo_chla": value, "match_status": "matched" if pd.notna(value) else "missing_model_value"})
    return pd.DataFrame(rows)

def extract_float_metadata(profiles):
    return profiles.sort_values("time").drop_duplicates("float_id").reset_index(drop=True)
