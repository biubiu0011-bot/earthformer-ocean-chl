from pathlib import Path
import sys

import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from earthformer_ocean_chl.argo.matching import match_profiles_to_nemo


def test_matching_returns_nearest_nemo_value():
    dataset = xr.Dataset({"chl": (("time", "depth", "lat", "lon"), [[[[0.42]]]])}, coords={"time": pd.to_datetime(["2020-01-01"]), "depth": [10.0], "lat": [20.0], "lon": [140.0]})
    profiles = pd.DataFrame({"time": pd.to_datetime(["2020-01-01"]), "depth": [10.0], "lat": [20.0], "lon": [140.0], "chla": [0.4]})
    result = match_profiles_to_nemo(profiles, dataset, "chl", "2D", 1.0, 5.0)
    assert result.loc[0, "nemo_chla"] == pytest.approx(0.42)
    assert result.loc[0, "match_status"] == "matched"
