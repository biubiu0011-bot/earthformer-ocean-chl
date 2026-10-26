from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from earthformer_ocean_chl.argo.profiles import (
    interpolate_profile,
    normalize_profile_table,
    process_profiles,
)


def test_normalize_profile_table_maps_raw_bgc_aliases():
    """Removing a raw-header alias mapping must fail this canonical table contract."""
    raw = pd.DataFrame(
        {
            "PLATFORM_CODE": ["5901234"],
            "DATE (YYYY-MM-DDTHH:MI:SSZ)": ["2020-01-02T03:04:05Z"],
            "LATITUDE (degree_north)": [20.5],
            "LONGITUDE (degree_east)": [140.25],
            "PRES (decibar)": [10.0],
            "CHLA (mg/m3)": [0.42],
        }
    )

    result = normalize_profile_table(raw)

    assert list(result.columns[:7]) == [
        "float_id",
        "profile_id",
        "time",
        "lat",
        "lon",
        "depth",
        "chla",
    ]
    assert result.loc[0, "float_id"] == "5901234"
    assert result.loc[0, "depth"] == pytest.approx(10.0)
    assert result.loc[0, "chla"] == pytest.approx(0.42)


def test_normalize_profile_table_rejects_missing_chla():
    """Removing chlorophyll observations must produce a useful ingestion error."""
    raw = pd.DataFrame(
        {
            "PLATFORM_CODE": ["5901234"],
            "DATE (YYYY-MM-DDTHH:MI:SSZ)": ["2020-01-02T03:04:05Z"],
            "LATITUDE (degree_north)": [20.5],
            "LONGITUDE (degree_east)": [140.25],
            "PRES (decibar)": [10.0],
        }
    )

    with pytest.raises(ValueError, match="chla"):
        normalize_profile_table(raw)


def test_merge_cli_help_runs_from_repository_root_without_installing_package():
    """Removing the source-root bootstrap must make direct CLI help fail."""
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/merge_bgc_argo_profiles.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--input-dir" in result.stdout


def test_interpolate_profile_returns_requested_depth_grid():
    """Replacing interpolation with endpoint copying must fail this depth-grid contract."""
    profile = pd.DataFrame(
        {"depth": [0.0, 10.0], "chla": [1.0, 3.0], "float_id": ["a", "a"]}
    )
    result = interpolate_profile(profile, depth_grid=[0.0, 5.0, 10.0])
    assert result["depth"].tolist() == [0.0, 5.0, 10.0]
    assert result["chla"].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_process_profiles_reports_too_short_profile():
    """Treating a one-point profile as valid must fail this quality-control contract."""
    profile = pd.DataFrame(
        {
            "float_id": ["a"],
            "profile_id": ["p"],
            "time": pd.to_datetime(["2020-01-01"], utc=True),
            "lat": [20.0],
            "lon": [140.0],
            "depth": [10.0],
            "chla": [0.2],
        }
    )
    processed, report = process_profiles(profile, depth_step=1.0, smoothing_window=None)
    assert processed.empty
    assert report["skipped_profiles"] == 1
