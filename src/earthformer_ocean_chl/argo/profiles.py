"""Canonical BGC-Argo profile table ingestion and processing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_ALIASES = {
    "float_id": ("float_id", "PLATFORM_CODE", "platform_code", "fubiao"),
    "profile_id": ("profile_id", "fubiaohao", "PROFILE_ID", "CYCLE_NUMBER"),
    "time": ("time", "DATE (YYYY-MM-DDTHH:MI:SSZ)", "JULD", "date"),
    "lat": ("lat", "LATITUDE (degree_north)", "LATITUDE", "latitude"),
    "lon": ("lon", "LONGITUDE (degree_east)", "LONGITUDE", "longitude"),
    "depth": ("depth", "PRES (decibar)", "PRES", "pressure"),
    "chla": ("chla", "CHLA (mg/m3)", "CHLA", "CHLA_ADJUSTED"),
}

REQUIRED_COLUMNS = ("float_id", "time", "lat", "lon", "depth", "chla")


def _first_present(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((column for column in aliases if column in frame.columns), None)


def normalize_profile_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common BGC-Argo headers to a canonical, typed profile table."""
    chosen = {
        canonical: _first_present(frame, aliases)
        for canonical, aliases in CANONICAL_ALIASES.items()
    }
    missing = [name for name in REQUIRED_COLUMNS if chosen[name] is None]
    if missing:
        raise ValueError(f"Missing required BGC-Argo columns: {', '.join(missing)}")

    result = pd.DataFrame(
        {
            canonical: frame[source]
            for canonical, source in chosen.items()
            if source is not None
        }
    )
    result["float_id"] = result["float_id"].astype(str)
    result["time"] = pd.to_datetime(result["time"], utc=True, errors="coerce")
    for column in ("lat", "lon", "depth", "chla"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if chosen["profile_id"] is None:
        result["profile_id"] = (
            result["float_id"] + "_" + result["time"].dt.strftime("%Y%m%dT%H%M%SZ")
        )
    else:
        result["profile_id"] = result["profile_id"].astype(str)

    ordered = ["float_id", "profile_id", "time", "lat", "lon", "depth", "chla"]
    return result.loc[:, ordered]


def merge_profile_csvs(input_dir: Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Read and normalize all matching profile CSV files in stable name order."""
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {input_dir}")
    return pd.concat(
        [normalize_profile_table(pd.read_csv(path)) for path in paths],
        ignore_index=True,
    )


def interpolate_profile(frame: pd.DataFrame, depth_grid: list[float] | np.ndarray) -> pd.DataFrame:
    """Linearly interpolate one valid profile onto an explicit depth grid."""
    valid = frame.dropna(subset=["depth", "chla"]).copy()
    valid["depth"] = pd.to_numeric(valid["depth"], errors="coerce")
    valid["chla"] = pd.to_numeric(valid["chla"], errors="coerce")
    valid = valid.dropna(subset=["depth", "chla"])
    valid = valid.groupby("depth", as_index=False, sort=True)["chla"].mean()
    if len(valid) < 2:
        raise ValueError("A profile needs at least two valid depth/chla points")
    grid = np.asarray(depth_grid, dtype=float)
    values = np.interp(grid, valid["depth"], valid["chla"], left=np.nan, right=np.nan)
    result = pd.DataFrame({"depth": grid, "chla": values})
    for column in frame.columns:
        if column not in result and column not in {"depth", "chla"}:
            result[column] = frame.iloc[0][column]
    return result


def process_profiles(
    frame: pd.DataFrame, depth_step: float = 1.0, smoothing_window: int | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Interpolate valid profiles between 0 and 300 m and report skipped profiles."""
    if depth_step <= 0:
        raise ValueError("depth_step must be positive")
    grid = np.arange(0.0, 300.0 + depth_step / 2, depth_step)
    processed: list[pd.DataFrame] = []
    skipped = 0
    for _, profile in frame.groupby("profile_id", sort=False):
        try:
            result = interpolate_profile(profile, grid)
        except ValueError:
            skipped += 1
            continue
        if smoothing_window is not None:
            if smoothing_window < 3 or smoothing_window % 2 == 0:
                raise ValueError("smoothing_window must be an odd integer of at least 3")
            valid = result["chla"].notna()
            if valid.sum() >= smoothing_window:
                from scipy.signal import savgol_filter

                result.loc[valid, "chla"] = savgol_filter(
                    result.loc[valid, "chla"], smoothing_window, 2
                )
        processed.append(result)
    output = pd.concat(processed, ignore_index=True) if processed else frame.iloc[0:0].copy()
    return output, {"input_profiles": frame["profile_id"].nunique(), "skipped_profiles": skipped}
