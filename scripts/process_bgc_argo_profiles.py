"""Interpolate canonical BGC-Argo profiles on a 0-300 m depth grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from earthformer_ocean_chl.argo.profiles import process_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--depth-step", type=float, default=1.0)
    parser.add_argument("--smoothing-window", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed, report = process_profiles(
        pd.read_csv(args.input_csv), args.depth_step, args.smoothing_window
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output_csv, index=False)
    args.output_csv.with_suffix(".report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
