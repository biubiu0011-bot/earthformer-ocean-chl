"""Merge raw BGC-Argo CSV profiles into a canonical table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from earthformer_ocean_chl.argo.profiles import merge_profile_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = merge_profile_csvs(args.input_dir, args.pattern)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(table)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
