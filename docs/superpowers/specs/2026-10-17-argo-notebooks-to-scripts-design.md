# Argo Notebook-to-Script Refactor Design

## Goal

Replace the reusable parts of the historical Argo notebooks with portable, command-line Python scripts. The public workflow must follow the thesis pipeline:

```text
raw BGC-Argo profiles -> profile QC and interpolation -> NEMO matching -> transfer samples -> comparison and plots
```

The refactor must not copy legacy server paths, credentials, notebook outputs, or unrelated CMIP6 experiments into public code.

## Scope

The source notebooks are local reference material in `notebooks/argo/` and remain ignored by Git. New scripts are the public implementation.

| Source notebook | New public file | Responsibility |
|---|---|---|
| `01_merge_bgc_argo.ipynb` | `scripts/merge_bgc_argo_profiles.py` | Merge raw profile CSV files; select and normalize canonical columns. |
| `02_process_bgc_argo.ipynb` | `scripts/process_bgc_argo_profiles.py` | Interpolate each profile to 0–300 m, apply profile smoothing and save a canonical processed table. |
| `02_match_all_float_coordinates.ipynb` | `scripts/match_argo_to_nemo.py` | Match processed profiles to NEMO in time, latitude, longitude and depth. |
| `03_inspect_input_output.ipynb` | `scripts/build_argo_transfer_samples.py` | Build NTHWC transfer input/output `.npy` samples compatible with the Argo data module. |
| `extract_unique_bgc_argo_metadata.ipynb` | `scripts/extract_argo_metadata.py` | Write one metadata row per float/platform. |
| `compare_datasets.ipynb` | `analysis/compare_argo_nemo.py` | Compute matched statistics and save comparison tables/figures. |
| `05_plot_single_float_profile.ipynb` | `analysis/plot_argo_profiles.py` | Plot selected processed float profiles. |
| `density_scatter_plot.ipynb` | `analysis/plot_density_scatter.py` | Draw a density-scatter figure from matched values. |

`01_match_single_float_coordinates.ipynb` is excluded: it depends on a distinct CMIP6/ConvLSTM experiment and is not part of the NEMO-to-Argo thesis pipeline. Archive-copy notebooks are also excluded.

## Script interface conventions

Every script will:

- use English identifiers and `argparse` with `--help`;
- accept explicit input and output paths, with defaults derived from `EARTHFORMER_OCEAN_CHL_DATA_ROOT` or `EARTHFORMER_OCEAN_CHL_RESULTS_ROOT`;
- fail early with clear errors when paths or required columns are absent;
- create output parent directories only when writing results;
- avoid loading project data at import time;
- write only documented artifacts.

The scripts will use the repository data contract: input tensors use `NTHWC`, six surface channels, 14 time steps, a 45×60 grid, and target tensors with 20 depth channels. Exact grid construction and normalization statistics remain explicit arguments or documented prerequisites rather than inferred values.

## Data contracts

### Profile table

The normalized profile table uses these canonical columns when available:

`float_id`, `profile_id`, `time`, `lat`, `lon`, `depth`, `chla`.

Raw BGC-Argo header aliases such as `PLATFORM_CODE`, `DATE (YYYY-MM-DDTHH:MI:SSZ)`, `LATITUDE (degree_north)`, and `LONGITUDE (degree_east)` are mapped at ingestion. Scripts retain additional source columns only when requested.

### Processed profiles

`process_bgc_argo_profiles.py` emits depth-resolved rows from 0 through 300 m at a configured step (default 1 m). It preserves profile identity/time/location and documents the interpolation and smoothing choices in output-sidecar metadata.

### NEMO matches

`match_argo_to_nemo.py` accepts an NEMO NetCDF glob, a variable name and match tolerances. It writes matched observation/model values plus the selected grid coordinates and depth, so downstream metrics remain auditable.

### Transfer samples

`build_argo_transfer_samples.py` writes `input_<index>.npy` and `output_<index>.npy` in a user-selected output directory. It validates final sample shape before writing. It does not silently choose channel order, depth selection, masks or normalization statistics.

## Error handling and validation

- Reject duplicate or non-numeric profile depths within a profile when they cannot be resolved deterministically.
- Skip or report profiles without enough valid depth/Chl-a points for interpolation.
- Reject transfer samples whose shape differs from the configured `[14, 45, 60, 6]` / `[14, 45, 60, 20]` contracts.
- Require a non-empty match result before statistical analysis.
- Write concise JSON metadata beside generated data, recording input paths, options, row/sample counts and skipped profile count.

## Tests

Add lightweight unit tests using synthetic data only:

1. Canonical BGC-Argo columns are selected and normalized.
2. A small profile interpolates to the requested depth grid.
3. A malformed profile is reported or skipped.
4. NEMO matching selects expected nearest coordinates in a tiny in-memory dataset.
5. Every public script exposes `--help` without needing data.

## Documentation changes

- Update `README.md` to name the new Argo scripts and distinguish them from ignored historical notebooks.
- Update `notebooks/README.md` to point users to the replacement scripts.
- Extend `data/README.md` with the processed-profile and transfer-sample contracts.
- Preserve `docs/transfer_learning.md` as the explanation of training, not as an executable preprocessing notebook.

## Non-goals

- Reproducing every exploratory plot or temporary notebook cell.
- Publishing private raw data, model weights, original server paths or credentials.
- Changing thesis channel order, depth layers, masks, normalization values or reported metrics.
