# Data

> English version: this document is maintained in English.

This repository does not redistribute raw satellite, NEMO, BGC-Argo, NetCDF, Numpy, or checkpoint files.

Set `EARTHFORMER_OCEAN_CHL_DATA_ROOT` to a local data directory before running scripts. The NEMO pre-training loader expects:

```text
${EARTHFORMER_OCEAN_CHL_DATA_ROOT}/
└── pre_train_data/
    ├── input_log_zscore/input_<index>.npy
    └── 2output_log_zscore_log10/output_<index>.npy
```

## Data sources and reproducibility checks

The thesis workflow combines surface Chl-a, SST, SSS, SSH, wind U/V, NEMO three-dimensional Chl-a fields, and BGC-Argo profiles. The documented data sources are CMEMS/CNR for surface Chl-a and SSS, OSTIA for SST, AVISO+/CMEMS for SSH, CCMP v3.1 for winds, NEMO for source-domain labels, and BGC-Argo for transfer/validation observations.

Before reproducing training, verify all of the following against the preprocessing records:

- Input channel order is surface Chl-a, SST, SSS, SSH, wind U, wind V.
- Input tensors use `NTHWC` layout with per-sample shape `[14, 45, 60, 6]`.
- Target tensors use per-sample shape `[14, 45, 60, 20]`; the 20 channels represent the documented 0–300 m depth layers.
- NEMO and BGC-Argo samples use the same land/ocean mask, spatial grid, time convention, depth selection, normalization statistics, and train/validation/test split definition.
- BGC-Argo profiles have documented quality-control rules, units, time parsing, coordinate convention, and handling for missing depth/Chl-a values before interpolation.

The public scripts deliberately require these choices as explicit inputs or documented prerequisites; they do not infer scientific parameters from filenames.

For Argo preprocessing, public scripts exchange a canonical CSV table with `float_id`, `profile_id`, `time`, `lat`, `lon`, `depth`, and `chla`. Processed profiles are interpolated over an explicitly selected 0–300 m depth grid. Transfer tensors must be prepared externally with the documented NTHWC shapes `[N, 14, 45, 60, 6]` and `[N, 14, 45, 60, 20]` before `build_argo_transfer_samples.py` writes them.

