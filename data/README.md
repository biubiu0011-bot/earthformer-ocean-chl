# Data

> English version. [Original README](README.md)

This repository does not redistribute raw satellite data, NEMO/BGC-Argo data, NetCDF files, NumPy arrays, or checkpoints.

Set `EARTHFORMER_OCEAN_CHL_DATA_ROOT` before running scripts. The NEMO pre-training loader expects:

```text
${EARTHFORMER_OCEAN_CHL_DATA_ROOT}/
└── pre_train_data/
    ├── input_log_zscore/input_<index>.npy
    └── 2output_log_zscore_log10/output_<index>.npy
```

Inputs are surface Chl-a, SST, SSS, SSH, wind U, and wind V in `[14, 45, 60, 6]` NTHWC layout. Targets are `[14, 45, 60, 20]` Chl-a fields for the documented 0-300 m layers. Before reproduction, confirm source permissions, channel order, masks, grid, time convention, depth selection, normalization, split definition, and BGC-Argo quality-control rules. See [README_EN.md](../README_EN.md) for the project workflow.
