# Earthformer-Ocean-Chl

> English version. [中文版 / Chinese version](README.md)

This project reconstructs daily three-dimensional chlorophyll-a (Chl-a) fields in the northwest Pacific through deep transfer learning. It extends [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer): a 14-day sequence of surface variables is used for NEMO pre-training and then transferred to sparse BGC-Argo profiles to reconstruct 20 daily Chl-a layers from 0 to 300 m.

> This is a public code-only release. Raw satellite, NEMO, and BGC-Argo data, checkpoints, logs, and generated results are not distributed. Read [data/README.md](data/README.md) and [docs/transfer_learning.md](docs/transfer_learning.md) before reproducing an experiment.

## Workflow

```text
Data processing -> NEMO pre-training -> Argo transfer -> baseline comparison -> interpretability
```

1. Prepare and quality-control surface observations, NEMO fields, and BGC-Argo profiles.
2. Pre-train Earthformer-ISTRB against complete NEMO three-dimensional fields.
3. Load the pre-trained checkpoint, freeze the encoder, and fine-tune the decoder for BGC-Argo transfer.
4. Compare with CNN, ConvLSTM, and ViT baselines and run ablations.
5. Use gradient attribution, sensitivity analysis, and attention analysis for interpretation.

## Method

- Input: `[batch, 14 days, 45, 60, 6 variables]` in NTHWC layout.
- Channels: surface Chl-a, SST, SSS, SSH, wind U, wind V.
- Output: `[batch, 14 days, 45, 60, 20 depth layers]` representing 0-300 m Chl-a.
- Model: Earthformer with 2D+1D improved spatiotemporal residual blocks (ISTRB).
- Loss: Hybrid Scale-aware Loss (Huber + MSE, `alpha=0.7`).

## Reported results

| Model on NEMO test data | Pearson's r | NSE | RMSE (mg/m³) | MAE (mg/m³) |
|---|---:|---:|---:|---:|
| CNN | 0.8589 | 0.7364 | 0.0900 | 0.0483 |
| ConvLSTM | 0.8746 | 0.7641 | 0.0851 | 0.0446 |
| ViT | 0.8633 | 0.7440 | 0.0887 | 0.0451 |
| Earthformer-ISTRB | **0.8993** | **0.8056** | **0.0741** | **0.0368** |

The reported Argo transfer test result is r = 0.8935, NSE = 0.7975, RMSE = 0.0722 mg/m³, and MAE = 0.0343 mg/m³.

## Repository map

| Area | Location |
|---|---|
| NEMO pre-training | `scripts/train_nemo_pretrain.py`, `configs/nemo_pretrain.yaml` |
| Argo transfer | `scripts/train_argo_transfer.py`, `configs/argo_transfer.yaml` |
| Custom data modules | `src/earthformer_ocean_chl/data/` |
| Earthformer-ISTRB and residual blocks | `src/earthformer/cuboid_transformer/` |
| Hybrid loss | `src/earthformer/losses/hybrid_scale_aware_loss.py` |
| Baselines | `baselines/` |
| Interpretability | `analysis/` |
| Local historical notebooks | `notebooks/` (ignored by Git) |

### Public Argo utilities

- `scripts/merge_bgc_argo_profiles.py`: merge and normalize raw profile CSV files.
- `scripts/process_bgc_argo_profiles.py`: interpolate profiles from 0 to 300 m and optionally smooth them.
- `scripts/match_argo_to_nemo.py`: nearest-neighbour matching between profiles and NEMO.
- `scripts/extract_argo_metadata.py`: one metadata record per float.
- `scripts/build_argo_transfer_samples.py`: validate and save prebuilt transfer tensors.
- `analysis/compare_argo_nemo.py`, `plot_argo_profiles.py`, and `plot_density_scatter.py`: reproducible comparison and figures.

## Installation

```bash
conda env create -f environment.yml
conda activate earthformer-ocean-chl
pip install -e .
```

Use Linux or WSL with an NVIDIA GPU where possible. CUDA, PyTorch, PyTorch Lightning, and Apex must be compatible. Apex, if required, must be compiled separately for the local CUDA/PyTorch combination.

## Data and paths

Set paths before running scripts:

```bash
export EARTHFORMER_OCEAN_CHL_DATA_ROOT=/path/to/ocean-chl-data
export EARTHFORMER_OCEAN_CHL_RESULTS_ROOT=/path/to/ocean-chl-results
```

See [data/README.md](data/README.md) for the expected NEMO sample layout, source data, channel order, and mandatory reproducibility checks. Never commit raw data, checkpoints, local paths, credentials, or generated training outputs.

## Run

```bash
python tests/test_repository_layout.py

python scripts/train_nemo_pretrain.py --gpus 1 \
  --cfg configs/nemo_pretrain.yaml --save nemo_pretrain

python scripts/train_argo_transfer.py --gpus 1 \
  --cfg configs/argo_transfer.yaml \
  --pretrain_ckpt /path/to/nemo_pretrain.ckpt --save argo_transfer
```

These are the current code entry points, not a claim that the thesis can be reproduced without preparing compatible data, masks, depth layers, and normalization statistics.

## License and citation

The upstream Earthformer code under `src/earthformer/` retains its [Apache-2.0 license](LICENSE) and notices. Cite the associated thesis and Gao et al., *Earthformer: Exploring Space-Time Transformers for Earth System Forecasting*, NeurIPS 2022.
