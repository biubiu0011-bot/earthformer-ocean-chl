# Argo Notebook-to-Script Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the reusable BGC-Argo notebook workflow with portable, tested Python command-line scripts for preprocessing, NEMO matching, transfer sample generation, and analysis.

**Architecture:** Put reusable profile and matching logic in `src/earthformer_ocean_chl/argo/`; keep `scripts/` as thin CLI entry points and `analysis/` as plotting/statistics entry points. Use explicit paths or the existing data/results environment variables, and keep historical notebooks ignored.

**Tech Stack:** Python 3.9, NumPy, pandas, xarray, SciPy, matplotlib, argparse, unittest/pytest-compatible tests.

**Spec:** `docs/superpowers/specs/2026-10-17-argo-notebooks-to-scripts-design.md`

## Global Constraints

- Use English identifiers and `argparse` `--help` for every public script.
- Do not embed original server paths, credentials, notebook outputs, or CMIP6-only workflow code.
- Do not load external data at import time.
- Preserve the NTHWC contract: inputs `[14, 45, 60, 6]`, outputs `[14, 45, 60, 20]`.
- Require explicit channel/depth/mask/normalization inputs; never infer thesis-critical values from filenames.
- Keep raw data, generated arrays, checkpoints and results outside Git.

## Review Focus

- Raw tables with alias headers or missing required columns must map deterministically or fail clearly; cover in Task 1.
- Profiles with duplicate, non-numeric, or too-few valid depths must be reported/skipped rather than silently interpolated; cover in Task 2.
- NEMO match tolerance failures must be distinguishable from missing source values; cover in Task 3.
- Transfer tensors with a wrong time/grid/channel/depth shape must be rejected before any `.npy` write; cover in Task 4.
- All scripts must run `--help` without requiring scientific datasets or optional visualization packages; cover in Task 6.

---

### Task 1: Canonical BGC-Argo profile ingestion

**Files:**
- Create: `src/earthformer_ocean_chl/argo/__init__.py`
- Create: `src/earthformer_ocean_chl/argo/profiles.py`
- Create: `scripts/merge_bgc_argo_profiles.py`
- Test: `tests/test_argo_profiles.py`

**Interfaces:**
- Produces: `normalize_profile_table(frame: pandas.DataFrame) -> pandas.DataFrame` with `float_id`, `profile_id`, `time`, `lat`, `lon`, `depth`, `chla`.
- Produces: `merge_profile_csvs(input_dir: Path, pattern: str) -> pandas.DataFrame` for Task 2.

- [ ] **Step 1: Write failing tests for aliases and missing required fields**

```python
def test_normalize_profile_table_maps_raw_bgc_aliases():
    result = normalize_profile_table(raw_bgc_frame)
    assert list(result.columns[:7]) == ["float_id", "profile_id", "time", "lat", "lon", "depth", "chla"]

def test_normalize_profile_table_rejects_missing_chla():
    with pytest.raises(ValueError, match="chla"):
        normalize_profile_table(frame_without_chla)
```

- [ ] **Step 2: Run the new test file and verify it fails because the module does not exist**

Run: `python -m pytest tests/test_argo_profiles.py -v`

Expected: FAIL with an import error for `earthformer_ocean_chl.argo.profiles`.

- [ ] **Step 3: Implement canonical-column mapping and CSV merge helpers**

Implement `normalize_profile_table` and `merge_profile_csvs` in `profiles.py`; parse timestamps, coerce numerical coordinates/depth/Chl-a, preserve stable source order, and reject absent canonical fields. Implement a thin CLI with `--input-dir`, `--pattern`, and `--output-csv`.

- [ ] **Step 4: Run Task 1 tests and the CLI help command**

Run: `python -m pytest tests/test_argo_profiles.py -v; python scripts/merge_bgc_argo_profiles.py --help`

Expected: PASS; help exits with status 0 without reading data.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/earthformer_ocean_chl/argo scripts/merge_bgc_argo_profiles.py tests/test_argo_profiles.py
git commit -m "feat: add portable BGC-Argo profile ingestion"
```

### Task 2: Profile interpolation and quality processing

**Files:**
- Modify: `src/earthformer_ocean_chl/argo/profiles.py`
- Create: `scripts/process_bgc_argo_profiles.py`
- Modify: `tests/test_argo_profiles.py`

**Interfaces:**
- Consumes: canonical frames from `normalize_profile_table`.
- Produces: `interpolate_profile(frame: pandas.DataFrame, depth_grid: numpy.ndarray) -> pandas.DataFrame` and `process_profiles(frame, depth_step, smoothing_window) -> tuple[pandas.DataFrame, dict]` for Tasks 3–4.

- [ ] **Step 1: Write failing interpolation and malformed-profile tests**

```python
def test_interpolate_profile_returns_requested_zero_to_300_depth_grid():
    result = interpolate_profile(valid_profile, np.arange(0, 301, 1))
    assert result["depth"].tolist() == list(range(301))

def test_process_profiles_reports_profile_with_only_one_valid_depth():
    _, report = process_profiles(single_point_profile, depth_step=1, smoothing_window=None)
    assert report["skipped_profiles"] == 1
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_argo_profiles.py -k "interpolate or skipped" -v`

Expected: FAIL because the interpolation functions are absent.

- [ ] **Step 3: Implement deterministic interpolation and optional smoothing**

Deduplicate equal depths by mean Chl-a, reject non-numeric depth values, linearly interpolate only inside valid depth bounds, and optionally apply an odd Savitzky-Golay window only when enough values exist. The CLI accepts `--input-csv`, `--output-csv`, `--depth-max`, `--depth-step`, and `--smoothing-window`, then writes a JSON processing report beside the CSV.

- [ ] **Step 4: Run Task 2 tests and CLI help**

Run: `python -m pytest tests/test_argo_profiles.py -v; python scripts/process_bgc_argo_profiles.py --help`

Expected: PASS; output depth grid and skip report match assertions.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/earthformer_ocean_chl/argo/profiles.py scripts/process_bgc_argo_profiles.py tests/test_argo_profiles.py
git commit -m "feat: process BGC-Argo depth profiles"
```

### Task 3: NEMO coordinate matching and metadata extraction

**Files:**
- Create: `src/earthformer_ocean_chl/argo/matching.py`
- Create: `scripts/match_argo_to_nemo.py`
- Create: `scripts/extract_argo_metadata.py`
- Create: `tests/test_argo_matching.py`

**Interfaces:**
- Consumes: processed profile table and an opened `xarray.Dataset`.
- Produces: `match_profiles_to_nemo(profiles, dataset, variable, time_tolerance, spatial_tolerance, depth_tolerance) -> pandas.DataFrame`.
- Produces: `extract_float_metadata(profiles: pandas.DataFrame) -> pandas.DataFrame`.

- [ ] **Step 1: Write failing nearest-match and tolerance tests**

```python
def test_match_profiles_to_nemo_returns_nearest_grid_value(tiny_profiles, tiny_nemo):
    result = match_profiles_to_nemo(tiny_profiles, tiny_nemo, "chl", "2D", 1.0, 5.0)
    assert result.loc[0, "nemo_chla"] == pytest.approx(0.42)

def test_match_profiles_to_nemo_marks_out_of_tolerance_rows_unmatched(tiny_profiles, tiny_nemo):
    result = match_profiles_to_nemo(far_profiles, tiny_nemo, "chl", "1D", 0.1, 1.0)
    assert result["match_status"].eq("out_of_tolerance").all()
```

- [ ] **Step 2: Run matching tests and verify they fail**

Run: `python -m pytest tests/test_argo_matching.py -v`

Expected: FAIL with an import error for `matching`.

- [ ] **Step 3: Implement explicit-dimension nearest NEMO matching**

Accept `time`, `lat`/`latitude`, `lon`/`longitude`, and `depth` dimension aliases; use xarray nearest selection, calculate selected-coordinate deltas, and assign `matched`, `out_of_tolerance`, or `missing_model_value` status. The matching CLI accepts an explicit NetCDF glob and writes an auditable CSV plus JSON summary. Metadata CLI selects one stable row per `float_id`.

- [ ] **Step 4: Run Task 3 tests and both help commands**

Run: `python -m pytest tests/test_argo_matching.py -v; python scripts/match_argo_to_nemo.py --help; python scripts/extract_argo_metadata.py --help`

Expected: PASS; each help command exits without data.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/earthformer_ocean_chl/argo/matching.py scripts/match_argo_to_nemo.py scripts/extract_argo_metadata.py tests/test_argo_matching.py
git commit -m "feat: match Argo profiles to NEMO"
```

### Task 4: Explicit Argo transfer-sample construction

**Files:**
- Create: `src/earthformer_ocean_chl/argo/samples.py`
- Create: `scripts/build_argo_transfer_samples.py`
- Create: `tests/test_argo_samples.py`

**Interfaces:**
- Produces: `validate_sample_shapes(inputs: numpy.ndarray, targets: numpy.ndarray, expected_input: tuple[int, int, int, int], expected_target: tuple[int, int, int, int]) -> None`.
- Produces: `write_transfer_samples(inputs, targets, output_root: Path) -> int`.

- [ ] **Step 1: Write failing valid-shape and wrong-shape tests**

```python
def test_write_transfer_samples_writes_matching_numbered_input_and_output_files(tmp_path):
    assert write_transfer_samples(inputs, targets, tmp_path) == 2
    assert (tmp_path / "transfer_input" / "input_0.npy").exists()

def test_validate_sample_shapes_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="6"):
        validate_sample_shapes(bad_inputs, targets, (14, 45, 60, 6), (14, 45, 60, 20))
```

- [ ] **Step 2: Run the sample tests and verify they fail**

Run: `python -m pytest tests/test_argo_samples.py -v`

Expected: FAIL with an import error for `samples`.

- [ ] **Step 3: Implement validation-first sample writing**

Validate sample count and every per-sample shape before creating output directories. CLI takes prepared input/target arrays plus `--output-root`; it requires supplied preprocessing provenance through a JSON argument/file and records it in metadata. Do not implement undocumented normalization or grid interpolation here.

- [ ] **Step 4: Run Task 4 tests and CLI help**

Run: `python -m pytest tests/test_argo_samples.py -v; python scripts/build_argo_transfer_samples.py --help`

Expected: PASS; no partial files are produced after a shape error.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/earthformer_ocean_chl/argo/samples.py scripts/build_argo_transfer_samples.py tests/test_argo_samples.py
git commit -m "feat: build validated Argo transfer samples"
```

### Task 5: Reproducible comparison and plot entry points

**Files:**
- Create: `analysis/compare_argo_nemo.py`
- Create: `analysis/plot_argo_profiles.py`
- Create: `analysis/plot_density_scatter.py`
- Create: `tests/test_argo_analysis.py`

**Interfaces:**
- Consumes: matched CSV columns `chla` and `nemo_chla`, plus optional profile fields.
- Produces: `calculate_comparison_metrics(frame: pandas.DataFrame) -> dict[str, float]` and output figures only from explicit `--output-dir`.

- [ ] **Step 1: Write failing metric and empty-match tests**

```python
def test_calculate_comparison_metrics_returns_rmse_mae_r_and_nse():
    metrics = calculate_comparison_metrics(matched_frame)
    assert set(metrics) >= {"rmse", "mae", "pearson_r", "nse"}

def test_calculate_comparison_metrics_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        calculate_comparison_metrics(empty_frame)
```

- [ ] **Step 2: Run analysis tests and verify they fail**

Run: `python -m pytest tests/test_argo_analysis.py -v`

Expected: FAIL with an import error for `compare_argo_nemo`.

- [ ] **Step 3: Implement focused analysis CLIs**

Implement metrics and a compact CSV/JSON report in `compare_argo_nemo.py`; implement profile and density-scatter plots with non-interactive matplotlib output. All three scripts require explicit source and output paths and avoid optional cartopy/seaborn dependencies.

- [ ] **Step 4: Run Task 5 tests and analysis help commands**

Run: `python -m pytest tests/test_argo_analysis.py -v; python analysis/compare_argo_nemo.py --help; python analysis/plot_argo_profiles.py --help; python analysis/plot_density_scatter.py --help`

Expected: PASS; help commands do not require input data.

- [ ] **Step 5: Commit Task 5**

```bash
git add analysis/compare_argo_nemo.py analysis/plot_argo_profiles.py analysis/plot_density_scatter.py tests/test_argo_analysis.py
git commit -m "feat: add reproducible Argo analysis scripts"
```

### Task 6: Documentation, ignore rules, and full verification

**Files:**
- Modify: `README.md`
- Modify: `data/README.md`
- Modify: `notebooks/README.md`
- Modify: `tests/test_repository_layout.py`
- Test: `tests/test_argo_profiles.py`
- Test: `tests/test_argo_matching.py`
- Test: `tests/test_argo_samples.py`
- Test: `tests/test_argo_analysis.py`

**Interfaces:**
- Consumes: public CLIs from Tasks 1–5.
- Produces: a README that identifies official Argo entry points and clearly labels notebooks as ignored local references.

- [ ] **Step 1: Write failing repository-layout assertions for the new public scripts**

```python
def test_public_argo_cli_scripts_exist():
    expected = ["scripts/merge_bgc_argo_profiles.py", "scripts/process_bgc_argo_profiles.py", "scripts/match_argo_to_nemo.py"]
    assert all((ROOT / path).is_file() for path in expected)
```

- [ ] **Step 2: Run the layout assertion and verify it fails until all script paths exist**

Run: `python -m pytest tests/test_repository_layout.py -v`

Expected: FAIL if any required public entry point is missing.

- [ ] **Step 3: Document public scripts and data contracts**

Update README run instructions and data README contracts with each input/output. Point notebook README to replacements and retain `notebooks/**/*.ipynb` ignore rule. Extend legacy-path scanning to include every public `.py`/`.sh` entry point.

- [ ] **Step 4: Run all tests, CLI help, syntax, and path scans**

Run: `python -m pytest tests -v; python -m compileall scripts analysis src/earthformer_ocean_chl; rg -n --glob '!notebooks/**' '/data/Lixiaoling|mycode/' .`

Expected: tests and compileall PASS; path scan returns no public-code matches.

- [ ] **Step 5: Commit Task 6**

```bash
git add README.md data/README.md notebooks/README.md tests/test_repository_layout.py
git commit -m "docs: document portable Argo preprocessing workflow"
```
