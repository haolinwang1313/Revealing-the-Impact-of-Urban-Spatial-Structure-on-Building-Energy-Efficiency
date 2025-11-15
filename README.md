# Urban Morphology & Energy Simulation Toolkit

This repository contains the assets required to preprocess urban spatial layers, compute morphology features, and run large-scale EnergyPlus simulations for district-scale energy studies.

## Contents

### `data/gpkg/`
Geopackage layers used throughout the pipeline (buildings, land use, transport, grids, and final energy tiles). Drop these into the same relative paths before running any script.

### `code/urban_form/`
Scripts to prepare and quantify morphology indicators on 250 m grids:
- `preprocess_xinwu.py`: clean and harmonize raw geospatial inputs.
- `calc_urban_form.py`, `calc_built_environment.py`, `calc_transportatio.py`, `compute_integration_variants.py`: compute form, built environment, and road-network metrics.

### `code/energy_sim/`
`simulate_building_energy.py` orchestrates EnergyPlus batch runs using the processed morphology/built environment data.

### `code/xai/`
SHAP utilities for trained XGBoost surrogates: `Allmodel/run_shap_analysis.py` (20 features) and `8xmodel/run_form_only_shap.py` (form-only). They generate dependence/summary/interaction figures for interpretability.

## Environment
Install dependencies via `pip install -r requirements.txt`. EnergyPlus binaries need to be available in `PATH` (see script header for the expected version).

## Usage
1. Preprocess spatial data and compute morphology metrics (`code/urban_form`).
2. Run `simulate_building_energy.py` to generate EnergyPlus outputs on the study grid.
3. Train XGBoost regressors (not included) and apply the SHAP scripts under `code/xai/` for model interpretation.
