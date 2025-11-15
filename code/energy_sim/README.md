# Energy Simulation Notes

- **Engine**: EnergyPlus 24.1 (run via `geomeppy.IDF`). Ensure the executable path in `simulate_building_energy.py` is set accordingly.
- **Temporal resolution**: Hourly outputs aggregated to annual cooling/heating/other electricity demand per 250 m grid cell.
- **Simulation scope**: 55,284 buildings across the Xinwu study area, with archetype distribution: 34,977 low-rise residential, 15,303 high-rise residential, 4,052 industrial, 454 education, 376 office, 122 commercial.
- **Runtime**: Full batch (including post-processing) completes in approximately 8 h 27 m 07 s on the project’s workstation.
- **Input requirements**:
  - Morphology/built-environment indicators from `code/urban_form/`.
  - Weather file: typical meteorological year for Wuxi (`.epw`).
  - Building height/footprint attributes sourced from the *3D-GloBFP: the first global three-dimensional building footprint dataset*, aligned with local cadastral data.
  - Building archetype templates stored under the original project `assets/` directory (not redistributed; provide equivalents before running).
- **Outputs**: EnergyPlus reports are aggregated into `xinwu_energy_250m.gpkg` for subsequent analysis/training.

Adjust simulation years, timesteps, or archetypes by editing the constants near the top of `simulate_building_energy.py`.
