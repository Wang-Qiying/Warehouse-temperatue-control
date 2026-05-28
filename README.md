# Refrigerated Warehouse Control Benchmark

This repository contains the modeling data, weather data interface, utility functions, and controller implementations used in a refrigerated-warehouse temperature-control study. The case study focuses on a pharmaceutical cold-storage warehouse operated within a 2--6 °C temperature range, with photovoltaic generation, time-of-use electricity prices, weather forecasts, and empirical violation-rate constraints.

The released code is intended to support reproduction of the paper's controller comparison and to make the data-processing and method-implementation details transparent. The BEAR simulation environment used in the paper is **not** included in this repository.

> Paper: `<Statistical Feedback Adaptive Chance Constrained Temperature Control of Low-Carbon Refrigerated Warehouses>`  
> Authors: `<Qiying Wang and Wei Wei and Songjie Feng>`  

---

## Repository contents

```text
.
├── warehouse_thermal_model.py      # Refrigerated warehouse thermal model and BEAR parameter builder
├── warehouse_helpers.py            # Utility functions for weather, pricing, PV, metrics, and output saving
├── warehouse_methods.py            # DP/LP solvers and baseline controller implementations
├── weather_data/                   # Weather observations and historical forecasts
│   ├── Warm-Dry/
│   ├── Warm-Humid/
│   ├── Hot-Dry/
│   ├── Very-Cold/
│   └── additional_cities/
└── output/                         # Generated experiment outputs; not required as input
```

The exact names of the weather-data subfolders and CSV files can be adjusted, but the experiment scripts should consistently map each weather case to the corresponding observation and forecast files.

---

## Code modules

### `warehouse_thermal_model.py`

This module defines the refrigerated warehouse model used to construct BEAR-compatible building parameters.

Main components:

- `WarehouseThermalConfig`: physical and BEAR-interface parameters for the warehouse.
- `make_pharma_cold_occupancy(...)`: occupancy and metabolic heat schedule for a pharmaceutical cold-storage warehouse.
- `build_bear_warehouse_parameter(...)`: converts processed weather data into the `parameter` dictionary expected by BEAR's `BuildingEnvReal` interface.

The default configuration represents a small refrigerated warehouse with:

- 100 m² floor area;
- 4 m room height;
- 8 kW maximum heating/cooling actuation power;
- 4 °C nominal target temperature;
- 2--6 °C operating temperature band used in the paper.

The function `build_bear_warehouse_parameter(...)` expects a weather table with at least:

```text
temp_air      outdoor air temperature, °C
ghi           global horizontal irradiance, W/m²
```

An optional `ground_temp` column can also be supplied. If it is absent, `temp_air` is used as the fallback ground/surface temperature.

### `warehouse_helpers.py`

This module contains shared utilities used by the experiment workflow:

- cold-storage occupancy and activity schedules;
- time-of-use electricity price generation;
- PV availability calculation;
- simple forecast perturbation utilities;
- BEAR-based thermal-parameter identification;
- adaptive safety-margin update rules used by SF-ACC/OA-SMPC-style controllers;
- energy and PV-consumption accounting;
- construction of per-case result DataFrames;
- violation-rate and performance-indicator calculation;
- summary-table generation and CSV output saving.

Typical output metrics include:

- total energy use;
- grid energy use;
- PV self-consumption;
- PV self-sufficiency;
- empirical violation rate;
- windowed violation rate;
- mean and maximum violation magnitude;
- maximum consecutive violation duration;
- temperature statistics;
- actuation statistics.

### `warehouse_methods.py`

This module contains the controller-side optimization and baseline implementations used in the comparison study.

Main components:

- `solve_dp(...)`: exact finite-horizon dynamic-programming controller based on a five-interval piecewise-linear recursion.
- `solve_dp_debug(...)`: debug version of the DP solver that returns value functions, slopes, state/action paths, and stage-level checks.
- `solve_lp(...)`: linear-programming baseline used to validate the DP solution.
- `validate_exact_dp_against_lp(...)`: batch validation utility for comparing exact DP and LP solutions.
- `plot_dp_stage_slopes(...)`: plotting utility for DP value-function slopes.
- `solve_lyapunov_hvac(...)`: Lyapunov-style baseline controller.
- `solve_rule_base(...)`: model-based rule-based baseline controller.

The paper compares SF-ACC, OA-SMPC, MPC, Lyapunov, and Rule-Based methods. The adaptive-margin update utilities in `warehouse_helpers.py` and the controller functions in `warehouse_methods.py` provide the reusable components for these comparisons.

---

## Weather data

The `weather_data/` folder contains paired observation and forecast data for the weather cases used in the paper.

The primary weather cases are:

| Case | Description |
|---|---|
| `Warm-Dry` | Warm and dry climate condition |
| `Warm-Humid` | Warm and humid climate condition |
| `Hot-Dry` | Hot and dry climate condition |
| `Very-Cold` | Very cold climate condition |

Additional candidate cities are also included as backup weather cases.

Each location contains:

1. **NOAA USCRN 15-minute observation data**  
   Used as the realized weather trajectory or observation truth.

2. **NBM 1-hour historical forecast data**  
   Used as the corresponding weather forecast sequence for control.

Recommended processed columns include:

```text
timestamp      local or UTC timestamp
temp_air       outdoor air temperature, °C
ghi            global horizontal irradiance, W/m²
ground_temp    optional ground/surface temperature, °C
```

For controller experiments, the weather processing scripts should align the 15-minute observation data and 1-hour forecast data to the simulation/control time grid used in the paper.

---

## External dependency: BEAR

The BEAR simulation environment used in the paper is not included in this repository.

The code in this repository can build BEAR-compatible model parameters, but it does not ship the BEAR environment itself. To run the full closed-loop simulation workflow, users need to install or obtain BEAR separately and make sure that the BEAR import path is available, for example:

```python
from BEAR.Env.env_building import BuildingEnvReal
```

Then the parameter dictionary returned by `build_bear_warehouse_parameter(...)` can be passed to BEAR:

```python
import pandas as pd
from warehouse_thermal_model import WarehouseThermalConfig, build_bear_warehouse_parameter

weather = pd.read_csv("weather_data/Warm-Dry/WD_true_data.csv")

cfg = WarehouseThermalConfig()
parameter, derived = build_bear_warehouse_parameter(
    weather,
    config=cfg,
    people_per_zone=[0, 0, 2],
    steps_per_hour=4,
)

# Requires external BEAR installation:
# from BEAR.Env.env_building import BuildingEnvReal
# env = BuildingEnvReal(parameter)
```

---

## Installation

A minimal Python environment can be created with:

```bash
conda create -n warehouse-control python=3.10
conda activate warehouse-control
pip install numpy pandas scipy matplotlib pvlib
```

If you run the full simulation experiments, install BEAR separately according to the version used in your project.

---

## Basic usage

### Build a BEAR-compatible warehouse model

```python
import pandas as pd
from warehouse_thermal_model import WarehouseThermalConfig, build_bear_warehouse_parameter

weather = pd.read_csv("weather_data/Warm-Dry/WD_true_data.csv")

parameter, derived = build_bear_warehouse_parameter(
    weather,
    config=WarehouseThermalConfig(),
    people_per_zone=[0, 0, 2],
    steps_per_hour=4,
)
```

### Solve one finite-horizon DP control step

```python
import numpy as np
from warehouse_methods import solve_dp

solution = solve_dp(
    fore_len=24,
    A=0.98,
    B=0.05,
    x0=4.0,
    w_fore=np.zeros(24),
    h=0.5,
    E_free=np.zeros(24),
    lam=np.ones(24),
    eta_c=2.0,
    eta_h=1.0,
    Tmax=6.0,
    Tmin=2.0,
    qmin=-1.0,
    qmax=1.0,
)

print(solution.q_star, solution.e_grid)
```

### Save one experiment result

```python
from warehouse_helpers import build_case_dataframe, save_case_output

df = build_case_dataframe(
    u=u,
    x=x,
    e_grid=e_grid,
    e_free_true=e_free_true,
    h_like=h_values,
    T_min=2.0,
    T_max=6.0,
)

save_case_output(
    df,
    method="SF-ACC",
    weather="Warm-Dry",
    location="ExampleCity",
    output_dir="output",
)
```

---

## Reproducibility notes

- The repository includes the modeling utilities, weather-data interface, controller implementations, and comparison-method components used by the paper.
- The released files do **not** include the BEAR simulation environment.
- Raw or processed weather files should be kept in `weather_data/` and mapped consistently to the paper's weather cases.
- Generated experiment outputs should be written to `output/` or another ignored results directory.
- Randomized forecast perturbations should use fixed seeds when exact reproducibility is required.
- When comparing controllers, use the same weather trajectory, forecast horizon, electricity price sequence, PV availability sequence, thermal parameters, and violation-rate settings across all methods.

---
