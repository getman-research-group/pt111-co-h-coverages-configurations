# Coverages and Configurations of CO* and H* on Pt(111) under Realistic Reaction Conditions

This repository contains the figure-generation scripts, processed plotting datasets, cluster-expansion files, grand canonical Monte Carlo (GCMC) simulation code, GCMC snapshots, and spatial-analysis utilities associated with the manuscript:

Sifat Hossain, Venkata Rohit Punyapu, Phillip Christopher and Rachel B. Getman  
"Coverages and Configurations of CO* and H* on Pt(111) under Realistic Reaction Conditions"

The processed `.pkl` files allow the manuscript figures to be regenerated without rerunning the upstream DFT, cluster-expansion, GCMC, or spatial-analysis calculations. The larger archived folders preserve the model files and representative/sampled structures used in the workflow.

## Repository Layout

```text
.
|-- README.md
|-- LICENSE.md
|-- LICENSE-CODE-MIT.txt
|-- LICENSE-DATA-CC-BY-4.0.txt
|-- CE_FEATURES/
|-- FIGURE_2/
|-- FIGURE_3/
|-- FIGURE_4/
|-- FIGURE_S1/
|-- FIGURE_S2/
|-- FIGURE_S3/
|-- FIGURE_S4/
|-- FIGURE_S5/
|-- FIGURE_S7/
|-- GCMC/
|-- GSCs/
|-- RECON_SORTED/
|-- SNAPSHOTS/
`-- SPATIAL_ANALYSIS/
```

Large collections of individual structure files are summarized below rather than listed exhaustively.

## Workflow Overview

```text
DFT calculations
      |
      v
Cluster-expansion fitting and validation
      |
      v
GCMC sampling of CO*/H* adlayers
      |
      v
Saved POSCAR snapshots
      |
      v
Spatial analysis of islands, low-coverage regions, and CO*-H* neighbors
      |
      v
Processed pickle datasets
      |
      v
Main-text and supplemental figures
```

The cluster-expansion models describe the energetics of CO* and H* configurations on Pt(111) under three surface charge states: `Negative`, `Neutral`, and `Positive`. The GCMC code samples adsorbate configurations using these energetics. The saved POSCAR snapshots are then analyzed to quantify CO* coverage, CO* islanding, low-coverage regions, H* coverage, and CO*-H* first-nearest-neighbor behavior.

## Figure Folders

Each figure folder contains a Python plotter and the processed `.pkl` dataset or datasets required by that plotter. Plotters use paths relative to their own folder.

| Folder | Contents | Purpose |
| --- | --- | --- |
| `FIGURE_2/` | `FIGURE_2_PLOTTER.py`, `FIGURE_2_CO_COVERAGE_HEATMAP.pkl` | Main-text CO* coverage heatmaps. |
| `FIGURE_3/` | `FIGURE_3_PLOTTER.py`, `FIGURE_3_CO_ISLANDING_DISTRIBUTION.pkl` | Main-text CO* island-size distributions. |
| `FIGURE_4/` | `FIGURE_4_PLOTTER.py`, `FIGURE_4A_THETA_LOW.pkl`, `FIGURE_4B_THETA_H.pkl`, `FIGURE_4C_THETA_COH_1NN.pkl` | Main-text spatial organization metrics: low-CO* regions, H* coverage, and CO*-H* 1NN coverage. |
| `FIGURE_S1/` | `FIGURE_S1_PLOTTER.py`, `FIGURE_S1_CO_BE_CHARGE.pkl` | Supplemental CO binding energy versus surface charge. |
| `FIGURE_S2/` | `FIGURE_S2_PLOTTER.py`, `FIGURE_S2_PRE_COVERAGE_BINDING_ENERGY.pkl` | Supplemental pre-coverage-dependent CO*/H* binding-energy data. |
| `FIGURE_S3/` | `FIGURE_S3_PLOTTER.py`, `FIGURE_S3_WINDOW_SENSITIVITY.pkl` | Supplemental window-width sensitivity data. |
| `FIGURE_S4/` | `FIGURE_S4_PLOTTER.py`, `FIGURE_S4_GCMC_EQUILIBRATION.pkl`, `FIGURE_S4.png` | Supplemental GCMC equilibration trace and convergence-window statistics. |
| `FIGURE_S5/` | `FIGURE_S5_PLOTTER.py`, `FIGURE_S5_PARITY.pkl`, `FIGURE_S5.png` | Supplemental cluster-expansion parity plots for training/testing data across charge states. |
| `FIGURE_S7/` | `FIGURE_S7_PLOTTER.py`, `FIGURE_S7_FUNCTIONALS.pkl`, `FIGURE_S7.png` | Supplemental comparison of CO* binding energies across density functionals and dispersion corrections. |

There is no `FIGURE_S6/` directory in the current archive.

## Reproducing Figures

Run each plotter from its own figure directory:

```bash
cd FIGURE_2
python FIGURE_2_PLOTTER.py

cd ../FIGURE_3
python FIGURE_3_PLOTTER.py

cd ../FIGURE_4
python FIGURE_4_PLOTTER.py
```

The supplemental figures can be regenerated in the same way:

```bash
cd FIGURE_S1
python FIGURE_S1_PLOTTER.py

cd ../FIGURE_S2
python FIGURE_S2_PLOTTER.py

cd ../FIGURE_S3
python FIGURE_S3_PLOTTER.py

cd ../FIGURE_S4
python FIGURE_S4_PLOTTER.py

cd ../FIGURE_S5
python FIGURE_S5_PLOTTER.py

cd ../FIGURE_S7
python FIGURE_S7_PLOTTER.py
```

Several plotters save `.png` output files in their own folder. If running on a headless machine, set a non-interactive Matplotlib backend, for example:

```bash
MPLBACKEND=Agg python FIGURE_S5_PLOTTER.py
```

On Windows PowerShell:

```powershell
$env:MPLBACKEND = "Agg"
py FIGURE_S5_PLOTTER.py
```

## Processed Dataset Notes

The processed pickle files are Python dictionaries. Their top-level keys are:

| Dataset | Top-level keys |
| --- | --- |
| `FIGURE_2_CO_COVERAGE_HEATMAP.pkl` | `columns`, `data` |
| `FIGURE_3_CO_ISLANDING_DISTRIBUTION.pkl` | `NEG`, `NEU`, `POS` |
| `FIGURE_4A_THETA_LOW.pkl` | `__type__`, `columns`, `index`, `data` |
| `FIGURE_4B_THETA_H.pkl` | `__type__`, `columns`, `index`, `data` |
| `FIGURE_4C_THETA_COH_1NN.pkl` | `__type__`, `columns`, `index`, `data` |
| `FIGURE_S1_CO_BE_CHARGE.pkl` | `charge_per_surface_pt`, `co_binding_energy_eV`, `highlight_indices` |
| `FIGURE_S2_PRE_COVERAGE_BINDING_ENERGY.pkl` | CO/H coverage arrays and charge-state binding-energy arrays |
| `FIGURE_S3_WINDOW_SENSITIVITY.pkl` | `window_width_ML`, `roughness`, `bootstrap_sd_ML`, `selected_window_width_ML` |
| `FIGURE_S4_GCMC_EQUILIBRATION.pkl` | `format`, `n_sites`, `columns` |
| `FIGURE_S5_PARITY.pkl` | `format`, `description`, `systems` |
| `FIGURE_S7_FUNCTIONALS.pkl` | `labels`, `binding_energy_eV`, `experimental_range_eV` |

The S7 dataset stores CO* binding energies and the experimental range using the sign convention plotted in `FIGURE_S7_PLOTTER.py`.

## Data Availability

The data and code supporting this article are available in this repository. The repository contains the processed numerical datasets underlying the main-text and supplemental figures, the corresponding figure-generation scripts, cluster-expansion inputs and outputs, archived structure files, reconstructed adsorbate configurations, GCMC-generated POSCAR snapshots, GCMC simulation code, and spatial-analysis code. GCMC simulation parameters are specified in `GCMC/GCMC_SIMULATION.py`.

## Cluster-Expansion Features

`CE_FEATURES/` contains 26 PNG images of cluster features, named with Roman numerals from `i.png` through `xxvi.png`. These images document the cluster basis used for the cluster-expansion models.

## Cluster-Expansion Models and Archived Structures

`GSCs/` contains charge-state-specific cluster-expansion files and archived ATAT structures:

```text
GSCs/
|-- Negative/
|-- Neutral/
`-- Positive/
```

Each charge-state folder contains:

- `clusters.out`: cluster definitions.
- `energy.eci`: fitted effective cluster interactions.
- `energy.ecimult`: ECI values with cluster multiplicities.
- `ERGS_NEG.csv`, `ERGS_NEU.csv`, or `ERGS_POS.csv`: energy data for the corresponding charge state.
- `ref_energy.in`: reference-energy data used with the model.
- `Structures/str_*/str.out`: ATAT-format archived structures.

Current archived structure counts:

| Charge state | `Structures/str_*` count |
| --- | ---: |
| Negative | 146 |
| Neutral | 153 |
| Positive | 146 |

## GCMC Simulation Code

`GCMC/` contains:

```text
GCMC/
|-- GCMC_SIMULATION.py
|-- lat.in
`-- str_0/
    `-- str.out
```

`GCMC_SIMULATION.py` implements a GCMC workflow for CO*/H* configurations. It uses ATAT-style structure files, proposes insertion, deletion, and movement moves for O/CO-site and H adsorbates, calls `clusterpredict energy` to evaluate energies, applies Metropolis acceptance criteria, and can save accepted snapshots at a specified interval. The script records sampled energies and adsorbate counts in output text files when run in the intended ATAT environment.

Important constants and settings, including temperature, chemical potentials, snapshot interval, and snapshot output behavior, are defined near the top of `GCMC_SIMULATION.py`.

The clusters.out, ref_energy.in, energy.eci, and energy.ecimult files corresponding to the surface being simulated must be copied into the simulation directory before running the GCMC simulation at a specified reaction condition.

## GCMC Snapshot Archive

`SNAPSHOTS/` contains the GCMC-generated POSCAR configurations. It is organized by charge state:

```text
SNAPSHOTS/
|-- Negative/
|-- Neutral/
`-- Positive/
```

Each charge-state folder currently contains 75 condition directories and 7,500 POSCAR files. Condition directories follow the naming pattern:

```text
POSCARS_<temperature>_H_<H-condition>_CO_<CO-condition>
```

Examples:

```text
POSCARS_200_H_0_CO_100
POSCARS_200_H_N2_CO_10000
POSCARS_400_H_P4_CO_1000
```

Each condition directory contains numbered files such as `POSCAR_1` through `POSCAR_100`.

## Spatial Analysis

`SPATIAL_ANALYSIS/` contains:

```text
SPATIAL_ANALYSIS/
|-- SPATIAL_ANALYSIS.py
`-- POSCAR
```

`SPATIAL_ANALYSIS.py` reads POSCAR-like structures with ASE, wraps atoms in the periodic x-y plane, identifies O-H nearest-neighbor pairs using a KDTree-based cutoff, detects O/CO islands with DBSCAN, prints summary statistics, and saves a visualization image.

Basic usage:

```bash
cd SPATIAL_ANALYSIS
python SPATIAL_ANALYSIS.py POSCAR --output spatial_analysis.png
```

Optional command-line arguments include:

- `--pair-cutoff`: O-H x-y cutoff in Angstrom.
- `--island-cutoff`: DBSCAN epsilon in Angstrom.
- `--min-island-size`: minimum island size.
- `--output`: output image path.

## Reconstructed Structures

`RECON_SORTED/` contains reconstructed POSCAR structures named `POSCAR_str_*_D_X` plus `coverage_report.xlsx`, which stores the associated coverage information. The current folder contains 68 files.


## Software Requirements

The plotting scripts use standard scientific Python packages, primarily:

- Python 3
- `numpy`
- `matplotlib`

The spatial-analysis script additionally requires:

- `ase`
- `scipy`
- `scikit-learn`

The GCMC workflow expects an ATAT environment with `clusterpredict` available on the command line. The manuscript calculations also rely on VASP and ATAT for the DFT and cluster-expansion stages described in the study.

## File Formats

Principal file formats in this archive include:

- `.pkl`: processed Python datasets used directly by plotting scripts.
- `.py`: plotting, GCMC, and spatial-analysis scripts.
- `.png`: generated figures and cluster-feature images.
- `str.out`: ATAT structure files.
- `lat.in`: ATAT lattice input.
- `energy.eci`, `energy.ecimult`, `clusters.out`, `ref_energy.in`: cluster-expansion model files.
- `POSCAR`, `POSCAR_*`: VASP POSCAR-format structures.
- `.csv`: tabulated cluster-expansion energy data.
- `.xlsx`: coverage report spreadsheet.

## Gas-Phase Chemical Potentials

Gas-phase chemical potentials used in the GCMC simulations were calculated using the previously published methodology cited in the manuscript. Because the methodology and implementation are unchanged from that prior work, the chemical-potential calculation code is not duplicated here.

## License

Source code in this repository is licensed under the MIT License. See `LICENSE-CODE-MIT.txt`.

Datasets, figure files, structure files, documentation, and other non-code research materials are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See `LICENSE-DATA-CC-BY-4.0.txt`.

## Citation

If you use the data or code in this repository, please cite the associated manuscript:

> S. Hossain, V. R. Punyapu, P. Christopher and R. B. Getman, "Coverages and Configurations of CO* and H* on Pt(111) under Realistic Reaction Conditions."

The complete journal citation and DOI can be added after publication.

## Contact

For questions regarding the data or computational workflow, please contact the corresponding author listed in the associated manuscript.
