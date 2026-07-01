# Proteinoid Spike Systems

Reference implementation for:

> Sharma, S., Mahmud, A., Tarabella, G., Mougoyannis, P., Adamatzky, A.
> **Graph-based Complexity and Computational Capabilities of Proteinoid Spike Systems.**
> arXiv:2504.10362v2 (2025).

This repository reproduces every figure and table in the paper from the raw
voltage-sensitive-dye (VSD) recordings of proteinoid spike trains.

## What's in here

The pipeline maps directly onto the paper:

| Paper section | Module | Script | Output |
| --- | --- | --- | --- |
| § II.A | `data_loading` | — | (loader used by all scripts) |
| § III, Fig. 3 | — | `plot_raw_spikes.py` | Fig. 3 |
| § III, Fig. 4 | `transformations` | `transform_and_plot.py` | Fig. 4 |
| § IV, Table III, Fig. 5 | `complexity` | `compute_complexity.py` | Table III, Fig. 5 |
| § V, Tables IV–V, Figs. 7–9 | `features`, `classifier` | `train_classifier.py` | Tables IV–V, Figs. 7, 8, 9 |

## Repository layout

```
.
├── data/raw/meta_data.csv     Raw VSD spike trains (5 datasets)
├── src/proteinoid_spikes/     Pipeline modules
├── scripts/                   One script per paper figure/table
├── notebooks/                 Original Jupyter notebooks (historical record)
├── results/figures/           Generated figures (created at runtime)
├── results/tables/            Generated tables (created at runtime)
└── docs/                      Supplementary documentation
```

## Installation

Requires Python 3.9, 3.10, 3.11, or 3.12 (TensorFlow does not yet support 3.13+).

Clone the repository:

```bash
git clone https://github.com/Adnan1729/Protocognitive_Agents
cd Protocognitive_Agents
```

### Quick setup (recommended)

From the repo root, run the setup script for your shell. It creates a
virtual environment in `.venv/` and installs the package in editable mode.

**Mac / Linux / Git Bash on Windows:**
```bash
bash setup.sh
source .venv/bin/activate        # Mac / Linux
source .venv/Scripts/activate    # Git Bash on Windows
```

**Windows PowerShell:**
```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

### Manual setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # or .venv/Scripts/activate on Windows
pip install --upgrade pip
pip install -e .
```

## Reproducing the paper

Run the full pipeline:

```bash
python scripts/reproduce_all.py
```

or run individual stages:

```bash
python scripts/plot_raw_spikes.py        # Fig. 3
python scripts/transform_and_plot.py     # Fig. 4
python scripts/compute_complexity.py     # Table III, Fig. 5
python scripts/train_classifier.py       # Tables IV-V, Figs. 7-9
```

All figures land in `results/figures/`, all tables in `results/tables/`.

## On exact reproduction of paper numbers

The numbers reported in the paper (accuracy 70.41 %, precision 70.59 %,
recall 55.81 %, F1 0.6234, AUC 0.73) came from an unseeded run of the
original Jupyter notebooks. The refactored pipeline pins seeds for
`numpy`, `tensorflow`, Python's `random`, and the sklearn validation
split, so re-running is deterministic going forward — but the resulting
numbers will differ slightly from the published ones for that reason.

The complexity metrics in Table III and the F1/F2 transformation are
fully deterministic and reproduce the published values exactly.

If you want to reproduce a different run, change `RANDOM_SEED` in
`src/proteinoid_spikes/config.py`.

### A note on the F1 sampling range

Equation 1 of the paper writes `t ∈ {0, 2, 3, ..., 19, 20}`. The original
notebook (and therefore this code) uses `t ∈ {1, 2, ..., 20}`. We
preserved the notebook's choice because that is what produced the
published figures and tables.

## Original notebooks

The two Jupyter notebooks in `notebooks/` are the historical artefacts
that produced the published numbers. They are kept verbatim as a record
of provenance. New work should use the modular `src/` code.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{sharma2025proteinoid,
  title   = {Graph-based Complexity and Computational Capabilities of Proteinoid Spike Systems},
  author  = {Sharma, Saksham and Mahmud, Adnan and Tarabella, Giuseppe and Mougoyannis, Panagiotis and Adamatzky, Andrew},
  journal = {arXiv preprint arXiv:2504.10362},
  year    = {2025}
}
```

A `CITATION.cff` file is also provided for GitHub's "Cite this repository"
feature.

## License

MIT. See `LICENSE`.