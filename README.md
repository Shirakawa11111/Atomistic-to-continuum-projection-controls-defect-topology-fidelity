# Atomistic-to-continuum projection controls defect-topology fidelity in phase-field-crystal modeling

Code and data accompanying the manuscript

> **Atomistic-to-continuum projection controls defect-topology fidelity in phase-field-crystal modeling**
> J. Bo, X.-W. Lei, T. Lu, T. Fujii.

Phase-field-crystal (PFC) and related continuum field models are routinely *seeded* from
atomistic configurations, yet the projection of discrete atoms onto a continuous density field
is non-unique. This repository contains everything needed to reproduce our benchmark showing
that this projection is **not a neutral pre-processing step but a control on the defect
structure the continuum model predicts**.

## Key results

- **Basin selection (the physics).** In a structural PFC that genuinely crystallizes graphene's
  honeycomb, the atoms-to-field projection selects which crystalline minimum the conserved dynamics
  relaxes into: a faithful projection recovers the correct 5–7 defect cores, while an unfaithful one
  falls into a distinct structure the dynamics cannot repair.
- **Three requirements for a faithful projection.** Sweeping each kernel's parameters shows a
  faithful projection must (i) be amplitude-normalized (model stability), (ii) keep spectral content
  up to the lattice's first reciprocal vector |G₁|, and (iii) place density at the atoms rather than
  on cell plateaus. A **localized Gaussian** kernel meets all three over a wide parameter band and is
  the safe default.
- **Two diagnosable failures.** A piecewise-constant **cell-indicator** kernel places density on
  Voronoi-cell plateaus and fails *intrinsically* (no width recovers the lattice). A **spectral**
  low-pass kernel fails only when its cutoff falls below |G₁| — which it does at the natural cutoff
  for the honeycomb but not for the coarser triangular lattice — recovering the correct basin only in
  a fragile high-cutoff resonance.
- **Numerically sound.** The structural-PFC relaxation has monotone free-energy descent, is grid- and
  time-step-converged, and the selected basin is stable under perturbation. The basin assignment is
  invariant to the topology detector (a bond-graph face detector that recovers the Stone–Wales core),
  the interatomic potential (AIREBO and Tersoff), and the structural-PFC parameters.

## Repository layout

```
src/route_a/        core library
  config.py           frozen reference parameters
  sc_ch_model.py      constrained Cahn-Hilliard / XPFC mesoscale testbed
  structural_pfc.py   peaked-C2 structural PFC (honeycomb, the linchpin model)
  triangular_pfc.py   single-mode structural PFC (triangular-lattice generality test)
  kernels.py          K1 spectral / K2 Gaussian / K3 Voronoi-Gaussian / K4 cell-indicator
  postprocess.py      P1 raw / P2 strict ratio / P3 rescale / P4 clip
  reference_density.py R1 MD-average / R2 perfect-crystal / R3 model-equilibrium
  cases.py            graphene defect-case generators (T0-T5)
  metrics.py          peak reconstruction, ring-statistics (shortest cycle), ring-L1 fidelity
  metrics_extra.py    independent Delaunay ring detector, coordination-F1, bond-angle, RMS
  nondim.py, lammps_io.py
scripts/            runners and figure generators (see "Reproducing", below)
experiments/        relaxed atomic structures (LAMMPS dumps) used as ground truth
outputs/            computed data tables and the publication figures
tests/              smoke / unit tests
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

A LAMMPS build with the `MANYBODY` package (AIREBO, Tersoff) and `lj/cut` is needed only to
*regenerate* the relaxed structures in `experiments/`; all downstream analysis runs from the
provided dumps with Python alone.

## Reproducing the figures

The relaxed atomic structures are included, so the continuum benchmark and every figure can be
reproduced without re-running any molecular dynamics:

```bash
export PYTHONPATH=$PWD/src

# main protocol matrix (constrained-Cahn-Hilliard testbed) -> outputs/production_v3/
python scripts/run_matrix_parallel.py

# structural-PFC linchpin + parameter scan -> outputs/structural_pfc{,_scan}/
python scripts/run_structural_pfc.py
python scripts/run_structural_pfc_scan.py

# triangular-lattice generality test -> outputs/triangular_pfc/
python scripts/run_triangular_pfc.py

# independent detectors, grid convergence, uncertainty quantification
python scripts/run_metrics_extra.py
python scripts/run_convergence.py
python scripts/uq_analysis.py
python scripts/uq_structural.py

# figures (F1-F9) -> outputs/figures/
python scripts/make_figures.py
python scripts/make_mechanism_figure.py
python scripts/make_structural_pfc_figure.py
python scripts/make_generality_figure.py
```

To regenerate the triangular Lennard-Jones cases from scratch (requires LAMMPS):

```bash
python scripts/make_triangular_cases.py        # writes LAMMPS inputs to experiments/triangular/
# then: for f in experiments/triangular/in.TR*; do lmp -in "$f"; done
```

## Data

- `experiments/hpc_package/relaxed_T*.dump` — AIREBO-relaxed graphene defect structures (T0–T5);
  `relaxed_tersoff_T*.dump` — the Tersoff cross-check.
- `experiments/triangular/relaxed_TR*.dump` — Lennard-Jones triangular-lattice cases.
- `outputs/production_v3/metrics.csv` — the canonical protocol × case × λ matrix.
- `outputs/{structural_pfc,structural_pfc_scan,triangular_pfc,uq,convergence_v2}/` — the
  structural-PFC, parameter-scan, generality, uncertainty, and convergence results.
- `outputs/figures/` — the figures in the manuscript.

## Citation

If you use this code or data, please cite the manuscript (above) and this repository.
A `CITATION.cff` and an archival DOI will be added on publication.

## License

Released under the MIT License (see `LICENSE`).
