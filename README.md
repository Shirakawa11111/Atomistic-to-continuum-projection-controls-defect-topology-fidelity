# Atomistic-to-continuum projection controls defect-topology fidelity in phase-field-crystal modeling

Code and data accompanying the manuscript

> **Atoms-to-field projection selects the practical-time crystallization basin in graphene phase-field-crystal modeling**
> J. Bo, X.-W. Lei, T. Lu, T. Fujii.

Phase-field-crystal (PFC) and related continuum field models are routinely *seeded* from
atomistic configurations, yet the projection of discrete atoms onto a continuous density field
is non-unique. This repository contains everything needed to reproduce our benchmark showing
that this projection is **not a neutral pre-processing step but a control on the defect
structure the continuum model predicts**.

## Key results

- **Crystallization-basin selection (the physics).** In a structural PFC that genuinely crystallizes
  graphene's honeycomb, the atoms-to-field projection selects which basin the conserved dynamics
  *begins* in. A faithful projection begins in the correct **crystalline basin** (a clean honeycomb;
  ring-L1 0.01–0.08 vs the atomistic ground truth); an unfaithful one is trapped in a **long-lived
  metastable disordered state** (ring-L1 0.7–2.7 at practical budgets). This is the dynamical
  amplification of a fidelity difference already present at the seed: the faithful *initial*
  reconstruction reproduces the exact atomic topology, including the Stone–Wales 5-7-7-5 core (one
  peak per atom), whereas the unfaithful ones cannot even resolve the atoms (~300–500 peaks for 1024
  atoms).
- **It is a long-lived metastable state, not a permanent basin (stated honestly).** Following the
  relaxation to 3.6×10⁵ steps (≫ the 2.5×10³ benchmark) shows the disordered tier is a robust,
  long-lived metastable incubation state — the free energy is stationary to ~10⁻⁴ per recorded
  1500-step chunk over a ~10⁴-step **incubation**, and ±10% perturbations decay and re-incubate
  (nucleating at the same step) while the crystalline basin is stable — that eventually **nucleates**
  into the crystalline minimum, reaching Δ*F*≤0.02 and a low-ring-L1 topology comparable to the
  faithful seed (~0.03–0.14). Its lifetime exceeds practical relaxation budgets by 1–2 orders of
  magnitude, so the projection controls the structure predicted at any practical relaxation; given
  unbounded relaxation both seeds crystallize. Under relaxation the metastable atomic-scale defect
  cores anneal toward the surrounding crystal, so "correct (crystalline) basin" means the field stays
  crystalline — not that the specific 5-7-7-5 core is frozen in.
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

# robustness checks (answering the advisor review)
python scripts/run_hyperparam_scan.py         # is each kernel failure intrinsic, or a bad knob? -> outputs/hyperparam_scan/
python scripts/run_detector_validation.py     # ring-detector audit; also writes figure F11 -> outputs/detector_validation/
python scripts/run_structural_convergence.py  # free-energy descent, grid/dt convergence, basin stability -> outputs/structural_convergence/
python scripts/run_fair_k1_basin.py           # does a fairly-tuned spectral kernel reach the correct basin? -> outputs/fair_k1_basin/

# strengthening round: the basin is a metastable basin (incubation -> nucleation)
python scripts/run_detector_sensitivity.py    # basin invariant to bond-graph cutoff r_bond in {1.3,1.4,1.5} -> outputs/detector_sensitivity/
python scripts/run_mixed_corners.py           # joint (mixed-corner) parameter robustness; 4 corners hold -> outputs/mixed_corners/
python scripts/run_landscape_long.py          # asymptotic probe to 3.6e5 steps: incubation -> nucleation -> outputs/landscape/landscape_long.json
python scripts/run_metastability.py           # +/-10% perturbation test: the disordered tier is a long-lived metastable incubation state -> outputs/landscape/metastability.json

# figures (F1-F12 + F_basin + F_basinvis + F_landscape) -> outputs/figures/
python scripts/make_basinvis_figure.py               # F_basinvis (relaxed crystalline-vs-disordered field)
python scripts/make_landscape_figure.py              # F_landscape (run_landscape_long first; incubation -> nucleation)
python scripts/make_figures.py                       # F1-F6
python scripts/make_mechanism_figure.py              # F7
python scripts/make_structural_pfc_figure.py         # F8
python scripts/make_generality_figure.py             # F9
python scripts/make_hyperparam_figure.py             # F10     (run_hyperparam_scan first)
python scripts/make_structural_convergence_figure.py # F12     (run_structural_convergence first)
python scripts/make_basin_figure.py                  # F_basin (run_fair_k1_basin first); F11 is written by run_detector_validation
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
- `outputs/{hyperparam_scan,detector_validation,structural_convergence,fair_k1_basin}/` — the
  advisor-review robustness checks: kernel-family hyperparameter sweep, ring-detector audit,
  numerical-soundness study, and the fairly-tuned spectral-kernel basin test.
- `outputs/detector_sensitivity/` — basin assignment vs the bond-graph cutoff r_bond ∈ {1.3,1.4,1.5}.
- `outputs/mixed_corners/` — joint (mixed-corner) structural-PFC parameter robustness (4 corners).
- `outputs/landscape/` — asymptotic free-energy/topology trajectories (`landscape_long.json`, to
  3.6×10⁵ steps) and the perturbation/metastability test (`metastability.json`).
- `outputs/figures/` — the figures in the manuscript (F1–F12, F_basin, F_basinvis, F_landscape).

## Citation

If you use this code or data, please cite the manuscript (above) and this repository.
A `CITATION.cff` and an archival DOI will be added on publication.

## License

Released under the MIT License (see `LICENSE`).
