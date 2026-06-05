#!/usr/bin/env python3
"""Numerical-soundness study for the structural-PFC relaxation (PRM advisor ask).

For TWO representative defects -- T2 (Stone--Wales 5-7-7-5) and T4 (low-angle
tilt GB) -- using the honeycomb StructuralPFCModel at the calibrated config
(c2_h1=1.05, c2_h2=0.6, chi=1.0, eta=0.25, lambda=0) under the K3/R2/P3 mapping,
this script establishes four properties and writes CSV + summary.json + an
energy_trajectory CSV (used by the figure):

  (1) FREE-ENERGY MONOTONICITY -- relax with the model and confirm the recorded
      ``energy_curve`` (from ``SCCHModel.relax``) decreases monotonically.
  (2) GRID CONVERGENCE -- relaxed ring-L1 at nx in {384,512,768}; max rel change.
  (3) TIME-STEP CONVERGENCE -- relaxed ring-L1 at dt in {5e-9,1e-8,2e-8}, each run
      to the SAME physical end time T_end = N_STEPS * dt_ref; max rel change.
  (4) BASIN STABILITY -- take the relaxed field, add small reproducible noise
      (numpy default_rng, fixed integer seed; amplitude 3% of the field max, in
      the 2-5% band), re-relax, and confirm the ring-L1 returns to essentially
      the same value (same basin).

Methodology note (matched evolution).  The conserved (Cahn--Hilliard) flow of the
structural PFC is still actively coarsening at the chunked stopping point -- the
recorded free energy keeps decreasing and the *atomic* ring-L1 oscillates by
O(0.02-0.03) as peaks micro-rearrange.  A per-chunk ``|dF|/|F| < tol`` early-stop
can therefore halt the reference and a perturbed copy at slightly different points
of the *same* trajectory and manufacture an apparent ring-L1 gap.  To compare
states fairly we evolve every run for a FIXED number of steps (early-stop
disabled via tol=0) so all comparisons -- across grid, dt, and the basin
perturbation -- are made at matched evolution.  Basin stability is then the
clean statement that a relaxed field and its noisy-perturbed re-relaxation, both
advanced the same number of steps, stay co-located (same ring-L1 to within the
trajectory's own oscillation, and near-identical free energy / field L2).

Follows the map+relax+ring_l1 pattern of scripts/run_structural_pfc.py.
Run with PYTHONPATH=$PWD/src from the project root.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from route_a.cases import get_case                                  # noqa: E402
from route_a.lammps_io import read_lammps_dump                       # noqa: E402
from route_a.nondim import Nondimensionalizer, setup_grid            # noqa: E402
from route_a.kernels import get_kernel                               # noqa: E402
from route_a.postprocess import get_postprocess                      # noqa: E402
from route_a.structural_pfc import StructuralPFCModel, honeycomb_k1  # noqa: E402
from route_a.metrics import reconstruct_peaks, ring_histogram, ring_l1  # noqa: E402
from route_a.config import A0_GRAPHENE_ANG                           # noqa: E402

# Calibrated structural-PFC config (matches run_structural_pfc.py CFG; eta=0.25 is
# the StructuralPFCModel default, lambda=0 via morphological_constraint_weight=0).
CFG = dict(c2_h1=1.05, c2_h2=0.6, chi=1.0, eta=0.25)
PROTO = ("K3", "R2", "P3")                                  # representative faithful mapping
DUMP_DIR = os.path.join(ROOT, "experiments", "hpc_package")
FMAP = {"T2": "relaxed_T2_stone_wales.dump", "T4": "relaxed_T4_low_angle_gb.dump"}
DEFECT_LABEL = {"T2": "T2 Stone-Wales", "T4": "T4 low-angle GB"}

# Fixed-step ("matched evolution") relaxation: early-stop disabled so every run
# advances exactly N_STEPS, recording the free energy every CHUNK steps.
N_STEPS = 2500
CHUNK = 250
MONO_REL_TOL = 1e-6          # tolerate per-step rises up to this fraction of |F| (round-off)
RL1_FLOOR = 0.02             # denominator floor for relative ring-L1 change (small but nonzero)
NOISE_FRAC = 0.03            # basin-perturbation amplitude (3% of field max; in the 2-5% band)
NOISE_SEED = 20260605        # fixed integer seed for the reproducible perturbation
# basin verdict: matched-step ring-L1 within the trajectory's own oscillation band
BASIN_ABS_TOL = 0.05         # absolute ring-L1 agreement (< the good/bad basin gap ~1)
BASIN_REL_TOL = 0.25         # OR relative agreement vs the reference value


def _prep(ck: str, nx: int) -> dict:
    """Map the relaxed atoms of case ``ck`` through K3/R2/P3 at grid resolution
    ``nx`` and return the seed field + everything needed to relax + score."""
    c = get_case(ck)
    atoms, _ = read_lammps_dump(os.path.join(DUMP_DIR, FMAP[ck]))
    nd = Nondimensionalizer((c.box[0], c.box[1]), (c.box[2], c.box[3]))
    a_nd = nd.nondimensionalize_coords(atoms)
    grid = setup_grid(nd.Lx_nd, nd.Ly_nd, nx)
    k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax)             # depends on box only, not nx
    nn = float(np.median(cKDTree(a_nd[:, :2]).query(a_nd[:, :2], k=2)[0][:, 1]))
    minsep = 0.7 * nn
    hgt = ring_histogram(a_nd[:, :2])                        # ground-truth ring histogram
    K, _, P = PROTO
    field = get_postprocess(P).apply(get_kernel(K).map(a_nd, grid))
    return dict(field=field, grid=grid, k1=k1, minsep=minsep, hgt=hgt)


def _ring_l1_of_field(field: np.ndarray, grid, minsep: float, hgt: dict) -> float:
    return ring_l1(ring_histogram(reconstruct_peaks(field, grid, minsep)), hgt)


def _make_model(field: np.ndarray, P: dict) -> StructuralPFCModel:
    grid = P["grid"]
    return StructuralPFCModel.for_field(
        field, grid.dx, P["k1"], x_bounds=grid.x_bounds, y_bounds=grid.y_bounds,
        morphological_constraint_weight=0.0, **CFG)


def _relax_fixed(field: np.ndarray, P: dict, dt: float, n_steps: int = N_STEPS) -> dict:
    """Advance ``field`` for exactly ``n_steps`` (early-stop disabled, tol=0) and
    return the model, the recorded free-energy curve, and the relaxed ring-L1."""
    m = _make_model(field, P)
    res = m.relax(dt=dt, tol=0.0, max_steps=n_steps, chunk=CHUNK)
    rl1 = _ring_l1_of_field(m.n, P["grid"], P["minsep"], P["hgt"])
    return dict(model=m, res=res, ring_l1=rl1, curve=list(res["energy_curve"]))


def _max_rel_change(values) -> float:
    """Max pairwise relative change across a small set of values, with a floor on
    the denominator so a (near-)zero reference does not blow the ratio up."""
    v = np.asarray(list(values), dtype=float)
    out = 0.0
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            denom = max(abs(v[i]), abs(v[j]), RL1_FLOOR)
            out = max(out, abs(v[i] - v[j]) / denom)
    return float(out)


def _abs_spread(values) -> float:
    v = np.asarray(list(values), dtype=float)
    return float(v.max() - v.min()) if v.size else 0.0


def _monotonicity(curve) -> dict:
    """Check the recorded free-energy curve is monotone non-increasing.

    Returns the worst per-step rise (absolute and relative to |F|) and a boolean
    tolerating only round-off-scale increases (<= MONO_REL_TOL * |F|).
    """
    c = np.asarray(curve, dtype=float)
    diffs = np.diff(c)                                       # F_{k+1} - F_k
    max_rise = float(diffs.max()) if diffs.size else 0.0     # > 0 means an increase occurred
    scale = max(np.abs(c).max(), 1e-300)
    return dict(
        n_points=int(c.size),
        F_init=float(c[0]) if c.size else float("nan"),
        F_final=float(c[-1]) if c.size else float("nan"),
        max_step_rise=max_rise,
        max_step_rise_rel=float(max_rise / scale),
        monotone=bool(max_rise / scale <= MONO_REL_TOL),
        curve=[float(x) for x in c],
    )


def run_case(ck: str, nx_list, dt_list, dt_ref) -> dict:
    """Run all four checks for a single defect case (matched evolution)."""
    t0 = time.time()
    rows = []                                                # flat per-run records for CSV

    # preps keyed by nx (the dt_ref reference grid nx=512 is reused everywhere)
    preps = {nx: _prep(ck, nx) for nx in sorted(set(nx_list) | {512})}
    Pref = preps[512]

    # ---- reference relaxation: nx=512, dt_ref, fixed N_STEPS ----
    base = _relax_fixed(Pref["field"], Pref, dt_ref)
    rl1_ref = base["ring_l1"]

    # (1) FREE-ENERGY MONOTONICITY (reference run's recorded energy_curve) ----
    mono = _monotonicity(base["curve"])

    # (2) GRID CONVERGENCE -- relaxed ring-L1 vs nx (reference dt, matched steps) ----
    grid_rl1 = {}
    for nx in nx_list:
        r = base if nx == 512 else _relax_fixed(preps[nx]["field"], preps[nx], dt_ref)
        grid_rl1[nx] = r["ring_l1"]
        rows.append(dict(case=ck, check="grid", nx=nx, dt=dt_ref, ring_l1=r["ring_l1"],
                         steps=r["res"]["steps"], F_init=r["res"]["F_init"],
                         F_final=r["res"]["F_final"], rel_drop=r["res"]["rel_drop"]))
    grid_max_rel = _max_rel_change(grid_rl1.values())

    # (3) TIME-STEP CONVERGENCE -- relaxed ring-L1 vs dt at a FIXED physical end
    #     time T_end = N_STEPS * dt_ref (smaller dt => more steps, same evolution),
    #     so this isolates the temporal-discretization error rather than total time.
    t_end = N_STEPS * dt_ref
    dt_rl1 = {}
    for dt in dt_list:
        n_dt = int(round(t_end / dt))
        r = base if dt == dt_ref else _relax_fixed(Pref["field"], Pref, dt, n_steps=n_dt)
        dt_rl1[dt] = r["ring_l1"]
        rows.append(dict(case=ck, check="dt", nx=512, dt=dt, ring_l1=r["ring_l1"],
                         steps=r["res"]["steps"], F_init=r["res"]["F_init"],
                         F_final=r["res"]["F_final"], rel_drop=r["res"]["rel_drop"]))
    dt_max_rel = _max_rel_change(dt_rl1.values())

    # (4) BASIN STABILITY -- take the relaxed field, add small fixed-seed noise,
    #     re-relax, and confirm the ring-L1 returns to essentially the same value.
    #     Because the conserved flow is still slowly coarsening at N_STEPS, we
    #     compare at MATCHED EVOLUTION: a clean copy of the relaxed field and the
    #     noisy copy are BOTH advanced the same N_STEPS from the relaxed baseline,
    #     so any residual difference is the basin response to the perturbation and
    #     not N steps of trajectory drift.  The genuine same-basin signals are the
    #     free-energy gap and the field-L2 difference between the two co-evolved
    #     states (both tiny if it is the same basin).
    relaxed_field = base["model"].n.copy()
    fmax = float(np.abs(relaxed_field).max())
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.normal(0.0, NOISE_FRAC * fmax, size=relaxed_field.shape)
    perturbed = relaxed_field + noise
    rl1_perturbed_init = _ring_l1_of_field(perturbed, Pref["grid"], Pref["minsep"], Pref["hgt"])
    # co-evolve the unperturbed reference and the perturbed copy by the same N_STEPS
    ref_cont = _relax_fixed(relaxed_field, Pref, dt_ref)        # reference at +N_STEPS
    rb = _relax_fixed(perturbed, Pref, dt_ref)                  # perturbed at +N_STEPS
    rl1_ref_matched = ref_cont["ring_l1"]
    rl1_basin = rb["ring_l1"]
    field_l2 = float(np.linalg.norm(rb["model"].n - ref_cont["model"].n)
                     / max(np.linalg.norm(ref_cont["model"].n), 1e-300))
    F_ref = ref_cont["res"]["F_final"]
    F_basin = rb["res"]["F_final"]
    F_rel_gap = abs(F_basin - F_ref) / max(abs(F_ref), 1e-300)
    basin_abs_change = abs(rl1_basin - rl1_ref_matched)
    basin_rel_change = basin_abs_change / max(abs(rl1_ref_matched), RL1_FLOOR)
    basin_stable = bool(basin_abs_change <= BASIN_ABS_TOL or basin_rel_change <= BASIN_REL_TOL)
    rows.append(dict(case=ck, check="basin", nx=512, dt=dt_ref, ring_l1=rl1_basin,
                     steps=rb["res"]["steps"], F_init=rb["res"]["F_init"],
                     F_final=rb["res"]["F_final"], rel_drop=rb["res"]["rel_drop"]))

    summary = dict(
        label=DEFECT_LABEL[ck],
        protocol="/".join(PROTO),
        n_steps=N_STEPS,
        ring_l1_reference=round(rl1_ref, 6),
        free_energy_monotonic=mono["monotone"],
        monotonicity=dict(n_points=mono["n_points"], F_init=mono["F_init"],
                          F_final=mono["F_final"], max_step_rise=mono["max_step_rise"],
                          max_step_rise_rel=mono["max_step_rise_rel"]),
        energy_curve=mono["curve"],
        grid=dict(nx=list(nx_list),
                  ring_l1={str(k): round(v, 6) for k, v in grid_rl1.items()},
                  abs_spread=round(_abs_spread(grid_rl1.values()), 6),
                  max_rel_change=round(grid_max_rel, 6)),
        dt=dict(dt=list(dt_list),
                ring_l1={f"{k:.0e}": round(v, 6) for k, v in dt_rl1.items()},
                abs_spread=round(_abs_spread(dt_rl1.values()), 6),
                max_rel_change=round(dt_max_rel, 6)),
        basin=dict(noise_frac=NOISE_FRAC, seed=NOISE_SEED, field_max=round(fmax, 6),
                   ring_l1_perturbed_before_rerelax=round(rl1_perturbed_init, 6),
                   ring_l1_reference_matched=round(rl1_ref_matched, 6),
                   ring_l1_after_rerelax=round(rl1_basin, 6),
                   abs_change_vs_reference=round(basin_abs_change, 6),
                   rel_change_vs_reference=round(basin_rel_change, 6),
                   field_l2_rel_vs_reference=round(field_l2, 6),
                   free_energy_rel_gap=round(F_rel_gap, 6),
                   basin_stable=basin_stable),
        wall_s=round(time.time() - t0, 1),
    )
    print(f"[{ck}] {DEFECT_LABEL[ck]}: mono={summary['free_energy_monotonic']} "
          f"grid(rl1 spread={summary['grid']['abs_spread']:.3f}, rel={grid_max_rel:.2f}) "
          f"dt(rl1 spread={summary['dt']['abs_spread']:.3f}, rel={dt_max_rel:.2f}) "
          f"basin {rl1_ref_matched:.4f}->{rl1_basin:.4f} (dRL1={basin_abs_change:.4f}, "
          f"dF/F={F_rel_gap:.2e}, dL2={field_l2:.2e}) stable={basin_stable} "
          f"[{summary['wall_s']}s]", flush=True)
    return dict(summary=summary, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="T2,T4")
    ap.add_argument("--nx-list", default="384,512,768")
    ap.add_argument("--dt-list", default="5e-9,1e-8,2e-8")
    ap.add_argument("--dt-ref", type=float, default=1e-8)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "structural_convergence"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = [c.strip() for c in args.cases.split(",")]
    nx_list = [int(x) for x in args.nx_list.split(",")]
    dt_list = [float(x) for x in args.dt_list.split(",")]

    print(f"structural convergence (matched evolution, {N_STEPS} steps): cases={cases} "
          f"nx={nx_list} dt={dt_list} dt_ref={args.dt_ref:.0e} config={CFG}", flush=True)
    t0 = time.time()

    results = {ck: run_case(ck, nx_list, dt_list, args.dt_ref) for ck in cases}

    # ---- flat CSV (one row per run) ----
    all_rows = [r for ck in cases for r in results[ck]["rows"]]
    keys = ["case", "check", "nx", "dt", "ring_l1", "steps", "F_init", "F_final", "rel_drop"]
    with open(os.path.join(args.out, "convergence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    # ---- energy trajectory CSV (the recorded monotone free-energy curves) ----
    with open(os.path.join(args.out, "energy_trajectory.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "checkpoint", "step", "free_energy"])
        for ck in cases:
            curve = results[ck]["summary"]["energy_curve"]
            for i, F in enumerate(curve):
                w.writerow([ck, i, i * CHUNK, F])

    # ---- summary.json (aggregate pass/fail across the two defects) ----
    per_case = {ck: results[ck]["summary"] for ck in cases}
    overall = dict(
        free_energy_monotonic=all(per_case[ck]["free_energy_monotonic"] for ck in cases),
        grid_max_rel_change=max(per_case[ck]["grid"]["max_rel_change"] for ck in cases),
        grid_max_abs_spread=max(per_case[ck]["grid"]["abs_spread"] for ck in cases),
        dt_max_rel_change=max(per_case[ck]["dt"]["max_rel_change"] for ck in cases),
        dt_max_abs_spread=max(per_case[ck]["dt"]["abs_spread"] for ck in cases),
        basin_stable=all(per_case[ck]["basin"]["basin_stable"] for ck in cases),
        basin_max_abs_change=max(per_case[ck]["basin"]["abs_change_vs_reference"] for ck in cases),
    )
    summary = dict(config=CFG, n_steps=N_STEPS, chunk=CHUNK, protocol="/".join(PROTO),
                   cases=cases, overall=overall, per_case=per_case)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nOVERALL: monotonic={overall['free_energy_monotonic']}  "
          f"grid(rel={overall['grid_max_rel_change']:.3f}, abs={overall['grid_max_abs_spread']:.3f})  "
          f"dt(rel={overall['dt_max_rel_change']:.3f}, abs={overall['dt_max_abs_spread']:.3f})  "
          f"basin_stable={overall['basin_stable']} (max dRL1={overall['basin_max_abs_change']:.3f})",
          flush=True)
    print(f"wrote {args.out}/convergence.csv + energy_trajectory.csv + summary.json "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
