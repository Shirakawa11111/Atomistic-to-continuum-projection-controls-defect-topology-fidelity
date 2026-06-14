#!/usr/bin/env python3
"""EXPERIMENT 2 (advisor I.2) -- free-energy landscape / basin evidence.

Question: when a GOOD projection (K3 Voronoi-Gaussian) and a BAD one
(K1 spectral at the naive cutoff k_cut=200 < |G1|) seed the *same* structural-PFC
relaxation, do they both descend monotonically to a STABLE free-energy plateau
(i.e. converged, not numerically stuck) yet settle at DISTINCT final free
energies and DISTINCT final topologies -- the signature of two different local
minima of the same energy functional?

For each defect case (T2 Stone-Wales, T4 low-angle GB) and each projection
(good=K3, bad=K1@200) we:
  (i)  evolve the structural PFC in fixed chunks (dt=1e-8), recording the free
       energy F(step) at every chunk -> the descent trajectory; and
  (ii) at the same chunk boundaries reconstruct atoms (local-maxima peaks) and
       compute the bond-graph ring-L1 vs the AIREBO ground truth -> topology(step).

Convergence uses the SAME rule as SCCHModel.relax: stop when the per-chunk
relative free-energy drop |dF|/|F| < tol (=1e-6) or max_steps (=2500) is reached.

Structural PFC (numerically sound, no morphological pin):
  c2_h1=1.05, c2_h2=0.6, chi=1.0, eta=0.25, lambda=0 (morph weight 0).
Mapping: P3.apply( get_kernel(KN, **kw).map(a_nd, grid) ); k1 = honeycomb_k1.
minsep = 0.7 * median_NN. Relaxation: tol=1e-6, max_steps=2500, chunk=300, dt=1e-8.

Output: outputs/landscape/landscape.json  (+ per-(case,proj) trajectories).
Figure: scripts/make_landscape_figure.py -> outputs/figures/F_landscape.png.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
import multiprocessing as mp

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
from route_a.metrics import reconstruct_peaks                        # noqa: E402
from route_a.metrics_extra import ring_histogram_bondgraph, ring_l1  # noqa: E402
from route_a.config import A0_GRAPHENE_ANG                           # noqa: E402

# structural-PFC config (numerically sound; lambda=0 via morph weight 0)
CFG = dict(c2_h1=1.05, c2_h2=0.6, chi=1.0, eta=0.25)
# relaxation schedule.  dt=1e-8 and chunk=300 follow the basin template
# (run_fair_k1_basin.py), but the basin run stops at max_steps=2500 -- which at
# this tiny dt is only ~9 chunks and leaves F STILL descending (no plateau).  To
# furnish the convergence evidence this experiment demands, we let the SAME
# tol=1e-6 per-chunk criterion actually trigger by extending the step budget; the
# 2500-step basin endpoint is then just an early snapshot ON these trajectories.
TOL, MAX_STEPS, CHUNK, DT = 1e-6, 15000, 300, 1e-8
BASIN_SNAPSHOT_STEPS = 2500  # the basin paper's reported endpoint, marked on the figure
# the two defect cases under test
CASES = ["T2", "T4"]
FMAP = {"T2": "relaxed_T2_stone_wales.dump", "T4": "relaxed_T4_low_angle_gb.dump"}
# the two projections: GOOD vs BAD
KCFG = {
    "good_K3":     ("K3", {}),                # Voronoi-Gaussian, resolves lattice
    "bad_K1@200":  ("K1", dict(k_cut=200.0)),  # spectral, naive cutoff < |G1| (~325)
}
KLABEL = {"good_K3": "good (K3)", "bad_K1@200": "bad (K1, k_cut=200)"}
RELAXED_DIR = os.path.join(ROOT, "experiments", "hpc_package")


def _prep(ck):
    c = get_case(ck)
    atoms, _ = read_lammps_dump(os.path.join(RELAXED_DIR, FMAP[ck]))
    nd = Nondimensionalizer((c.box[0], c.box[1]), (c.box[2], c.box[3]))
    a_nd = nd.nondimensionalize_coords(atoms)
    grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
    k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax)
    nn = float(np.median(cKDTree(a_nd[:, :2]).query(a_nd[:, :2], k=2)[0][:, 1]))
    gt = ring_histogram_bondgraph(a_nd[:, :2])
    return dict(name=c.name, a_nd=a_nd, grid=grid, k1=k1, minsep=0.7 * nn, gt=gt)


def _ring_l1_of_field(field, grid, minsep, gt):
    pk = reconstruct_peaks(field, grid, minsep)
    return ring_l1(ring_histogram_bondgraph(pk), gt)


def _work(task):
    ck, kname = task
    P = _prep(ck)
    grid = P["grid"]
    kn, kw = KCFG[kname]
    # map: kernel -> P3 rescale -> initial field
    field0 = get_postprocess("P3").apply(get_kernel(kn, **kw).map(P["a_nd"], grid))

    m = StructuralPFCModel.for_field(
        field0, grid.dx, P["k1"],
        x_bounds=grid.x_bounds, y_bounds=grid.y_bounds,
        morphological_constraint_weight=0.0, **CFG,
    )

    # record initial state (step 0)
    steps = [0]
    F = [m.free_energy()]
    L1 = [_ring_l1_of_field(m.n, grid, P["minsep"], P["gt"])]

    F_prev = F[0]
    done = 0
    converged = False
    snap_F = snap_L1 = None  # values at the 2500-step basin snapshot
    while done < MAX_STEPS:
        this = min(CHUNK, MAX_STEPS - done)
        m.evolve(dt=DT, steps=this, scheme="semi-implicit", mobility=1.0)
        done += this
        F_now = m.free_energy()
        l1_now = _ring_l1_of_field(m.n, grid, P["minsep"], P["gt"])
        steps.append(done)
        F.append(F_now)
        L1.append(l1_now)
        if snap_F is None and done >= BASIN_SNAPSHOT_STEPS:
            snap_F, snap_L1 = float(F_now), float(l1_now)
        rel = abs(F_now - F_prev) / max(abs(F_prev), 1e-300)
        F_prev = F_now
        if not m.check_density_health():
            break
        if rel < TOL:
            converged = True
            break

    F = np.asarray(F, float)
    L1 = np.asarray(L1, float)
    # monotone descent check: every recorded chunk must not increase F (allow tiny
    # numerical slack relative to the total drop).
    dF = np.diff(F)
    total_drop = float(F[0] - F[-1])
    slack = 1e-9 * max(abs(F[0]), 1e-30)
    n_up = int(np.sum(dF > slack))
    monotone = bool(n_up == 0)
    # plateau check: final relative slope small (stable, not still sliding)
    final_rel = abs(F[-1] - F[-2]) / max(abs(F[-2]), 1e-300) if len(F) > 1 else 0.0

    return dict(
        case=P["name"], case_key=ck, proj=kname, proj_label=KLABEL[kname],
        steps=[int(s) for s in steps],
        free_energy=[float(x) for x in F],
        ring_l1=[float(x) for x in L1],
        F_init=float(F[0]), F_final=float(F[-1]),
        ringL1_init=float(L1[0]), ringL1_final=float(L1[-1]),
        n5_gt=int(P["gt"].get(5, 0)), n7_gt=int(P["gt"].get(7, 0)),
        converged=bool(converged), steps_run=int(done),
        monotone=monotone, n_increasing_chunks=n_up, total_drop=total_drop,
        final_rel_drop=float(final_rel),
        healthy=bool(m.check_density_health()),
        nmax=float(np.abs(m.n).max()),
    )


def main():
    out = os.path.join(ROOT, "outputs", "landscape")
    os.makedirs(out, exist_ok=True)

    tasks = [(ck, k) for ck in CASES for k in KCFG]
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    t0 = time.time()
    with mp.Pool(min(len(tasks), os.cpu_count() or 4)) as pool:
        rows = pool.map(_work, tasks)
    dt_run = time.time() - t0

    # index by (case_key, proj) for easy figure consumption
    data = {f"{r['case_key']}__{r['proj']}": r for r in rows}
    json.dump(data, open(os.path.join(out, "landscape.json"), "w"), indent=2)

    # console report
    print(f"\nran {len(rows)} relaxations in {dt_run:.0f}s\n", flush=True)
    print(f"{'case':22s} {'proj':16s} | {'steps':>6s} {'conv':>5s} {'mono':>5s} "
          f"{'F_init':>11s} {'F_final':>11s} | {'L1_init':>7s} {'L1_final':>8s}")
    for ck in CASES:
        for k in KCFG:
            r = data[f"{ck}__{k}"]
            print(f"{r['case']:22s} {k:16s} | {r['steps_run']:6d} "
                  f"{str(r['converged']):>5s} {str(r['monotone']):>5s} "
                  f"{r['F_init']:11.4e} {r['F_final']:11.4e} | "
                  f"{r['ringL1_init']:7.3f} {r['ringL1_final']:8.3f}")

    print("\n-- per-case good-vs-bad final free energy / topology --")
    for ck in CASES:
        g = data[f"{ck}__good_K3"]
        b = data[f"{ck}__bad_K1@200"]
        dF = b["F_final"] - g["F_final"]
        print(f"{g['case']:22s}  F_final good={g['F_final']:.4e}  bad={b['F_final']:.4e}  "
              f"(bad-good={dF:+.4e})  |  ringL1 good={g['ringL1_final']:.3f} "
              f"bad={b['ringL1_final']:.3f}")

    all_mono = all(data[f"{ck}__{k}"]["monotone"] for ck in CASES for k in KCFG)
    print(f"\nfe_monotone (all trajectories): {all_mono}")
    print(f"wrote {os.path.join(out, 'landscape.json')}", flush=True)


if __name__ == "__main__":
    main()
