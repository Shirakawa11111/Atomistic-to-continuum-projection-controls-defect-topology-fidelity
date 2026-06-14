#!/usr/bin/env python3
"""Asymptotic basin probe: do the GOOD (K3) and BAD (K1@200) seeds settle at
DISTINCT free-energy plateaus, or does the bad one slowly slide into the good
basin?  Same structural PFC and dt=1e-8 as the benchmark, but run to a real
plateau (per-chunk relative |dF| < 1e-7 sustained, or MAX_STEPS), recording F and
bond-graph ring-L1 coarsely.  Writes outputs/landscape/landscape_long.json.
"""
from __future__ import annotations
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from route_a.cases import get_case
from route_a.lammps_io import read_lammps_dump
from route_a.nondim import Nondimensionalizer, setup_grid
from route_a.kernels import get_kernel
from route_a.postprocess import get_postprocess
from route_a.structural_pfc import StructuralPFCModel, honeycomb_k1
from route_a.metrics import reconstruct_peaks
from route_a.metrics_extra import ring_histogram_bondgraph, ring_l1
from route_a.config import A0_GRAPHENE_ANG

CFG = dict(c2_h1=1.05, c2_h2=0.6, chi=1.0, eta=0.25)
DT, CHUNK, MAX_STEPS = 1e-8, 1500, 360000
PLATEAU_REL, PLATEAU_HITS = 1e-7, 4   # stop after 4 consecutive sub-threshold chunks
CASES = ["T2", "T4"]
FMAP = {"T2": "relaxed_T2_stone_wales.dump", "T4": "relaxed_T4_low_angle_gb.dump"}
KCFG = {"good_K3": ("K3", {}), "bad_K1@200": ("K1", dict(k_cut=200.0))}
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


def _l1(field, grid, minsep, gt):
    return ring_l1(ring_histogram_bondgraph(reconstruct_peaks(field, grid, minsep)), gt)


def _work(task):
    ck, kname = task
    P = _prep(ck)
    grid = P["grid"]
    kn, kw = KCFG[kname]
    field0 = get_postprocess("P3").apply(get_kernel(kn, **kw).map(P["a_nd"], grid))
    m = StructuralPFCModel.for_field(
        field0, grid.dx, P["k1"], x_bounds=grid.x_bounds, y_bounds=grid.y_bounds,
        morphological_constraint_weight=0.0, **CFG)
    steps, F, L1 = [0], [m.free_energy()], [_l1(m.n, grid, P["minsep"], P["gt"])]
    Fp, done, hits, plateaued = F[0], 0, 0, False
    while done < MAX_STEPS:
        this = min(CHUNK, MAX_STEPS - done)
        m.evolve(dt=DT, steps=this, scheme="semi-implicit", mobility=1.0)
        done += this
        Fn = m.free_energy()
        steps.append(done); F.append(Fn)
        L1.append(_l1(m.n, grid, P["minsep"], P["gt"]))
        rel = abs(Fn - Fp) / max(abs(Fp), 1e-300); Fp = Fn
        if not m.check_density_health():
            break
        hits = hits + 1 if rel < PLATEAU_REL else 0
        if hits >= PLATEAU_HITS:
            plateaued = True
            break
    return dict(case=P["name"], case_key=ck, proj=kname,
                steps=[int(s) for s in steps],
                free_energy=[float(x) for x in F], ring_l1=[float(x) for x in L1],
                F_final=float(F[-1]), ringL1_final=float(L1[-1]),
                steps_run=int(done), plateaued=bool(plateaued),
                healthy=bool(m.check_density_health()), n5_gt=int(P["gt"].get(5, 0)),
                n7_gt=int(P["gt"].get(7, 0)))


def main():
    out = os.path.join(ROOT, "outputs", "landscape"); os.makedirs(out, exist_ok=True)
    tasks = [(ck, k) for ck in CASES for k in KCFG]
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    t0 = time.time()
    with mp.Pool(len(tasks)) as pool:
        rows = pool.map(_work, tasks)
    data = {f"{r['case_key']}__{r['proj']}": r for r in rows}
    json.dump(data, open(os.path.join(out, "landscape_long.json"), "w"), indent=2)
    print(f"ran {len(rows)} long trajectories in {time.time()-t0:.0f}s\n")
    for ck in CASES:
        g, b = data[f"{ck}__good_K3"], data[f"{ck}__bad_K1@200"]
        print(f"{ck}: GOOD  steps={g['steps_run']:7d} plateau={g['plateaued']} "
              f"F={g['F_final']:+.5f} L1={g['ringL1_final']:.3f}")
        print(f"{ck}: BAD   steps={b['steps_run']:7d} plateau={b['plateaued']} "
              f"F={b['F_final']:+.5f} L1={b['ringL1_final']:.3f}")
        print(f"     -> dF(bad-good)={b['F_final']-g['F_final']:+.5f}   "
              f"distinct_basin={b['F_final']-g['F_final']>1e-3}\n")


if __name__ == "__main__":
    main()
