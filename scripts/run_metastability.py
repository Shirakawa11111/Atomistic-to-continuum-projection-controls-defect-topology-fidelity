#!/usr/bin/env python3
"""Metastability test: is the disordered state the unfaithful projection relaxes
into a GENUINE (metastable) basin, or just a point on a slow downhill path?

Each branch deterministically relaxes the SAME seed to an incubation-window time
t*, then perturbs the live field in place (additive zero-mean noise, mass
conserved) and continues. If a finite perturbation does NOT immediately trigger
nucleation -- the field re-incubates on the flat free-energy plateau before later
nucleating -- the disordered state is a true metastable basin (a local attractor),
not a saddle being passed through. As a control, the GOOD (crystalline) state is
perturbed the same way and must stay crystalline (a stable basin).

Writes outputs/landscape/metastability.json.
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
DT, CHUNK = 1e-8, 1500
FMAP = {"T2": "relaxed_T2_stone_wales.dump", "T4": "relaxed_T4_low_angle_gb.dump"}
KCFG = {"good_K3": ("K3", {}), "bad_K1@200": ("K1", dict(k_cut=200.0))}
RELAXED_DIR = os.path.join(ROOT, "experiments", "hpc_package")
NUCLEATED = 0.2  # ring-L1 below this = crystalline basin

# (case, proj, t_star, t_end) -- t_star sits in the incubation plateau
BRANCHES = []
for eps in (0.0, 0.02, 0.05, 0.10):
    BRANCHES.append(("T4", "bad_K1@200", 12000, 66000, eps, 11))
for eps in (0.0, 0.05):
    BRANCHES.append(("T4", "good_K3", 12000, 66000, eps, 11))
for eps in (0.0, 0.05, 0.10):
    BRANCHES.append(("T2", "bad_K1@200", 7500, 42000, eps, 11))
for eps in (0.0, 0.05):
    BRANCHES.append(("T2", "good_K3", 7500, 42000, eps, 11))


def _prep(ck):
    c = get_case(ck)
    atoms, _ = read_lammps_dump(os.path.join(RELAXED_DIR, FMAP[ck]))
    nd = Nondimensionalizer((c.box[0], c.box[1]), (c.box[2], c.box[3]))
    a_nd = nd.nondimensionalize_coords(atoms)
    grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
    k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax)
    nn = float(np.median(cKDTree(a_nd[:, :2]).query(a_nd[:, :2], k=2)[0][:, 1]))
    gt = ring_histogram_bondgraph(a_nd[:, :2])
    return dict(a_nd=a_nd, grid=grid, k1=k1, minsep=0.7 * nn, gt=gt)


def _l1(field, grid, minsep, gt):
    return ring_l1(ring_histogram_bondgraph(reconstruct_peaks(field, grid, minsep)), gt)


def _evolve_record(m, P, start, stop):
    steps, F, L1 = [], [], []
    done = start
    while done < stop:
        this = min(CHUNK, stop - done)
        m.evolve(dt=DT, steps=this, scheme="semi-implicit", mobility=1.0)
        done += this
        steps.append(done); F.append(float(m.free_energy()))
        L1.append(float(_l1(m.n, P["grid"], P["minsep"], P["gt"])))
        if not m.check_density_health():
            break
    return steps, F, L1


def _work(task):
    ck, proj, t_star, t_end, eps, seed = task
    P = _prep(ck)
    grid = P["grid"]
    kn, kw = KCFG[proj]
    field0 = get_postprocess("P3").apply(get_kernel(kn, **kw).map(P["a_nd"], grid))
    m = StructuralPFCModel.for_field(
        field0, grid.dx, P["k1"], x_bounds=grid.x_bounds, y_bounds=grid.y_bounds,
        morphological_constraint_weight=0.0, **CFG)
    # deterministic relax to the incubation point
    m.evolve(dt=DT, steps=t_star, scheme="semi-implicit", mobility=1.0)
    F_star = float(m.free_energy())
    L1_star = float(_l1(m.n, grid, P["minsep"], P["gt"]))
    # perturb in place, mass-conserving
    if eps > 0.0:
        rng = np.random.default_rng(seed)
        mean0 = float(m.n.mean())
        noise = rng.standard_normal(m.n.shape) * (eps * float(m.n.std()))
        m.n = m.n + noise
        m.n = m.n - (float(m.n.mean()) - mean0)
    F_pert = float(m.free_energy())
    L1_pert = float(_l1(m.n, grid, P["minsep"], P["gt"]))
    # continue, record
    steps, F, L1 = _evolve_record(m, P, t_star, t_end)
    L1 = np.asarray(L1); steps_a = np.asarray(steps)
    below = np.where(L1 < NUCLEATED)[0]
    nuc_step = int(steps_a[below[0]]) if len(below) else None
    return dict(case=ck, proj=proj, eps=eps, t_star=t_star,
                F_star=F_star, L1_star=L1_star, F_pert=F_pert, L1_pert=L1_pert,
                steps=steps, free_energy=F, ring_l1=[float(x) for x in L1],
                nucleation_step=nuc_step, L1_final=float(L1[-1]),
                F_final=float(F[-1]))


def main():
    out = os.path.join(ROOT, "outputs", "landscape"); os.makedirs(out, exist_ok=True)
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    t0 = time.time()
    with mp.Pool(min(len(BRANCHES), os.cpu_count() or 4)) as pool:
        rows = pool.map(_work, BRANCHES)
    data = {f"{r['case']}__{r['proj']}__eps{r['eps']:.2f}": r for r in rows}
    json.dump(data, open(os.path.join(out, "metastability.json"), "w"), indent=2)
    print(f"ran {len(rows)} branches in {time.time()-t0:.0f}s\n")
    hdr = f"{'case':4s} {'proj':11s} {'eps':>5s} | {'L1*':>6s} {'L1_pert':>8s} {'nuc_step':>9s} {'L1_fin':>7s}"
    print(hdr); print("-"*len(hdr))
    for r in rows:
        ns = r["nucleation_step"]
        print(f"{r['case']:4s} {r['proj']:11s} {r['eps']:5.2f} | {r['L1_star']:6.3f} "
              f"{r['L1_pert']:8.3f} {str(ns):>9s} {r['L1_final']:7.3f}")
    print("\nInterpretation:")
    print("  bad eps=0 nucleates late (incubation) -> disordered state is long-lived")
    print("  bad eps>0 does NOT nucleate immediately (re-incubates) -> genuine metastable basin")
    print("  good eps>0 stays crystalline (L1 low) -> crystalline basin is stable")


if __name__ == "__main__":
    main()
