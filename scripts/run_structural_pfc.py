#!/usr/bin/env python3
"""Linchpin: run the mapping protocols through a properly-scaled STRUCTURAL PFC
(peak C2 at the honeycomb reciprocal lattice) and record mapping fidelity at
initialisation AND after relaxation. Tests whether the K2/K3 >> K1 ranking
survives in a model that actually crystallises the lattice (unlike the SC-CH
testbed). Parallel over (case x protocol x lambda)."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time, multiprocessing as mp
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

CFG = dict(c2_h1=1.05, c2_h2=0.6, chi=1.0)   # calibrated: perfect honeycomb stable at lambda=0
_PREP = {}


def _prep(case_keys, relaxed_dir):
    fmap = {"T0": "relaxed_T0_pristine.dump", "T1": "relaxed_T1_vacancy.dump",
            "T2": "relaxed_T2_stone_wales.dump", "T3": "relaxed_T3_dislocation_dipole.dump",
            "T4": "relaxed_T4_low_angle_gb.dump", "T5": "relaxed_T5_high_angle_gb.dump"}
    out = {}
    for ck in case_keys:
        c = get_case(ck)
        atoms, _ = read_lammps_dump(os.path.join(relaxed_dir, fmap[ck]))
        nd = Nondimensionalizer((c.box[0], c.box[1]), (c.box[2], c.box[3]))
        a_nd = nd.nondimensionalize_coords(atoms)
        grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
        k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax)
        nn = float(np.median(cKDTree(a_nd[:, :2]).query(a_nd[:, :2], k=2)[0][:, 1]))
        hgt = ring_histogram(a_nd[:, :2])
        out[ck] = dict(name=c.name, a_nd=a_nd, grid=grid, k1=k1, minsep=0.7 * nn, hgt=hgt)
    return out


def _rh(f, grid, minsep):
    return ring_histogram(reconstruct_peaks(f, grid, minsep))


def _work(task):
    ck, proto, lam = task
    P = _PREP[ck]; grid = P["grid"]
    K, R, Pp = proto.split("/")
    field = get_postprocess(Pp).apply(get_kernel(K).map(P["a_nd"], grid))
    hi = _rh(field, grid, P["minsep"])
    m = StructuralPFCModel.for_field(field, grid.dx, P["k1"], x_bounds=grid.x_bounds,
                                     y_bounds=grid.y_bounds, morphological_constraint_weight=lam, **CFG)
    res = m.relax(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)
    hr = _rh(m.n, grid, P["minsep"])
    return dict(case=P["name"], protocol=proto, lam=lam, k1=P["k1"],
                n5_gt=P["hgt"].get(5, 0), n7_gt=P["hgt"].get(7, 0),
                ringL1_init=ring_l1(hi, P["hgt"]), ringL1_relaxed=ring_l1(hr, P["hgt"]),
                n5_init=hi.get(5, 0), n7_init=hi.get(7, 0),
                n5_relaxed=hr.get(5, 0), n7_relaxed=hr.get(7, 0),
                healthy=bool(m.check_density_health()), nmax=float(np.abs(m.n).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncores", type=int, default=60)
    ap.add_argument("--relaxed-dir", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "structural_pfc"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cases = ["T0", "T1", "T2", "T3", "T4", "T5"]
    protos = ["K1/R2/P3", "K2/R2/P3", "K3/R2/P3", "K3/R2/P4", "K4/R2/P3"]
    lams = [0.0, 0.01]
    global _PREP
    _PREP = _prep(cases, args.relaxed_dir)
    tasks = [(ck, p, lam) for ck in cases for p in protos for lam in lams]
    print(f"structural-PFC tasks={len(tasks)} ncores={args.ncores}", flush=True)
    t0 = time.time()
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    with mp.Pool(args.ncores) as pool:
        rows = pool.map(_work, tasks)
    import csv
    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "structural_pfc.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # ranking check at lambda=0: per case, is K1 the worst by relaxed ringL1?
    summary = {"n_rows": len(rows), "per_case": {}}
    for ck in cases:
        for lam in lams:
            sub = [r for r in rows if r["case"].startswith(ck) and r["lam"] == lam]
            sub.sort(key=lambda r: r["ringL1_relaxed"])
            if sub:
                worst = sub[-1]["protocol"]; best = sub[0]["protocol"]
                summary["per_case"][f"{ck}_lam{lam}"] = dict(
                    best=best, best_rl1=round(sub[0]["ringL1_relaxed"], 3),
                    worst=worst, worst_rl1=round(sub[-1]["ringL1_relaxed"], 3),
                    K1_is_worst=("K1" in worst))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    nworst = sum(1 for v in summary["per_case"].values() if v["K1_is_worst"])
    print(f"ran {len(rows)} rows in {time.time()-t0:.0f}s; K1 is worst in "
          f"{nworst}/{len(summary['per_case'])} (case,lambda) cells", flush=True)
    print(f"wrote {args.out}/structural_pfc.csv + summary.json", flush=True)


if __name__ == "__main__":
    main()
