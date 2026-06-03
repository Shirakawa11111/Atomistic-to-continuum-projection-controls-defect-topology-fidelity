#!/usr/bin/env python3
"""Non-honeycomb generality test: run the mapping protocols through a single-mode
TRIANGULAR structural PFC and measure defect fidelity by the COORDINATION-number
histogram (perfect triangular = all 6-fold; defects introduce 5-/7-fold), the
triangular-lattice analog of the honeycomb ring-L1. Tests whether the two-tier
ranking (Voronoi-Gaussian {K2,K3} << non-Gaussian {K1,K4}) transfers to a
different lattice. lambda=0. Parallel over (case x protocol)."""
from __future__ import annotations
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, time, csv, multiprocessing as mp
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from route_a.lammps_io import read_lammps_dump                        # noqa: E402
from route_a.nondim import Nondimensionalizer, setup_grid             # noqa: E402
from route_a.kernels import get_kernel                                # noqa: E402
from route_a.postprocess import get_postprocess                       # noqa: E402
from route_a.triangular_pfc import TriangularPFCModel, triangular_k1  # noqa: E402
from route_a.metrics import reconstruct_peaks                         # noqa: E402
from route_a.metrics_extra import self_bond_cutoff, _coordination, ring_l1  # noqa: E402

CFG = dict(c2_h=1.05, eta=0.25, chi=1.0)   # validated self-sustaining triangular PFC
GOOD = {"K2/R2/P3", "K3/R2/P3", "K3/R2/P4"}
BAD = {"K1/R2/P3", "K4/R2/P3"}
PROTOS = ["K1/R2/P3", "K2/R2/P3", "K3/R2/P3", "K3/R2/P4", "K4/R2/P3"]
CASES = ["TR0_pristine", "TR1_vacancy", "TR2_void", "TR3_interstitial"]
TRI_DIR = os.path.join(ROOT, "experiments", "triangular")
_PREP = {}


def coord_hist(p):
    """Coordination histogram {coord: count} over interior points (self-calibrated cutoff)."""
    p = np.asarray(p)[:, :2]
    if len(p) < 4:
        return {}
    bc = self_bond_cutoff(p)
    if bc <= 0:
        return {}
    c = _coordination(p, bc)
    lo = p.min(0) + bc; hi = p.max(0) - bc
    interior = np.all((p > lo) & (p < hi), axis=1)
    return dict(Counter(c[interior].tolist()))


def _prep(case_keys, tri_dir):
    out = {}
    for ck in case_keys:
        atoms, box = read_lammps_dump(os.path.join(tri_dir, f"relaxed_{ck}.dump"))
        nd = Nondimensionalizer((box[0], box[1]), (box[2], box[3]))
        a_nd = nd.nondimensionalize_coords(atoms)
        grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
        nn = float(np.median(cKDTree(a_nd[:, :2]).query(a_nd[:, :2], k=2)[0][:, 1]))
        k1 = triangular_k1(nn)   # NN spacing IS the lattice constant for a triangular lattice
        out[ck] = dict(a_nd=a_nd, grid=grid, k1=k1, minsep=0.7 * nn, hgt=coord_hist(a_nd))
    return out


def _work(task):
    ck, proto = task
    P = _PREP[ck]; grid = P["grid"]
    K, R, Pp = proto.split("/")
    field = get_postprocess(Pp).apply(get_kernel(K).map(P["a_nd"], grid))
    hi = coord_hist(reconstruct_peaks(field, grid, P["minsep"]))
    m = TriangularPFCModel.for_field(field, grid.dx, P["k1"], x_bounds=grid.x_bounds,
                                     y_bounds=grid.y_bounds, morphological_constraint_weight=0.0, **CFG)
    m.relax(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)
    hr = coord_hist(reconstruct_peaks(m.n, grid, P["minsep"]))
    return dict(case=ck, protocol=proto, tier=("good" if proto in GOOD else "bad"),
                coordL1_init=ring_l1(hi, P["hgt"]), coordL1_relaxed=ring_l1(hr, P["hgt"]),
                n5_gt=P["hgt"].get(5, 0), n7_gt=P["hgt"].get(7, 0),
                n5_relaxed=hr.get(5, 0), n6_relaxed=hr.get(6, 0), n7_relaxed=hr.get(7, 0),
                healthy=bool(m.check_density_health()), nmax=float(np.abs(m.n).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncores", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "triangular_pfc"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global _PREP
    _PREP = _prep(CASES, TRI_DIR)
    print("ground-truth coordination:", {ck: _PREP[ck]["hgt"] for ck in CASES}, flush=True)
    tasks = [(ck, p) for ck in CASES for p in PROTOS]
    t0 = time.time()
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    with mp.Pool(min(args.ncores, len(tasks))) as pool:
        rows = pool.map(_work, tasks)
    with open(os.path.join(args.out, "triangular.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # per-case two-tier separation on relaxed coord-L1
    summary = {"per_case": {}, "n_clean": 0, "n_cases": 0, "worst_gap": None}
    worst = np.inf
    for ck in CASES:
        sub = [r for r in rows if r["case"] == ck]
        good = [r["coordL1_relaxed"] for r in sub if r["tier"] == "good" and r["healthy"]]
        bad = [r["coordL1_relaxed"] for r in sub if r["tier"] == "bad" and r["healthy"]]
        if good and bad:
            gmax, bmin = max(good), min(bad)
            clean = bool(gmax < bmin)
            gap = float(bmin / gmax) if gmax > 1e-9 else float("inf")
            summary["per_case"][ck] = dict(good_max=round(gmax, 3), bad_min=round(bmin, 3),
                                           clean=clean, gap=round(gap, 1) if np.isfinite(gap) else None)
            summary["n_cases"] += 1; summary["n_clean"] += int(clean)
            if ck != "TR0_pristine":
                worst = min(worst, gap)
    summary["worst_gap_defects"] = round(worst, 1) if np.isfinite(worst) else None
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"ran {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)
    print(f"two-tier clean in {summary['n_clean']}/{summary['n_cases']} cases; "
          f"worst defect gap = {summary['worst_gap_defects']}x", flush=True)
    for ck, v in summary["per_case"].items():
        print(f"  {ck:18s} good_max={v['good_max']:.3f} bad_min={v['bad_min']:.3f} "
              f"clean={v['clean']} gap={v['gap']}x", flush=True)


if __name__ == "__main__":
    main()
