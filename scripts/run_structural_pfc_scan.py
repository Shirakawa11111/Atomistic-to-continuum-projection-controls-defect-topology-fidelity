#!/usr/bin/env python3
"""Structural-PFC PARAMETER ROBUSTNESS SCAN (PRM revision; de-risk the linchpin).

Reviewer concern: the peaked-C2 structural PFC is tested at a SINGLE
parameterisation (h1=1.05, h2=0.6, w=0.10 k1), so the two-tier ranking might be
"tuned to the answer." Here we vary the structural-PFC parameters ONE-AT-A-TIME
around the calibrated point and check the ranking survives:

    Voronoi-localised Gaussian {K2,K3}+rescale  <<  non-Gaussian {K1,K4}
    (lower relaxed ring-L1 == better defect-topology fidelity)

Note: the *initialisation* ring-L1 is model-independent (pure mapping), so it is
IDENTICAL across configs by construction. The scan therefore tests whether the
RELAXED ranking — the dynamical, model-dependent outcome — is parameter-robust.
lambda=0 (structural-PFC native regime). Parallel over (config x case x protocol).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time, csv, multiprocessing as mp
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

# Calibrated baseline (matches run_structural_pfc.py): h1=1.05, h2=0.6, w=0.10 k1, chi=1.0.
BASE = dict(c2_h1=1.05, c2_h2=0.6, w_ratio=0.10, chi=1.0)
GOOD = {"K2/R2/P3", "K3/R2/P3", "K3/R2/P4"}   # Voronoi-localised Gaussian + rescale
BAD = {"K1/R2/P3", "K4/R2/P3"}                # non-Gaussian (spectral / cell-indicator)
PROTOS = ["K1/R2/P3", "K2/R2/P3", "K3/R2/P3", "K3/R2/P4", "K4/R2/P3"]
CASES = ["T0", "T1", "T2", "T3", "T4", "T5"]


def _cfgs():
    """Baseline + one-at-a-time variations of (h1, h2, peak width)."""
    out = [("base", dict(BASE))]
    for h1 in (0.95, 1.15):
        c = dict(BASE); c["c2_h1"] = h1; out.append((f"h1_{h1:.2f}", c))
    for h2 in (0.40, 0.80):
        c = dict(BASE); c["c2_h2"] = h2; out.append((f"h2_{h2:.2f}", c))
    for wr in (0.07, 0.13):
        c = dict(BASE); c["w_ratio"] = wr; out.append((f"w_{wr:.2f}", c))
    return out


CFGS = _cfgs()
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
    cfg_name, cfg, ck, proto = task
    P = _PREP[ck]; grid = P["grid"]
    K, R, Pp = proto.split("/")
    field = get_postprocess(Pp).apply(get_kernel(K).map(P["a_nd"], grid))
    hi = _rh(field, grid, P["minsep"])
    c2_w = cfg["w_ratio"] * P["k1"]
    m = StructuralPFCModel.for_field(
        field, grid.dx, P["k1"], x_bounds=grid.x_bounds, y_bounds=grid.y_bounds,
        morphological_constraint_weight=0.0,
        c2_h1=cfg["c2_h1"], c2_h2=cfg["c2_h2"], c2_w=c2_w, chi=cfg["chi"])
    m.relax(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)
    hr = _rh(m.n, grid, P["minsep"])
    return dict(config=cfg_name, case=P["name"], protocol=proto, tier=("good" if proto in GOOD else "bad"),
                c2_h1=cfg["c2_h1"], c2_h2=cfg["c2_h2"], w_ratio=cfg["w_ratio"],
                ringL1_init=ring_l1(hi, P["hgt"]), ringL1_relaxed=ring_l1(hr, P["hgt"]),
                healthy=bool(m.check_density_health()), nmax=float(np.abs(m.n).max()))


def _summarise(rows):
    """Per-(config,case): does every healthy GOOD protocol beat every healthy BAD one?"""
    summary = {"n_rows": len(rows), "configs": [c[0] for c in CFGS], "per_config": {}}
    worst_gap_overall = np.inf
    n_clean = n_cells = 0
    for cfg_name, _ in CFGS:
        cells = []
        for ck in CASES:
            cname = get_case(ck).name
            sub = [r for r in rows if r["config"] == cfg_name and r["case"] == cname]
            good = [r["ringL1_relaxed"] for r in sub if r["tier"] == "good" and r["healthy"]]
            bad = [r["ringL1_relaxed"] for r in sub if r["tier"] == "bad" and r["healthy"]]
            if not good or not bad:
                continue
            gmax, bmin = max(good), min(bad)
            clean = bool(gmax < bmin)
            gap = float(bmin / gmax) if gmax > 1e-9 else float("inf")
            cells.append(dict(case=ck, good_max=round(gmax, 3), bad_min=round(bmin, 3),
                              clean=clean, gap=round(gap, 1) if np.isfinite(gap) else None))
            n_cells += 1; n_clean += int(clean)
            worst_gap_overall = min(worst_gap_overall, gap)
        finite_gaps = [c["gap"] for c in cells if c["gap"] is not None]
        n_unhealthy = sum(1 for r in rows if r["config"] == cfg_name and not r["healthy"])
        summary["per_config"][cfg_name] = dict(
            n_cases=len(cells), n_clean=sum(c["clean"] for c in cells),
            min_gap=round(min(finite_gaps), 1) if finite_gaps else None,
            median_gap=round(float(np.median(finite_gaps)), 1) if finite_gaps else None,
            n_unhealthy=n_unhealthy, cells=cells)
    summary["overall"] = dict(
        n_cells=n_cells, n_clean=n_clean,
        clean_fraction=round(n_clean / n_cells, 3) if n_cells else None,
        worst_gap=round(worst_gap_overall, 1) if np.isfinite(worst_gap_overall) else None)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncores", type=int, default=128)
    ap.add_argument("--relaxed-dir", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "structural_pfc_scan"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global _PREP
    _PREP = _prep(CASES, args.relaxed_dir)
    tasks = [(name, cfg, ck, p) for (name, cfg) in CFGS for ck in CASES for p in PROTOS]
    print(f"structural-PFC SCAN: {len(CFGS)} configs x {len(CASES)} cases x {len(PROTOS)} protos "
          f"= {len(tasks)} tasks, ncores={args.ncores}", flush=True)
    t0 = time.time()
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    with mp.Pool(args.ncores) as pool:
        rows = pool.map(_work, tasks)
    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "scan.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary = _summarise(rows)
    with open(os.path.join(args.out, "scan_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    ov = summary["overall"]
    print(f"ran {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)
    print(f"two-tier separation clean in {ov['n_clean']}/{ov['n_cells']} (config,case) cells; "
          f"worst good-vs-bad gap = {ov['worst_gap']}x", flush=True)
    print(f"wrote {args.out}/scan.csv + scan_summary.json", flush=True)


if __name__ == "__main__":
    main()
