#!/usr/bin/env python3
"""EXPERIMENT 4 (advisor I.4) -- JOINT (mixed-corner) structural-PFC robustness.

The published structural-PFC parameter scan (run_structural_pfc_scan.py) varies
(h1, h2, peak-width) ONE-AT-A-TIME around the calibrated point. A reviewer can
object that a one-at-a-time scan misses *interaction* effects: maybe the basin
assignment only survives because the other two parameters were held at their
sweet spot. This script closes that gap by moving h1, h2, and the peak width
TOGETHER to the corners of the joint parameter box and re-running the decisive
basin test at each corner.

Decisive claim (must hold at EVERY mixed corner):
  * the Voronoi-localised Gaussian seeds {K2, K3} relax into the CORRECT basin
    (relaxed bond-graph ring-L1 small vs the AIREBO-relaxed ground truth), while
  * the cell-indicator K4 and the naive default-cutoff K1@200 relax into the
    WRONG basin (ring-L1 large).
If GOOD-max < BAD-min with a margin at every corner, the basin assignment is
robust to JOINT parameter variation, not just one-at-a-time.

Mixed corners (h1, h2, w_ratio), with the structural-PFC native regime lambda=0:
  C1  low h1  + high width : (0.95, 0.60, 0.13)
  C2  high h2 + low  width : (1.05, 0.80, 0.07)
  C3  low h1  + low  h2    : (0.95, 0.40, 0.10)
plus the calibrated baseline (1.05, 0.60, 0.10) as an anchor.

Structural PFC: chi=1.0, eta=0.25, lambda=0 (morphological_constraint_weight=0).
Map: get_postprocess("P3").apply(get_kernel(KN, **kw).map(a_nd, grid)).
Relax: tol=1e-6, max_steps=2500, chunk=300, dt=1e-8.
k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax);  c2_w = w_ratio * k1.
minsep = 0.7 * median_NN.  Detector: route_a.metrics_extra bond-graph + ring_l1.

Writes outputs/mixed_corners/mixed_corners.csv + summary.json.
Run with PYTHONPATH=$PWD/src from the repo root.
"""
from __future__ import annotations
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, json, time, multiprocessing as mp
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

# Fixed structural-PFC params shared across corners (lambda=0 native regime).
FIXED = dict(chi=1.0, eta=0.25)

# Calibrated baseline (matches run_structural_pfc.py): h1=1.05, h2=0.6, w=0.10 k1.
BASE = dict(c2_h1=1.05, c2_h2=0.60, w_ratio=0.10)

# JOINT mixed corners: h1, h2, width move TOGETHER.
CORNERS = [
    ("baseline",            dict(c2_h1=1.05, c2_h2=0.60, w_ratio=0.10)),
    ("C1_lowh1_highw",      dict(c2_h1=0.95, c2_h2=0.60, w_ratio=0.13)),
    ("C2_highh2_loww",      dict(c2_h1=1.05, c2_h2=0.80, w_ratio=0.07)),
    ("C3_lowh1_lowh2",      dict(c2_h1=0.95, c2_h2=0.40, w_ratio=0.10)),
]

# Kernels to seed from. GOOD = Voronoi-localised Gaussians (should stay correct).
# BAD = cell-indicator K4 + naive default-cutoff spectral K1@200 (should stay wrong).
KCFG = {
    "K2":      ("K2", {}),
    "K3":      ("K3", {}),
    "K1@200":  ("K1", dict(k_cut=200.0)),
    "K4":      ("K4", {}),
}
GOOD = {"K2", "K3"}
BAD = {"K1@200", "K4"}

CASES = ["T0", "T1", "T2", "T3", "T4", "T5"]
FMAP = {"T0": "relaxed_T0_pristine.dump", "T1": "relaxed_T1_vacancy.dump",
        "T2": "relaxed_T2_stone_wales.dump", "T3": "relaxed_T3_dislocation_dipole.dump",
        "T4": "relaxed_T4_low_angle_gb.dump", "T5": "relaxed_T5_high_angle_gb.dump"}
RELAXED_DIR = os.path.join(ROOT, "experiments", "hpc_package")
_PREP = {}


def _prep():
    out = {}
    for ck in CASES:
        c = get_case(ck)
        atoms, _ = read_lammps_dump(os.path.join(RELAXED_DIR, FMAP[ck]))
        nd = Nondimensionalizer((c.box[0], c.box[1]), (c.box[2], c.box[3]))
        a_nd = nd.nondimensionalize_coords(atoms)
        grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
        k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax)
        nn = float(np.median(cKDTree(a_nd[:, :2]).query(a_nd[:, :2], k=2)[0][:, 1]))
        gt = ring_histogram_bondgraph(a_nd[:, :2])  # AIREBO-relaxed ground truth
        out[ck] = dict(name=c.name, a_nd=a_nd, grid=grid, k1=k1, minsep=0.7 * nn, gt=gt)
    return out


def _bg(field, grid, minsep):
    return ring_histogram_bondgraph(reconstruct_peaks(field, grid, minsep))


def _work(task):
    corner_name, cfg, ck, kname = task
    P = _PREP[ck]; grid = P["grid"]
    kn, kw = KCFG[kname]
    field = get_postprocess("P3").apply(get_kernel(kn, **kw).map(P["a_nd"], grid))
    hi = _bg(field, grid, P["minsep"])
    c2_w = cfg["w_ratio"] * P["k1"]
    m = StructuralPFCModel.for_field(
        field, grid.dx, P["k1"], x_bounds=grid.x_bounds, y_bounds=grid.y_bounds,
        morphological_constraint_weight=0.0,
        c2_h1=cfg["c2_h1"], c2_h2=cfg["c2_h2"], c2_w=c2_w, **FIXED)
    m.relax(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)
    hr = _bg(m.n, grid, P["minsep"])
    return dict(
        corner=corner_name, case=P["name"], case_key=ck, kernel=kname,
        tier=("good" if kname in GOOD else "bad"),
        c2_h1=cfg["c2_h1"], c2_h2=cfg["c2_h2"], w_ratio=cfg["w_ratio"],
        n5_gt=P["gt"].get(5, 0), n7_gt=P["gt"].get(7, 0),
        ringL1_init=round(ring_l1(hi, P["gt"]), 4),
        ringL1_relaxed=round(ring_l1(hr, P["gt"]), 4),
        n5_relaxed=hr.get(5, 0), n7_relaxed=hr.get(7, 0),
        healthy=bool(m.check_density_health()),
        nmax=round(float(np.abs(m.n).max()), 4))


def _summarise(rows):
    """Per (corner, defect-case): does every healthy GOOD seed beat every healthy
    BAD seed (relaxed ring-L1)? T0 (pristine) has no rings to discriminate, so the
    basin claim is evaluated on defects T1-T5 only (T0 is reported for health)."""
    DEF = CASES[1:]  # T1..T5
    summary = {"corners": [c[0] for c in CORNERS], "per_corner": {}}
    all_hold = True
    any_unhealthy = False
    worst_gap_overall = np.inf
    for corner_name, cfg in CORNERS:
        cells = []
        corner_hold = True
        n_unhealthy = sum(1 for r in rows if r["corner"] == corner_name and not r["healthy"])
        any_unhealthy = any_unhealthy or n_unhealthy > 0
        for ck in DEF:
            cname = get_case(ck).name
            sub = [r for r in rows if r["corner"] == corner_name and r["case"] == cname]
            good = [r["ringL1_relaxed"] for r in sub if r["tier"] == "good" and r["healthy"]]
            bad = [r["ringL1_relaxed"] for r in sub if r["tier"] == "bad" and r["healthy"]]
            if not good or not bad:
                corner_hold = False
                cells.append(dict(case=ck, note="missing-healthy-seed", clean=False))
                continue
            gmax, bmin = max(good), min(bad)
            clean = bool(gmax < bmin)
            gap = float(bmin / gmax) if gmax > 1e-9 else float("inf")
            corner_hold = corner_hold and clean
            worst_gap_overall = min(worst_gap_overall, gap)
            cells.append(dict(case=ck, good_max=round(gmax, 3), bad_min=round(bmin, 3),
                              clean=clean, gap=round(gap, 2) if np.isfinite(gap) else None))
        summary["per_corner"][corner_name] = dict(
            params=dict(c2_h1=cfg["c2_h1"], c2_h2=cfg["c2_h2"], w_ratio=cfg["w_ratio"]),
            hold=corner_hold, n_unhealthy=n_unhealthy, cells=cells)
        all_hold = all_hold and corner_hold
    summary["all_hold"] = bool(all_hold and not any_unhealthy)
    summary["any_unhealthy"] = bool(any_unhealthy)
    summary["worst_good_vs_bad_gap"] = (round(worst_gap_overall, 2)
                                        if np.isfinite(worst_gap_overall) else None)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncores", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "mixed_corners"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global _PREP
    _PREP = _prep()
    print("GT bond-graph (n5,n7) per case:",
          {ck: (_PREP[ck]["gt"].get(5, 0), _PREP[ck]["gt"].get(7, 0)) for ck in CASES}, flush=True)
    tasks = [(name, cfg, ck, k) for (name, cfg) in CORNERS for ck in CASES for k in KCFG]
    print(f"MIXED-CORNER scan: {len(CORNERS)} corners x {len(CASES)} cases x {len(KCFG)} kernels "
          f"= {len(tasks)} relaxations, ncores={args.ncores}", flush=True)
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    t0 = time.time()
    with mp.Pool(args.ncores) as pool:
        rows = pool.map(_work, tasks)
    # stable sort for readable CSV: corner, case, kernel
    korder = {k: i for i, k in enumerate(KCFG)}
    rows.sort(key=lambda r: ([c[0] for c in CORNERS].index(r["corner"]),
                             CASES.index(r["case_key"]), korder[r["kernel"]]))
    with open(os.path.join(args.out, "mixed_corners.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = _summarise(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ------- compact console table -------
    print(f"\nran {len(rows)} relaxations in {time.time()-t0:.0f}s\n", flush=True)
    print("Per-corner relaxed ring-L1 (defects T1-T5), GOOD={K2,K3} vs BAD={K1@200,K4}:")
    hdr = f"{'corner':18s} {'(h1,h2,w)':18s} | {'case':5s} | {'good_max':>8s} {'bad_min':>8s} {'gap':>6s} {'clean':>6s}"
    print(hdr); print("-" * len(hdr))
    for corner_name, _ in CORNERS:
        pc = summary["per_corner"][corner_name]
        p = pc["params"]
        ptxt = f"({p['c2_h1']:.2f},{p['c2_h2']:.2f},{p['w_ratio']:.2f})"
        for i, cell in enumerate(pc["cells"]):
            cn = ptxt if i == 0 else ""
            nm = corner_name if i == 0 else ""
            if "good_max" in cell:
                gap = cell["gap"]
                print(f"{nm:18s} {cn:18s} | {cell['case']:5s} | "
                      f"{cell['good_max']:8.3f} {cell['bad_min']:8.3f} "
                      f"{(gap if gap is not None else float('nan')):6.2f} "
                      f"{('YES' if cell['clean'] else 'NO'):>6s}")
            else:
                print(f"{nm:18s} {cn:18s} | {cell['case']:5s} | {cell.get('note',''):>24s}")
        uh = pc["n_unhealthy"]
        print(f"{'':18s} {'':18s} | corner hold={pc['hold']}  n_unhealthy={uh}")
        print("-" * len(hdr))
    print(f"\nworst GOOD-vs-BAD gap over all corners/defects = {summary['worst_good_vs_bad_gap']}x")
    print(f"ALL CORNERS HOLD (basin assignment robust to joint variation): {summary['all_hold']}")
    if summary["any_unhealthy"]:
        print("WARNING: at least one relaxation was unhealthy (blow-up) -- inspect CSV.")
    print(f"\nwrote {args.out}/mixed_corners.csv + summary.json", flush=True)


if __name__ == "__main__":
    main()
