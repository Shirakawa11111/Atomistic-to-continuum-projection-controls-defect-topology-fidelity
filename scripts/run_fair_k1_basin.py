#!/usr/bin/env python3
"""DECISIVE re-run: does a FAIRLY-TUNED spectral K1 reach the correct basin?

The hyperparameter scan showed K1's default cutoff k_cut=200 sits BELOW the atomic
|G1|~325, so the default K1 cannot represent the lattice. Tuning k_cut above |G1|
(~470) restores the *initialisation* fidelity to ~K2/K3 level. The open question
that decides the paper's headline: under the structural-PFC dynamics, does the
fairly-tuned K1 seed relax into the CORRECT 5-7 basin (like K2/K3) or still fall
into the WRONG basin? We answer it with the corrected bond-graph ring detector.

Configs (all P3, structural PFC c2_h1=1.05,c2_h2=0.6,chi=1.0,lambda=0):
  K1@200  spectral, default cutoff (below |G1|)   -- the "broken" baseline
  K1@470  spectral, fair cutoff (above |G1|)       -- the fair test
  K2,K3,K4 at defaults
Reports init + relaxed bond-graph ring-L1 vs the AIREBO-relaxed ground truth.
"""
from __future__ import annotations
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import csv, json, time, multiprocessing as mp
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

CFG = dict(c2_h1=1.05, c2_h2=0.6, chi=1.0)
CASES = ["T0", "T1", "T2", "T3", "T4", "T5"]
FMAP = {"T0": "relaxed_T0_pristine.dump", "T1": "relaxed_T1_vacancy.dump",
        "T2": "relaxed_T2_stone_wales.dump", "T3": "relaxed_T3_dislocation_dipole.dump",
        "T4": "relaxed_T4_low_angle_gb.dump", "T5": "relaxed_T5_high_angle_gb.dump"}
# kernel config -> (kernel-name, kwargs)
KCFG = {
    "K1@200_default": ("K1", dict(k_cut=200.0)),
    "K1@470_fair":    ("K1", dict(k_cut=470.0)),
    "K2":             ("K2", {}),
    "K3":             ("K3", {}),
    "K4":             ("K4", {}),
}
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
        gt = ring_histogram_bondgraph(a_nd[:, :2])
        out[ck] = dict(name=c.name, a_nd=a_nd, grid=grid, k1=k1, minsep=0.7 * nn, gt=gt)
    return out


def _bg(field, grid, minsep):
    return ring_histogram_bondgraph(reconstruct_peaks(field, grid, minsep))


def _work(task):
    ck, kname = task
    P = _PREP[ck]; grid = P["grid"]
    kn, kw = KCFG[kname]
    field = get_postprocess("P3").apply(get_kernel(kn, **kw).map(P["a_nd"], grid))
    hi = _bg(field, grid, P["minsep"])
    m = StructuralPFCModel.for_field(field, grid.dx, P["k1"], x_bounds=grid.x_bounds,
                                     y_bounds=grid.y_bounds, morphological_constraint_weight=0.0, **CFG)
    m.relax(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)
    hr = _bg(m.n, grid, P["minsep"])
    return dict(case=P["name"], kernel=kname,
                n5_gt=P["gt"].get(5, 0), n7_gt=P["gt"].get(7, 0),
                ringL1_init=round(ring_l1(hi, P["gt"]), 4),
                ringL1_relaxed=round(ring_l1(hr, P["gt"]), 4),
                n5_relaxed=hr.get(5, 0), n7_relaxed=hr.get(7, 0),
                healthy=bool(m.check_density_health()), nmax=round(float(np.abs(m.n).max()), 3))


def main():
    out = os.path.join(ROOT, "outputs", "fair_k1_basin")
    os.makedirs(out, exist_ok=True)
    global _PREP
    _PREP = _prep()
    print("GT bond-graph 5/7 per case:", {ck: (_PREP[ck]["gt"].get(5, 0), _PREP[ck]["gt"].get(7, 0)) for ck in CASES}, flush=True)
    tasks = [(ck, k) for ck in CASES for k in KCFG]
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    t0 = time.time()
    with mp.Pool(min(len(tasks), os.cpu_count() or 6)) as pool:
        rows = pool.map(_work, tasks)
    with open(os.path.join(out, "fair_k1_basin.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # summary: mean relaxed ring-L1 per kernel (defects only, T1-T5) + per-case table
    print(f"\nran {len(rows)} relaxations in {time.time()-t0:.0f}s\n", flush=True)
    print(f"{'kernel':16s} | " + " ".join(f"{ck:>6s}" for ck in CASES) + " | mean(T1-5)_relaxed")
    summ = {}
    for k in KCFG:
        rel = {r["case"][:2]: r["ringL1_relaxed"] for r in rows if r["kernel"] == k}
        # map case names back to T0..T5 order
        relx = [next(r["ringL1_relaxed"] for r in rows if r["kernel"] == k and r["case"].startswith(ck)) for ck in CASES]
        mean_def = float(np.mean(relx[1:]))
        summ[k] = dict(per_case_relaxed=relx, mean_defects_relaxed=round(mean_def, 4))
        print(f"{k:16s} | " + " ".join(f"{v:6.3f}" for v in relx) + f" | {mean_def:.3f}")
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    # verdict
    k1f = summ["K1@470_fair"]["mean_defects_relaxed"]
    k3 = summ["K3"]["mean_defects_relaxed"]
    k4 = summ["K4"]["mean_defects_relaxed"]
    print(f"\nVERDICT: fair-K1 mean relaxed ring-L1 = {k1f:.3f}  (K3={k3:.3f}, K4={k4:.3f})")
    if k1f < 0.3:
        print("  -> fair-K1 reaches the CORRECT basin: 'spectral fails' collapses to cutoff-sensitivity.")
    elif k1f > 0.6:
        print("  -> fair-K1 STILL falls into the WRONG basin: even optimally-tuned, spectral fails dynamically.")
    else:
        print("  -> intermediate; inspect per-case.")
    print(f"wrote {out}/fair_k1_basin.csv + summary.json", flush=True)


if __name__ == "__main__":
    main()
