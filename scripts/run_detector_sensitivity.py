#!/usr/bin/env python3
"""EXPERIMENT 3 (advisor I.3) -- bond-graph detector cutoff sensitivity.

The basin classification (correct-basin {K2,K3,fair-K1} vs wrong-basin
{naive-K1,K4}) must NOT depend on the ring detector's bond cutoff. The
bond-graph ring detector defines a bond as a KD-tree pair shorter than
``r_bond = self_bond_cutoff_factor * median_NN`` (the default factor is 1.4;
see route_a.metrics_extra.self_bond_cutoff). Here we re-score the SAME relaxed
structural-PFC fields at three cutoffs -- r_bond = {1.3, 1.4, 1.5} x median NN
-- and confirm the correct/wrong assignment is invariant.

Method (one relaxation per (case,kernel), then re-scored at all 3 cutoffs):
  * structural PFC: c2_h1=1.05, c2_h2=0.6, chi=1.0, eta=0.25, lambda=0,
    morphological_constraint_weight=0 (StructuralPFCModel.for_field).
  * map:  get_postprocess("P3").apply(get_kernel(KN,**kw).map(a_nd,grid))
  * relax(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)
  * k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax); minsep = 0.7*median_NN
  * peaks via reconstruct_peaks; rings via ring_histogram_bondgraph with an
    EXPLICIT bond_cutoff = factor * median_NN (median NN measured on the AIREBO
    ground-truth atoms, so the SAME physical cutoff is applied to the relaxed
    peak set and the ground-truth atoms for each factor -> a fair like-for-like
    ring-L1).

Six defects T0..T5, five projections (K2, K3, K1@200 naive, K1@470 fair, K4).
Writes outputs/detector_sensitivity/{detector_sensitivity.csv, summary.json,
sensitivity_table.txt}.
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

CFG = dict(c2_h1=1.05, c2_h2=0.6, chi=1.0, eta=0.25)
CASES = ["T0", "T1", "T2", "T3", "T4", "T5"]
FMAP = {"T0": "relaxed_T0_pristine.dump", "T1": "relaxed_T1_vacancy.dump",
        "T2": "relaxed_T2_stone_wales.dump", "T3": "relaxed_T3_dislocation_dipole.dump",
        "T4": "relaxed_T4_low_angle_gb.dump", "T5": "relaxed_T5_high_angle_gb.dump"}
# kernel config -> (kernel-name, kwargs), with basin tier label
KCFG = {
    "K2":             ("K2", {},                  "correct"),
    "K3":             ("K3", {},                  "correct"),
    "K1@470_fair":    ("K1", dict(k_cut=470.0),   "correct"),
    "K1@200_default": ("K1", dict(k_cut=200.0),   "wrong"),
    "K4":             ("K4", {},                  "wrong"),
}
CORRECT = ["K2", "K3", "K1@470_fair"]
WRONG = ["K1@200_default", "K4"]
# r_bond cutoff factors x median nearest-neighbour distance
FACTORS = [1.3, 1.4, 1.5]
RELAXED_DIR = os.path.join(ROOT, "experiments", "hpc_package")
_PREP = {}


def _median_nn(pts2d: np.ndarray) -> float:
    return float(np.median(cKDTree(pts2d).query(pts2d, k=2)[0][:, 1]))


def _prep():
    out = {}
    for ck in CASES:
        c = get_case(ck)
        atoms, _ = read_lammps_dump(os.path.join(RELAXED_DIR, FMAP[ck]))
        nd = Nondimensionalizer((c.box[0], c.box[1]), (c.box[2], c.box[3]))
        a_nd = nd.nondimensionalize_coords(atoms)
        grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
        k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax)
        gt_xy = a_nd[:, :2]
        nn = _median_nn(gt_xy)  # median NN of the GT atoms -> defines r_bond
        out[ck] = dict(name=c.name, a_nd=a_nd, grid=grid, k1=k1,
                       minsep=0.7 * nn, median_nn=nn, gt_xy=gt_xy)
    return out


def _work(task):
    """Relax once per (case,kernel); save relaxed + init peak sets for re-scoring."""
    ck, kname = task
    P = _PREP[ck]; grid = P["grid"]
    kn, kw, _tier = KCFG[kname]
    field = get_postprocess("P3").apply(get_kernel(kn, **kw).map(P["a_nd"], grid))
    init_pk = reconstruct_peaks(field, grid, P["minsep"])
    m = StructuralPFCModel.for_field(field, grid.dx, P["k1"], x_bounds=grid.x_bounds,
                                     y_bounds=grid.y_bounds,
                                     morphological_constraint_weight=0.0, **CFG)
    m.relax(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)
    relaxed_pk = reconstruct_peaks(m.n, grid, P["minsep"])
    return dict(case=ck, name=P["name"], kernel=kname,
                init_pk=init_pk, relaxed_pk=relaxed_pk,
                healthy=bool(m.check_density_health()),
                nmax=round(float(np.abs(m.n).max()), 3))


def main():
    out = os.path.join(ROOT, "outputs", "detector_sensitivity")
    os.makedirs(out, exist_ok=True)
    global _PREP
    _PREP = _prep()
    print("median NN per case (nd units):",
          {ck: round(_PREP[ck]["median_nn"], 5) for ck in CASES}, flush=True)

    tasks = [(ck, k) for ck in CASES for k in KCFG]
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    t0 = time.time()
    with mp.Pool(min(len(tasks), os.cpu_count() or 6)) as pool:
        raw = pool.map(_work, tasks)
    print(f"ran {len(raw)} relaxations in {time.time()-t0:.0f}s\n", flush=True)

    # Re-score each relaxed/init peak set at every cutoff factor; recompute the
    # GT histogram at the SAME explicit cutoff so ring-L1 is like-for-like.
    rows = []
    for rec in raw:
        ck = rec["case"]; P = _PREP[ck]; nn = P["median_nn"]
        for fac in FACTORS:
            rbond = fac * nn
            gt = ring_histogram_bondgraph(P["gt_xy"], bond_cutoff=rbond)
            hi = ring_histogram_bondgraph(rec["init_pk"], bond_cutoff=rbond)
            hr = ring_histogram_bondgraph(rec["relaxed_pk"], bond_cutoff=rbond)
            rows.append(dict(
                case=rec["name"], case_id=ck, kernel=rec["kernel"],
                tier=KCFG[rec["kernel"]][2], r_bond_factor=fac,
                r_bond_nd=round(rbond, 5),
                n5_gt=gt.get(5, 0), n7_gt=gt.get(7, 0),
                ringL1_init=round(ring_l1(hi, gt), 4),
                ringL1_relaxed=round(ring_l1(hr, gt), 4),
                n5_relaxed=hr.get(5, 0), n7_relaxed=hr.get(7, 0),
                healthy=rec["healthy"], nmax=rec["nmax"]))

    fields = list(rows[0].keys())
    with open(os.path.join(out, "detector_sensitivity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    # ----- per-cutoff per-kernel summary (defects T1-T5; mean relaxed ring-L1) -----
    def relaxed_vec(kernel, fac):
        return [next(r["ringL1_relaxed"] for r in rows
                     if r["kernel"] == kernel and r["case_id"] == ck
                     and r["r_bond_factor"] == fac) for ck in CASES]

    summ = {}
    for fac in FACTORS:
        summ[f"{fac}"] = {}
        for k in KCFG:
            vec = relaxed_vec(k, fac)
            summ[f"{fac}"][k] = dict(
                tier=KCFG[k][2], per_case_relaxed=[round(v, 4) for v in vec],
                mean_defects_relaxed=round(float(np.mean(vec[1:])), 4))

    # invariance test: at each cutoff, max(correct mean) < min(wrong mean)
    invariant = True
    gaps = {}
    for fac in FACTORS:
        cmax = max(summ[f"{fac}"][k]["mean_defects_relaxed"] for k in CORRECT)
        wmin = min(summ[f"{fac}"][k]["mean_defects_relaxed"] for k in WRONG)
        gaps[f"{fac}"] = dict(correct_max=round(cmax, 4), wrong_min=round(wmin, 4),
                              separated=bool(cmax < wmin))
        invariant = invariant and (cmax < wmin)

    json.dump(dict(factors=FACTORS, summary=summ, gaps=gaps,
                   invariant=bool(invariant),
                   correct_tier=CORRECT, wrong_tier=WRONG),
              open(os.path.join(out, "summary.json"), "w"), indent=2)

    # ----- compact human-readable table -----
    lines = []
    lines.append("Bond-graph detector cutoff sensitivity (relaxed structural-PFC, lambda=0)")
    lines.append("mean relaxed ring-L1 over defects T1-T5 (lower = correct basin)\n")
    header = f"{'kernel':16s} {'tier':8s} | " + "  ".join(f"x{fac}" for fac in FACTORS)
    lines.append(header)
    lines.append("-" * len(header))
    for k in list(CORRECT) + list(WRONG):
        cells = "  ".join(f"{summ[f'{fac}'][k]['mean_defects_relaxed']:5.3f}" for fac in FACTORS)
        lines.append(f"{k:16s} {KCFG[k][2]:8s} | {cells}")
    lines.append("-" * len(header))
    for fac in FACTORS:
        g = gaps[f"{fac}"]
        lines.append(f"  r_bond=x{fac}: correct_max={g['correct_max']:.3f}  "
                     f"wrong_min={g['wrong_min']:.3f}  separated={g['separated']}")
    lines.append("")
    lines.append(f"ASSIGNMENT INVARIANT across r_bond in {{1.3,1.4,1.5}}: {invariant}")
    table = "\n".join(lines)
    with open(os.path.join(out, "sensitivity_table.txt"), "w") as fh:
        fh.write(table + "\n")

    print(table, flush=True)
    print(f"\nwrote {out}/detector_sensitivity.csv + summary.json + sensitivity_table.txt",
          flush=True)


if __name__ == "__main__":
    main()
