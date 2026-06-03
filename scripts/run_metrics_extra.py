#!/usr/bin/env python3
"""Independent-detector robustness audit of the two-tier fidelity claim.

The headline result (``metrics.ring_l1_init``) uses ONE ring detector (a per-edge
shortest-cycle search). This script re-scores the SAME mapping protocols with two
*algorithmically independent* probes from ``metrics_extra``:

  1. a Delaunay planar-face ring detector  -> ``ring_l1_delaunay``
  2. non-ring geometry fidelity vs the ground-truth atoms:
        coordination-F1 mismatch (graphene = 3-fold) -> ``coord_f1_mismatch``
        bond-angle-distribution L1                   -> ``bond_angle_l1``
        peak-localisation RMS (greedy NN match)      -> ``loc_rms``

For each case T0-T5 we load the LAMMPS-relaxed atoms, map them to a field through
each protocol exactly as the pipeline does (Nondimensionalizer from the *case*
box, setup_grid at nx=512, kernel.map, postprocess.apply), reconstruct peaks with
``metrics.reconstruct_peaks`` (min_sep = 0.7 * median nearest-neighbour), and
compare the reconstruction to the ground-truth atoms under each independent metric.
The shortest-cycle ``ring_l1_init`` is also recomputed here so all numbers come
from one consistent run.

Output: outputs/metrics_extra/metrics_extra.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from route_a.cases import get_case  # noqa: E402
from route_a.lammps_io import read_lammps_dump  # noqa: E402
from route_a.nondim import Nondimensionalizer, setup_grid  # noqa: E402
from route_a.kernels import get_kernel  # noqa: E402
from route_a.postprocess import get_postprocess  # noqa: E402
import route_a.metrics as M  # noqa: E402
import route_a.metrics_extra as MX  # noqa: E402


CASES = ["T0", "T1", "T2", "T3", "T4", "T5"]
PROTOCOLS = [  # (K, R, P) — the two-tier claim's protocols, all R-independent (P3/P4)
    ("K1", "R2", "P3"),
    ("K2", "R2", "P3"),
    ("K3", "R2", "P3"),
    ("K3", "R2", "P4"),
    ("K4", "R2", "P3"),
]
NX = 512
DUMP_TMPL = os.path.join(REPO, "experiments", "hpc_package", "relaxed_{name}.dump")
OUT_DIR = os.path.join(REPO, "outputs", "metrics_extra")
OUT_CSV = os.path.join(OUT_DIR, "metrics_extra.csv")


def median_nn(points_nd: np.ndarray) -> float:
    p = np.asarray(points_nd, float)[:, :2]
    if len(p) < 2:
        return 1.0
    d, _ = cKDTree(p).query(p, k=2)
    return float(np.median(d[:, 1]))


def run_one(case_key: str, K: str, R: str, P: str) -> dict:
    case = get_case(case_key)
    atoms, _dump_box = read_lammps_dump(DUMP_TMPL.format(name=case.name))

    # nondimensionalize with the CASE box (pipeline convention) so the field and
    # the ground-truth atoms share one coordinate frame.
    nd = Nondimensionalizer((case.box[0], case.box[1]), (case.box[2], case.box[3]))
    atoms_nd = nd.nondimensionalize_coords(atoms)[:, :2]
    defects_nd = (
        nd.nondimensionalize_2d(case.defect_centers) if case.defect_centers else None
    )
    grid = setup_grid(nd.Lx_nd, nd.Ly_nd, NX)

    # map -> postprocess (P3/P4 are reference-free, so rho0=None)
    kernel = get_kernel(K)
    post = get_postprocess(P)
    rho = kernel.map(atoms_nd, grid, defects_nd=defects_nd)
    field_init = post.apply(rho, rho0=None)

    # reconstruct atom positions from the mapped field
    min_sep = 0.7 * median_nn(atoms_nd)
    peaks = M.reconstruct_peaks(field_init, grid, min_sep_nd=min_sep)

    # --- (headline) shortest-cycle ring-L1, recomputed for a consistent baseline
    h_gt_sssr = M.ring_histogram(atoms_nd)
    h_pk_sssr = M.ring_histogram(peaks)
    ring_l1_sssr = M.ring_l1(h_pk_sssr, h_gt_sssr)

    # --- (a) INDEPENDENT Delaunay-face ring-L1
    ring_l1_del, h_pk_del, h_gt_del = MX.ring_l1_delaunay_vs_truth(peaks, atoms_nd)

    # --- (b) NON-ring geometry fidelity
    cf = MX.coordination_f1(peaks, atoms_nd)
    ba_l1 = MX.bond_angle_l1(peaks, atoms_nd)
    loc = MX.peak_localisation_rms(peaks, atoms_nd)

    return dict(
        case=case.name,
        K=K, R=R, P=P, protocol=f"{K}/{R}/{P}",
        nx=NX, ny=grid.NY, dx=grid.dx,
        n_atoms_gt=int(len(atoms_nd)),
        n_peaks=int(len(peaks)),
        peak_count_err=abs(len(peaks) - len(atoms_nd)) / max(len(atoms_nd), 1),
        min_sep_nd=min_sep,
        # headline detector (recomputed)
        ring_l1_sssr=ring_l1_sssr,
        n5_gt_sssr=h_gt_sssr.get(5, 0), n6_gt_sssr=h_gt_sssr.get(6, 0),
        n7_gt_sssr=h_gt_sssr.get(7, 0),
        n5_pk_sssr=h_pk_sssr.get(5, 0), n6_pk_sssr=h_pk_sssr.get(6, 0),
        n7_pk_sssr=h_pk_sssr.get(7, 0),
        # independent detector (a)
        ring_l1_delaunay=ring_l1_del,
        n5_gt_del=h_gt_del.get(5, 0), n6_gt_del=h_gt_del.get(6, 0),
        n7_gt_del=h_gt_del.get(7, 0),
        n5_pk_del=h_pk_del.get(5, 0), n6_pk_del=h_pk_del.get(6, 0),
        n7_pk_del=h_pk_del.get(7, 0),
        # geometry fidelity (b)
        coord_f1=cf["coord_f1"], coord_f1_mismatch=cf["coord_f1_mismatch"],
        coord_precision=cf["coord_precision"], coord_recall=cf["coord_recall"],
        coord_n_matched=cf["n_matched"],
        bond_angle_l1=ba_l1,
        loc_rms=loc["loc_rms"], loc_matched_frac=loc["loc_matched_frac"],
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for case_key in CASES:
        for (K, R, P) in PROTOCOLS:
            print(f"[run] {case_key:>3}  {K}/{R}/{P} ...", flush=True)
            row = run_one(case_key, K, R, P)
            rows.append(row)
            print(
                f"      ring_l1_sssr={row['ring_l1_sssr']:.4f}  "
                f"ring_l1_delaunay={row['ring_l1_delaunay']:.4f}  "
                f"coord_f1_mismatch={row['coord_f1_mismatch']:.4f}  "
                f"bond_angle_l1={row['bond_angle_l1']:.4f}  "
                f"loc_rms={row['loc_rms']:.4e}  n_peaks={row['n_peaks']}",
                flush=True,
            )

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[done] wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
