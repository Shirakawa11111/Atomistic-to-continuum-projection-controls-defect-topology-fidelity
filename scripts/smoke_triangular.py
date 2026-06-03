#!/usr/bin/env python3
"""GATE: does the single-mode TriangularPFCModel self-sustain a perfect triangular
lattice? Construct a triangular sheet, map with K3/P3, relax, and check the
reconstructed peaks keep ~6-fold coordination (i.e. the model crystallizes/holds
the triangular lattice, the analog of the honeycomb self-sustain check)."""
from __future__ import annotations
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from route_a.nondim import Nondimensionalizer, setup_grid
from route_a.kernels import get_kernel
from route_a.postprocess import get_postprocess
from route_a.metrics import reconstruct_peaks
from route_a.metrics_extra import self_bond_cutoff, _coordination
from route_a.triangular_pfc import TriangularPFCModel, triangular_k1


def tri_lattice(nx, ny, a=1.0):
    pts = [(i * a + (j % 2) * 0.5 * a, j * a * np.sqrt(3) / 2)
           for j in range(ny) for i in range(nx)]
    return np.array(pts)


def coord_hist(p):
    p = np.asarray(p)[:, :2]
    bc = self_bond_cutoff(p)
    c = _coordination(p, bc)
    lo = p.min(0) + bc; hi = p.max(0) - bc
    interior = np.all((p > lo) & (p < hi), axis=1)
    return dict(sorted(Counter(c[interior].tolist()).items())), int(interior.sum())


def main():
    a = 1.0; nx = ny = 18
    at2 = tri_lattice(nx, ny, a)
    Lx, Ly = nx * a, ny * a * np.sqrt(3) / 2
    atoms = np.column_stack([at2, np.zeros(len(at2))])
    nd = Nondimensionalizer((0, Lx), (0, Ly))
    a_nd = nd.nondimensionalize_coords(atoms)
    grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
    a_NN_nd = a / nd.Lmax
    k1 = triangular_k1(a_NN_nd)
    print(f"atoms={len(a_nd)}  a_NN_nd={a_NN_nd:.4f}  k1={k1:.2f}")
    print("GT coord:", coord_hist(a_nd))

    field = get_postprocess("P3").apply(get_kernel("K3").map(a_nd, grid))
    nn = float(np.median(cKDTree(a_nd[:, :2]).query(a_nd[:, :2], k=2)[0][:, 1]))
    pk0 = reconstruct_peaks(field, grid, 0.7 * nn)
    print("init  coord:", coord_hist(pk0), " npk=", len(pk0))

    for c2_h, eta in [(1.05, 0.25), (1.05, 0.6), (1.3, 0.25), (0.9, 0.25)]:
        m = TriangularPFCModel.for_field(field, grid.dx, k1, x_bounds=grid.x_bounds,
                                         y_bounds=grid.y_bounds, c2_h=c2_h, eta=eta, chi=1.0)
        res = m.relax(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)
        pk = reconstruct_peaks(m.n, grid, 0.7 * nn)
        ch, ni = coord_hist(pk)
        frac6 = ch.get(6, 0) / max(ni, 1)
        print(f"  c2_h={c2_h} eta={eta}: conv={res['converged']} steps={res['steps']} "
              f"healthy={m.check_density_health()} nmax={np.abs(m.n).max():.2f} "
              f"npk={len(pk)} coord={ch} frac6={frac6:.2f}")


if __name__ == "__main__":
    main()
