"""Benchmark case suite T0–T5 — atomic input structures.

Generators produce *unrelaxed* honeycomb structures (physical Å). The HPC stage
relaxes them with AIREBO (0 K minimisation; short finite-T MD for the R1
reference). The relaxed structure is what the MD→ψ protocol then maps.

* **T0** pristine honeycomb (~1024 atoms) — fidelity.
* **T1** single vacancy — local energetics.
* **T2** Stone–Wales (5-7-7-5) — bond rotation, stress field.
* **T3** 5-7 dislocation dipole — long-range 1/r stress (built in builders_57).
* **T4** 5° low-angle tilt GB — Read–Shockley (built in builders_gb).
* **T5** 15° high-angle tilt GB — high defect density (built in builders_gb).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .config import BOND_GRAPHENE_ANG

# 4-atom rectangular graphene cell (units of a_cc): tiles to honeycomb.
_CELL = np.array([(0.0, 0.0), (1.0, 0.0), (1.5, np.sqrt(3) / 2), (2.5, np.sqrt(3) / 2)])
_CELL_X = 3.0           # cell width  (a_cc units)
_CELL_Y = np.sqrt(3.0)  # cell height (a_cc units)


@dataclass
class Case:
    name: str
    atoms: np.ndarray                       # (N, 3) physical Å, z=0
    box: Tuple[float, float, float, float, float, float]  # xlo,xhi,ylo,yhi,zlo,zhi
    description: str
    defect_centers: List[Tuple[float, float]] = field(default_factory=list)
    perfect_atoms: Optional[np.ndarray] = None   # pristine reference (R2), same box
    periodic: Tuple[bool, bool, bool] = (True, True, False)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


def honeycomb_lattice(
    nx_cells: int, ny_cells: int, a_cc: float = BOND_GRAPHENE_ANG, z_vacuum: float = 20.0
):
    """Return (atoms (N,3), box) for a periodic honeycomb patch."""
    pts = []
    for i in range(nx_cells):
        for j in range(ny_cells):
            pts.append(_CELL + np.array([_CELL_X * i, _CELL_Y * j]))
    xy = np.vstack(pts) * a_cc
    atoms = np.column_stack([xy, np.zeros(len(xy))])
    Lx = _CELL_X * nx_cells * a_cc
    Ly = _CELL_Y * ny_cells * a_cc
    box = (0.0, Lx, 0.0, Ly, -z_vacuum, z_vacuum)
    return atoms, box


def _nearest_index(atoms: np.ndarray, point: Tuple[float, float]) -> int:
    d = (atoms[:, 0] - point[0]) ** 2 + (atoms[:, 1] - point[1]) ** 2
    return int(np.argmin(d))


# --------------------------------------------------------------------------
# T0 / T1 / T2
# --------------------------------------------------------------------------
def make_T0(nx_cells: int = 16, ny_cells: int = 16, a_cc: float = BOND_GRAPHENE_ANG) -> Case:
    atoms, box = honeycomb_lattice(nx_cells, ny_cells, a_cc)
    return Case(
        name="T0_pristine",
        atoms=atoms,
        box=box,
        description=f"pristine honeycomb, {len(atoms)} atoms",
        perfect_atoms=atoms.copy(),
    )


def make_T1(nx_cells: int = 16, ny_cells: int = 16, a_cc: float = BOND_GRAPHENE_ANG) -> Case:
    atoms, box = honeycomb_lattice(nx_cells, ny_cells, a_cc)
    cx, cy = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    idx = _nearest_index(atoms, (cx, cy))
    removed = tuple(atoms[idx, :2])
    kept = np.delete(atoms, idx, axis=0)
    return Case(
        name="T1_vacancy",
        atoms=kept,
        box=box,
        description=f"single vacancy, {len(kept)} atoms",
        defect_centers=[removed],
        perfect_atoms=atoms.copy(),
    )


def make_T2(nx_cells: int = 16, ny_cells: int = 16, a_cc: float = BOND_GRAPHENE_ANG) -> Case:
    """Stone–Wales: rotate the central C–C bond by 90° → 5-7-7-5."""
    atoms, box = honeycomb_lattice(nx_cells, ny_cells, a_cc)
    cx, cy = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    i = _nearest_index(atoms, (cx, cy))
    # nearest neighbour of i (bond partner) at distance ~ a_cc
    d = np.sqrt(((atoms[:, :2] - atoms[i, :2]) ** 2).sum(1))
    d[i] = np.inf
    j = int(np.argmin(d))
    mid = 0.5 * (atoms[i, :2] + atoms[j, :2])
    new = atoms.copy()
    for k in (i, j):
        v = atoms[k, :2] - mid
        new[k, 0] = mid[0] - v[1]   # 90° rotation about midpoint
        new[k, 1] = mid[1] + v[0]
    return Case(
        name="T2_stone_wales",
        atoms=new,
        box=box,
        description="Stone–Wales 5-7-7-5 (90° bond rotation)",
        defect_centers=[tuple(mid)],
        perfect_atoms=atoms.copy(),
    )


# --------------------------------------------------------------------------
# T3 / T4 / T5 — extended defects (relaxed by LAMMPS to final form)
# --------------------------------------------------------------------------
def _rotate_about(pts: np.ndarray, deg: float, center: np.ndarray) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    Rm = np.array([[c, -s], [s, c]])
    out = pts.copy()
    out[:, :2] = (pts[:, :2] - center) @ Rm.T + center
    return out


def _prune_overlaps(atoms: np.ndarray, min_dist: float) -> np.ndarray:
    tree = cKDTree(atoms[:, :2])
    remove = set()
    for i, j in tree.query_pairs(min_dist):
        if i in remove or j in remove:
            continue
        remove.add(j)
    keep = [k for k in range(len(atoms)) if k not in remove]
    return atoms[keep]


def make_T3(
    seg_cells: int = 4, nx_cells: int = 18, ny_cells: int = 12, a_cc: float = BOND_GRAPHENE_ANG
) -> Case:
    """5-7 dislocation dipole: remove a finite horizontal atom segment; the two
    ends relax (LAMMPS) into a 5-7 | 7-5 dipole with long-range 1/r stress."""
    atoms, box = honeycomb_lattice(nx_cells, ny_cells, a_cc)
    cx, cy = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    seg = 0.5 * seg_cells * _CELL_X * a_cc
    band = (np.abs(atoms[:, 1] - cy) < 0.45 * a_cc) & (np.abs(atoms[:, 0] - cx) < seg)
    kept = atoms[~band]
    return Case(
        name="T3_dislocation_dipole",
        atoms=kept,
        box=box,
        description=f"5-7 dislocation dipole (removed {int(band.sum())}-atom segment; relax with LAMMPS)",
        defect_centers=[(cx - seg, cy), (cx + seg, cy)],
        perfect_atoms=atoms.copy(),
    )


def build_tilt_gb(
    theta_deg: float,
    name: str,
    nx_cells: int = 18,
    ny_cells: int = 12,
    a_cc: float = BOND_GRAPHENE_ANG,
    prune_frac: float = 0.6,
    z_vacuum: float = 20.0,
) -> Case:
    """Symmetric tilt grain boundary by stitching two grains rotated by ±θ/2.

    Periodic box (two interfaces under PBC at x=0 and x=Lx/2). Overlaps at the
    interface are pruned; AIREBO minimisation establishes the relaxed 5-7
    dislocation-array GB. The angle is nominal (not restricted to CSL values)."""
    Lx = _CELL_X * nx_cells * a_cc
    Ly = _CELL_Y * ny_cells * a_cc
    box = (0.0, Lx, 0.0, Ly, -z_vacuum, z_vacuum)
    center = np.array([Lx / 2.0, Ly / 2.0])
    pad = 8
    src, _ = honeycomb_lattice(nx_cells + 2 * pad, ny_cells + 2 * pad, a_cc, z_vacuum)
    src[:, 0] -= pad * _CELL_X * a_cc
    src[:, 1] -= pad * _CELL_Y * a_cc
    left = _rotate_about(src, +theta_deg / 2.0, center)
    right = _rotate_about(src, -theta_deg / 2.0, center)
    lm = (left[:, 0] >= 0) & (left[:, 0] <= Lx / 2) & (left[:, 1] >= 0) & (left[:, 1] <= Ly)
    rm = (right[:, 0] > Lx / 2) & (right[:, 0] <= Lx) & (right[:, 1] >= 0) & (right[:, 1] <= Ly)
    merged = np.vstack([left[lm], right[rm]])
    merged = _prune_overlaps(merged, prune_frac * a_cc)
    merged[:, 2] = 0.0
    perfect, _ = honeycomb_lattice(nx_cells, ny_cells, a_cc)
    return Case(
        name=name,
        atoms=merged,
        box=box,
        description=f"symmetric tilt GB θ≈{theta_deg}° (nominal; relax with LAMMPS)",
        defect_centers=[(Lx / 2, Ly / 2), (0.0, Ly / 2)],
        perfect_atoms=perfect,
    )


def make_T4(nx_cells: int = 18, ny_cells: int = 12, a_cc: float = BOND_GRAPHENE_ANG) -> Case:
    return build_tilt_gb(5.0, "T4_low_angle_gb", nx_cells, ny_cells, a_cc)


def make_T5(nx_cells: int = 18, ny_cells: int = 12, a_cc: float = BOND_GRAPHENE_ANG) -> Case:
    return build_tilt_gb(15.0, "T5_high_angle_gb", nx_cells, ny_cells, a_cc)


# Registry
CASES: Dict[str, Callable[..., Case]] = {
    "T0": make_T0,
    "T1": make_T1,
    "T2": make_T2,
    "T3": make_T3,
    "T4": make_T4,
    "T5": make_T5,
}


def get_case(name: str, **kwargs) -> Case:
    key = name.split("_")[0] if name not in CASES else name
    if key not in CASES:
        raise KeyError(f"unknown case {name!r}; available: {sorted(CASES)}")
    return CASES[key](**kwargs)
