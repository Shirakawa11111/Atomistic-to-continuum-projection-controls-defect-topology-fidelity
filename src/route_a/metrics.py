"""Comparison metrics for the protocol benchmark.

Headline observables (post-relaxation): formation-energy proxy (model free
energy, computed in the pipeline) and topology retention τ. Secondary: density
conservation, field roughness. Diagnostic: ΔF_init, spectral high-k fraction.

Topology is measured by reconstructing atom positions from field peaks
(self-calibrating bond cutoff) and counting 5-/7-membered rings with a minimum
cycle basis. Ring counts are reported via 2 detectors (cycle-basis and a
coordination-defect proxy); the intersection guards against detector artefacts.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from .nondim import GridSpec


# --- scalar field metrics -------------------------------------------------
def density_conservation(rho: np.ndarray, grid: GridSpec, n_atoms: int) -> float:
    """Relative atom-count error |∫ρ − N| / N."""
    return float(abs(rho.sum() * grid.dx**2 - n_atoms) / max(n_atoms, 1))


def field_roughness(field: np.ndarray, grid: GridSpec) -> float:
    """Mean gradient magnitude ⟨|∇field|⟩."""
    gy, gx = np.gradient(field, grid.dx)
    return float(np.mean(np.hypot(gx, gy)))


def spectral_highk_fraction(
    field: np.ndarray, grid: GridSpec, k_split: Optional[float] = None
) -> float:
    """Fraction of spectral power above ``k_split`` (default: half-Nyquist)."""
    F = np.fft.fft2(field - field.mean())
    P = np.abs(F) ** 2
    kx = np.fft.fftfreq(grid.NX, d=grid.dx) * 2 * np.pi
    ky = np.fft.fftfreq(grid.NY, d=grid.dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.hypot(KX, KY)
    if k_split is None:
        k_split = 0.5 * K.max()
    tot = P.sum()
    if tot <= 0:
        return 0.0
    return float(P[K > k_split].sum() / tot)


def field_l2_change(field_a: np.ndarray, field_b: np.ndarray) -> float:
    """Relative L2 change ‖b−a‖ / ‖a‖ (how much relaxation moved the field)."""
    na = np.linalg.norm(field_a)
    return float(np.linalg.norm(field_b - field_a) / max(na, 1e-300))


# --- atom reconstruction & topology --------------------------------------
def reconstruct_peaks(
    field: np.ndarray, grid: GridSpec, min_sep_nd: float, rel_threshold: float = 0.50
) -> np.ndarray:
    """Local-maxima atom reconstruction, separated by ≥ ``min_sep_nd``.

    The acceptance threshold is ``rel_threshold`` × the *median* local-maximum
    height (robust to a few tall outlier peaks, e.g. K3 defect-modulated spikes
    that would otherwise crush normal peaks below a global-max threshold).
    """
    size = max(int(round(min_sep_nd / grid.dx)), 1)
    mx = maximum_filter(field, size=size, mode="nearest")
    is_max = (field == mx) & (field > 0)
    if not np.any(is_max):
        return np.empty((0, 2))
    med_peak = float(np.median(field[is_max]))
    thr = rel_threshold * med_peak
    ys, xs = np.where(is_max & (field > thr))
    if len(xs) == 0:
        return np.empty((0, 2))
    coords = np.column_stack([xs * grid.dx, ys * grid.dx])
    vals = field[ys, xs]
    # greedy dedup within min_sep (kills plateau over-counting), keep highest peaks
    tree = cKDTree(coords)
    taken = np.zeros(len(coords), dtype=bool)
    keep = []
    for idx in np.argsort(-vals):
        if taken[idx]:
            continue
        keep.append(idx)
        taken[tree.query_ball_point(coords[idx], 0.99 * min_sep_nd)] = True
    return coords[keep]


def _self_bond_cutoff(points: np.ndarray) -> float:
    """1.4× median nearest-neighbour distance (self-calibrating bond cutoff)."""
    if len(points) < 2:
        return 0.0
    d, _ = cKDTree(points).query(points, k=2)
    nn = float(np.median(d[:, 1]))
    return 1.4 * nn


def _shortest_ring_through_edge(adj, src: int, dst: int, max_ring: int):
    """Smallest cycle through edge (src,dst): BFS for the shortest src→dst path
    that avoids the direct edge. Returns a frozenset of node ids, or None."""
    parent = {src: -1}
    q = deque([(src, 0)])
    while q:
        node, d = q.popleft()
        if d >= max_ring:
            continue
        for nb in adj[node]:
            if node == src and nb == dst:   # forbid the direct edge
                continue
            if nb == dst:
                path = [dst, node]
                p = parent[node]
                while p != -1:
                    path.append(p)
                    p = parent[p]
                return frozenset(path) if len(path) <= max_ring else None
            if nb not in parent:
                parent[nb] = node
                q.append((nb, d + 1))
    return None


def ring_histogram(
    points: np.ndarray, bond_cutoff: Optional[float] = None, max_ring: int = 9
) -> Dict[int, int]:
    """Ring-size histogram via fast per-edge shortest-cycle (SSSR-like)."""
    if len(points) < 3:
        return {}
    if bond_cutoff is None:
        bond_cutoff = _self_bond_cutoff(points)
    if bond_cutoff <= 0:
        return {}
    pairs = cKDTree(points).query_pairs(bond_cutoff)
    adj: list = [[] for _ in range(len(points))]
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)
    rings = set()
    for i, j in pairs:
        r = _shortest_ring_through_edge(adj, i, j, max_ring)
        if r is not None:
            rings.add(r)
    hist: Dict[int, int] = {}
    for r in rings:
        L = len(r)
        if 3 <= L <= max_ring:
            hist[L] = hist.get(L, 0) + 1
    return hist


def coordination_defect_count(points: np.ndarray, bond_cutoff: Optional[float] = None) -> int:
    """Detector 2: atoms whose coordination ≠ 3 (graphene-anomalous), interior only."""
    if len(points) < 3:
        return 0
    if bond_cutoff is None:
        bond_cutoff = _self_bond_cutoff(points)
    if bond_cutoff <= 0:
        return 0
    tree = cKDTree(points)
    coord = np.zeros(len(points), dtype=int)
    for i, j in tree.query_pairs(bond_cutoff):
        coord[i] += 1
        coord[j] += 1
    # restrict to interior to avoid free-edge undercoordination
    lo = points.min(0) + bond_cutoff
    hi = points.max(0) - bond_cutoff
    interior = np.all((points > lo) & (points < hi), axis=1)
    return int(np.sum((coord != 3) & interior))


def topology_metrics(
    field_init: np.ndarray,
    field_relaxed: np.ndarray,
    grid: GridSpec,
    min_sep_nd: float,
) -> dict:
    """Ring/topology retention between initial and relaxed fields (2 detectors)."""
    p0 = reconstruct_peaks(field_init, grid, min_sep_nd)
    p1 = reconstruct_peaks(field_relaxed, grid, min_sep_nd)
    h0 = ring_histogram(p0)
    h1 = ring_histogram(p1)
    n57_0 = h0.get(5, 0) + h0.get(7, 0)
    n57_1 = h1.get(5, 0) + h1.get(7, 0)

    def _tau(a: int, b: int) -> float:
        if a > 0:
            return b / a
        return 1.0 if b == 0 else float("nan")

    return dict(
        n_atoms_init=int(len(p0)),
        n_atoms_relaxed=int(len(p1)),
        n5_init=h0.get(5, 0),
        n7_init=h0.get(7, 0),
        n6_init=h0.get(6, 0),
        n5_relaxed=h1.get(5, 0),
        n7_relaxed=h1.get(7, 0),
        n6_relaxed=h1.get(6, 0),
        tau_57=_tau(n57_0, n57_1),                     # detector 1: ring-basis
        coord_defects_init=coordination_defect_count(p0),
        coord_defects_relaxed=coordination_defect_count(p1),  # detector 2: coordination
    )


def ring_l1(hist_a: Dict[int, int], hist_b: Dict[int, int]) -> float:
    """Normalised L1 distance between two ring histograms (vs hist_b total)."""
    keys = set(hist_a) | set(hist_b)
    num = sum(abs(hist_a.get(k, 0) - hist_b.get(k, 0)) for k in keys)
    den = max(sum(hist_b.values()), 1)
    return num / den


def init_fidelity_metrics(
    field_init: np.ndarray, atoms_nd: np.ndarray, grid: GridSpec, min_sep_nd: float
) -> dict:
    """Mapping fidelity at INITIALISATION: how well ψ_init reproduces the true
    (LAMMPS-relaxed atoms') topology. Used because the SC-CH relaxation washes out
    atomic 5-7 topology, so fidelity must be judged on the mapped field, not after.
    """
    pts = atoms_nd[:, :2] if atoms_nd.shape[1] > 2 else atoms_nd
    h_gt = ring_histogram(pts)                     # ground truth from atoms
    p_init = reconstruct_peaks(field_init, grid, min_sep_nd)
    h_init = ring_histogram(p_init)
    n57_gt = h_gt.get(5, 0) + h_gt.get(7, 0)
    n57_init = h_init.get(5, 0) + h_init.get(7, 0)
    return dict(
        n5_gt=h_gt.get(5, 0), n6_gt=h_gt.get(6, 0), n7_gt=h_gt.get(7, 0),
        n_atoms_gt=int(len(pts)), n_atoms_init_recon=int(len(p_init)),
        atom_recon_err=abs(len(p_init) - len(pts)) / max(len(pts), 1),
        ring_l1_init=ring_l1(h_init, h_gt),        # 0 = perfect topology match
        tau_init_57=(n57_init / n57_gt if n57_gt > 0
                     else (1.0 if n57_init == 0 else float("nan"))),
    )
