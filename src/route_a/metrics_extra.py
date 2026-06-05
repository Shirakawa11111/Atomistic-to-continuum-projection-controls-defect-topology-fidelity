"""Independent fidelity metrics for the protocol-robustness audit.

The headline ``metrics.ring_l1_init`` rests on ONE ring detector
(``metrics.ring_histogram``: a per-edge shortest-cycle / SSSR search on a
bond-cutoff graph). A sceptical reviewer could argue the two-tier protocol
ranking is an artefact of that single detector. This module provides
*algorithmically independent* fidelity probes so the ranking can be re-tested:

(a) ``ring_histogram_delaunay`` — a ring-size histogram from **planar-face
    traversal of a pruned Delaunay triangulation**. This shares no code path
    with the shortest-cycle detector: it builds the Delaunay simplices, deletes
    triangulation edges longer than a self-calibrating bond cutoff, then walks
    the combinatorial planar embedding (next-edge-clockwise-around-each-vertex)
    to enumerate the bounded faces and read their sizes. Different algorithm,
    different failure modes.

(b) Non-ring *geometry* fidelity vs the ground-truth atoms, none of which
    counts rings:
      * ``coordination_f1`` — F1 of the "3-fold-coordinated" label
        (graphene coordination = 3) between reconstruction and atoms.
      * ``bond_angle_l1`` — normalised L1 distance between bond-angle
        histograms (purely geometric, distribution-level).
      * ``peak_localisation_rms`` — RMS position error after greedy
        nearest-neighbour matching of reconstructed peaks to atoms.

All routines take *non-dimensional* peak / atom coordinates (the same units the
pipeline maps in) so the reconstruction and the ground truth are compared on an
identical footing.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay, cKDTree


# --------------------------------------------------------------------------
# shared: self-calibrating bond cutoff (independent re-derivation)
# --------------------------------------------------------------------------
def self_bond_cutoff(points: np.ndarray, factor: float = 1.4) -> float:
    """``factor`` × median nearest-neighbour distance.

    Re-derived here (not imported from ``metrics``) so this module is a fully
    standalone second opinion; the default ``factor=1.4`` matches the headline
    detector's cutoff convention so any ranking difference is the *algorithm*,
    not the bond definition.
    """
    p = _xy(points)
    if len(p) < 2:
        return 0.0
    d, _ = cKDTree(p).query(p, k=2)
    nn = float(np.median(d[:, 1]))
    return factor * nn


def _xy(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return p[:, :2] if p.ndim == 2 and p.shape[1] > 2 else p


def _bond_adjacency(points: np.ndarray, bond_cutoff: float) -> list:
    """Undirected adjacency list from a KD-tree bond cutoff (shared helper)."""
    adj: list = [[] for _ in range(len(points))]
    for i, j in cKDTree(points).query_pairs(bond_cutoff):
        adj[i].append(j)
        adj[j].append(i)
    return adj


# --------------------------------------------------------------------------
# (a) INDEPENDENT ring detector: planar-face traversal of pruned Delaunay
# --------------------------------------------------------------------------
def ring_histogram_delaunay(
    points: np.ndarray,
    bond_cutoff: Optional[float] = None,
    max_ring: int = 12,
) -> Dict[int, int]:
    """Ring-size histogram via planar-face enumeration of a pruned Delaunay graph.

    Algorithm (distinct from the SSSR shortest-cycle detector):
      1. Delaunay-triangulate the points.
      2. Keep only triangulation edges shorter than ``bond_cutoff`` (the true
         graphene bonds; long "convex-hull-closing" Delaunay edges are dropped).
      3. Treat the surviving edges as a straight-line planar graph and traverse
         its combinatorial embedding: for each directed half-edge (u→v), the
         next half-edge around the face is (v→w) where w is the neighbour of v
         most-clockwise from the incoming direction. Walking until return
         enumerates every bounded face exactly once; its length is the ring size.
      4. Drop the single outer (unbounded) face — identified as the longest face
         and/or any face that encloses all others — so only interior rings count.

    Returns ``{ring_size: count}`` for ``3 <= ring_size <= max_ring``.
    """
    pts = _xy(points)
    if len(pts) < 3:
        return {}
    if bond_cutoff is None:
        bond_cutoff = self_bond_cutoff(pts)
    if bond_cutoff <= 0:
        return {}

    # 1-2. Delaunay edges, pruned by bond length.
    try:
        tri = Delaunay(pts)
    except Exception:
        return {}
    edges = set()
    s = tri.simplices
    for a, b, c in s:
        for u, v in ((a, b), (b, c), (c, a)):
            e = (int(u), int(v)) if u < v else (int(v), int(u))
            edges.add(e)
    bc2 = bond_cutoff * bond_cutoff
    kept = [
        (u, v)
        for (u, v) in edges
        if ((pts[u, 0] - pts[v, 0]) ** 2 + (pts[u, 1] - pts[v, 1]) ** 2) <= bc2
    ]
    if not kept:
        return {}

    # adjacency, with neighbours sorted by polar angle (for clockwise walk)
    nbrs: Dict[int, list] = {}
    for u, v in kept:
        nbrs.setdefault(u, []).append(v)
        nbrs.setdefault(v, []).append(u)
    ang: Dict[Tuple[int, int], float] = {}
    for u, vs in nbrs.items():
        for v in vs:
            ang[(u, v)] = np.arctan2(pts[v, 1] - pts[u, 1], pts[v, 0] - pts[u, 0])
        vs.sort(key=lambda v: ang[(u, v)])

    def next_halfedge(u: int, v: int) -> Tuple[int, int]:
        """Half-edge after (u->v) around the face: at v, take the neighbour
        immediately clockwise from the reverse direction (v->u)."""
        ring = nbrs[v]
        # index of u in v's cyclic neighbour order, step one clockwise
        iu = ring.index(u)
        w = ring[(iu - 1) % len(ring)]
        return v, w

    # 3. walk every directed half-edge exactly once. A face whose length exceeds
    #    max_ring (+slack) before closing is the unbounded outer boundary (or a
    #    huge degenerate face): it is dropped here, so only small bounded faces
    #    survive. Each interior face is enumerated exactly once.
    visited = set()
    faces_nodes = []
    face_areas = []
    for u0, v0 in kept:
        for a, b in ((u0, v0), (v0, u0)):
            if (a, b) in visited:
                continue
            face = [a]
            cu, cv = a, b
            ok = True
            for _ in range(max_ring + 4):  # guard against runaway / outer face
                visited.add((cu, cv))
                face.append(cv)
                cu, cv = next_halfedge(cu, cv)
                if (cu, cv) == (a, b):
                    break
            else:
                ok = False  # longer than max_ring (+slack): outer/degenerate face
            if ok:
                nodes = face[:-1]  # drop the repeated closing node (a ... a)
                if 3 <= len(nodes) <= max_ring:
                    faces_nodes.append(nodes)
                    face_areas.append(_signed_area2(pts[nodes]))

    if not faces_nodes:
        return {}

    # 4. orientation of the interior faces is fixed by the (consistent) half-edge
    #    walk but its sign depends on the coordinate handedness, so detect it from
    #    the data: interior faces share one sign; any opposite-signed face is the
    #    outer boundary (kept short only for tiny/degenerate clusters) and is
    #    dropped. Also drop area-outlier faces (a closed outer hull).
    areas = np.asarray(face_areas)
    interior_sign = np.sign(np.median(areas)) or 1.0
    mag = np.abs(areas)
    # an outer hull, if present, is a gross area outlier; flag via robust z-score
    med = np.median(mag)
    mad = np.median(np.abs(mag - med)) or (med if med > 0 else 1.0)
    hist: Dict[int, int] = {}
    for nodes, area2 in zip(faces_nodes, areas):
        if np.sign(area2) != interior_sign:
            continue  # outer / reversed face
        if abs(area2) - med > 6.0 * 1.4826 * mad:
            continue  # area outlier (degenerate outer hull)
        L = len(nodes)
        hist[L] = hist.get(L, 0) + 1
    return hist


def _signed_area2(poly: np.ndarray) -> float:
    """Twice the signed area of a polygon (shoelace); sign encodes orientation."""
    x = poly[:, 0]
    y = poly[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ring_l1(hist_a: Dict[int, int], hist_b: Dict[int, int]) -> float:
    """Normalised L1 distance between ring histograms (vs ``hist_b`` total).

    Re-derived locally to keep this module independent of ``metrics``; numerically
    identical definition so the two detectors are compared like-for-like.
    """
    keys = set(hist_a) | set(hist_b)
    num = sum(abs(hist_a.get(k, 0) - hist_b.get(k, 0)) for k in keys)
    den = max(sum(hist_b.values()), 1)
    return num / den


def ring_l1_delaunay_vs_truth(
    peaks_nd: np.ndarray, atoms_nd: np.ndarray
) -> Tuple[float, Dict[int, int], Dict[int, int]]:
    """Independent (Delaunay) ring-L1 of reconstructed peaks vs ground-truth atoms.

    Returns ``(ring_l1, hist_peaks, hist_atoms)``.
    """
    h_gt = ring_histogram_delaunay(atoms_nd)
    h_pk = ring_histogram_delaunay(peaks_nd)
    return ring_l1(h_pk, h_gt), h_pk, h_gt


# --------------------------------------------------------------------------
# (a') CORRECTED PRIMARY ring detector: bond-graph planar-face enumeration
# --------------------------------------------------------------------------
def ring_histogram_bondgraph(points, bond_cutoff=None, max_ring: int = 12):
    """Ring histogram = bounded-face sizes of the self-calibrated bond graph's
    planar embedding. This is the trustworthy primary detector: unlike the
    shortest-cycle (SSSR) detector — which can MISS a heptagon when two large
    rings share an edge (e.g. one of the two Stone--Wales heptagons) — and unlike
    the pruned-Delaunay detector — whose area-outlier filter can delete large
    interior rings — this enumerates every bounded face and removes ONLY the outer
    face by orientation sign (no area-magnitude cut), so it recovers the full
    Stone--Wales 5-7-7-5 core ($n_5=2,\\,n_7=2$). For a 2D bonded sheet the bounded
    faces are exactly the rings. Same 1.4x-median bond convention as the others.
    """
    pts = _xy(points)
    if len(pts) < 3:
        return {}
    if bond_cutoff is None:
        bond_cutoff = self_bond_cutoff(pts)
    if bond_cutoff <= 0:
        return {}
    adj = [set() for _ in range(len(pts))]
    for i, j in cKDTree(pts).query_pairs(bond_cutoff):
        adj[i].add(j)
        adj[j].add(i)
    nb = {}
    for u in range(len(pts)):
        vs = list(adj[u])
        vs.sort(key=lambda v: np.arctan2(pts[v, 1] - pts[u, 1], pts[v, 0] - pts[u, 0]))
        nb[u] = vs

    def _next(u, v):
        ring = nb[v]
        iu = ring.index(u)
        return v, ring[(iu - 1) % len(ring)]

    visited = set()
    faces = []
    for u0 in range(len(pts)):
        for v0 in adj[u0]:
            if (u0, v0) in visited:
                continue
            face = []
            cu, cv = u0, v0
            ok = True
            for _ in range(max_ring + 2):
                visited.add((cu, cv))
                face.append(cu)
                cu, cv = _next(cu, cv)
                if (cu, cv) == (u0, v0):
                    break
            else:
                ok = False  # longer than max_ring -> outer boundary, drop
            if ok and 3 <= len(face) <= max_ring:
                faces.append(face)
    if not faces:
        return {}
    areas = np.array([_signed_area2(pts[f]) for f in faces])
    interior_sign = np.sign(np.median(areas)) or 1.0
    hist = {}
    for f, ar in zip(faces, areas):
        if np.sign(ar) != interior_sign:
            continue  # outer face -- the ONLY rejection
        hist[len(f)] = hist.get(len(f), 0) + 1
    return hist


# --------------------------------------------------------------------------
# (b) NON-ring geometry fidelity vs ground-truth atoms
# --------------------------------------------------------------------------
def _coordination(points: np.ndarray, bond_cutoff: float) -> np.ndarray:
    adj = _bond_adjacency(points, bond_cutoff)
    return np.array([len(a) for a in adj], dtype=int)


def coordination_f1(
    peaks_nd: np.ndarray,
    atoms_nd: np.ndarray,
    bond_cutoff: Optional[float] = None,
    target_coord: int = 3,
) -> Dict[str, float]:
    """Coordination-number agreement (graphene = 3-fold), interior atoms only.

    We match reconstructed peaks to ground-truth atoms by greedy nearest
    neighbour, then score the binary label "coordination == ``target_coord``"
    with precision / recall / F1. Returns F1 and a 1-F1 "mismatch" (0 = perfect).
    Free-edge atoms are excluded on both sides (edge undercoordination is not a
    mapping defect).
    """
    pk = _xy(peaks_nd)
    at = _xy(atoms_nd)
    out = dict(coord_f1=float("nan"), coord_f1_mismatch=float("nan"),
               coord_precision=float("nan"), coord_recall=float("nan"),
               n_matched=0)
    if len(pk) < 3 or len(at) < 3:
        return out
    bc_at = bond_cutoff or self_bond_cutoff(at)
    bc_pk = bond_cutoff or self_bond_cutoff(pk)
    if bc_at <= 0 or bc_pk <= 0:
        return out

    coord_at = _coordination(at, bc_at)
    coord_pk = _coordination(pk, bc_pk)

    # interior mask on the atom side (true geometry sets the frame)
    lo = at.min(0) + bc_at
    hi = at.max(0) - bc_at
    interior_at = np.all((at > lo) & (at < hi), axis=1)

    # greedy nearest-neighbour match peaks -> atoms (one-to-one, within bc_at)
    match = _greedy_match(pk, at, max_dist=bc_at)
    # for each interior atom that found a peak, compare the 3-fold label
    y_true, y_pred = [], []
    for ai, pi in match.items():
        if not interior_at[ai]:
            continue
        y_true.append(1 if coord_at[ai] == target_coord else 0)
        y_pred.append(1 if coord_pk[pi] == target_coord else 0)
    if not y_true:
        return out
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    out.update(coord_f1=float(f1), coord_f1_mismatch=float(1.0 - f1),
               coord_precision=float(prec), coord_recall=float(rec),
               n_matched=int(len(y_true)))
    return out


def bond_angle_l1(
    peaks_nd: np.ndarray,
    atoms_nd: np.ndarray,
    bond_cutoff: Optional[float] = None,
    nbins: int = 36,
) -> float:
    """Normalised L1 distance between bond-angle histograms (0..180 deg).

    For every atom, every pair of its bonds defines an angle; the distribution of
    those angles is a purely geometric fingerprint (graphene peaks at 120 deg).
    L1 is normalised so identical distributions give 0 and disjoint give 2 -> we
    return half that, i.e. total-variation in [0,1].
    """
    ha = _bond_angle_hist(_xy(peaks_nd), bond_cutoff, nbins)
    hb = _bond_angle_hist(_xy(atoms_nd), bond_cutoff, nbins)
    if ha is None or hb is None:
        return float("nan")
    return 0.5 * float(np.abs(ha - hb).sum())


def _bond_angle_hist(points: np.ndarray, bond_cutoff: Optional[float], nbins: int):
    if len(points) < 3:
        return None
    bc = bond_cutoff or self_bond_cutoff(points)
    if bc <= 0:
        return None
    adj = _bond_adjacency(points, bc)
    angles = []
    for i, nb in enumerate(adj):
        if len(nb) < 2:
            continue
        v = points[nb] - points[i]
        norm = np.linalg.norm(v, axis=1)
        good = norm > 1e-12
        v = v[good] / norm[good, None]
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                c = float(np.clip(np.dot(v[a], v[b]), -1.0, 1.0))
                angles.append(np.degrees(np.arccos(c)))
    if not angles:
        return None
    h, _ = np.histogram(angles, bins=nbins, range=(0.0, 180.0))
    tot = h.sum()
    return h / tot if tot > 0 else h.astype(float)


def peak_localisation_rms(
    peaks_nd: np.ndarray, atoms_nd: np.ndarray, max_dist: Optional[float] = None
) -> Dict[str, float]:
    """RMS position error of reconstructed peaks vs atoms after greedy NN matching.

    Distances are non-dimensional (same units as the mapping). Reported with the
    matched fraction so a tiny well-placed subset cannot masquerade as a good map.
    """
    pk = _xy(peaks_nd)
    at = _xy(atoms_nd)
    if len(pk) == 0 or len(at) == 0:
        return dict(loc_rms=float("nan"), loc_matched_frac=0.0, n_matched=0)
    if max_dist is None:
        max_dist = self_bond_cutoff(at) if len(at) > 1 else np.inf
    match = _greedy_match(pk, at, max_dist=max_dist)
    if not match:
        return dict(loc_rms=float("nan"), loc_matched_frac=0.0, n_matched=0)
    d2 = [float(np.sum((at[ai] - pk[pi]) ** 2)) for ai, pi in match.items()]
    rms = float(np.sqrt(np.mean(d2)))
    return dict(loc_rms=rms, loc_matched_frac=len(match) / len(at), n_matched=len(match))


def _greedy_match(src: np.ndarray, dst: np.ndarray, max_dist: float) -> Dict[int, int]:
    """Greedy one-to-one nearest-neighbour match dst_index -> src_index.

    Pairs are accepted in increasing distance order; each src and dst used once;
    pairs beyond ``max_dist`` are rejected. Returns ``{dst_idx: src_idx}``.
    """
    if len(src) == 0 or len(dst) == 0:
        return {}
    tree = cKDTree(src)
    k = min(len(src), 6)
    dists, idxs = tree.query(dst, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    cand = []
    for di in range(len(dst)):
        for kk in range(k):
            d = dists[di, kk]
            if np.isfinite(d) and d <= max_dist:
                cand.append((d, di, int(idxs[di, kk])))
    cand.sort()
    used_src = set()
    used_dst = set()
    match: Dict[int, int] = {}
    for d, di, si in cand:
        if di in used_dst or si in used_src:
            continue
        match[di] = si
        used_dst.add(di)
        used_src.add(si)
    return match
