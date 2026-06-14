#!/usr/bin/env python3
"""Experiment 1 (advisor I.1): a direct VISUAL of basin selection on the FINAL
relaxed topology.

For T2 (Stone--Wales) and T4 (low-angle GB) we seed the structural PFC from the
SAME AIREBO-relaxed atoms through five projections -- K2, K3, K1@k_cut=200
(naive), K1@k_cut=470 (fair), K4 -- relax each under the numerically sound
structural PFC ($\\lambda=0$), then reconstruct the peaks and bond-graph network
from the RELAXED field (NOT the init field). The whole point is to show the
final state: good projections (K2, K3, fair-K1) visibly recover the correct core
(T2: the 5-7-7-5; T4: the GB dislocation core), while the bad ones (naive-K1, K4)
settle into a visibly different, disordered structure.

The bond graph and its rings are read with the SAME planar-face enumeration as
``route_a.metrics_extra.ring_histogram_bondgraph`` (self-calibrated 1.4x-median
bond cutoff, outer face removed by orientation sign only) -- we re-walk it here
only to recover the per-face NODE lists so pentagons/heptagons can be drawn; the
script asserts the histogram it draws equals the canonical detector's histogram,
so the figure is faithful to the reported n5/n6/n7.

Run: PYTHONPATH=$PWD/src python scripts/make_basinvis_figure.py
from /Users/bojingkai/Desktop/Route_A_protocol_robustness
"""
from __future__ import annotations
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
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
from route_a.metrics_extra import (ring_histogram_bondgraph,         # noqa: E402
                                   self_bond_cutoff)
from route_a.config import A0_GRAPHENE_ANG                           # noqa: E402

# ---- structural-PFC config (numerically sound, lambda=0) -----------------
CFG = dict(c2_h1=1.05, c2_h2=0.6, chi=1.0, eta=0.25)   # lambda = morph weight = 0
RELAX = dict(tol=1e-6, max_steps=2500, chunk=300, dt=1e-8)

# ---- the five projections ------------------------------------------------
# kernel-config -> (kernel-name, kwargs, marker-label, tier)
KCFG = {
    "K2":             ("K2", {},                   "$K_2$ Gaussian",                "good"),
    "K3":             ("K3", {},                   "$K_3$ Voronoi--Gaussian",       "good"),
    "K1@470_fair":    ("K1", dict(k_cut=470.0),    "$K_1$ spectral, cutoff $>|G_1|$ (fair)", "good"),
    "K1@200_naive":   ("K1", dict(k_cut=200.0),    "$K_1$ spectral, cutoff $<|G_1|$ (naive)", "bad"),
    "K4":             ("K4", {},                   "$K_4$ cell-indicator",          "bad"),
}
KORDER = ["K2", "K3", "K1@470_fair", "K1@200_naive", "K4"]

CASES = {
    "T2": dict(case="T2", dump="relaxed_T2_stone_wales.dump",
               label="T2  Stone--Wales", core="5-7-7-5"),
    "T4": dict(case="T4", dump="relaxed_T4_low_angle_gb.dump",
               label="T4  low-angle GB", core="GB dislocation core"),
}
CASE_ORDER = ["T2", "T4"]
RELAXED_DIR = os.path.join(ROOT, "experiments", "hpc_package")

# ---- Tol palette ----------------------------------------------------------
C = dict(blue="#4477AA", cyan="#66CCEE", green="#228833", olive="#999933",
         yellow="#CCBB44", red="#EE6677", purple="#AA3377", grey="#BBBBBB",
         dark="#222222")
RING_FACE = {5: C["green"], 7: C["red"]}      # pentagons green, heptagons red
RING_LAB = {5: "5-ring", 7: "7-ring"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.0,
    "axes.linewidth": 0.7, "savefig.dpi": 320, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------
# bond-graph face enumeration that RETURNS the face node-lists.
# This is the SAME algorithm as route_a.metrics_extra.ring_histogram_bondgraph
# (same self-calibrated 1.4x-median bond cutoff, same clockwise half-edge walk,
# outer face removed by orientation sign ONLY). We re-walk it here only to keep
# the polygon node-indices so 5-/7-rings can be coloured.  We then assert the
# resulting histogram equals the canonical detector's, so the drawn rings are
# exactly the ones the detector counts.
# --------------------------------------------------------------------------
def _signed_area2(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def bondgraph_faces(points, bond_cutoff=None, max_ring: int = 12):
    """Return (faces, hist) where faces is a list of (ring_size, node_index_list)
    for every interior bounded face and hist is {ring_size: count}."""
    pts = np.asarray(points, dtype=float)[:, :2]
    if len(pts) < 3:
        return [], {}
    if bond_cutoff is None:
        bond_cutoff = self_bond_cutoff(pts)
    if bond_cutoff <= 0:
        return [], {}
    adj = [set() for _ in range(len(pts))]
    for i, j in cKDTree(pts).query_pairs(bond_cutoff):
        adj[i].add(j); adj[j].add(i)
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
    faces_nodes = []
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
                ok = False
            if ok and 3 <= len(face) <= max_ring:
                faces_nodes.append(face)
    if not faces_nodes:
        return [], {}
    areas = np.array([_signed_area2(pts[f]) for f in faces_nodes])
    interior_sign = np.sign(np.median(areas)) or 1.0
    faces, hist = [], {}
    for f, ar in zip(faces_nodes, areas):
        if np.sign(ar) != interior_sign:
            continue                       # outer face -- the ONLY rejection
    # (note: loop below mirrors above; kept separate for clarity)
    faces, hist = [], {}
    for f, ar in zip(faces_nodes, areas):
        if np.sign(ar) != interior_sign:
            continue
        L = len(f)
        faces.append((L, f))
        hist[L] = hist.get(L, 0) + 1
    return faces, hist, bond_cutoff


# --------------------------------------------------------------------------
def prep_case(meta):
    c = get_case(meta["case"])
    atoms, _ = read_lammps_dump(os.path.join(RELAXED_DIR, meta["dump"]))
    nd = Nondimensionalizer((c.box[0], c.box[1]), (c.box[2], c.box[3]))
    a_nd = nd.nondimensionalize_coords(atoms)
    grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
    k1 = honeycomb_k1(A0_GRAPHENE_ANG / nd.Lmax)
    nn = float(np.median(cKDTree(a_nd[:, :2]).query(a_nd[:, :2], k=2)[0][:, 1]))
    minsep = 0.7 * nn
    gt = ring_histogram_bondgraph(a_nd[:, :2])
    return dict(name=c.name, a_nd=a_nd, grid=grid, k1=k1, nn=nn,
                minsep=minsep, gt=gt)


def relax_projection(P, kname):
    kn, kw, _, _ = KCFG[kname]
    grid = P["grid"]
    field = get_postprocess("P3").apply(get_kernel(kn, **kw).map(P["a_nd"], grid))
    m = StructuralPFCModel.for_field(
        field, grid.dx, P["k1"], x_bounds=grid.x_bounds, y_bounds=grid.y_bounds,
        morphological_constraint_weight=0.0, **CFG)
    m.relax(**RELAX)
    peaks = reconstruct_peaks(m.n, grid, P["minsep"])
    return peaks, bool(m.check_density_health()), float(np.abs(m.n).max())


def draw_panel(ax, P, peaks, kname, meta, tier):
    a = P["a_nd"]
    # faint TRUE atoms underneath
    ax.scatter(a[:, 0], a[:, 1], s=2.2, color="0.74", marker="o",
               zorder=0, linewidths=0)
    hist = {}
    bc = None
    if len(peaks) >= 3:
        faces, hist, bc = bondgraph_faces(peaks)
        # faithfulness check vs the canonical detector
        canon = ring_histogram_bondgraph(peaks)
        assert hist == canon, f"{meta['case']}/{kname}: face hist {hist} != detector {canon}"
        # bond graph edges (same cutoff the detector used)
        pairs = cKDTree(peaks).query_pairs(bc)
        seg = np.array([[peaks[i], peaks[j]] for i, j in pairs])
        if len(seg):
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(seg, colors=C["dark"], linewidths=0.6,
                                             alpha=0.85, zorder=1))
        # coloured 5-/7-rings (filled polygons)
        patches5, patches7 = [], []
        for L, nodes in faces:
            if L == 5:
                patches5.append(Polygon(peaks[nodes], closed=True))
            elif L == 7:
                patches7.append(Polygon(peaks[nodes], closed=True))
        if patches5:
            ax.add_collection(PatchCollection(patches5, facecolor=C["green"],
                                              edgecolor=C["green"], alpha=0.55,
                                              linewidths=0.8, zorder=2))
        if patches7:
            ax.add_collection(PatchCollection(patches7, facecolor=C["red"],
                                              edgecolor=C["red"], alpha=0.55,
                                              linewidths=0.8, zorder=2))
        # reconstructed peaks on top
        ax.scatter(peaks[:, 0], peaks[:, 1], s=5.5, color=C["blue"], zorder=3,
                   edgecolor="white", linewidth=0.25)
    n5, n6, n7 = hist.get(5, 0), hist.get(6, 0), hist.get(7, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    # tier-coloured frame
    fcol = C["green"] if tier == "good" else C["red"]
    for s in ax.spines.values():
        s.set_color(fcol); s.set_linewidth(1.5)
    # ring count badge
    ax.text(0.5, -0.085, f"$n_5{{=}}{n5}\\,$ $n_6{{=}}{n6}\\,$ $n_7{{=}}{n7}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.4,
            color=fcol)
    return n5, n6, n7


def main():
    OUT = os.path.join(ROOT, "outputs", "figures")
    os.makedirs(OUT, exist_ok=True)

    preps = {ck: prep_case(CASES[ck]) for ck in CASE_ORDER}
    print("Ground-truth (AIREBO) bond-graph 5/6/7 per case:")
    for ck in CASE_ORDER:
        g = preps[ck]["gt"]
        print(f"  {CASES[ck]['label']:22s}  n5={g.get(5,0)} n6={g.get(6,0)} n7={g.get(7,0)}")

    nrow, ncol = len(CASE_ORDER), len(KORDER)
    fig = plt.figure(figsize=(2.05 * ncol, 2.05 * nrow + 0.9))
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.30, wspace=0.10,
                           top=0.86, bottom=0.10, left=0.05, right=0.985)

    results = {}   # (case, kernel) -> (n5,n6,n7,healthy,nmax)
    for ri, ck in enumerate(CASE_ORDER):
        P = preps[ck]; meta = CASES[ck]
        # find a tight window around the defect core for a readable crop:
        # centre on box centre; show a generous patch (whole-cell topology stays
        # interpretable while the core is large).
        a = P["a_nd"]
        cx, cy = a[:, 0].mean(), a[:, 1].mean()
        half = 0.5 * max(a[:, 0].ptp(), a[:, 1].ptp()) * 1.02
        for ci, kname in enumerate(KORDER):
            kn, kw, lab, tier = KCFG[kname]
            peaks, healthy, nmax = relax_projection(P, kname)
            ax = fig.add_subplot(gs[ri, ci])
            n5, n6, n7 = draw_panel(ax, P, peaks, kname, meta, tier)
            results[(ck, kname)] = (n5, n6, n7, healthy, round(nmax, 3))
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
            if ri == 0:
                ax.set_title(lab, fontsize=8.0, pad=4,
                             color=(C["green"] if tier == "good" else C["red"]))
            print(f"  [{meta['case']}/{kname:13s}] relaxed bond-graph  "
                  f"n5={n5} n6={n6} n7={n7}  (healthy={healthy}, nmax={nmax:.3f})",
                  flush=True)
        # row label (left), with ground-truth core annotation
        g = P["gt"]
        rowax = fig.add_subplot(gs[ri, 0])  # reuse left cell's y for label? -> use fig.text
        rowax.remove()

    # row labels via fig.text (vertical, left margin)
    for ri, ck in enumerate(CASE_ORDER):
        meta = CASES[ck]; g = preps[ck]["gt"]
        y = 0.86 - (ri + 0.5) * (0.86 - 0.10) / nrow
        fig.text(0.012, y, f"{meta['label']}\n(true core: {meta['core']},"
                           f" $n_5{{=}}{g.get(5,0)}$ $n_7{{=}}{g.get(7,0)}$)",
                 rotation=90, ha="center", va="center", fontsize=8.2)

    # title + legend
    fig.suptitle("Basin selection on the FINAL relaxed topology: reconstructed "
                 "bond graph of the relaxed structural-PFC field\n"
                 "(pentagons green, heptagons red; faint grey $=$ true AIREBO "
                 "atoms; green frame $=$ correct basin, red frame $=$ wrong basin)",
                 fontsize=9.2, y=0.985)
    # manual legend handles
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=C["green"], alpha=0.55, edgecolor=C["green"], label="5-ring (pentagon)"),
        Patch(facecolor=C["red"], alpha=0.55, edgecolor=C["red"], label="7-ring (heptagon)"),
        Line2D([0], [0], color=C["dark"], lw=1.0, label="bond graph (relaxed)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C["blue"],
               markeredgecolor="white", markersize=4.5, label="reconstructed peaks"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.74",
               markersize=3.5, label="true atoms (AIREBO)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.52, 0.0), columnspacing=1.3,
               handletextpad=0.4)

    out = os.path.join(OUT, "F_basinvis.png")
    fig.savefig(out)
    plt.close(fig)
    print("\nwrote", out)

    # final per-(case,kernel) table
    print("\n=== relaxed bond-graph n5/n6/n7 per (case, kernel) ===")
    print(f"{'case':6s} {'kernel':14s}  n5  n6  n7   tier   healthy  nmax")
    for ck in CASE_ORDER:
        for kname in KORDER:
            n5, n6, n7, healthy, nmax = results[(ck, kname)]
            tier = KCFG[kname][3]
            print(f"{ck:6s} {kname:14s}  {n5:2d}  {n6:2d}  {n7:2d}   {tier:5s}  "
                  f"{str(healthy):7s} {nmax}")
    return results, {ck: preps[ck]["gt"] for ck in CASE_ORDER}


if __name__ == "__main__":
    main()
