#!/usr/bin/env python3
"""Mechanism figure (PRM): WHY the kernels differ + WHY raw/strict are inadmissible.
Top 2x2: psi_init zoom + reconstructed peaks/bonds for the spectral K1 vs the
Voronoi-Gaussian K3, around a Stone-Wales defect. Bottom: field-amplitude
distributions for P1/P2/P3/P4 vs the model's bounded-amplitude stable band."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))
from route_a.cases import get_case
from route_a.lammps_io import read_lammps_dump
from route_a.nondim import Nondimensionalizer, setup_grid
from route_a.kernels import get_kernel
from route_a.postprocess import get_postprocess
from route_a.metrics import reconstruct_peaks

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.linewidth": 0.7, "savefig.dpi": 320, "savefig.bbox": "tight",
})
C = dict(blue="#4477AA", red="#EE6677", green="#228833", yellow="#CCBB44", purple="#AA3377")


def main():
    c = get_case("T2")
    atoms, _ = read_lammps_dump(os.path.join(ROOT, "experiments/hpc_package/relaxed_T2_stone_wales.dump"))
    nd = Nondimensionalizer((c.box[0], c.box[1]), (c.box[2], c.box[3]))
    a = nd.nondimensionalize_coords(atoms)
    grid = setup_grid(nd.Lx_nd, nd.Ly_nd, 512)
    X = np.arange(grid.NX) * grid.dx
    Y = np.arange(grid.NY) * grid.dx
    cx, cy = a[:, 0].mean(), a[:, 1].mean()
    nn = float(np.median(cKDTree(a[:, :2]).query(a[:, :2], k=2)[0][:, 1]))
    half = 9 * nn
    bond = 1.45 * nn

    fig = plt.figure(figsize=(6.9, 5.4))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.32, wspace=0.18)

    def panel_field(ax, field, title):
        ax.imshow(field, origin="lower", extent=[0, X[-1], 0, Y[-1]], cmap="magma",
                  vmin=np.percentile(field, 2), vmax=np.percentile(field, 99.5))
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)

    def panel_recon(ax, field, title):
        p = reconstruct_peaks(field, grid, 0.7 * nn)
        m = (np.abs(p[:, 0] - cx) < half) & (np.abs(p[:, 1] - cy) < half)
        pp = p[m]
        if len(pp) > 2:
            for i, j in cKDTree(pp).query_pairs(bond):
                ax.plot([pp[i, 0], pp[j, 0]], [pp[i, 1], pp[j, 1]], "-", color=C["blue"], lw=0.8, zorder=1)
            ax.scatter(pp[:, 0], pp[:, 1], s=14, color=C["red"], zorder=2, edgecolor="white", linewidth=0.3)
        ax.scatter(a[:, 0], a[:, 1], s=5, color="0.6", marker="x", zorder=0)
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)

    fK1 = get_postprocess("P3").apply(get_kernel("K1").map(a, grid))
    fK3 = get_postprocess("P3").apply(get_kernel("K3").map(a, grid))
    panel_field(fig.add_subplot(gs[0, 0]), fK1, "(a) $K_1$ spectral: $\\psi_{\\rm init}$")
    panel_field(fig.add_subplot(gs[0, 1]), fK3, "(b) $K_3$ Voronoi--Gaussian: $\\psi_{\\rm init}$")
    panel_recon(fig.add_subplot(gs[1, 0]), fK1, "(c) $K_1$ reconstructed peaks + bonds")
    panel_recon(fig.add_subplot(gs[1, 1]), fK3, "(d) $K_3$ reconstructed peaks + bonds")

    # (e) initial-field amplitude RANGE per post-processing vs the model's stable band.
    # P2 needs a genuine reference: rho0 = the perfect crystal (T0) mapped through K3,
    # so psi = rho/rho0 - 1 actually diverges where rho0 is small (using rho0=rho would
    # give psi == 0 and hide the inadmissibility).
    ax = fig.add_subplot(gs[2, :])
    rhoT2 = get_kernel("K3").map(a, grid)
    atoms0, _ = read_lammps_dump(os.path.join(ROOT, "experiments/hpc_package/relaxed_T0_pristine.dump"))
    a0 = nd.nondimensionalize_coords(atoms0)
    rho0 = get_kernel("K3").map(a0, grid)
    items = [
        ("$P_1$ raw", rhoT2),
        ("$P_2$ strict", get_postprocess("P2").apply(rhoT2, rho0=rho0)),
        ("$P_3$ rescale", get_postprocess("P3").apply(rhoT2)),
        ("$P_4$ clip", get_postprocess("P4").apply(rhoT2)),
    ]
    band = 1.5  # bounded-amplitude stable band |n| <~ 1.5 (quartic free energy)
    ax.axvspan(-band, band, color=C["green"], alpha=0.13, zorder=0)
    ax.axvline(band, color=C["green"], lw=0.8, ls=(0, (4, 2)), alpha=0.7, zorder=1)
    yticks, ylabels = [], []
    for i, (lab, f) in enumerate(items):
        v = f.ravel()
        lo, q1, med, q3, hi = np.percentile(v, [0.1, 25, 50, 75, 99.9])
        y = len(items) - i
        yticks.append(y); ylabels.append(lab)
        escapes = (hi > band) or (lo < -band)
        col = C["red"] if escapes else C["green"]
        ax.plot([max(lo, -2.8), hi], [y, y], color=col, lw=1.3, solid_capstyle="round", zorder=2)
        ax.plot([q1, q3], [y, y], color=col, lw=7, alpha=0.45, solid_capstyle="round", zorder=2)
        ax.scatter([med], [y], facecolor="white", edgecolor=col, s=30, linewidth=1.3, zorder=4)
        ax.text(hi * (2.2 if escapes else 1.0) + (0 if escapes else 0.4), y,
                ("escapes band" if escapes else "in band"),
                ha="left", va="center", fontsize=6.6, color=col, style="italic")
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.7)
    ax.set_xlim(-3, 4e4)
    ax.set_ylim(0.45, len(items) + 0.75)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("initial-field amplitude $n$  (symlog; box $=$ interquartile, whisker $=$ 0.1--99.9\\%)")
    ax.set_title("(e) Post-processing controls admissibility", fontsize=9)
    ax.text(0, len(items) + 0.5, "model's stable band", fontsize=7, color="0.30", ha="center", va="center")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = os.path.join(ROOT, "outputs", "figures", "F7_mechanism.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
