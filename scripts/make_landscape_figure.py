#!/usr/bin/env python3
"""F_landscape -- the projection selects which basin the conserved dynamics starts in.

Consumes outputs/landscape/landscape_long.json (run_landscape_long.py), the full
relaxation of T2 (Stone--Wales) and T4 (low-angle GB) seeded through a GOOD
projection (K3 Voronoi-Gaussian) and a BAD one (K1 spectral at the naive cutoff
k_cut=200 < |G1|), recorded to 360k steps.

Story (per case, two panels):
  (top)  free energy F(step): the faithful seed descends straight into the
         crystalline basin; the unfaithful seed sits on a near-stationary
         INCUBATION plateau (a metastable disordered basin) for ~10^4 steps, then
         NUCLEATES and descends into the SAME crystalline basin.
  (bot)  ring-L1(step) vs ground truth: faithful is in the correct basin from the
         start; unfaithful stays disordered through the incubation, then drops into
         the correct basin at the nucleation step.

The benchmark relaxation budget (2500 steps, dashed) sits INSIDE the unfaithful
seed's incubation -- which is why, at any practical budget, the projection controls
the predicted structure, even though the disordered basin is metastable, not
permanent.

Run: PYTHONPATH=$PWD/src python scripts/make_landscape_figure.py
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")
LAND = os.path.join(ROOT, "outputs", "landscape")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "savefig.dpi": 320, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})
C = dict(green="#228833", red="#EE6677", grey="#999999", dark="#222222", amber="#CC8800")
COLF = 6.9
BUDGET = 2500
NUC = 0.2

CASES = [("T2", "Stone--Wales"), ("T4", "low-angle GB")]
GOOD, BAD = "good_K3", "bad_K1@200"


def _arr(rec, key):
    return np.asarray(rec["steps"], float), np.asarray(rec[key], float)


def _nuc_step(rec):
    s, L = _arr(rec, "ring_l1")
    below = np.where(L < NUC)[0]
    return s[below[0]] if len(below) else None


def main():
    data = json.load(open(os.path.join(LAND, "landscape_long.json")))
    fig, axes = plt.subplots(2, 2, figsize=(COLF, 4.8), sharex="col")

    for j, (ck, title) in enumerate(CASES):
        g, b = data[f"{ck}__{GOOD}"], data[f"{ck}__{BAD}"]
        axF, axL = axes[0, j], axes[1, j]
        nuc = _nuc_step(b)

        # ---- free energy ----
        for rec, col in ((g, C["green"]), (b, C["red"])):
            s, F = _arr(rec, "free_energy")
            axF.plot(np.maximum(s, 300), F, "-", color=col)
        axF.axvline(BUDGET, color=C["dark"], ls=(0, (4, 3)), lw=0.8)
        if nuc:
            axF.axvline(nuc, color=C["amber"], ls=(0, (1, 1.5)), lw=0.9)
        axF.set_xscale("log")
        axF.set_title(f"{ck}: {title}", fontsize=9.5)
        if j == 0:
            axF.set_ylabel("free energy $F$")
        # annotate incubation plateau + nucleation
        s_b, F_b = _arr(b, "free_energy")
        axF.annotate("incubation\n(metastable\ndisordered basin)",
                     xy=(BUDGET * 1.4, F_b[0]), xytext=(330, F_b[0] - 0.12 * (F_b[0] - F_b[-1])),
                     fontsize=6.2, color=C["red"], va="top")
        if nuc:
            axF.annotate("nucleation", xy=(nuc, np.interp(nuc, s_b, F_b)),
                         xytext=(nuc * 1.25, F_b[0] - 0.45 * (F_b[0] - F_b[-1])),
                         fontsize=6.4, color=C["amber"],
                         arrowprops=dict(arrowstyle="->", color=C["amber"], lw=0.8))

        # ---- ring-L1 ----
        for rec, col, lab in ((g, C["green"], "faithful ($K_3$)"),
                              (b, C["red"], "unfaithful ($K_1$, $k_{\\mathrm{cut}}{=}200$)")):
            s, L = _arr(rec, "ring_l1")
            axL.plot(np.maximum(s, 300), L, "-", color=col, label=lab)
        axL.axhspan(0.0, NUC, color=C["green"], alpha=0.08, zorder=0)
        axL.axvline(BUDGET, color=C["dark"], ls=(0, (4, 3)), lw=0.8)
        if nuc:
            axL.axvline(nuc, color=C["amber"], ls=(0, (1, 1.5)), lw=0.9)
        axL.set_xscale("log")
        axL.set_xlabel("relaxation step")
        if j == 0:
            axL.set_ylabel("ring-L1 vs ground truth")
        axL.text(0.97, 0.5, "correct\n(crystalline)\nbasin", transform=axL.transAxes,
                 fontsize=6.2, ha="right", va="center", color=C["green"])

    # budget label (once, top-left panel)
    axes[0, 0].text(BUDGET, axes[0, 0].get_ylim()[1], "benchmark\nbudget", fontsize=6.0,
                    color=C["dark"], ha="center", va="top")
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, columnspacing=1.6,
               handletextpad=0.5, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 1.05))

    out = os.path.join(OUT, "F_landscape.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)
    for ck, _ in CASES:
        g, b = data[f"{ck}__{GOOD}"], data[f"{ck}__{BAD}"]
        print(f"  {ck}: nucleation(bad) @ step {_nuc_step(b)};  "
              f"F_final good={g['F_final']:+.3f} bad={b['F_final']:+.3f};  "
              f"L1_final good={g['ringL1_final']:.3f} bad={b['ringL1_final']:.3f}")


if __name__ == "__main__":
    main()
