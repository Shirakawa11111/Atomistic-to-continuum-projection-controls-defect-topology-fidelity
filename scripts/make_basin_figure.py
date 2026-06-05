#!/usr/bin/env python3
"""New headline figure: the projection selects the defect basin.

In the numerically sound structural PFC (lambda=0), the relaxed defect topology
(bond-graph ring-L1 vs the AIREBO ground truth) separates into a CORRECT-basin
tier and a WRONG-basin tier -- but the split is NOT "Gaussian vs non-Gaussian":
a localized Gaussian (K2,K3) AND a spectral kernel whose cutoff resolves the
lattice (K1 at k_cut=470 > |G1|) all reach the correct 5-7 minimum, whereas the
spectral kernel at its naive cutoff (k_cut=200 < |G1|, a RESOLUTION failure) and
the cell-indicator (K4, an intrinsic PLACEMENT failure) fall into a distinct
structure the conserved dynamics cannot repair. Source: outputs/fair_k1_basin/.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.6,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    "savefig.dpi": 320, "savefig.bbox": "tight", "figure.constrained_layout.use": True,
})
C = dict(blue="#4477AA", cyan="#66CCEE", green="#228833", olive="#999933",
         red="#EE6677", purple="#AA3377", grey="#BBBBBB", dark="#222222")
COLF = 6.9
CASE_LAB = {"T0_pristine": "T0", "T1_vacancy": "T1\nvac", "T2_stone_wales": "T2\nSW",
            "T3_dislocation_dipole": "T3\ndipole", "T4_low_angle_gb": "T4\nlow-$\\theta$",
            "T5_high_angle_gb": "T5\nhigh-$\\theta$"}
CASE_ORDER = ["T0_pristine", "T1_vacancy", "T2_stone_wales",
              "T3_dislocation_dipole", "T4_low_angle_gb", "T5_high_angle_gb"]
# kernel -> (marker, color, label, tier)
KMETA = {
    "K2":             ("o", C["green"], "$K_2$ Gaussian", "good"),
    "K3":             ("s", C["blue"],  "$K_3$ Voronoi-Gaussian", "good"),
    "K1@470_fair":    ("D", C["cyan"],  "$K_1$ spectral, cutoff $>|G_1|$", "good"),
    "K1@200_default": ("v", C["red"],   "$K_1$ spectral, naive cutoff $<|G_1|$", "bad"),
    "K4":             ("^", C["purple"],"$K_4$ cell-indicator", "bad"),
}
KORDER = ["K2", "K3", "K1@470_fair", "K1@200_default", "K4"]


def main():
    df = pd.read_csv(os.path.join(ROOT, "outputs", "fair_k1_basin", "fair_k1_basin.csv"))
    fig, ax = plt.subplots(figsize=(COLF, 3.0))
    ax.axhspan(-0.05, 0.15, color=C["green"], alpha=0.08, zorder=0)
    ax.axhspan(0.5, 3.0, color=C["red"], alpha=0.06, zorder=0)
    dx = np.linspace(-0.26, 0.26, len(KORDER))
    for ki, k in enumerate(KORDER):
        mk, col, lab, tier = KMETA[k]
        ys = [float(df[(df.kernel == k) & (df.case == c)]["ringL1_relaxed"].iloc[0]) for c in CASE_ORDER]
        ax.scatter(np.arange(len(CASE_ORDER)) + dx[ki], ys, marker=mk, s=42,
                   facecolor=col, edgecolor="white", linewidth=0.5, zorder=3, label=lab)
    ax.set_xticks(range(len(CASE_ORDER)))
    ax.set_xticklabels([CASE_LAB[c] for c in CASE_ORDER])
    ax.set_ylim(-0.08, 2.85)
    ax.set_ylabel("relaxed ring-L1  (lower = correct)")
    ax.text(0.015, 0.085, "correct 5--7 basin", transform=ax.transAxes, fontsize=7.5,
            color=C["green"], va="center")
    ax.text(0.015, 0.62, "distinct, unrepaired basin", transform=ax.transAxes, fontsize=7.5,
            color=C["red"], va="center")
    ax.set_title("The projection selects the defect basin (structural PFC, $\\lambda{=}0$)", fontsize=9)
    ax.legend(loc="upper center", ncol=3, columnspacing=1.0, handletextpad=0.3,
              bbox_to_anchor=(0.5, 1.0), fontsize=7.0)
    fig.savefig(os.path.join(OUT, "F_basin.png"))
    plt.close(fig)
    # console summary
    for k in KORDER:
        m = float(df[(df.kernel == k) & (~df.case.str.startswith("T0"))]["ringL1_relaxed"].mean())
        print(f"{k:18s} mean relaxed ring-L1 (T1-T5) = {m:.3f}")
    print("wrote", os.path.join(OUT, "F_basin.png"))


if __name__ == "__main__":
    main()
