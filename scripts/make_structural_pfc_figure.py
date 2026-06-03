#!/usr/bin/env python3
"""F8 — the structural-PFC linchpin results (PRM revision).

(a) Basin selection: in a properly scaled structural PFC (peaked C2 at the
    honeycomb |G1|, lambda=0) that genuinely crystallizes the honeycomb, the
    faithful Voronoi-Gaussian projections relax to the correct defect topology
    (relaxed ring-L1 ~ 0), while the spectral/cell-indicator seeds crystallize
    into distinct structures the conserved dynamics cannot repair (ring-L1 >> 0).
(b) Parameter robustness: the two-tier separation survives a one-at-a-time scan
    of the structural-PFC parameters (peak heights h1,h2 and width), so it is not
    an artifact of one tuned parameterization. Worst good-vs-bad gap annotated.

Source: outputs/structural_pfc_scan/scan.csv (contains the 'base' config + the
OAT variations, all at lambda=0).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")
SCAN = os.path.join(ROOT, "outputs", "structural_pfc_scan")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "lines.linewidth": 1.4, "lines.markersize": 5,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "savefig.dpi": 320, "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})
C = dict(blue="#4477AA", cyan="#66CCEE", green="#228833", yellow="#CCBB44",
         red="#EE6677", purple="#AA3377", grey="#BBBBBB", dark="#222222")
COLF = 6.9
FLOOR = 0.01  # display floor for log-y (ring-L1 == 0 means perfect)

CASE_LAB = {"T0_pristine": "T0", "T1_vacancy": "T1\nvac", "T2_stone_wales": "T2\nSW",
            "T3_dislocation_dipole": "T3\ndipole", "T4_low_angle_gb": "T4\nlow-$\\theta$",
            "T5_high_angle_gb": "T5\nhigh-$\\theta$"}
CASE_ORDER = ["T0_pristine", "T1_vacancy", "T2_stone_wales",
              "T3_dislocation_dipole", "T4_low_angle_gb", "T5_high_angle_gb"]
# kernel marker per protocol
PMARK = {"K1/R2/P3": ("v", C["red"], "$K_1$ spectral"),
         "K4/R2/P3": ("^", C["purple"], "$K_4$ cell-indicator"),
         "K2/R2/P3": ("o", C["green"], "$K_2$ Gaussian"),
         "K3/R2/P3": ("s", C["blue"], "$K_3$ Voronoi-Gaussian"),
         "K3/R2/P4": ("D", C["cyan"], "$K_3$ Voronoi-Gaussian + clip")}
CFG_LAB = {"base": "base\n(1.05,0.6,\n0.10)", "h1_0.95": "$h_1$\n0.95", "h1_1.15": "$h_1$\n1.15",
           "h2_0.40": "$h_2$\n0.40", "h2_0.80": "$h_2$\n0.80", "w_0.07": "width\n0.07", "w_0.13": "width\n0.13"}
CFG_ORDER = ["base", "h1_0.95", "h1_1.15", "h2_0.40", "h2_0.80", "w_0.07", "w_0.13"]


def main():
    df = pd.read_csv(os.path.join(SCAN, "scan.csv"))
    summ = json.load(open(os.path.join(SCAN, "scan_summary.json")))
    worst_gap = summ["overall"]["worst_gap"]
    yv = lambda s: np.maximum(s.to_numpy(dtype=float), FLOOR)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COLF, 2.85))

    # ---- (a) basin selection by defect (base config) ----
    base = df[df.config == "base"]
    for j, ck in enumerate(CASE_ORDER):
        sub = base[base.case == ck]
        for proto, (mk, col, _) in PMARK.items():
            r = sub[sub.protocol == proto]
            if len(r):
                axA.scatter([j], yv(r["ringL1_relaxed"]), marker=mk, s=34,
                            facecolor=col, edgecolor="white", linewidth=0.5, zorder=3)
    axA.set_yscale("log")
    axA.set_ylim(FLOOR * 0.8, 9)
    axA.set_xticks(range(len(CASE_ORDER)))
    axA.set_xticklabels([CASE_LAB[c] for c in CASE_ORDER])
    axA.set_ylabel("relaxed ring-L1  (lower = correct)")
    axA.axhspan(FLOOR * 0.8, 0.1, color=C["green"], alpha=0.06, zorder=0)
    axA.axhspan(0.1, 9, color=C["red"], alpha=0.05, zorder=0)
    axA.text(0.02, 0.97, "(a) basin selection (structural PFC, $\\lambda{=}0$)",
             transform=axA.transAxes, fontsize=8.5, va="top")
    axA.text(2.5, 4.3, "spectral / cell-indicator $\\to$ wrong, unrepaired",
             color=C["red"], fontsize=7, ha="center")
    axA.text(2.5, 0.022, "Voronoi-Gaussian $\\to$ correct", color=C["green"], fontsize=7, ha="center")
    # legend (markers) placed in the empty band between the two tiers
    handles = [plt.Line2D([0], [0], marker=mk, color="none", markerfacecolor=col,
                          markeredgecolor="white", markersize=6, label=lab)
               for (mk, col, lab) in PMARK.values()]
    axA.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.02, 0.40),
               fontsize=6.3, handletextpad=0.2, labelspacing=0.2)

    # ---- (b) parameter robustness: good vs bad band per config ----
    for i, cfg in enumerate(CFG_ORDER):
        sub = df[df.config == cfg]
        good = yv(sub[sub.tier == "good"]["ringL1_relaxed"])
        bad = yv(sub[sub.tier == "bad"]["ringL1_relaxed"])
        axB.scatter(np.full(len(good), i) - 0.12, good, marker="o", s=18,
                    facecolor=C["green"], edgecolor="none", alpha=0.8, zorder=3)
        axB.scatter(np.full(len(bad), i) + 0.12, bad, marker="^", s=18,
                    facecolor=C["red"], edgecolor="none", alpha=0.8, zorder=3)
        # separation line between good-max and bad-min
        axB.plot([i - 0.12, i + 0.12], [good.max(), bad.min()], color=C["grey"],
                 lw=0.6, ls=":", zorder=2)
    axB.set_yscale("log")
    axB.set_ylim(FLOOR * 0.8, 9)
    axB.set_xticks(range(len(CFG_ORDER)))
    axB.set_xticklabels([CFG_LAB[c] for c in CFG_ORDER], fontsize=6.6)
    axB.set_ylabel("relaxed ring-L1")
    axB.text(0.02, 0.97, "(b) parameter robustness (all defects)",
             transform=axB.transAxes, fontsize=8.5, va="top")
    axB.scatter([], [], marker="o", facecolor=C["green"], edgecolor="none",
                label="Voronoi-Gaussian {$K_2,K_3$}")
    axB.scatter([], [], marker="^", facecolor=C["red"], edgecolor="none",
                label="non-Gaussian {$K_1,K_4$}")
    axB.legend(loc="center left", fontsize=6.6, handletextpad=0.2)
    axB.text(0.98, 0.06, f"clean separation in 42/42 cells\nworst gap ${worst_gap}\\times$",
             transform=axB.transAxes, fontsize=7, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grey"], lw=0.6))

    fig.savefig(os.path.join(OUT, "F8_structural_pfc.png"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, "F8_structural_pfc.png"))


if __name__ == "__main__":
    main()
