#!/usr/bin/env python3
"""F10 — best-achievable initialisation ring-L1 by kernel family (hyperparameter scan).

Tests the advisor MUST-FIX: are the spectral (K1) and cell-indicator (K4)
failures INTRINSIC to the family, or just a bad knob? For each family we sweep
its single width/cutoff over a wide band and report the BEST-achievable
init ring-L1 (model-independent: atoms->field->peaks->ring-L1 vs the GT atoms).

(a) Best-achievable ring-L1 per family (bars). K4's piecewise-constant 1/area
    field has an intrinsic FLOOR at ~1.0 (its maxima sit on Voronoi plateaus, not
    atoms) — no smoothing width recovers the topology. K1/K2/K3 can all reach
    ~0.02, so the spectral kernel is NOT intrinsically floored at init-fidelity:
    its default cutoff is simply mis-set below the atomic |G1|.
(b) Full sweep curves. K2/K3 are robust (flat-low over a contiguous band); K4 is
    a flat ceiling (always ~1.0); K1 is a fragile, non-monotonic resonance — low
    only in an isolated high-cutoff window where it degenerates to band-limited
    Dirac deposition (k_cut >> atomic |G1|, dashed line).

Source: outputs/hyperparam_scan/scan.csv (+ summary.json).
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
SCAN = os.path.join(ROOT, "outputs", "hyperparam_scan")
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

FAM_LAB = {"K1": "$K_1$\nspectral", "K2": "$K_2$\nGaussian",
           "K3": "$K_3$\nVoronoi-\nGaussian", "K4": "$K_4$\ncell-\nindicator"}
FAM_COL = {"K1": C["red"], "K2": C["green"], "K3": C["blue"], "K4": C["purple"]}
FAM_MK = {"K1": "v", "K2": "o", "K3": "s", "K4": "^"}
FAM_ORDER = ["K1", "K2", "K3", "K4"]


def main():
    df = pd.read_csv(os.path.join(SCAN, "scan.csv"))
    summ = json.load(open(os.path.join(SCAN, "summary.json")))
    k_atom = summ["median_atomic_k"]
    worst_good = summ["worst_good_band_K2K3"]

    # best-achievable per family over the FULL sweep (band U wide)
    best = {f: float(df[df.family == f]["mean_ringL1"].min()) for f in FAM_ORDER}
    k1_opt = summ["k1_resonance"]
    k4_floor = best["K4"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COLF, 3.25),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.35]))

    # ---- (a) best-achievable ring-L1 per family (bars, log-y) ----
    yb = [max(best[f], FLOOR) for f in FAM_ORDER]
    bars = axA.bar(range(len(FAM_ORDER)), yb,
                   color=[FAM_COL[f] for f in FAM_ORDER], width=0.66,
                   edgecolor="white", linewidth=0.6, zorder=3)
    axA.set_yscale("log")
    axA.set_ylim(FLOOR * 0.8, 3.2)
    axA.set_xticks(range(len(FAM_ORDER)))
    axA.set_xticklabels([FAM_LAB[f] for f in FAM_ORDER], fontsize=7.6)
    axA.set_ylabel("best-achievable init ring-L1")
    # shade tiers
    axA.axhspan(FLOOR * 0.8, 0.1, color=C["green"], alpha=0.06, zorder=0)
    axA.axhspan(0.1, 3.2, color=C["red"], alpha=0.05, zorder=0)
    for b, f in zip(bars, FAM_ORDER):
        axA.text(b.get_x() + b.get_width() / 2, max(best[f], FLOOR) * 1.18,
                 f"{best[f]:.2f}", ha="center", fontsize=7.8, zorder=4)
    axA.text(0.03, 0.97, "(a) best over the swept width/cutoff",
             transform=axA.transAxes, fontsize=8.3, va="top")
    axA.annotate("intrinsic floor\n(no width works)", xy=(3, k4_floor),
                 xytext=(2.05, 0.30), fontsize=6.8, color=C["purple"], ha="center",
                 arrowprops=dict(arrowstyle="->", color=C["purple"], lw=0.8))

    # ---- (b) full sweep curves vs smoothing length sigma (log-log) ----
    for f in FAM_ORDER:
        sub = df[(df.family == f) & (df.regime == "wide")].copy()
        sub = sub.sort_values("sigma_equiv")
        x = sub["sigma_equiv"].to_numpy()
        y = np.maximum(sub["mean_ringL1"].to_numpy(dtype=float), FLOOR)
        axB.plot(x, y, FAM_MK[f] + "-", color=FAM_COL[f], mfc=FAM_COL[f],
                 mec="white", mew=0.4, ms=3.4, lw=1.0, label=FAM_LAB[f].replace("\n", " "))
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("smoothing length $\\sigma$  (= $1/k_{\\mathrm{cut}}$ for $K_1$)")
    axB.set_ylabel("mean init ring-L1  (6 defects)")
    axB.set_ylim(FLOOR * 0.8, 6.0)
    # atomic |G1| -> sigma_atom = 1/k_atom: below this K1 cannot represent atoms
    sig_atom = 1.0 / k_atom
    axB.axvline(sig_atom, color=C["dark"], ls=(0, (4, 3)), lw=0.8, zorder=1)
    axB.text(sig_atom * 1.16, 0.013, "$\\sigma{=}1/|G_1|$", rotation=90,
             fontsize=6.6, ha="left", va="bottom", color=C["dark"])
    axB.axhspan(FLOOR * 0.8, 0.1, color=C["green"], alpha=0.05, zorder=0)
    axB.text(0.03, 0.965, "(b) full hyperparameter sweep",
             transform=axB.transAxes, fontsize=8.3, va="top")
    axB.text(0.96, 0.90, "$K_4$ ceiling $\\approx$ 1", transform=axB.transAxes,
             fontsize=7.0, ha="right", va="center", color=C["purple"])
    axB.text(0.96, 0.30, "$K_1$ low only in a\nfragile resonance", transform=axB.transAxes,
             fontsize=7.0, ha="right", va="center", color=C["red"])

    # shared family legend at the top (frees panel (b) of its crowded bottom legend)
    h, l = axB.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, columnspacing=1.3, handletextpad=0.4,
               fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(os.path.join(OUT, "F10_hyperparam.png"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, "F10_hyperparam.png"))
    print(f"  best per family: " + "  ".join(f"{f}={best[f]:.3f}" for f in FAM_ORDER))
    print(f"  worst K2/K3 in comparable band = {worst_good:.3f}; "
          f"K4 floor = {k4_floor:.3f}; K1 fragile-opt = {k1_opt['wide_opt_mean_ringL1']:.3f} "
          f"at k_cut={k1_opt['wide_opt_kcut']:.0f}")


if __name__ == "__main__":
    main()
