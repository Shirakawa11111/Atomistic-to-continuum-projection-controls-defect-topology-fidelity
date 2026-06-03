#!/usr/bin/env python3
"""F9 — universal vs lattice-specific projection requirements.

Compares the relaxed structural-PFC fidelity error of each kernel on TWO lattices,
each normalized to that lattice's best kernel (so the two metrics are comparable):
  * honeycomb  : ring-L1 (outputs/structural_pfc/structural_pfc.csv, lambda=0)
  * triangular : coordination-L1 (outputs/triangular_pfc/triangular.csv, lambda=0)

Message: the cell-indicator (K4) failure and the Gaussian-locality requirement are
UNIVERSAL (K4 tall on both lattices), whereas the spectral-kernel (K1) failure is
specific to the multi-sublattice honeycomb (K1 tall on honeycomb, modest on the
single-mode triangular lattice). This delimits the mechanism precisely.
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
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
    "savefig.dpi": 320, "savefig.bbox": "tight", "figure.constrained_layout.use": True,
})
C = dict(blue="#4477AA", red="#EE6677", grey="#BBBBBB")
COL1 = 3.37
KORDER = ["K1", "K2", "K3", "K4"]
KLAB = {"K1": "$K_1$\nspectral", "K2": "$K_2$\nGaussian",
        "K3": "$K_3$\nVoronoi-\nGaussian", "K4": "$K_4$\ncell-\nindicator"}


def kernel_means(csv, col):
    df = pd.read_csv(csv)
    if "lam" in df.columns:
        df = df[df.lam == 0.0]
    df["K"] = df["protocol"].str.split("/").str[0]
    # exclude pristine control from the defect-fidelity average where present
    if "case" in df.columns:
        df = df[~df["case"].astype(str).str.contains("TR0|pristine|T0")]
    return df.groupby("K")[col].mean()


def main():
    hc = kernel_means(os.path.join(ROOT, "outputs", "structural_pfc", "structural_pfc.csv"),
                      "ringL1_relaxed")
    tr = kernel_means(os.path.join(ROOT, "outputs", "triangular_pfc", "triangular.csv"),
                      "coordL1_relaxed")
    # relative to each lattice's best kernel (floor to avoid divide-by-zero)
    hc_rel = {k: max(hc.get(k, np.nan), 1e-3) / max(hc.min(), 1e-3) for k in KORDER}
    tr_rel = {k: max(tr.get(k, np.nan), 1e-3) / max(tr.min(), 1e-3) for k in KORDER}

    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    x = np.arange(len(KORDER)); w = 0.38
    hcv = [hc_rel[k] for k in KORDER]
    trv = [tr_rel[k] for k in KORDER]
    ax.bar(x - w / 2, hcv, w, color=C["blue"], label="honeycomb (ring-L1)",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, trv, w, color=C["red"], label="triangular (coord-L1)",
           edgecolor="white", linewidth=0.5)
    ax.axhline(10, color=C["grey"], lw=0.7, ls="--")
    ax.text(3.4, 11, "order of\nmagnitude", color=C["grey"], fontsize=6.5, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.7, max(max(hcv), max(trv)) * 1.8)
    ax.set_xticks(x); ax.set_xticklabels([KLAB[k] for k in KORDER], fontsize=7)
    ax.set_ylabel("relaxed fidelity error\n($\\times$ best kernel)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Universal vs lattice-specific projection failure", fontsize=8.5)
    fig.savefig(os.path.join(OUT, "F9_generality.png"))
    plt.close(fig)
    print("honeycomb rel:", {k: round(v, 1) for k, v in hc_rel.items()})
    print("triangular rel:", {k: round(v, 1) for k, v in tr_rel.items()})
    print("wrote", os.path.join(OUT, "F9_generality.png"))


if __name__ == "__main__":
    main()
