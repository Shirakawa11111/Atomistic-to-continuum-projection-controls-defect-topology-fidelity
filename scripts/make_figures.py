#!/usr/bin/env python3
"""Publication-quality figures (PRB style) from the corrected production_v3 matrix
and the corrected convergence sweep.

F1 admissibility · F2 variance decomposition · F3 protocol-CV by defect ·
F4 mapping-fidelity heatmap + decision tree · F5 Ef spread · F6 Ef convergence.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- publication style ----
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
# Tol bright, colour-blind safe
C = dict(blue="#4477AA", cyan="#66CCEE", green="#228833", yellow="#CCBB44",
         red="#EE6677", purple="#AA3377", grey="#BBBBBB", dark="#222222")
COL1, COLF = 3.37, 6.9        # single- and full-column widths (in)

CASE_LABEL = {"T0_pristine": "T0\npristine", "T1_vacancy": "T1\nvacancy",
              "T2_stone_wales": "T2\nSW", "T3_dislocation_dipole": "T3\ndipole",
              "T4_low_angle_gb": "T4\nlow-$\\theta$ GB", "T5_high_angle_gb": "T5\nhigh-$\\theta$ GB"}
CASE_ORDER = ["T0_pristine", "T1_vacancy", "T2_stone_wales",
              "T3_dislocation_dipole", "T4_low_angle_gb", "T5_high_angle_gb"]
DEF = CASE_ORDER[1:]
PLAB = {"K1/R2/P2": "K1/P2", "K1/R2/P3": "K1/P3", "K1/R2/P4": "K1/P4",
        "K2/R2/P3": "K2/P3", "K2/R2/P4": "K2/P4", "K3/R2/P3": "K3/P3",
        "K3/R2/P4": "K3/P4", "K3/R3/P2": "K3/P2(R3)", "K4/R2/P2": "K4/P2",
        "K4/R2/P3": "K4/P3", "K4/R2/P4": "K4/P4"}


def cv(s):
    v = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return v.std() / abs(v.mean()) if len(v) > 1 and abs(v.mean()) > 1e-12 else np.nan


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(ROOT, "outputs", "production_v3", "metrics.csv"))
    ref = df[df.lam == 0.01].copy()
    A = ref[ref.admissible == True].copy()
    AD = A[A.case != "T0_pristine"]

    # ---- F1 admissibility ----
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    adm = (ref.groupby("P")["admissible"].mean().reindex(["P1", "P2", "P3", "P4"]) * 100)
    cols = [C["red"], C["yellow"], C["green"], C["blue"]]
    bars = ax.bar(["$P_1$ raw", "$P_2$ strict", "$P_3$ rescale", "$P_4$ clip"],
                  adm.values, color=cols, width=0.66, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("admissible runs (\\%)"); ax.set_ylim(0, 108)
    for b, v in zip(bars, adm.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_yticks([0, 50, 100])
    _save(fig, "F1_admissibility.png")

    # ---- F2 variance decomposition ----
    obs = ["ring_l1_init", "l2_change", "roughness_init"]
    olab = ["fidelity\n(ring L1)", "relaxation\n$\\|\\Delta\\psi\\|$", "roughness"]
    sp, sc = [], []
    for o in obs:
        gm = AD[o].mean()
        pe = {p: AD[AD.protocol == p][o].mean() - gm for p in AD.protocol.unique()}
        ce = {c: AD[AD.case == c][o].mean() - gm for c in AD.case.unique()}
        sp.append(np.var([pe[p] for p in AD.protocol])); sc.append(np.var([ce[c] for c in AD.case]))
    x = np.arange(len(obs)); w = 0.34
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    ax.bar(x - w / 2, sp, w, label="protocol", color=C["blue"], edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, sc, w, label="case", color=C["grey"], edgecolor="white", linewidth=0.5)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(olab)
    ax.set_ylabel("variance component"); ax.legend(loc="upper left")
    for i, (a, b) in enumerate(zip(sp, sc)):
        ax.text(i, a * 1.7, f"{a/max(b,1e-12):.0f}$\\times$", ha="center", fontsize=8.5,
                color=C["dark"], fontweight="bold")
    ax.set_ylim(min(sc) / 3, max(sp) * 6)
    _save(fig, "F2_variance_decomposition.png")

    # ---- F3 protocol-CV by defect ----
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    for o, lab, col, mk in zip(obs, ["fidelity", "$\\|\\Delta\\psi\\|$", "roughness"],
                               [C["blue"], C["green"], C["red"]], ["o", "s", "^"]):
        ys = [cv(A[A.case == c][o]) for c in DEF]
        ax.plot(range(len(DEF)), ys, mk + "-", label=lab, color=col, mfc=col, mec="white", mew=0.5)
    ax.axhline(0.20, ls=(0, (4, 3)), color=C["dark"], lw=0.8)
    ax.text(len(DEF) - 1, 0.23, "$D_1$ threshold", fontsize=7.5, ha="right", color=C["dark"])
    ax.set_xticks(range(len(DEF))); ax.set_xticklabels([CASE_LABEL[c] for c in DEF])
    ax.set_ylabel("protocol-induced CV"); ax.set_ylim(0, 1.32); ax.legend(ncol=3, loc="upper center",
                                                                          columnspacing=1.0, handletextpad=0.4)
    _save(fig, "F3_cv_by_defect.png")

    # ---- F4 fidelity heatmap + decision tree (full width) ----
    protos = [p for p in PLAB if p in set(A.protocol)]
    M = np.full((len(protos), len(CASE_ORDER)), np.nan)
    for i, p in enumerate(protos):
        for j, c in enumerate(CASE_ORDER):
            v = A[(A.protocol == p) & (A.case == c)]["ring_l1_init"]
            if len(v):
                M[i, j] = v.iloc[0]
    cmap = LinearSegmentedColormap.from_list("fid", ["#FFFFFF", C["cyan"], C["blue"], C["purple"]])
    fig, ax = plt.subplots(figsize=(COLF, 3.5))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=np.nanpercentile(M, 92))
    ax.set_xticks(range(len(CASE_ORDER))); ax.set_xticklabels([CASE_LABEL[c] for c in CASE_ORDER])
    ax.set_yticks(range(len(protos))); ax.set_yticklabels([PLAB[p] for p in protos])
    ax.set_xticks(np.arange(-.5, len(CASE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(protos), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0); ax.tick_params(which="minor", length=0)
    for j in range(len(CASE_ORDER)):
        col = M[:, j]
        if np.isfinite(col).any():
            bi = int(np.nanargmin(col))
            ax.scatter(j, bi, marker="*", s=90, color="white", edgecolor=C["dark"], linewidth=0.5, zorder=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("ring L1  (0 = perfect topology match)")
    ax.set_title("Mapping fidelity by protocol and defect  ($\\star$ = best per defect)", fontsize=9)
    _save(fig, "F4_fidelity_decision_tree.png")

    # ---- F5 Ef spread ----
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    stds = [A[A.case == c]["Ef_rel"].std() for c in DEF]
    sprd = [A[A.case == c]["Ef_rel"].max() - A[A.case == c]["Ef_rel"].min() for c in DEF]
    x = np.arange(len(DEF)); w = 0.34
    ax.bar(x - w / 2, stds, w, label="std", color=C["green"], edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, sprd, w, label="full range", color=C["purple"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([CASE_LABEL[c] for c in DEF])
    ax.set_ylabel("protocol-induced $\\Delta E_f^{\\mathrm{rel}}$"); ax.legend()
    _save(fig, "F5_Ef_spread.png")

    # ---- F6 Ef convergence (corrected convergence_v2 if present) ----
    cpath = os.path.join(ROOT, "outputs", "convergence_v2", "convergence.csv")
    if not os.path.exists(cpath):
        cpath = os.path.join(ROOT, "outputs", "convergence", "convergence.csv")
    cd = pd.read_csv(cpath); nxs = sorted(cd.nx_sweep.unique())
    cd["Ef_nx"] = np.nan
    for (pr, nx), g in cd.groupby(["protocol", "nx_sweep"]):
        t0 = g[g.case == "T0_pristine"]["F_final"]
        if len(t0) and abs(float(t0.iloc[0])) > 1e-12:
            cd.loc[g.index, "Ef_nx"] = (g["F_final"] - float(t0.iloc[0])) / abs(float(t0.iloc[0]))
    b = cd[cd.protocol == "K3/R2/P4"]
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    cyc = [(C["blue"], "o"), (C["green"], "s"), (C["red"], "^"), (C["purple"], "D"), (C["yellow"], "v")]
    for c, (col, mk) in zip(DEF, cyc):
        g = b[b.case == c].sort_values("nx_sweep")
        ax.plot(g.nx_sweep, g.Ef_nx, mk + "-", color=col, mfc=col, mec="white", mew=0.5,
                label=CASE_LABEL[c].replace("\n", " "))
    ax.set_xlabel("grid $n_x$ (long edge)"); ax.set_ylabel("$E_f^{\\mathrm{rel}}$  ($K_3/R_2/P_4$)")
    ax.set_xticks(nxs); ax.legend(ncol=2, columnspacing=1.0, handletextpad=0.4, loc="center right")
    _save(fig, "F6_convergence.png")

    print("wrote 6 publication figures to", OUT)


if __name__ == "__main__":
    main()
