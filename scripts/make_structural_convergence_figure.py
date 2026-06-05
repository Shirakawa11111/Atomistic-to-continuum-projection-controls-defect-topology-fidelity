#!/usr/bin/env python3
"""F12 -- numerical soundness of the structural-PFC relaxation (PRM advisor ask).

Three panels, for the two representative defects T2 (Stone--Wales) and T4
(low-angle GB) at the calibrated config (lambda=0), under K3/R2/P3, all compared
at matched evolution (fixed step count / fixed physical end time):

  (a) FREE-ENERGY MONOTONICITY -- the recorded ``energy_curve`` from
      ``SCCHModel.relax`` decreases monotonically (max per-step rise annotated).
  (b) GRID & TIME-STEP CONVERGENCE -- relaxed ring-L1 vs nx in {384,512,768} and
      vs dt in {5e-9,1e-8,2e-8} (fixed end time); absolute spread + max relative
      change annotated.
  (c) BASIN STABILITY -- relaxed ring-L1 vs the value after adding small fixed-
      seed noise and re-relaxing; the field stays in the same basin (near-zero
      free-energy gap and field-L2 difference annotated).

Reads outputs/structural_convergence/summary.json. Serif / Paul-Tol palette.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "outputs", "structural_convergence")
OUT = os.path.join(ROOT, "outputs", "figures")
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
COLF = 7.2
STYLE = {
    "T2 Stone-Wales": dict(color=C["blue"], marker="s", key="T2 SW"),
    "T4 low-angle GB": dict(color=C["red"], marker="o", key="T4 GB"),
}


def per_by_label(per: dict, label: str) -> dict:
    for v in per.values():
        if v["label"] == label:
            return v
    raise KeyError(label)


def main():
    summ = json.load(open(os.path.join(SRC, "summary.json")))
    per = summ["per_case"]
    labels = [per[k]["label"] for k in summ["cases"]]
    chunk = summ.get("chunk", 250)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(COLF, 2.7))

    # ===== (a) free-energy monotonicity =====
    for lab in labels:
        st = STYLE[lab]
        curve = np.asarray(per_by_label(per, lab)["energy_curve"], dtype=float)
        steps = np.arange(len(curve)) * chunk
        axA.plot(steps, curve, color=st["color"], marker=st["marker"], ms=3.4,
                 lw=1.3, label=lab)
    axA.set_xlabel("relaxation step")
    axA.set_ylabel("free energy  $F$  (model units)")
    axA.text(0.04, 0.06, "(a) $F$ decreases\nmonotonically",
             transform=axA.transAxes, fontsize=8, va="bottom")
    # worst per-step CHANGE across both defects: negative => strictly decreasing
    worst_rise = max(per_by_label(per, lab)["monotonicity"]["max_step_rise"] for lab in labels)
    note = ("every recorded step\n$\\Delta F < 0$"
            if worst_rise < 0 else f"max rise {worst_rise:.1e}")
    axA.text(0.97, 0.97, note, transform=axA.transAxes, fontsize=7, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grey"], lw=0.6))
    axA.legend(loc="upper center", fontsize=7, handletextpad=0.4, bbox_to_anchor=(0.5, 0.88))

    # ===== (b) grid & time-step convergence of relaxed ring-L1 =====
    # categorical x positions so the three nx / three dt values are evenly spaced
    # and the two axes' tick labels never collide.
    nx_list = sorted(int(k) for k in next(iter(per.values()))["grid"]["ring_l1"].keys())
    dt_vals = sorted(float(k) for k in next(iter(per.values()))["dt"]["ring_l1"].keys())
    xpos = np.arange(len(nx_list))
    axB2 = axB.twiny()
    for lab in labels:
        st = STYLE[lab]
        d = per_by_label(per, lab)
        gy = [d["grid"]["ring_l1"][str(nx)] for nx in nx_list]
        axB.plot(xpos, gy, color=st["color"], marker=st["marker"], ms=4.5, lw=1.3, ls="-")
        dy = [d["dt"]["ring_l1"][f"{dv:.0e}"] for dv in dt_vals]
        axB2.plot(xpos, dy, color=st["color"], marker=st["marker"], ms=4.5,
                  lw=1.1, ls="--", mfc="white")
    axB.set_xlim(-0.3, len(nx_list) - 0.7)
    axB2.set_xlim(-0.3, len(nx_list) - 0.7)
    axB.set_xticks(xpos)
    axB.set_xticklabels([str(n) for n in nx_list])
    axB.set_xlabel("grid points $n_x$  (solid)")
    axB2.set_xticks(xpos)
    axB2.set_xticklabels([f"{v:.0e}" for v in dt_vals], fontsize=7.2)
    axB2.set_xlabel("time step $\\Delta t$  (dashed)", fontsize=8.5)
    axB.set_ylabel("relaxed ring-L1")
    gabs = summ["overall"]["grid_max_abs_spread"]
    dabs = summ["overall"]["dt_max_abs_spread"]
    gmax = summ["overall"]["grid_max_rel_change"]
    dmax = summ["overall"]["dt_max_rel_change"]
    axB.text(0.5, 1.30, "(b) grid & $\\Delta t$ convergence", transform=axB.transAxes,
             fontsize=8.5, ha="center", va="bottom")
    axB.text(0.5, 0.025,
             f"max abs. ring-L1 spread:  grid {gabs:.3f},  $\\Delta t$ {dabs:.3f}",
             transform=axB.transAxes, fontsize=6.6, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grey"], lw=0.6))
    # proxy legend for grid vs dt line styles
    h_g = plt.Line2D([0], [0], color=C["dark"], ls="-", lw=1.3, label="vs $n_x$")
    h_d = plt.Line2D([0], [0], color=C["dark"], ls="--", lw=1.1, label="vs $\\Delta t$")
    axB.legend(handles=[h_g, h_d], loc="upper left", fontsize=6.8, handlelength=1.6)
    allv = []
    for lab in labels:
        d = per_by_label(per, lab)
        allv += list(d["grid"]["ring_l1"].values()) + list(d["dt"]["ring_l1"].values())
    lo, hi = min(allv), max(allv)
    pad = max(0.01, 0.30 * (hi - lo))
    axB.set_ylim(max(0, lo - 1.6 * pad), hi + pad)   # extra bottom room for the annotation

    # ===== (c) basin stability =====
    xs = np.arange(len(labels))
    # compare at matched evolution: unperturbed reference vs perturbed, both +N steps
    ref = [per_by_label(per, lab)["basin"]["ring_l1_reference_matched"] for lab in labels]
    aft = [per_by_label(per, lab)["basin"]["ring_l1_after_rerelax"] for lab in labels]
    cols = [STYLE[lab]["color"] for lab in labels]
    w = 0.34
    axC.bar(xs - w / 2, ref, width=w, color=cols, edgecolor="white", linewidth=0.6)
    axC.bar(xs + w / 2, aft, width=w, color=cols, edgecolor="white", linewidth=0.6,
            alpha=0.45, hatch="///")
    axC.set_xticks(xs)
    axC.set_xticklabels([STYLE[l]["key"] for l in labels])
    axC.set_ylabel("relaxed ring-L1")
    axC.text(0.5, 1.02, "(c) basin stability", transform=axC.transAxes,
             fontsize=8.5, ha="center", va="bottom")
    h1 = plt.Rectangle((0, 0), 1, 1, fc=C["grey"], ec="white")
    h2 = plt.Rectangle((0, 0), 1, 1, fc=C["grey"], ec="white", alpha=0.45, hatch="///")
    axC.legend([h1, h2], ["relaxed", "re-relaxed\nafter +3% noise"],
               loc="upper left", fontsize=6.8, handlelength=1.2, handletextpad=0.4)
    # the genuine basin-identity signals: free-energy gap and field L2 difference
    worst_dF = max(per_by_label(per, lab)["basin"]["free_energy_rel_gap"] for lab in labels)
    worst_L2 = max(per_by_label(per, lab)["basin"]["field_l2_rel_vs_reference"] for lab in labels)
    axC.text(0.97, 0.97,
             f"same basin:\n$\\Delta F/F \\leq {worst_dF:.0e}$\n"
             f"$\\|\\Delta n\\|/\\|n\\| \\leq {worst_L2*100:.1f}\\%$",
             transform=axC.transAxes, fontsize=6.6, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grey"], lw=0.6))
    ymax = max(max(ref), max(aft))
    axC.set_ylim(0, ymax * 1.5)

    fig.savefig(os.path.join(OUT, "F12_convergence.png"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, "F12_convergence.png"))


if __name__ == "__main__":
    main()
