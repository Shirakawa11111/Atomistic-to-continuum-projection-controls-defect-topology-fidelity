#!/usr/bin/env python3
"""Variance dominance on the CONSTRAINT-FREE, properly-scaled models.

The headline sigma^2_protocol / sigma^2_case ratio in the paper is computed on the
SC-CH testbed at lambda=0.01, which carries two caveats: the morphological
constraint inflates the protocol component, and the testbed is mesoscale (relative
only). Here we recompute the SAME estimator (uq_analysis.paper_ratio_from_matrix
+ the noise-corrected ANOVA/REML decomposition) on:

  (1) the structural PFC, relaxed ring-L1, lambda=0  -> properly scaled AND no constraint
  (2) the triangular structural PFC, relaxed coord-L1 -> a different lattice

If the dominance survives here, it is not an artifact of the constraint or the
mesoscale testbed. Defect cases only (exclude pristine).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from uq_analysis import paper_ratio_from_matrix, anova_varcomp_balanced, bootstrap_ratio, ci_summary

N_BOOT = 5000
SEED = 20260604


def ratio_on(csv, value_col, exclude_prefix, lam=None):
    df = pd.read_csv(csv)
    if lam is not None and "lam" in df.columns:
        df = df[df["lam"] == lam]
    df = df[~df["case"].astype(str).str.startswith(exclude_prefix)].copy()
    piv = df.pivot_table(index="case", columns="protocol", values=value_col, aggfunc="mean")
    Y = piv.to_numpy(dtype=float)                 # rows = cases, cols = protocols
    point, sp, sc = paper_ratio_from_matrix(Y)
    vc = anova_varcomp_balanced(Y)
    rng = np.random.default_rng(SEED)
    bc = ci_summary(bootstrap_ratio(Y, "case", N_BOOT, rng))
    return dict(
        n_cases=int(Y.shape[0]), n_protocols=int(Y.shape[1]),
        cases=list(piv.index), protocols=list(piv.columns),
        point_ratio=float(point), sigma2_protocol=float(sp), sigma2_case=float(sc),
        boot_case_lo=bc["lo"], boot_case_hi=("inf" if bc.get("hi_is_inf") else bc["hi"]),
        boot_case_median=bc["median"], boot_case_P_gt_10=bc["P_gt_10"],
        boot_case_frac_inf=bc["frac_degenerate_inf"],
        noise_corrected_ratio=(vc["ratio_noise_corrected"] if vc else None),
        s2_protocol_nc=(vc["s2_protocol"] if vc else None),
        s2_case_nc=(vc["s2_case"] if vc else None),
    )


def main():
    out = {}
    out["structural_honeycomb_ringL1_relaxed_lam0"] = ratio_on(
        os.path.join(ROOT, "outputs", "structural_pfc", "structural_pfc.csv"),
        "ringL1_relaxed", "T0", lam=0.0)
    out["triangular_coordL1_relaxed_lam0"] = ratio_on(
        os.path.join(ROOT, "outputs", "triangular_pfc", "triangular.csv"),
        "coordL1_relaxed", "TR0", lam=None)
    os.makedirs(os.path.join(ROOT, "outputs", "uq"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "uq", "uq_structural.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    for k, v in out.items():
        nc = v["noise_corrected_ratio"]
        nc_s = ("inf" if nc == float("inf") else (f"{nc:.1f}" if nc is not None else "NA"))
        print(f"\n{k}:")
        print(f"  cases={v['n_cases']} protocols={v['n_protocols']}")
        print(f"  point ratio sigma2_protocol/sigma2_case = {v['point_ratio']:.1f}  "
              f"(sp={v['sigma2_protocol']:.4g}, sc={v['sigma2_case']:.4g})")
        print(f"  bootstrap-over-cases 95% lower bound = {v['boot_case_lo']:.1f}  "
              f"median {v['boot_case_median']:.1f}  P(ratio>10)={v['boot_case_P_gt_10']:.3f}")
        print(f"  noise-corrected (REML/ANOVA) ratio = {nc_s}")
    print("\nwrote outputs/uq/uq_structural.json")


if __name__ == "__main__":
    main()
