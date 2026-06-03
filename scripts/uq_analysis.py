#!/usr/bin/env python3
"""
Rigorous uncertainty quantification for the protocol-vs-case variance result.

The paper (docs/PRODUCTION_FINDINGS.md, Finding B; Fig. F2 in scripts/make_figures.py)
reports a between-group variance ratio

    ratio_o  =  Var_pop( {mean_p(o)}_protocols )  /  Var_pop( {mean_c(o)}_cases )

for o in {ring_l1_init, l2_change, roughness_init}, evaluated on
    admissible rows, lam = 0.01, defect cases only (exclude T0_pristine),
giving the headline "protocol matters 30-43x more than which defect".

This script attaches real UQ to that point estimate:

(1) Bootstrap 95% CIs on the ratio (resample the 5 defect CASES with replacement;
    separately resample the PROTOCOLS with replacement; and a combined two-way
    bootstrap). >= N_BOOT resamples each (default 5000).

(2) Leave-one-defect-out (LODO): drop each of T1..T5 in turn, recompute the ratio
    on the remaining 4 cases, report the min-max range.

(3) A random-effects variance-components decomposition:
    - statsmodels MixedLM with `case` as the grouping factor and `protocol` as a
      crossed variance component (groups=dummy, vc_formula). If statsmodels is not
      installed we fall back to a closed-form balanced two-way ANOVA / REML estimate
      and say so.  This decomposition is *noise-corrected* (it subtracts the
      residual cell-to-cell variance), so its protocol/case variance ratio is the
      more conservative, statistically defensible version of the headline number.

Everything is also recomputed at lam = 0 for comparison.

Outputs:
    outputs/uq/uq.json    -- full machine-readable results
    outputs/uq/uq_summary.csv  -- compact table of point estimates + CIs
    outputs/uq/uq.png     -- bootstrap distributions + LODO + components figure

Run:
    cd /Users/bojingkai/Desktop/Route_A_protocol_robustness
    PYTHONPATH=$PWD/src python3 scripts/uq_analysis.py
"""
from __future__ import annotations

import json
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

# MixedLM near a variance-component boundary (s2_case is small) emits expected
# ConvergenceWarnings; we report convergence flags explicitly, so silence noise.
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*[Cc]onvergence.*")
warnings.filterwarnings("ignore", message=".*Hessian.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "outputs", "production_v3", "metrics.csv")
OUTDIR = os.path.join(ROOT, "outputs", "uq")

OBS = ["ring_l1_init", "l2_change", "roughness_init"]
OBS_LABEL = {"ring_l1_init": "ring L1 (fidelity)",
             "l2_change": "||Delta psi|| (relaxation)",
             "roughness_init": "roughness"}
N_BOOT = 5000
SEED = 20260603
CI = (2.5, 97.5)          # percentile CI endpoints
PAPER_POINT = {"ring_l1_init": 42.9, "l2_change": 29.6, "roughness_init": 31.9}

try:
    import statsmodels.api as sm        # noqa: F401
    import statsmodels.formula.api as smf
    HAVE_SM = True
    import statsmodels
    SM_VERSION = statsmodels.__version__
except Exception:                       # pragma: no cover
    HAVE_SM = False
    SM_VERSION = None


# --------------------------------------------------------------------------- #
# data loading / filtering
# --------------------------------------------------------------------------- #
def load_panel(lam: float) -> pd.DataFrame:
    """admissible rows at the given lam, defect cases only (exclude T0_*)."""
    df = pd.read_csv(METRICS)
    sub = df[(df["admissible"] == True) & (df["lam"] == lam)
             & (~df["case"].str.startswith("T0"))].copy()
    return sub


def balanced_panel_matrix(sub: pd.DataFrame, obs: str):
    """Return (cases, protocols, Y) where Y[i,j] = obs for case i, protocol j.

    The full grid is used; any missing cell is NaN.  For lam=0.01 / lam=0 the
    grid is fully balanced (5 cases x 11 protocols).
    """
    cases = sorted(sub["case"].unique())
    protocols = sorted(sub["protocol"].unique())
    Y = np.full((len(cases), len(protocols)), np.nan)
    piv = sub.pivot_table(index="case", columns="protocol", values=obs, aggfunc="mean")
    piv = piv.reindex(index=cases, columns=protocols)
    Y = piv.to_numpy(dtype=float)
    return cases, protocols, Y


# --------------------------------------------------------------------------- #
# the paper's between-group variance ratio (naive, NOT noise-corrected)
# --------------------------------------------------------------------------- #
def paper_ratio_from_matrix(Y: np.ndarray):
    """Reproduce make_figures.py F2 estimator from a case x protocol matrix.

    sp = population variance of protocol means (column means, minus grand mean)
    sc = population variance of case     means (row    means, minus grand mean)
    ratio = sp / sc.   np.var uses ddof=0 (population), matching the paper.
    Rows = cases, columns = protocols.
    """
    gm = np.nanmean(Y)
    case_means = np.nanmean(Y, axis=1) - gm          # one per case (row)
    protocol_means = np.nanmean(Y, axis=0) - gm      # one per protocol (column)
    sp = np.nanvar(protocol_means)                   # ddof=0
    sc = np.nanvar(case_means)
    ratio = sp / sc if sc > 0 else np.inf
    return ratio, sp, sc


# --------------------------------------------------------------------------- #
# closed-form balanced two-way ANOVA / REML variance components (noise corrected)
# --------------------------------------------------------------------------- #
def anova_varcomp_balanced(Y: np.ndarray):
    """Random-effects variance components for a balanced two-way crossed design
    with one observation per cell (no replication), additive model

        y_ij = mu + a_i (case) + b_j (protocol) + e_ij,
        a_i ~ N(0, s2_case), b_j ~ N(0, s2_protocol), e_ij ~ N(0, s2_resid).

    Standard ANOVA mean squares (Type I / balanced) give unbiased estimators:
        E[MS_case]     = s2_resid + J * s2_case
        E[MS_protocol] = s2_resid + I * s2_protocol
        E[MS_resid]    = s2_resid
    with I = #cases, J = #protocols.  Solve:
        s2_resid     = MS_resid
        s2_case      = (MS_case     - MS_resid) / J
        s2_protocol  = (MS_protocol - MS_resid) / I
    Negative estimates are clamped to 0 (REML-consistent boundary behaviour).

    Returns dict with the three components, the noise-corrected ratio, dof, and MS.
    Requires a complete (no-NaN) matrix.
    """
    if np.isnan(Y).any():
        return None
    I, J = Y.shape
    grand = Y.mean()
    row_means = Y.mean(axis=1)            # cases
    col_means = Y.mean(axis=0)            # protocols
    # sums of squares
    ss_case = J * np.sum((row_means - grand) ** 2)
    ss_protocol = I * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((Y - grand) ** 2)
    ss_resid = ss_total - ss_case - ss_protocol
    df_case = I - 1
    df_protocol = J - 1
    df_resid = (I - 1) * (J - 1)
    ms_case = ss_case / df_case
    ms_protocol = ss_protocol / df_protocol
    ms_resid = ss_resid / df_resid if df_resid > 0 else np.nan
    s2_resid = ms_resid
    s2_case = (ms_case - ms_resid) / J
    s2_protocol = (ms_protocol - ms_resid) / I
    s2_case_c = max(s2_case, 0.0)
    s2_protocol_c = max(s2_protocol, 0.0)
    ratio_corr = (s2_protocol_c / s2_case_c) if s2_case_c > 0 else np.inf
    return {
        "s2_case_raw": float(s2_case),
        "s2_protocol_raw": float(s2_protocol),
        "s2_case": float(s2_case_c),
        "s2_protocol": float(s2_protocol_c),
        "s2_resid": float(s2_resid),
        "ratio_noise_corrected": float(ratio_corr),
        "ms_case": float(ms_case),
        "ms_protocol": float(ms_protocol),
        "ms_resid": float(ms_resid),
        "df_case": int(df_case),
        "df_protocol": int(df_protocol),
        "df_resid": int(df_resid),
    }


# --------------------------------------------------------------------------- #
# statsmodels MixedLM crossed random effects
# --------------------------------------------------------------------------- #
def mixedlm_varcomp(sub: pd.DataFrame, obs: str):
    """Crossed random-effects variance components via statsmodels MixedLM.

    Model: y ~ 1, with `case` and `protocol` BOTH as crossed random intercepts.
    Implemented with a single dummy group (all rows) and two variance components
    (vc_formula) so the two factors are crossed rather than nested.  Residual
    variance is the model scale.  Returns components or None on failure.
    """
    if not HAVE_SM:
        return None
    d = sub[["case", "protocol", obs]].dropna().copy()
    d = d.rename(columns={obs: "y"})
    # scale y to O(1) to help the optimiser, then rescale variances back.
    s = d["y"].std()
    if s == 0 or not np.isfinite(s):
        return None
    d["yz"] = d["y"] / s
    d["grp"] = 1
    d["case_c"] = d["case"].astype("category")
    d["protocol_c"] = d["protocol"].astype("category")
    vc = {"case": "0 + C(case_c)", "protocol": "0 + C(protocol_c)"}
    try:
        md = smf.mixedlm("yz ~ 1", d, groups=d["grp"],
                         vc_formula=vc, re_formula="0")
        mdf = md.fit(reml=True, method=["lbfgs"], maxiter=1000)
        scale = mdf.scale                      # residual var (on yz scale)
        # mdf.vcomp is a numpy array ordered by md.exog_vc.names
        vc_names = list(md.exog_vc.names)
        vc_vals = np.asarray(mdf.vcomp, dtype=float).ravel()
        s2 = s * s                             # rescale variances back to y-units
        out = {}
        for name, val in zip(vc_names, vc_vals):
            kk = "protocol" if "protocol" in name else ("case" if "case" in name else name)
            out[f"s2_{kk}"] = float(val) * s2
        out["s2_resid"] = float(scale) * s2
        if "s2_case" in out and "s2_protocol" in out:
            out["ratio_noise_corrected"] = (
                out["s2_protocol"] / out["s2_case"] if out["s2_case"] > 0 else np.inf)
        out["converged"] = bool(mdf.converged)
        out["method"] = "statsmodels MixedLM REML, crossed VC (case + protocol)"
        return out
    except Exception as e:                      # pragma: no cover
        return {"error": repr(e)}


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_ratio(Y: np.ndarray, axis: str, n_boot: int, rng: np.random.Generator):
    """Bootstrap the paper ratio by resampling rows (cases), columns (protocols),
    or both ('two_way'), with replacement.

    Y rows = cases, columns = protocols.
    Returns the FULL array of ratio values (length n_boot) INCLUDING +inf draws
    (which occur when a resample makes the denominator sigma2_case == 0, i.e. all
    drawn cases identical).  Downstream summaries decide how to treat them.
    """
    I, J = Y.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        if axis == "case":
            ri = rng.integers(0, I, I)
            Yb = Y[ri, :]
        elif axis == "protocol":
            cj = rng.integers(0, J, J)
            Yb = Y[:, cj]
        elif axis == "two_way":
            ri = rng.integers(0, I, I)
            cj = rng.integers(0, J, J)
            Yb = Y[np.ix_(ri, cj)]
        else:
            raise ValueError(axis)
        r, _, _ = paper_ratio_from_matrix(Yb)
        out[b] = r
    return out


def ci_summary(samples: np.ndarray):
    """Robust summary of a bootstrap ratio distribution with a heavy upper tail.

    A variance-ratio with only 5 groups has an unstable denominator: a fraction
    of resamples drive sigma2_case -> 0 and the ratio -> +inf.  We therefore:
      * report the fraction of degenerate (inf) draws,
      * compute percentiles INCLUDING inf (so the upper CI is honest: it is +inf
        if the requested percentile lands in the degenerate mass; otherwise finite),
      * also report finite-only percentiles for reference,
      * report decision-relevant exceedance probabilities P(ratio > {1,5,10,20}),
        which are robust to the tail because they only need ordering.
    The LOWER CI bound is the stable, load-bearing quantity for a ">= k x" claim.
    """
    n = int(samples.size)
    if n == 0:
        return dict(lo=None, hi=None, median=None, n=0)
    finite = samples[np.isfinite(samples)]
    n_inf = int(np.sum(~np.isfinite(samples)))
    frac_inf = n_inf / n
    # percentiles including inf: sort with inf at the top
    s_sorted = np.sort(samples)  # inf sorts to the end
    def pct_incl(p):
        idx = min(int(np.ceil(p / 100.0 * n)) - 1, n - 1)
        idx = max(idx, 0)
        v = s_sorted[idx]
        return float(v) if np.isfinite(v) else float("inf")
    lo = pct_incl(CI[0])
    hi = pct_incl(CI[1])
    out = dict(
        lo=lo,
        hi=(None if not np.isfinite(hi) else hi),
        hi_is_inf=bool(not np.isfinite(hi)),
        median=float(np.median(finite)) if finite.size else None,
        mean_finite=float(np.mean(finite)) if finite.size else None,
        p05_finite=float(np.percentile(finite, 5)) if finite.size else None,
        p16_finite=float(np.percentile(finite, 16)) if finite.size else None,
        p84_finite=float(np.percentile(finite, 84)) if finite.size else None,
        p95_finite=float(np.percentile(finite, 95)) if finite.size else None,
        hi_finite_p975=float(np.percentile(finite, 97.5)) if finite.size else None,
        frac_degenerate_inf=frac_inf,
        n=n,
        # decision-relevant exceedance probs (inf counts as > any threshold)
        P_gt_1=float(np.mean(samples > 1.0)),
        P_gt_5=float(np.mean(samples > 5.0)),
        P_gt_10=float(np.mean(samples > 10.0)),
        P_gt_20=float(np.mean(samples > 20.0)),
    )
    return out


def bootstrap_reml_components(Y: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Bootstrap 95% CIs on the NOISE-CORRECTED (closed-form REML/ANOVA) variance
    components and their ratio, by two-way resampling (cases x protocols) of the
    balanced cell matrix.  Returns CI dicts for each component and the ratio.

    Note: resampling rows/cols can create duplicate groups, which deflates the
    between-group SS for that factor; the MoM estimator stays unbiased in
    expectation but the bootstrap honestly widens the CI.  We clamp components at
    0 (REML boundary).  The ratio CI uses finite-inclusive percentiles like the
    naive bootstrap.
    """
    I, J = Y.shape
    sp = np.empty(n_boot); sc = np.empty(n_boot); sr = np.empty(n_boot)
    rr = np.empty(n_boot)
    for b in range(n_boot):
        ri = rng.integers(0, I, I)
        cj = rng.integers(0, J, J)
        Yb = Y[np.ix_(ri, cj)]
        vc = anova_varcomp_balanced(Yb)
        if vc is None:
            sp[b] = sc[b] = sr[b] = rr[b] = np.nan
            continue
        sp[b] = vc["s2_protocol"]; sc[b] = vc["s2_case"]; sr[b] = vc["s2_resid"]
        rr[b] = vc["ratio_noise_corrected"]
    def _ci(a):
        af = a[np.isfinite(a)]
        if af.size == 0:
            return dict(lo=None, hi=None, median=None)
        return dict(lo=float(np.percentile(af, CI[0])),
                    hi=float(np.percentile(af, CI[1])),
                    median=float(np.median(af)))
    rr_ci = ci_summary(rr)   # ratio uses inf-aware summary
    return dict(
        s2_protocol_ci=_ci(sp),
        s2_case_ci=_ci(sc),
        s2_resid_ci=_ci(sr),
        ratio_ci=dict(lo=rr_ci["lo"],
                      hi=(None if rr_ci.get("hi_is_inf") else rr_ci["hi"]),
                      hi_is_inf=bool(rr_ci.get("hi_is_inf")),
                      hi_finite_p975=rr_ci.get("hi_finite_p975"),
                      median=rr_ci["median"],
                      frac_degenerate_inf=rr_ci.get("frac_degenerate_inf")),
    )


def balanced_levelcount_ratio(Y: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Robustness check against the 'more protocol levels' objection.

    The naive ratio compares the spread of 11 protocol means to 5 case means.
    A skeptic could argue protocol simply has more levels -> wider mean spread.
    Here we repeatedly subsample exactly 5 protocols (matching the 5 cases) and
    recompute the ratio, so both factors have the SAME number of levels.
    A persistently large ratio shows the dominance is not a level-count artifact.
    """
    I, J = Y.shape
    k = I  # match number of cases
    if J < k:
        return None
    rs = []
    for _ in range(n_boot):
        cols = rng.choice(J, k, replace=False)
        r, _, _ = paper_ratio_from_matrix(Y[:, cols])
        if np.isfinite(r):
            rs.append(r)
    rs = np.array(rs)
    if rs.size == 0:
        return None
    return dict(
        median=float(np.median(rs)),
        lo=float(np.percentile(rs, CI[0])),
        hi=float(np.percentile(rs, CI[1])),
        P_gt_1=float(np.mean(rs > 1)),
        P_gt_10=float(np.mean(rs > 10)),
        n_protocols_subsampled=int(k),
        n=int(rs.size),
    )


def leave_one_defect_out(sub: pd.DataFrame, obs: str):
    """Drop each case in turn, recompute the paper ratio on remaining cases."""
    cases = sorted(sub["case"].unique())
    res = OrderedDict()
    for drop in cases:
        s2 = sub[sub["case"] != drop]
        _, _, Y = balanced_panel_matrix(s2, obs)
        r, sp, sc = paper_ratio_from_matrix(Y)
        res[drop] = float(r)
    vals = np.array(list(res.values()))
    return dict(per_drop=res, min=float(vals.min()), max=float(vals.max()))


# --------------------------------------------------------------------------- #
# driver for one lam
# --------------------------------------------------------------------------- #
def analyse_lam(lam: float, rng: np.random.Generator):
    sub = load_panel(lam)
    cases = sorted(sub["case"].unique())
    protocols = sorted(sub["protocol"].unique())
    res = OrderedDict()
    res["lam"] = lam
    res["n_rows"] = int(len(sub))
    res["n_cases"] = len(cases)
    res["n_protocols"] = len(protocols)
    res["cases"] = cases
    res["protocols"] = protocols
    res["observables"] = OrderedDict()

    for o in OBS:
        cases_o, protos_o, Y = balanced_panel_matrix(sub, o)
        complete = not np.isnan(Y).any()
        point, sp, sc = paper_ratio_from_matrix(Y)

        # bootstrap (independent rng draws but reproducible via the shared rng)
        bs_case = bootstrap_ratio(Y, "case", N_BOOT, rng)
        bs_proto = bootstrap_ratio(Y, "protocol", N_BOOT, rng)
        bs_two = bootstrap_ratio(Y, "two_way", N_BOOT, rng)

        lodo = leave_one_defect_out(sub, o)
        vc_anova = anova_varcomp_balanced(Y)
        vc_mixed = mixedlm_varcomp(sub, o) if HAVE_SM else None
        vc_boot = bootstrap_reml_components(Y, N_BOOT, rng)
        balanced_lc = balanced_levelcount_ratio(Y, N_BOOT, rng)

        res["observables"][o] = OrderedDict(
            label=OBS_LABEL[o],
            balanced_complete_grid=bool(complete),
            point_ratio=float(point),
            sigma2_protocol_naive=float(sp),
            sigma2_case_naive=float(sc),
            paper_reported_ratio=PAPER_POINT[o],
            bootstrap=OrderedDict(
                resample_cases=ci_summary(bs_case),
                resample_protocols=ci_summary(bs_proto),
                resample_two_way=ci_summary(bs_two),
            ),
            leave_one_defect_out=lodo,
            balanced_levelcount_check=balanced_lc,
            varcomp_anova_reml_closed_form=vc_anova,
            varcomp_anova_reml_bootstrap_ci=vc_boot,
            varcomp_statsmodels_mixedlm=vc_mixed,
        )
    return res, {o: (load_panel(lam), o) for o in OBS}


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[uq] matplotlib unavailable, skipping figure ({e})")
        return False

    rng = np.random.default_rng(SEED + 99)
    r01 = results["lam=0.01"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # row 0: bootstrap distributions (resample cases) per observable -- log-x,
    # because the case-resample ratio has a heavy upper tail (denominator -> 0).
    for k, o in enumerate(OBS):
        ax = axes[0, k]
        ob = r01["observables"][o]
        sub01 = load_panel(0.01)
        _, _, Y = balanced_panel_matrix(sub01, o)
        bs = bootstrap_ratio(Y, "case", N_BOOT, rng)
        bsf = bs[np.isfinite(bs) & (bs > 0)]
        bins = np.logspace(np.log10(max(bsf.min(), 1e-2)), np.log10(bsf.max()), 60)
        ax.hist(bsf, bins=bins, color="#3b6fb6", alpha=0.85, edgecolor="white", linewidth=0.2)
        ax.set_xscale("log")
        d = ob["bootstrap"]["resample_cases"]
        pt = ob["point_ratio"]
        lo = d["lo"]
        hi_f = d["hi_finite_p975"]
        ax.axvline(pt, color="#c23b22", lw=2, label=f"point={pt:.1f}")
        ax.axvline(lo, color="k", ls="--", lw=1, label=f"2.5% = {lo:.1f}")
        ax.axvline(hi_f, color="0.4", ls=":", lw=1, label=f"finite 97.5% = {hi_f:.0f}")
        ax.axvline(1.0, color="#7ab648", lw=1, label="ratio = 1")
        ax.set_title(f"{OBS_LABEL[o]}\ncase-resample (frac_inf={d['frac_degenerate_inf']:.3f})",
                     fontsize=9)
        ax.set_xlabel("sigma2_protocol / sigma2_case  (log)")
        ax.legend(fontsize=6.5)

    # row 1, col 0: LODO ranges
    ax = axes[1, 0]
    width = 0.25
    xs = np.arange(len(OBS))
    for di, o in enumerate(OBS):
        ob = r01["observables"][o]
        per = ob["leave_one_defect_out"]["per_drop"]
        ys = list(per.values())
        labels = [k.split("_")[0] for k in per.keys()]
        ax.scatter([di] * len(ys), ys, s=28, color="#3b6fb6", zorder=3)
        ax.scatter([di], [ob["point_ratio"]], marker="*", s=120, color="#c23b22", zorder=4)
        for y, lb in zip(ys, labels):
            ax.annotate(lb, (di, y), fontsize=6, xytext=(4, 0),
                        textcoords="offset points", va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([OBS_LABEL[o].split(" ")[0] for o in OBS], fontsize=8)
    ax.set_ylabel("ratio (leave-one-defect-out)")
    ax.set_title("LODO: drop each defect (star = full)", fontsize=9)

    # row 1, col 1: naive vs noise-corrected ratio
    ax = axes[1, 1]
    naive = [r01["observables"][o]["point_ratio"] for o in OBS]
    corr = []
    for o in OBS:
        vc = r01["observables"][o]["varcomp_anova_reml_closed_form"]
        corr.append(vc["ratio_noise_corrected"] if vc else np.nan)
    ax.bar(xs - 0.2, naive, 0.4, label="naive (paper)", color="#3b6fb6")
    ax.bar(xs + 0.2, corr, 0.4, label="noise-corrected (REML)", color="#7ab648")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([OBS_LABEL[o].split(" ")[0] for o in OBS], fontsize=8)
    ax.set_ylabel("protocol/case variance ratio")
    ax.set_title("naive vs noise-corrected", fontsize=9)
    ax.legend(fontsize=7)
    for i, (a, b) in enumerate(zip(naive, corr)):
        ax.text(i - 0.2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=7)
        if np.isfinite(b):
            ax.text(i + 0.2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=7)

    # row 1, col 2: variance components (REML) stacked, ring_l1
    ax = axes[1, 2]
    comp_names = ["s2_protocol", "s2_case", "s2_resid"]
    colors = {"s2_protocol": "#3b6fb6", "s2_case": "#9aa0a6", "s2_resid": "#d9b310"}
    for di, o in enumerate(OBS):
        vc = r01["observables"][o]["varcomp_anova_reml_closed_form"]
        if not vc:
            continue
        bottom = 0.0
        tot = sum(max(vc[c], 0) for c in comp_names)
        for c in comp_names:
            val = max(vc[c], 0) / tot * 100 if tot > 0 else 0
            ax.bar(di, val, bottom=bottom, color=colors[c],
                   label=c if di == 0 else None, edgecolor="white", linewidth=0.4)
            bottom += val
    ax.set_xticks(xs)
    ax.set_xticklabels([OBS_LABEL[o].split(" ")[0] for o in OBS], fontsize=8)
    ax.set_ylabel("% of explained variance")
    ax.set_title("REML variance components", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("UQ on protocol-vs-case variance ratio (lam=0.01, admissible, defects only)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    results = OrderedDict()
    results["meta"] = OrderedDict(
        metrics_csv=os.path.relpath(METRICS, ROOT),
        filter="admissible == True, defect cases only (exclude T0_*)",
        observables=OBS,
        n_boot=N_BOOT,
        ci_percentiles=list(CI),
        seed=SEED,
        statsmodels_available=HAVE_SM,
        statsmodels_version=SM_VERSION,
        estimator=("paper ratio = Var_pop(protocol means) / Var_pop(case means), "
                   "ddof=0, group means averaged over the other factor "
                   "(reproduces make_figures.py F2)."),
        note=("varcomp_* give the noise-corrected random-effects decomposition; "
              "the naive paper ratio does NOT subtract residual cell variance."),
    )

    res01, _ = analyse_lam(0.01, rng)
    res00, _ = analyse_lam(0.0, rng)
    results["lam=0.01"] = res01
    results["lam=0.0"] = res00

    # ---- write JSON ----
    jpath = os.path.join(OUTDIR, "uq.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[uq] wrote {jpath}")

    # ---- write compact CSV ----
    rows = []
    for lam_key in ["lam=0.01", "lam=0.0"]:
        R = results[lam_key]
        for o in OBS:
            ob = R["observables"][o]
            bc = ob["bootstrap"]["resample_cases"]
            bp = ob["bootstrap"]["resample_protocols"]
            bt = ob["bootstrap"]["resample_two_way"]
            lodo = ob["leave_one_defect_out"]
            vca = ob["varcomp_anova_reml_closed_form"] or {}
            vcm = ob["varcomp_statsmodels_mixedlm"] or {}
            def rnd(v, d=3):
                return None if v is None else round(v, d)
            rows.append(OrderedDict(
                lam=R["lam"],
                observable=o,
                point_ratio=round(ob["point_ratio"], 3),
                paper_reported=ob["paper_reported_ratio"],
                # case-resample: lower bound is the stable, load-bearing quantity
                boot_case_lo=rnd(bc["lo"]),
                boot_case_hi=("inf" if bc.get("hi_is_inf") else rnd(bc["hi"])),
                boot_case_hi_finite_p975=rnd(bc.get("hi_finite_p975")),
                boot_case_median=rnd(bc["median"]),
                boot_case_frac_inf=rnd(bc.get("frac_degenerate_inf"), 4),
                boot_case_P_gt_10=rnd(bc.get("P_gt_10"), 4),
                boot_case_P_gt_1=rnd(bc.get("P_gt_1"), 4),
                # protocol-resample
                boot_proto_lo=rnd(bp["lo"]),
                boot_proto_hi=("inf" if bp.get("hi_is_inf") else rnd(bp["hi"])),
                boot_proto_median=rnd(bp["median"]),
                # two-way resample
                boot_twoway_lo=rnd(bt["lo"]),
                boot_twoway_hi=("inf" if bt.get("hi_is_inf") else rnd(bt["hi"])),
                lodo_min=round(lodo["min"], 3),
                lodo_max=round(lodo["max"], 3),
                balanced_lc_median=rnd((ob.get("balanced_levelcount_check") or {}).get("median")),
                balanced_lc_P_gt_10=rnd((ob.get("balanced_levelcount_check") or {}).get("P_gt_10"), 4),
                ratio_noise_corrected=(round(vca.get("ratio_noise_corrected"), 3)
                                       if vca.get("ratio_noise_corrected") is not None
                                       and np.isfinite(vca.get("ratio_noise_corrected", np.inf))
                                       else vca.get("ratio_noise_corrected")),
                ratio_nc_boot_lo=rnd((ob.get("varcomp_anova_reml_bootstrap_ci") or {})
                                     .get("ratio_ci", {}).get("lo")),
                ratio_nc_boot_hi=(
                    "inf" if (ob.get("varcomp_anova_reml_bootstrap_ci") or {})
                    .get("ratio_ci", {}).get("hi_is_inf")
                    else rnd((ob.get("varcomp_anova_reml_bootstrap_ci") or {})
                             .get("ratio_ci", {}).get("hi"))),
                s2_protocol_reml=(round(vca.get("s2_protocol"), 6)
                                  if vca.get("s2_protocol") is not None else None),
                s2_case_reml=(round(vca.get("s2_case"), 6)
                              if vca.get("s2_case") is not None else None),
                s2_resid_reml=(round(vca.get("s2_resid"), 6)
                               if vca.get("s2_resid") is not None else None),
                mixedlm_ratio=(round(vcm.get("ratio_noise_corrected"), 3)
                               if isinstance(vcm.get("ratio_noise_corrected"), (int, float))
                               and np.isfinite(vcm.get("ratio_noise_corrected", np.inf))
                               else vcm.get("ratio_noise_corrected")),
            ))
    cpath = os.path.join(OUTDIR, "uq_summary.csv")
    pd.DataFrame(rows).to_csv(cpath, index=False)
    print(f"[uq] wrote {cpath}")

    # ---- figure ----
    fpath = os.path.join(OUTDIR, "uq.png")
    ok = make_figure(results, fpath)
    if ok:
        print(f"[uq] wrote {fpath}")

    def fmt_hi(d):
        return "inf" if d.get("hi_is_inf") else f"{d['hi']:.1f}"

    # ---- console headline ----
    print("\n================ HEADLINE (lam=0.01) ================")
    for o in OBS:
        ob = results["lam=0.01"]["observables"][o]
        bc = ob["bootstrap"]["resample_cases"]
        bp = ob["bootstrap"]["resample_protocols"]
        lodo = ob["leave_one_defect_out"]
        vca = ob["varcomp_anova_reml_closed_form"]
        nc = vca["ratio_noise_corrected"] if vca else float("nan")
        print(f"\n{o} ({OBS_LABEL[o]}):")
        print(f"  point ratio            = {ob['point_ratio']:.1f}  (paper {ob['paper_reported_ratio']})")
        print(f"  bootstrap (cases)   95% CI = [{bc['lo']:.1f}, {fmt_hi(bc)}]  median {bc['median']:.1f}  "
              f"(finite p97.5={bc['hi_finite_p975']:.0f}, frac_inf={bc['frac_degenerate_inf']:.3f})")
        print(f"      P(ratio>1)={bc['P_gt_1']:.3f}  P(ratio>10)={bc['P_gt_10']:.3f}  P(ratio>20)={bc['P_gt_20']:.3f}")
        print(f"  bootstrap (protocols) 95% CI = [{bp['lo']:.1f}, {fmt_hi(bp)}]  median {bp['median']:.1f}")
        print(f"  leave-one-defect-out range = [{lodo['min']:.1f}, {lodo['max']:.1f}]")
        vcb = ob.get("varcomp_anova_reml_bootstrap_ci", {}).get("ratio_ci", {})
        vcm = ob.get("varcomp_statsmodels_mixedlm") or {}
        rcb_hi = "inf" if vcb.get("hi_is_inf") else (f"{vcb.get('hi'):.1f}" if vcb.get("hi") is not None else "NA")
        mlm = (f"{vcm.get('ratio_noise_corrected'):.1f}"
               if isinstance(vcm.get("ratio_noise_corrected"), (int, float))
               and np.isfinite(vcm.get("ratio_noise_corrected", np.inf)) else "NA")
        print(f"  noise-corrected (closed-form REML) ratio = {nc:.1f}   "
              f"boot 95% CI = [{vcb.get('lo'):.1f}, {rcb_hi}]")
        print(f"  noise-corrected (statsmodels MixedLM)  ratio = {mlm}   "
              f"[s2_proto={vcm.get('s2_protocol'):.4g}, s2_case={vcm.get('s2_case'):.4g}, "
              f"s2_resid={vcm.get('s2_resid'):.4g}]" if mlm != "NA" else
              f"  noise-corrected (statsmodels MixedLM)  ratio = {mlm}")
        blc = ob.get("balanced_levelcount_check") or {}
        if blc:
            print(f"  balanced level-count (5 protocols vs 5 cases): median={blc['median']:.1f} "
                  f"[{blc['lo']:.1f},{blc['hi']:.1f}]  P(>10)={blc['P_gt_10']:.3f}  "
                  f"(rules out 'more protocol levels' artifact)")
    print("\n================ lam=0 comparison ================")
    for o in OBS:
        ob = results["lam=0.0"]["observables"][o]
        bc = ob["bootstrap"]["resample_cases"]
        lodo = ob["leave_one_defect_out"]
        vca = ob["varcomp_anova_reml_closed_form"]
        nc = vca["ratio_noise_corrected"] if vca else float("nan")
        print(f"  {o:16s}: point {ob['point_ratio']:6.1f}  boot(cases) [{bc['lo']:.1f},{fmt_hi(bc)}]  "
              f"LODO [{lodo['min']:.1f},{lodo['max']:.1f}]  REML {nc:.1f}")
    print("=====================================================")


if __name__ == "__main__":
    main()
