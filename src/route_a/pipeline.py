"""Pipeline: Pipeline(K, R, P)(case) → metrics, plus the experiment matrix,
variance decomposition, and the pre-registered discrimination gate.

The discrimination criterion **D** (declared before production data exist):
  D1 (magnitude): for ≥1 case and ≥1 headline observable, the protocol-induced
       coefficient of variation CV_protocol ≥ 0.20;  AND
  D2 (specificity): the protocol ranking on a headline observable changes between
       ≥2 cases (Spearman rank correlation < 0.5).
D met  → defect-type decision tree (Outcome A).
D not  → robustness certificate (Outcome B).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from . import metrics as M
from .cases import Case
from .config import RELAX_TOL, REFERENCE_STEPS, DEFAULT_NX_LONG_EDGE, REFERENCE_XPFC_PARAMS
from .kernels import get_kernel
from .nondim import Nondimensionalizer, setup_grid
from .postprocess import get_postprocess
from .reference_density import get_reference
from .sc_ch_model import SCCHModel


@dataclass(frozen=True)
class Protocol:
    K: str
    R: str
    P: str

    def __str__(self) -> str:
        return f"{self.K}/{self.R}/{self.P}"


BASELINE = Protocol("K3", "R2", "P4")


def reduced_protocol_matrix() -> List[Protocol]:
    """~18 protocols: K×P under R2 (16) + the R axis under K3/P2 (R1, R3)."""
    protos = [Protocol(K, "R2", P)
              for K in ("K1", "K2", "K3", "K4")
              for P in ("P1", "P2", "P3", "P4")]
    protos.append(Protocol("K3", "R1", "P2"))
    protos.append(Protocol("K3", "R3", "P2"))
    return protos


def _median_nn(points_nd: np.ndarray) -> float:
    p = np.asarray(points_nd, float)[:, :2]
    if len(p) < 2:
        return 1.0
    d, _ = cKDTree(p).query(p, k=2)
    return float(np.median(d[:, 1]))


def _model_relax_factory(grid, max_steps: int, tol: float,
                         overrides: Optional[Dict] = None) -> Callable[[np.ndarray], np.ndarray]:
    def relax(rho0: np.ndarray) -> np.ndarray:
        m = SCCHModel.reference(rho0, dx=grid.dx, x_bounds=grid.x_bounds,
                                y_bounds=grid.y_bounds, **(overrides or {}))
        m.relax(tol=tol, max_steps=max_steps)
        return m.n
    return relax


def run_protocol(
    case: Case,
    protocol: Protocol,
    *,
    nx_long_edge: int = DEFAULT_NX_LONG_EDGE,
    relax_tol: float = RELAX_TOL,
    relax_max_steps: int = REFERENCE_STEPS,
    relax_chunk: int = 200,
    md_snapshots: Optional[List[np.ndarray]] = None,
    relaxed_atoms: Optional[np.ndarray] = None,
    model_overrides: Optional[Dict] = None,
) -> Dict:
    """Map ``case`` through ``protocol``, relax under the fixed model, return metrics.

    ``relaxed_atoms`` (from LAMMPS) supersedes ``case.atoms`` when provided.
    ``md_snapshots`` (finite-T MD, physical Å) feed the true R1 time-average.
    """
    atoms = relaxed_atoms if relaxed_atoms is not None else case.atoms
    nd = Nondimensionalizer(
        (case.box[0], case.box[1]), (case.box[2], case.box[3])
    )
    atoms_nd = nd.nondimensionalize_coords(atoms)
    defects_nd = (
        nd.nondimensionalize_2d(case.defect_centers) if case.defect_centers else None
    )
    perfect_nd = (
        nd.nondimensionalize_coords(case.perfect_atoms)
        if case.perfect_atoms is not None else None
    )
    snaps_nd = (
        [nd.nondimensionalize_coords(s) for s in md_snapshots] if md_snapshots else None
    )
    grid = setup_grid(nd.Lx_nd, nd.Ly_nd, nx_long_edge)

    kernel = get_kernel(protocol.K)
    post = get_postprocess(protocol.P)
    rho = kernel.map(atoms_nd, grid, defects_nd=defects_nd)

    rho0 = None
    r1_fell_back = False
    if post.needs_reference:
        ref = get_reference(protocol.R)
        rho0 = ref.compute(
            kernel, grid,
            perfect_atoms_nd=perfect_nd,
            md_snapshots_nd=snaps_nd,
            model_relax=_model_relax_factory(grid, relax_max_steps, relax_tol, model_overrides),
            defects_nd=defects_nd,
        )
        r1_fell_back = bool(getattr(ref, "fell_back", False))

    field_init = post.apply(rho, rho0=rho0)
    model = SCCHModel.reference(
        field_init, dx=grid.dx, x_bounds=grid.x_bounds, y_bounds=grid.y_bounds,
        **(model_overrides or {})
    )
    lam = (model_overrides or {}).get(
        "morphological_constraint_weight",
        REFERENCE_XPFC_PARAMS["morphological_constraint_weight"],
    )
    F_init = model.free_energy()
    res = model.relax(tol=relax_tol, max_steps=relax_max_steps, chunk=relax_chunk)
    field_relaxed = model.n.copy()
    # protocol-admissibility gate: did the field stay in the model's valid range?
    admissible = bool(
        model.check_density_health()
        and np.isfinite(res["F_final"]) and abs(res["F_final"]) < 1e6
    )

    min_sep = 0.7 * _median_nn(atoms_nd)
    topo = M.topology_metrics(field_init, field_relaxed, grid, min_sep_nd=min_sep)
    fidelity = M.init_fidelity_metrics(field_init, atoms_nd, grid, min_sep)

    row = dict(
        case=case.name,
        K=protocol.K, R=protocol.R, P=protocol.P, protocol=str(protocol),
        lam=lam,
        n_atoms=case.n_atoms,
        nx=grid.NX, ny=grid.NY, dx=grid.dx,
        F_init=F_init, F_final=res["F_final"], rel_drop=res["rel_drop"],
        converged=res["converged"], steps=res["steps"],
        admissible=admissible,
        conservation=M.density_conservation(rho, grid, case.n_atoms),
        roughness_init=M.field_roughness(field_init, grid),
        highk_init=M.spectral_highk_fraction(field_init, grid),
        l2_change=M.field_l2_change(field_init, field_relaxed),
        r1_fell_back=r1_fell_back,
        **topo, **fidelity,
    )
    return row


def run_matrix(
    cases: List[Case],
    protocols: Optional[List[Protocol]] = None,
    *,
    seeds: Optional[List[int]] = None,
    progress: bool = True,
    **kwargs,
) -> List[Dict]:
    """Run all (case × protocol [× seed]) combinations; return metric rows."""
    protocols = protocols or reduced_protocol_matrix()
    seeds = seeds or [0]
    rows: List[Dict] = []
    total = len(cases) * len(protocols) * len(seeds)
    i = 0
    for case in cases:
        for proto in protocols:
            for seed in seeds:
                i += 1
                if progress:
                    print(f"[{i:3d}/{total}] {case.name:20s} {proto}  seed={seed}")
                row = run_protocol(case, proto, **kwargs)
                row["seed"] = seed
                rows.append(row)
    return rows


# --- analysis -------------------------------------------------------------
def _finite(values) -> np.ndarray:
    v = np.asarray(values, float)
    return v[np.isfinite(v)]


def protocol_cv_by_case(rows: List[Dict], observable: str) -> Dict[str, float]:
    """Per case: CV of ``observable`` across protocols (std / |mean|)."""
    out: Dict[str, float] = {}
    cases = sorted({r["case"] for r in rows})
    for c in cases:
        vals = _finite([r[observable] for r in rows if r["case"] == c])
        if len(vals) >= 2 and abs(vals.mean()) > 1e-300:
            out[c] = float(vals.std() / abs(vals.mean()))
        else:
            out[c] = float("nan")
    return out


def variance_components(rows: List[Dict], observable: str) -> Dict[str, float]:
    """Two-way decomposition of ``observable`` into protocol / case / residual."""
    vals = np.array([r[observable] for r in rows], float)
    cases = np.array([r["case"] for r in rows])
    protos = np.array([r["protocol"] for r in rows])
    m = np.isfinite(vals)
    vals, cases, protos = vals[m], cases[m], protos[m]
    if len(vals) < 3:
        return dict(total=float("nan"), sigma2_protocol=float("nan"),
                    sigma2_case=float("nan"), sigma2_resid=float("nan"))
    grand = vals.mean()
    proto_eff = {p: vals[protos == p].mean() - grand for p in np.unique(protos)}
    case_eff = {c: vals[cases == c].mean() - grand for c in np.unique(cases)}
    s2_proto = float(np.var([proto_eff[p] for p in protos]))
    s2_case = float(np.var([case_eff[c] for c in cases]))
    resid = vals - grand - np.array([proto_eff[p] for p in protos]) - np.array([case_eff[c] for c in cases])
    return dict(
        total=float(np.var(vals)),
        sigma2_protocol=s2_proto,
        sigma2_case=s2_case,
        sigma2_resid=float(np.var(resid)),
    )


def add_relative_observables(rows: List[Dict], baseline_prefix: str = "T0") -> List[Dict]:
    """Add scale-invariant formation-energy proxy ``Ef_rel`` per (protocol, case).

    ``Ef_rel = (F_final[case] − F_final[T0]) / |F_final[T0]|`` using the *same
    protocol*'s pristine baseline. This is commensurable across protocols even
    when the field scale differs wildly (e.g. P1 raw vs P4 normalised), unlike
    the raw extensive free energy F_final.
    """
    # key by (protocol, λ) so the two λ branches do NOT collide on case name
    by: Dict[tuple, Dict[str, Dict]] = {}
    for r in rows:
        by.setdefault((r["protocol"], r.get("lam", None)), {})[r["case"]] = r
    for _key, d in by.items():
        base = next((rr for c, rr in d.items() if c.startswith(baseline_prefix)), None)
        Fb = base["F_final"] if base else None
        Fb_ok = (Fb is not None and np.isfinite(Fb) and abs(Fb) > 1e-300
                 and bool(base.get("admissible", True)))
        for r in d.values():
            if Fb_ok and r.get("admissible", True) and np.isfinite(r["F_final"]):
                r["Ef_rel"] = (r["F_final"] - Fb) / abs(Fb)
            else:
                r["Ef_rel"] = float("nan")
    return rows


def discrimination_verdict(
    rows: List[Dict],
    observables: Tuple[str, ...] = ("Ef_rel", "ring_l1_init", "l2_change", "roughness_init"),
    cv_threshold: float = 0.20,
    exclude_cases: Tuple[str, ...] = ("T0",),
    admissible_only: bool = True,
) -> Dict:
    """Evaluate the pre-registered criterion D over scale-robust headline observables,
    restricted to admissible protocols (diverged P1/P2 runs excluded)."""
    if "Ef_rel" in observables and rows and "Ef_rel" not in rows[0]:
        add_relative_observables(rows)
    n_total = len(rows)
    if admissible_only:
        rows = [r for r in rows if r.get("admissible", True)]
    n_admissible = len(rows)
    rows = [r for r in rows
            if not any(r["case"].startswith(x) for x in exclude_cases)]
    d1_hits = []
    max_cv = 0.0
    for obs in observables:
        cvbycase = protocol_cv_by_case(rows, obs)
        for c, cv in cvbycase.items():
            if math.isfinite(cv):
                max_cv = max(max_cv, cv)
                if cv >= cv_threshold:
                    d1_hits.append((obs, c, cv))
    # D2: protocol ranking changes across cases (use first observable available)
    d2_met, min_rho = False, 1.0
    cases = sorted({r["case"] for r in rows})
    protos = sorted({r["protocol"] for r in rows})
    for obs in observables:
        ranks = {}
        for c in cases:
            vals = {r["protocol"]: r[obs] for r in rows if r["case"] == c}
            v = np.array([vals.get(p, np.nan) for p in protos], float)
            if np.isfinite(v).sum() >= 3:
                ranks[c] = v
        cs = [c for c in cases if c in ranks]
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                rho = _spearman(ranks[cs[a]], ranks[cs[b]])
                if math.isfinite(rho):
                    min_rho = min(min_rho, rho)
                    if rho < 0.5:
                        d2_met = True
    d1_met = len(d1_hits) > 0
    return dict(
        D1_met=d1_met, D2_met=d2_met, D_met=(d1_met and d2_met),
        max_cv=max_cv, cv_threshold=cv_threshold, d1_hits=d1_hits,
        min_rank_corr=min_rho,
        n_admissible=n_admissible, n_total=n_total,
        observables=list(observables),
        outcome="A: decision tree" if (d1_met and d2_met) else "B: robustness certificate",
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m]))
    rb = np.argsort(np.argsort(b[m]))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rows_to_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
