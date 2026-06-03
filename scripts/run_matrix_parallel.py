#!/usr/bin/env python3
"""Parallel SC-CH matrix runner — distributes (case × protocol × λ) over N cores.

Each worker runs ONE pipeline evaluation single-threaded (numpy/scipy threads
pinned to 1); the parallelism is across the embarrassingly-parallel run list.
Designed to saturate the 128-core HPC.

    PYTHONPATH=src python3 scripts/run_matrix_parallel.py \
        --nx 512 --max-steps 20000 --lams 0.01,0.0 --ncores 120 \
        --relaxed-dir . --md-dir . --out outputs/production
"""
from __future__ import annotations

# pin BLAS/FFT threads to 1 BEFORE numpy is imported (so forked workers inherit)
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
import multiprocessing as mp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from route_a.cases import get_case                       # noqa: E402
from route_a.lammps_io import read_lammps_dump            # noqa: E402
from route_a.pipeline import (                             # noqa: E402
    reduced_protocol_matrix, run_protocol, Protocol, add_relative_observables,
    discrimination_verdict, variance_components, protocol_cv_by_case, rows_to_csv,
)

# module globals — set in main() before Pool() so forked workers inherit them
_CASES: dict = {}
_RELAXED: dict = {}
_MD: dict = {}


def _work(task):
    case_key, proto_tuple, lam, nx, max_steps = task
    c = _CASES[case_key]
    row = run_protocol(
        c, Protocol(*proto_tuple), nx_long_edge=nx, relax_max_steps=max_steps,
        relaxed_atoms=_RELAXED.get(case_key),
        md_snapshots=_MD.get(case_key),
        model_overrides={"morphological_constraint_weight": lam},
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--lams", default="0.01,0.0")
    ap.add_argument("--ncores", type=int, default=120)
    ap.add_argument("--relaxed-dir", default=ROOT)
    ap.add_argument("--md-dir", default=None, help="dir with md_<case>_*.dump for R1 snapshots")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "production"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = [get_case(k) for k in ["T0", "T1", "T2", "T3", "T4", "T5"]]
    global _CASES, _RELAXED, _MD
    _CASES = {c.name: c for c in cases}
    for c in cases:
        rp = os.path.join(args.relaxed_dir, f"relaxed_{c.name}.dump")
        if os.path.exists(rp):
            _RELAXED[c.name], _ = read_lammps_dump(rp)
        if args.md_dir:
            snaps = []
            for f in sorted(glob.glob(os.path.join(args.md_dir, f"md_{c.name}_*.dump"))):
                try:
                    atoms, _ = read_lammps_dump(f)
                    snaps.append(atoms)
                except Exception:
                    pass
            if snaps:
                _MD[c.name] = snaps
    print(f"relaxed: {sorted(_RELAXED)}  md_snapshots: "
          f"{{{', '.join(f'{k}:{len(v)}' for k, v in _MD.items())}}}", flush=True)

    protos = reduced_protocol_matrix()
    lams = [float(x) for x in args.lams.split(",")]
    tasks = [(c.name, (p.K, p.R, p.P), lam, args.nx, args.max_steps)
             for lam in lams for c in cases for p in protos]
    print(f"tasks={len(tasks)}  ncores={args.ncores}  nx={args.nx}  max_steps={args.max_steps}",
          flush=True)

    t0 = time.time()
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    with mp.Pool(args.ncores) as pool:
        rows = pool.map(_work, tasks)
    print(f"ran {len(rows)} rows in {time.time()-t0:.0f}s "
          f"({len(rows)/max(time.time()-t0,1):.1f} rows/s)", flush=True)

    add_relative_observables(rows)
    rows_to_csv(rows, os.path.join(args.out, "metrics.csv"))

    summary = dict(nx=args.nx, max_steps=args.max_steps, lams=lams, n_rows=len(rows),
                   relaxed=sorted(_RELAXED), md_snapshots={k: len(v) for k, v in _MD.items()},
                   per_lam={})
    for lam in lams:
        sub = [r for r in rows if abs(r["lam"] - lam) < 1e-12]
        verdict = discrimination_verdict(sub)
        adm = [r for r in sub if r.get("admissible", True)]  # diagnostics on admissible subset
        obs_list = ["Ef_rel", "ring_l1_init", "l2_change", "roughness_init"]
        vc = {o: variance_components(adm, o) for o in obs_list}
        cv = {o: protocol_cv_by_case(adm, o) for o in obs_list}
        n_adm_protos = len({r["protocol"] for r in adm})
        summary["per_lam"][str(lam)] = dict(verdict=verdict, variance_components=vc,
                                            protocol_cv_by_case=cv,
                                            n_admissible_protocols=n_adm_protos)
        print(f"λ={lam}: {verdict['outcome']}  D1={verdict['D1_met']} D2={verdict['D2_met']} "
              f"max_cv={verdict['max_cv']:.3f}", flush=True)
    outcomes = {k: v["verdict"]["outcome"][0] for k, v in summary["per_lam"].items()}
    summary["lambda_robust"] = len(set(outcomes.values())) == 1
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"λ-robust (same outcome both λ): {summary['lambda_robust']}", flush=True)
    print(f"wrote {args.out}/metrics.csv + summary.json", flush=True)


if __name__ == "__main__":
    main()
