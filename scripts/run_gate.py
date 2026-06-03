#!/usr/bin/env python3
"""Discrimination gate on LAMMPS-relaxed structures, at λ=0.01 (reference) and λ=0.

Produces the pre-registered Outcome A/B verdict and the λ-robustness comparison
(the PRB mechanism evidence: is the protocol-sensitivity verdict an artefact of
the morphological constraint?).

    python scripts/run_gate.py --nx 256 --max-steps 4000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from route_a.cases import get_case                       # noqa: E402
from route_a.lammps_io import read_lammps_dump            # noqa: E402
from route_a.pipeline import (                             # noqa: E402
    reduced_protocol_matrix, run_protocol, add_relative_observables,
    discrimination_verdict, variance_components, protocol_cv_by_case, rows_to_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--lams", default="0.01,0.0")
    ap.add_argument("--relaxed-dir", default=os.path.join(ROOT, "experiments", "hpc_package"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "gate"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = [get_case(k) for k in ["T0", "T1", "T2", "T3", "T4", "T5"]]
    relaxed = {}
    for c in cases:
        p = os.path.join(args.relaxed_dir, f"relaxed_{c.name}.dump")
        if os.path.exists(p):
            atoms, _ = read_lammps_dump(p)
            relaxed[c.name] = atoms
    print(f"relaxed structures loaded: {sorted(relaxed)}", flush=True)

    protos = reduced_protocol_matrix()
    lams = [float(x) for x in args.lams.split(",")]
    summary = dict(nx=args.nx, max_steps=args.max_steps, lams=lams,
                   used_relaxed=sorted(relaxed), per_lam={})
    t0 = time.time()
    for lam in lams:
        rows = []
        n = len(cases) * len(protos)
        i = 0
        for c in cases:
            for p in protos:
                i += 1
                row = run_protocol(
                    c, p, nx_long_edge=args.nx, relax_max_steps=args.max_steps,
                    relaxed_atoms=relaxed.get(c.name),
                    model_overrides={"morphological_constraint_weight": lam},
                )
                rows.append(row)
                if i % 18 == 0:
                    print(f"  λ={lam}: {i}/{n}  ({c.name})  t={time.time()-t0:.0f}s", flush=True)
        add_relative_observables(rows)
        rows_to_csv(rows, os.path.join(args.out, f"metrics_lam{lam}.csv"))
        verdict = discrimination_verdict(rows)
        vc = {obs: variance_components(rows, obs)
              for obs in ["Ef_rel", "tau_57", "l2_change"]}
        cv = {obs: protocol_cv_by_case(rows, obs) for obs in ["Ef_rel", "tau_57", "l2_change"]}
        summary["per_lam"][str(lam)] = dict(verdict=verdict, variance_components=vc,
                                            protocol_cv_by_case=cv)
        print(f"λ={lam}: {verdict['outcome']}  D1={verdict['D1_met']} D2={verdict['D2_met']} "
              f"max_cv={verdict['max_cv']:.3f}", flush=True)

    # λ-robustness: do the two λ give the same outcome?
    outcomes = {k: v["verdict"]["outcome"][0] for k, v in summary["per_lam"].items()}
    summary["lambda_robust"] = len(set(outcomes.values())) == 1
    summary["outcomes_by_lam"] = outcomes
    with open(os.path.join(args.out, "gate_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\n=== GATE DONE ({time.time()-t0:.0f}s) ===", flush=True)
    for k in summary["per_lam"]:
        print(f"  λ={k}: {summary['per_lam'][k]['verdict']['outcome']}", flush=True)
    print(f"  λ-robust (same outcome both λ): {summary['lambda_robust']}", flush=True)
    print(f"  wrote {args.out}/gate_summary.json", flush=True)


if __name__ == "__main__":
    main()
