#!/usr/bin/env python3
"""Run the reduced protocol matrix over T0–T5 and write metrics + verdict.

By default uses *unrelaxed* generated structures — this validates the pipeline
and produces provisional numbers, NOT final science. Final runs pass
LAMMPS-relaxed structures (and finite-T MD snapshots) via --relaxed-dir.

    python scripts/run_local_matrix.py                 # quick pipeline check
    python scripts/run_local_matrix.py --production    # nx=512, full relax
    python scripts/run_local_matrix.py --relaxed-dir experiments/hpc_package
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from route_a.cases import get_case                      # noqa: E402
from route_a.lammps_io import read_lammps_dump           # noqa: E402
from route_a.config import REFERENCE_STEPS               # noqa: E402
from route_a.pipeline import (                            # noqa: E402
    reduced_protocol_matrix, run_matrix, add_relative_observables,
    discrimination_verdict, variance_components, protocol_cv_by_case, rows_to_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", action="store_true", help="nx=512, full relaxation")
    ap.add_argument("--relaxed-dir", default=None,
                    help="dir with relaxed_<case>.dump from LAMMPS (uses relaxed atoms)")
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "local_matrix"))
    args = ap.parse_args()

    nx = args.nx or (512 if args.production else 192)
    max_steps = args.max_steps or (REFERENCE_STEPS if args.production else 1500)
    os.makedirs(args.out, exist_ok=True)

    cases = [get_case(k) for k in ["T0", "T1", "T2", "T3", "T4", "T5"]]
    # optionally swap in LAMMPS-relaxed atoms
    relaxed = {}
    if args.relaxed_dir:
        for c in cases:
            p = os.path.join(args.relaxed_dir, f"relaxed_{c.name}.dump")
            if os.path.exists(p):
                atoms, _ = read_lammps_dump(p)
                relaxed[c.name] = atoms
        print(f"loaded relaxed structures for: {sorted(relaxed)}")

    protos = reduced_protocol_matrix()
    print(f"cases={len(cases)} protocols={len(protos)} nx={nx} max_steps={max_steps} "
          f"→ {len(cases) * len(protos)} runs")

    t0 = time.time()
    rows = []
    total = len(cases) * len(protos)
    i = 0
    for c in cases:
        for p in protos:
            i += 1
            print(f"[{i:3d}/{total}] {c.name:24s} {p}")
            row = run_matrix([c], [p], nx_long_edge=nx, relax_max_steps=max_steps,
                             relaxed_atoms=relaxed.get(c.name),
                             progress=False)[0]
            rows.append(row)
    add_relative_observables(rows)
    rows_to_csv(rows, os.path.join(args.out, "metrics.csv"))

    verdict = discrimination_verdict(rows)
    vc = {obs: variance_components(rows, obs)
          for obs in ["Ef_rel", "tau_57", "l2_change", "roughness_init"]}
    cv = {obs: protocol_cv_by_case(rows, obs) for obs in ["Ef_rel", "tau_57", "l2_change"]}
    summary = dict(nx=nx, max_steps=max_steps, n_rows=len(rows),
                   used_relaxed=sorted(relaxed), verdict=verdict,
                   variance_components=vc, protocol_cv_by_case=cv,
                   note="unrelaxed inputs unless --relaxed-dir given; not final science")
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\nran {len(rows)} rows in {time.time() - t0:.1f}s")
    print(f"verdict: {verdict['outcome']}  (D1={verdict['D1_met']} D2={verdict['D2_met']} "
          f"max_cv={verdict['max_cv']:.2f})")
    print(f"wrote {args.out}/metrics.csv and summary.json")


if __name__ == "__main__":
    main()
