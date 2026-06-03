#!/usr/bin/env python3
"""Grid-convergence sweep for the recommended protocols (Richardson).

Recommended + baseline protocols × T0–T5 × nx ∈ {256,384,512,768}, λ=0.01,
parallel over cores. Lets us check convergence order ≥ 1 of the key observables.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, sys, time, multiprocessing as mp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from route_a.cases import get_case                       # noqa: E402
from route_a.lammps_io import read_lammps_dump            # noqa: E402
from route_a.pipeline import run_protocol, Protocol, add_relative_observables, rows_to_csv  # noqa: E402

_CASES, _RELAXED = {}, {}


def _work(t):
    ck, proto, nx = t
    r = run_protocol(_CASES[ck], Protocol(*proto), nx_long_edge=nx,
                     relax_max_steps=20000, relaxed_atoms=_RELAXED.get(ck))
    r["nx_sweep"] = nx
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncores", type=int, default=120)
    ap.add_argument("--relaxed-dir", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "convergence"))
    ap.add_argument("--nx-list", default="256,384,512,768")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    protocols = [("K2", "R2", "P3"), ("K3", "R2", "P3"), ("K3", "R2", "P4")]
    nx_list = [int(x) for x in args.nx_list.split(",")]
    cases = [get_case(k) for k in ["T0", "T1", "T2", "T3", "T4", "T5"]]
    global _CASES, _RELAXED
    _CASES = {c.name: c for c in cases}
    for c in cases:
        p = os.path.join(args.relaxed_dir, f"relaxed_{c.name}.dump")
        if os.path.exists(p):
            _RELAXED[c.name], _ = read_lammps_dump(p)
    tasks = [(c.name, p, nx) for c in cases for p in protocols for nx in nx_list]
    print(f"convergence tasks={len(tasks)} nx_list={nx_list}", flush=True)
    t0 = time.time()
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    with mp.Pool(args.ncores) as pool:
        rows = pool.map(_work, tasks)
    add_relative_observables(rows)
    rows_to_csv(rows, os.path.join(args.out, "convergence.csv"))
    print(f"convergence done {len(rows)} rows in {time.time()-t0:.0f}s -> {args.out}/convergence.csv",
          flush=True)


if __name__ == "__main__":
    main()
