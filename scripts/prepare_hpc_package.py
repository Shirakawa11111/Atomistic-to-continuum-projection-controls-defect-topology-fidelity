#!/usr/bin/env python3
"""Generate the HPC package for T0–T5: LAMMPS data + AIREBO minimisation/MD
scripts + manifest + sync helper. Launch is gated (run sync/relax manually).

    python scripts/prepare_hpc_package.py

HPC = nanolab2023 via Tailscale (`ssh hpc`); LAMMPS at /usr/bin/lmp.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from route_a.cases import CASES, get_case            # noqa: E402
from route_a.lammps_io import (                       # noqa: E402
    write_lammps_data, minimize_script, md_snapshots_script,
)

PKG = os.path.join(ROOT, "experiments", "hpc_package")
POTENTIAL_SRC = (
    "/Users/bojingkai/Desktop/20250612-2/research_extensions/md_xpfc_routes/"
    "experiments/paper_A_remote_ready/CH.airebo"
)
REMOTE_ROOT = "route_A_protocol_robustness"  # under ssh hpc:~/


def main() -> None:
    os.makedirs(PKG, exist_ok=True)
    cases_dir = os.path.join(PKG, "cases")
    os.makedirs(cases_dir, exist_ok=True)

    manifest = {"hpc": "nanolab2023 (ssh hpc, Tailscale)", "lammps": "/usr/bin/lmp",
                "potential": "CH.airebo", "remote_root": REMOTE_ROOT, "cases": []}
    for key in ["T0", "T1", "T2", "T3", "T4", "T5"]:
        c = get_case(key)
        data = f"cases/{c.name}.data"
        min_in = f"minimize_{c.name}.in"
        md_in = f"md_{c.name}.in"
        relaxed_dump = f"relaxed_{c.name}.dump"
        relaxed_data = f"relaxed_{c.name}.data"
        md_dump = f"md_{c.name}_*.dump"
        write_lammps_data(os.path.join(cases_dir, f"{c.name}.data"), c.atoms, c.box)
        with open(os.path.join(PKG, min_in), "w") as fh:
            fh.write(minimize_script(data, relaxed_dump, relaxed_data, periodic=c.periodic))
        with open(os.path.join(PKG, md_in), "w") as fh:
            fh.write(md_snapshots_script(relaxed_data, f"md_{c.name}_*.dump", periodic=c.periodic))
        manifest["cases"].append(dict(
            name=c.name, key=key, n_atoms=c.n_atoms, data=data,
            minimize=min_in, md=md_in,
            relaxed_dump=relaxed_dump, relaxed_data=relaxed_data, md_dump=md_dump,
            defect_centers=c.defect_centers, description=c.description,
        ))

    with open(os.path.join(PKG, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # run-all helpers
    with open(os.path.join(PKG, "run_all_minimize.sh"), "w") as fh:
        fh.write("#!/bin/bash\nset -e\nLMP=${LMP:-/usr/bin/lmp}\n")
        for cm in manifest["cases"]:
            fh.write(f'echo "== minimise {cm["name"]} =="\n$LMP -in {cm["minimize"]}\n')
    with open(os.path.join(PKG, "run_all_md.sh"), "w") as fh:
        fh.write("#!/bin/bash\nset -e\nLMP=${LMP:-/usr/bin/lmp}\n")
        for cm in manifest["cases"]:
            fh.write(f'echo "== MD {cm["name"]} =="\n$LMP -in {cm["md"]}\n')
    with open(os.path.join(PKG, "sync_to_hpc.sh"), "w") as fh:
        fh.write(
            "#!/bin/bash\n# Sync package to HPC (gated — run manually after approval)\n"
            f"set -e\nrsync -avz --progress '{PKG}/' hpc:~/{REMOTE_ROOT}/\n"
            f'echo "synced to hpc:~/{REMOTE_ROOT}/"\n'
        )
    with open(os.path.join(PKG, "fetch_from_hpc.sh"), "w") as fh:
        fh.write(
            "#!/bin/bash\nset -e\n"
            f"rsync -avz 'hpc:~/{REMOTE_ROOT}/relaxed_*.dump' '{PKG}/'\n"
            f"rsync -avz 'hpc:~/{REMOTE_ROOT}/relaxed_*.data' '{PKG}/'\n"
            f"rsync -avz 'hpc:~/{REMOTE_ROOT}/md_*_*.dump' '{PKG}/' || true\n"
        )
    for s in ["run_all_minimize.sh", "run_all_md.sh", "sync_to_hpc.sh", "fetch_from_hpc.sh"]:
        os.chmod(os.path.join(PKG, s), 0o755)

    if os.path.exists(POTENTIAL_SRC):
        shutil.copy(POTENTIAL_SRC, os.path.join(PKG, "CH.airebo"))
        pot = "copied"
    else:
        pot = "MISSING — place CH.airebo in the package before syncing"

    print(f"HPC package written to {PKG}")
    print(f"  cases: {len(manifest['cases'])}  potential: {pot}")
    print("  next (gated): bash experiments/hpc_package/sync_to_hpc.sh  (after approval)")


if __name__ == "__main__":
    main()
