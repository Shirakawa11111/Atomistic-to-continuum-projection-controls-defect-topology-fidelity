"""LAMMPS I/O and AIREBO script templates.

Generators emit ``*.data`` files; the HPC stage runs 0 K minimisation (the
relaxed reference) and a short finite-T MD (R1 density snapshots). Readers parse
the relaxed dump back into atom arrays for the MD→ψ mapping.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# writers / readers
# --------------------------------------------------------------------------
def write_lammps_data(
    path: str,
    atoms: np.ndarray,
    box: Tuple[float, float, float, float, float, float],
    atom_type: int = 1,
    mass: float = 12.011,
) -> None:
    """Write an ``atomic``-style LAMMPS data file."""
    a = np.asarray(atoms, float)
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(len(a))])
    xlo, xhi, ylo, yhi, zlo, zhi = box
    lines = [
        "# Route A case — graphene (AIREBO)",
        "",
        f"{len(a)} atoms",
        "1 atom types",
        "",
        f"{xlo:.6f} {xhi:.6f} xlo xhi",
        f"{ylo:.6f} {yhi:.6f} ylo yhi",
        f"{zlo:.6f} {zhi:.6f} zlo zhi",
        "",
        "Masses",
        "",
        f"1 {mass}",
        "",
        "Atoms  # atomic",
        "",
    ]
    for i, (x, y, z) in enumerate(a, start=1):
        lines.append(f"{i} {atom_type} {x:.6f} {y:.6f} {z:.6f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def read_lammps_data(path: str):
    """Parse an ``atomic``-style data file → (atoms (N,3), box)."""
    with open(path) as fh:
        lines = fh.readlines()
    box = [0.0] * 6
    atoms = []
    i = 0
    in_atoms = False
    while i < len(lines):
        s = lines[i].strip()
        if s.endswith("xlo xhi"):
            box[0], box[1] = map(float, s.split()[:2])
        elif s.endswith("ylo yhi"):
            box[2], box[3] = map(float, s.split()[:2])
        elif s.endswith("zlo zhi"):
            box[4], box[5] = map(float, s.split()[:2])
        elif s.startswith("Atoms"):
            in_atoms = True
            i += 2
            continue
        elif in_atoms and s and not s.startswith("#"):
            p = s.split()
            if len(p) >= 5:
                atoms.append([float(p[2]), float(p[3]), float(p[4])])
            else:
                in_atoms = False
        i += 1
    return np.asarray(atoms, float), tuple(box)


def read_lammps_dump(path: str, last_only: bool = True):
    """Parse a LAMMPS text dump (id type x y z). Returns (atoms, box) of the last
    frame, or a list of (atoms, box) frames if ``last_only`` is False."""
    frames: List[Tuple[np.ndarray, tuple]] = []
    with open(path) as fh:
        lines = fh.readlines()
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            natoms = int(lines[i + 1])
            i += 2
            continue
        if lines[i].startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i + 1].split()[:2])
            ylo, yhi = map(float, lines[i + 2].split()[:2])
            zlo, zhi = map(float, lines[i + 3].split()[:2])
            box = (xlo, xhi, ylo, yhi, zlo, zhi)
            i += 4
            continue
        if lines[i].startswith("ITEM: ATOMS"):
            cols = lines[i].split()[2:]
            ix, iy, iz = cols.index("x"), cols.index("y"), cols.index("z")
            coords = []
            for k in range(1, natoms + 1):
                p = lines[i + k].split()
                coords.append([float(p[ix]), float(p[iy]), float(p[iz])])
            frames.append((np.asarray(coords, float), box))
            i += natoms + 1
            continue
        i += 1
    if not frames:
        raise ValueError(f"no frames parsed from dump {path}")
    return frames[-1] if last_only else frames


# --------------------------------------------------------------------------
# AIREBO script templates
# --------------------------------------------------------------------------
def minimize_script(
    data_file: str,
    relaxed_dump: str,
    relaxed_data: str,
    potential: str = "CH.airebo",
    periodic: Tuple[bool, bool, bool] = (True, True, False),
    etol: float = 1e-8,
    ftol: float = 1e-8,
    maxiter: int = 100000,
) -> str:
    bx = "p" if periodic[0] else "s"
    by = "p" if periodic[1] else "s"
    bz = "p" if periodic[2] else "f"
    return f"""# Route A — 0 K AIREBO minimisation (relaxed reference)
units           metal
dimension       3
boundary        {bx} {by} {bz}
atom_style      atomic
read_data       {data_file}
pair_style      airebo 3.0 1 1
pair_coeff      * * {potential} C
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
thermo          100
thermo_style    custom step pe fmax lx ly
min_style       cg
minimize        {etol} {ftol} {maxiter} {maxiter * 10}
fix             relax all box/relax x 0.0 y 0.0 couple none vmax 0.001
minimize        {etol} {ftol} {maxiter} {maxiter * 10}
unfix           relax
min_style       cg
minimize        {etol} {ftol} {maxiter} {maxiter * 10}
write_dump      all custom {relaxed_dump} id type x y z modify sort id
write_data      {relaxed_data}
print           "MINIMIZE_DONE pe = $(pe) atoms = $(atoms)"
"""


def md_snapshots_script(
    data_file: str,
    dump_pattern: str,
    potential: str = "CH.airebo",
    temperature: float = 300.0,
    periodic: Tuple[bool, bool, bool] = (True, True, False),
    equil_steps: int = 20000,
    sample_steps: int = 40000,
    dump_every: int = 2000,
    seed: int = 12345,
) -> str:
    """Short finite-T NVT run dumping snapshots for the R1 time-average."""
    bx = "p" if periodic[0] else "s"
    by = "p" if periodic[1] else "s"
    bz = "p" if periodic[2] else "f"
    return f"""# Route A — finite-T MD for R1 time-average density
units           metal
dimension       3
boundary        {bx} {by} {bz}
atom_style      atomic
read_data       {data_file}
pair_style      airebo 3.0 1 1
pair_coeff      * * {potential} C
neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
velocity        all create {temperature} {seed} mom yes rot yes dist gaussian
fix             nvt all nvt temp {temperature} {temperature} 0.1
thermo          1000
thermo_style    custom step temp pe
timestep        0.0005
run             {equil_steps}
dump            d1 all custom {dump_every} {dump_pattern} id type x y z
dump_modify     d1 sort id
run             {sample_steps}
print           "MD_DONE T = {temperature}"
"""
