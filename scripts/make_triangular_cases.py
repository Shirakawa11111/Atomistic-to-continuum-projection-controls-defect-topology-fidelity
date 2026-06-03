#!/usr/bin/env python3
"""Emit LAMMPS (2D LJ) input scripts for a triangular-lattice defect suite — the
non-honeycomb generality test. Each script builds a triangular LJ crystal, makes
a defect, relaxes (atoms + box), and dumps the relaxed configuration. Defects are
chosen to produce clean 5-/7-fold *coordination* signatures (the triangular-lattice
analog of the honeycomb 5-7 rings):

  TR0 pristine       — all 6-fold (control)
  TR1 monovacancy    — six 5-fold around the hole
  TR2 divacancy      — relaxes to a 5-7-5-7 cluster
  TR3 interstitial   — split interstitial, 7-fold core

Run on HPC: for f in experiments/triangular/in.TR*; do lmp -in $f; done
"""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "triangular")
os.makedirs(OUT, exist_ok=True)

NX, NY = 26, 15          # ~ square box (Ly_hex = NY*sqrt3 ~ NX) so grid resolution matches
RHO = 0.90               # reduced density ~ LJ-equilibrium triangular spacing

HEADER = f"""# 2D LJ triangular lattice — {{name}}
units           lj
dimension       2
boundary        p p p
atom_style      atomic
lattice         hex {RHO}
region          box block 0 {NX} 0 {NY} -0.25 0.25
create_box      1 box
create_atoms    1 box
mass            1 1.0
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
"""

# defect blocks act near the box centre; xlo/xhi/ylo/yhi are valid LAMMPS thermo keywords
DEFECTS = {
    "TR0_pristine": "",
    "TR1_vacancy": """
variable cx equal 0.5*(xlo+xhi)
variable cy equal 0.5*(ylo+yhi)
region hole sphere ${cx} ${cy} 0.0 0.30 units box
delete_atoms region hole compress yes
""",
    "TR2_void": """
variable cx equal 0.5*(xlo+xhi)
variable cy equal 0.5*(ylo+yhi)
region hole sphere ${cx} ${cy} 0.0 1.20 units box
delete_atoms region hole compress yes
""",
    "TR3_interstitial": """
variable cx equal 0.5*(xlo+xhi)+0.5
variable cy equal 0.5*(ylo+yhi)+0.30
create_atoms 1 single ${cx} ${cy} 0.0 units box
""",
}

FOOTER = """
# relax: atoms, then box+atoms, then atoms (cg throughout; 2D)
fix             enf2d all enforce2d
min_style       cg
minimize        1e-12 1e-12 20000 200000
fix             rbox all box/relax x 0.0 y 0.0
minimize        1e-12 1e-12 20000 200000
unfix           rbox
minimize        1e-12 1e-12 20000 200000
write_dump      all custom {out} id type x y z modify sort id
print           "DONE {name} natoms $(count(all))"
"""


def main():
    for name, defect in DEFECTS.items():
        out_dump = f"relaxed_{name}.dump"   # relative — LAMMPS writes in its run dir (HPC)
        script = HEADER.format(name=name)
        script += defect
        script += FOOTER.format(out=out_dump, name=name)
        path = os.path.join(OUT, f"in.{name}")
        with open(path, "w") as fh:
            fh.write(script)
        print("wrote", path)


if __name__ == "__main__":
    main()
