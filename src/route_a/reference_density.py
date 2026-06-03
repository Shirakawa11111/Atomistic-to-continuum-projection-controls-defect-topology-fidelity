"""Reference-density axis R — the field ρ₀(r) used to normalise ρ→ψ.

ρ₀ is **field-valued** (not a scalar) and is mapped through the *same* kernel K
as the structure being benchmarked, so ψ = ρ/ρ₀ − 1 measures the structure's
deviation from a reference *as seen through the same instrument*. A scalar ρ₀
would make the three levels nearly degenerate; the field form makes R1 (thermally
smeared) genuinely differ from R2 (sharp perfect crystal).

Structural note: ρ₀ is consumed **only by postprocess P2**. Under P1/P3/P4 the
R axis has no effect — this is a real degeneracy of the protocol family and is
reported as such.

Levels:
* **R1** finite-T MD time-average: mean of mapped snapshots from a short ~300 K
  run. Falls back to a single snapshot (flagged) when trajectories are absent.
* **R2** perfect-crystal reference: a pristine lattice mapped through K.
* **R3** model-equilibrium reference: R2's field relaxed under the fixed model.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .kernels import KernelInterface
from .nondim import GridSpec


class RefDensityInterface:
    name: str = "R?"
    description: str = ""

    def compute(
        self,
        kernel: KernelInterface,
        grid: GridSpec,
        *,
        perfect_atoms_nd: Optional[np.ndarray] = None,
        md_snapshots_nd: Optional[List[np.ndarray]] = None,
        model_relax: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        defects_nd: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}: {self.description}>"


class R1MDTimeAverage(RefDensityInterface):
    name = "R1"
    description = "finite-T MD time-average density field"

    def __init__(self):
        self.fell_back = False  # set True if no trajectory was available

    def compute(self, kernel, grid, *, perfect_atoms_nd=None, md_snapshots_nd=None,
                model_relax=None, defects_nd=None):
        if md_snapshots_nd:
            acc = np.zeros(grid.shape, dtype=np.float64)
            for snap in md_snapshots_nd:
                acc += kernel.map(snap, grid, defects_nd=defects_nd)
            self.fell_back = False
            return acc / len(md_snapshots_nd)
        # fallback: single configuration (honest flag — NOT a true time average)
        if perfect_atoms_nd is None:
            raise ValueError("R1 needs md_snapshots_nd or a perfect_atoms_nd fallback")
        self.fell_back = True
        return kernel.map(perfect_atoms_nd, grid, defects_nd=defects_nd)


class R2PerfectCrystal(RefDensityInterface):
    name = "R2"
    description = "perfect-crystal reference density field"

    def compute(self, kernel, grid, *, perfect_atoms_nd=None, md_snapshots_nd=None,
                model_relax=None, defects_nd=None):
        if perfect_atoms_nd is None:
            raise ValueError("R2 needs perfect_atoms_nd")
        return kernel.map(perfect_atoms_nd, grid, defects_nd=None)


class R3ModelEquilibrium(RefDensityInterface):
    name = "R3"
    description = "model-equilibrium reference density field"

    def compute(self, kernel, grid, *, perfect_atoms_nd=None, md_snapshots_nd=None,
                model_relax=None, defects_nd=None):
        if perfect_atoms_nd is None or model_relax is None:
            raise ValueError("R3 needs perfect_atoms_nd and a model_relax callable")
        rho0 = kernel.map(perfect_atoms_nd, grid, defects_nd=None)
        return model_relax(rho0)


REFERENCES = {
    "R1": R1MDTimeAverage,
    "R2": R2PerfectCrystal,
    "R3": R3ModelEquilibrium,
}


def get_reference(name: str) -> RefDensityInterface:
    if name not in REFERENCES:
        raise KeyError(f"unknown reference {name!r}; choose from {list(REFERENCES)}")
    return REFERENCES[name]()
