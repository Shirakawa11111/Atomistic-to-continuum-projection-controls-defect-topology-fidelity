"""Single-mode structural PFC for a TRIANGULAR lattice (non-honeycomb generality test).

The honeycomb structural PFC (``structural_pfc.StructuralPFCModel``) needs two
$C_2$ peaks (at $|G_1|$ and $\\sqrt3|G_1|$) plus the three-point term to select the
two-atom basis. The triangular lattice is the *canonical single-mode* PFC: a
single Gaussian peak in $\\hat C_2(k)$ at the first reciprocal magnitude
$|G_1|=4\\pi/(\\sqrt3\\,a)$ ($a$ = nearest-neighbour spacing), with the cubic
$-\\tfrac{\\eta}{6}n^3$ term selecting the triangular phase (three first-shell
wavevectors at $120^\\circ$ resonate). We switch the three-point term OFF ($X=0$).

We use this as a second, non-honeycomb continuum model to test whether the
atoms-to-field projection ranking (Voronoi-localised Gaussian $\\gg$ non-Gaussian)
transfers to a different lattice, where defects appear as 5-/7-fold
*coordination* rather than 5-/7-membered rings.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .sc_ch_model import SCCHModel


class TriangularPFCModel(SCCHModel):
    """Single-mode XPFC with one Gaussian $C_2$ peak at the triangular $|G_1|$."""

    def __init__(
        self,
        shape: Tuple[int, int],
        dx: float,
        n_init: Optional[np.ndarray],
        *,
        k1: float,
        c2_h: float = 1.05,
        c2_w: Optional[float] = None,
        eta: float = 0.25,
        chi: float = 1.0,
        morphological_constraint_weight: float = 0.0,
        vacuum_threshold_for_mask: float = 0.05,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._k1 = float(k1)
        self._c2_h = float(c2_h)
        self._c2_w = float(c2_w) if c2_w is not None else 0.10 * self._k1
        super().__init__(
            shape=shape, dx=dx, n_init=n_init,
            a0=1.0, R=1.0, X=0.0, m=3, eta=eta, chi=chi, r0_a0_ratio=1.0,
            morphological_constraint_weight=morphological_constraint_weight,
            vacuum_threshold_for_mask=vacuum_threshold_for_mask,
            x_bounds=x_bounds, y_bounds=y_bounds,
        )

    def _build_C2_hat(self) -> np.ndarray:
        k = np.sqrt(self.K2)
        return self._c2_h * np.exp(-((k - self._k1) ** 2) / (2 * self._c2_w ** 2))

    @classmethod
    def for_field(cls, field, dx, k1, *, x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), **kw):
        return cls(shape=field.shape, dx=dx, n_init=field, k1=k1,
                   x_bounds=x_bounds, y_bounds=y_bounds, **kw)


def triangular_k1(a_nd: float) -> float:
    """First reciprocal-lattice magnitude $|G_1|=4\\pi/(\\sqrt3\\,a)$ for NN spacing $a$."""
    return 4.0 * np.pi / np.sqrt(3.0) / a_nd
