"""Properly-scaled structural PFC (peak C2 kernel) — the linchpin baseline.

The frozen SC-CH testbed uses a monotone $C_2(k)=-2R J_1(r_0k)/(r_0k)$ with no
finite-$k$ peak: it never crystallises, so atomic structure is held only by the
morphological constraint and 5-7 topology washes out under relaxation. This
model instead places Gaussian peaks in the two-point correlation at the
honeycomb reciprocal-lattice magnitudes $|G_1|$ (=k1) and $|G_2|=\\sqrt3|G_1|$,
with peak height $>1$ so modes near $k_1$ are linearly unstable and the field
crystallises a honeycomb at the *atomic* scale; the three-point term selects the
2-atom (3-fold) basis. We use it to test whether the mapping-protocol fidelity
ranking survives in a model that actually resolves the lattice.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .sc_ch_model import SCCHModel


class StructuralPFCModel(SCCHModel):
    """XPFC with a peaked two-point correlation at the honeycomb reciprocal lattice."""

    def __init__(
        self,
        shape: Tuple[int, int],
        dx: float,
        n_init: Optional[np.ndarray],
        *,
        k1: float,
        c2_h1: float = 1.2,
        c2_h2: float = 0.8,
        c2_w: Optional[float] = None,
        X: float = 2.5,
        eta: float = 0.25,
        chi: float = 0.5,
        morphological_constraint_weight: float = 0.0,
        vacuum_threshold_for_mask: float = 0.05,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        # set peak-kernel params BEFORE super().__init__ (it calls _build_C2_hat)
        self._k1 = float(k1)
        self._k2 = np.sqrt(3.0) * self._k1
        self._c2_h1 = float(c2_h1)
        self._c2_h2 = float(c2_h2)
        self._c2_w = float(c2_w) if c2_w is not None else 0.10 * self._k1
        # r0 places the J_3 three-point resonance at k1 (J_3 peaks near 4.20)
        r0 = 4.201 / self._k1
        super().__init__(
            shape=shape, dx=dx, n_init=n_init,
            a0=1.0, R=1.0, X=X, m=3, eta=eta, chi=chi, r0_a0_ratio=r0,
            morphological_constraint_weight=morphological_constraint_weight,
            vacuum_threshold_for_mask=vacuum_threshold_for_mask,
            x_bounds=x_bounds, y_bounds=y_bounds,
        )

    def _build_C2_hat(self) -> np.ndarray:
        k = np.sqrt(self.K2)
        c2 = (self._c2_h1 * np.exp(-((k - self._k1) ** 2) / (2 * self._c2_w ** 2))
              + self._c2_h2 * np.exp(-((k - self._k2) ** 2) / (2 * self._c2_w ** 2)))
        return c2

    @classmethod
    def for_field(cls, field, dx, k1, *, x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0), **kw):
        return cls(shape=field.shape, dx=dx, n_init=field, k1=k1,
                   x_bounds=x_bounds, y_bounds=y_bounds, **kw)


def honeycomb_k1(a0_nd: float) -> float:
    """First reciprocal-lattice magnitude $|G_1|=4\\pi/(\\sqrt3\\,a_0)$ (non-dim)."""
    return 4.0 * np.pi / np.sqrt(3.0) / a0_nd
