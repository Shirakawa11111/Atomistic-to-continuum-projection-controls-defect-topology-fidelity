"""Non-dimensionalization and grid setup — faithful to Paper 1's workflow.

The mapping pipeline runs in non-dimensional coordinates: atom positions are
scaled by ``1/Lmax`` so the long edge becomes ~1, the field grid is laid out on
that box, and the SC-CH model evolves there. This module reproduces
``Nondimensionalizer`` and ``GrapheneSimulation.setup_grid`` so the *fixed model*
sees exactly the Paper 1 conventions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


class Nondimensionalizer:
    """Scale physical (x, y) by 1/Lmax to a non-dimensional box (z preserved)."""

    def __init__(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min
        self.Lmax = max(self.Lx, self.Ly)
        self.scale = 1.0 / self.Lmax
        self.Lx_nd = self.Lx * self.scale
        self.Ly_nd = self.Ly * self.scale

    @classmethod
    def from_atoms(cls, atoms: np.ndarray, margin: float = 0.0) -> "Nondimensionalizer":
        a = np.asarray(atoms, dtype=float)
        xb = (a[:, 0].min() - margin, a[:, 0].max() + margin)
        yb = (a[:, 1].min() - margin, a[:, 1].max() + margin)
        return cls(xb, yb)

    def nondimensionalize_2d(self, coords: Sequence[Tuple[float, float]]) -> np.ndarray:
        out = []
        for c in coords:
            out.append(((c[0] - self.x_min) * self.scale, (c[1] - self.y_min) * self.scale))
        return np.asarray(out, dtype=float)

    def nondimensionalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Non-dimensionalize an (N, ≥2) array, preserving any trailing columns (e.g. z)."""
        a = np.asarray(coords, dtype=float).copy()
        a[:, 0] = (a[:, 0] - self.x_min) * self.scale
        a[:, 1] = (a[:, 1] - self.y_min) * self.scale
        return a

    def dimensionalize_field(self, phi_nd: np.ndarray, rho_ref: float) -> np.ndarray:
        return phi_nd * rho_ref + rho_ref

    def dimensionalize_grid(
        self, X_nd: np.ndarray, Y_nd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return X_nd * self.Lmax + self.x_min, Y_nd * self.Lmax + self.y_min


@dataclass
class GridSpec:
    NX: int
    NY: int
    dx: float
    Lx_nd: float
    Ly_nd: float

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.NY, self.NX)

    @property
    def x_bounds(self) -> Tuple[float, float]:
        return (0.0, self.Lx_nd)

    @property
    def y_bounds(self) -> Tuple[float, float]:
        return (0.0, self.Ly_nd)


def setup_grid(Lx_nd: float, Ly_nd: float, nx_long_edge: int = 512) -> GridSpec:
    """Reproduce ``GrapheneSimulation.setup_grid``.

    ``nx_long_edge`` points along the long edge; the short edge is proportional;
    ``NY`` is forced even (FFT-friendly); ``dx`` is set by the long edge.
    """
    NX = int(nx_long_edge)
    if Lx_nd >= Ly_nd:
        NY = int(round(NX * Ly_nd / Lx_nd))
    else:
        NY = int(round(NX * Lx_nd / Ly_nd))
    NY += NY % 2  # force even
    dx = Lx_nd / (NX - 1) if Lx_nd >= Ly_nd else Ly_nd / (NY - 1)
    return GridSpec(NX=NX, NY=NY, dx=dx, Lx_nd=Lx_nd, Ly_nd=Ly_nd)
