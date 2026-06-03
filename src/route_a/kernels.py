"""Kernel axis K — atoms → raw density field ρ(r).

Four kernels spanning distinct mapping characters, all normalised so that
∫ρ dA = N_atoms:

* **K1** Dirac deposition + spectral hard low-pass (minimal, spectral).
* **K2** constant-width Gaussian smearing (uniform).
* **K3** Voronoi-cell-area-modulated Gaussian (Paper 1 baseline): wider
  Gaussians in larger cells, σᵢ ∝ √(areaᵢ/⟨area⟩), with near-defect modulation.
* **K4** Voronoi 1/area piecewise-constant ("cell indicator") + Gaussian smooth.

Cell areas use a pixel-Voronoi estimate (nearest-atom labelling via cKDTree):
robust, automatically box-clipped, and convergent to the exact clipped polygon
area as the grid is refined.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

from .nondim import GridSpec
from . import config


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _as_xy(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return p[:, :2] if p.shape[1] > 2 else p


def pixel_centers(grid: GridSpec):
    """Pixel-centre coordinates (uniform spacing dx on both axes)."""
    X = np.arange(grid.NX) * grid.dx
    Y = np.arange(grid.NY) * grid.dx
    return X, Y


def deposit_cic(points: np.ndarray, weights: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Cloud-in-cell deposition onto the grid (mass-conserving, box-clipped)."""
    pts = _as_xy(points)
    rho = np.zeros(grid.shape, dtype=np.float64)
    fx = pts[:, 0] / grid.dx
    fy = pts[:, 1] / grid.dx
    nx, ny = grid.NX, grid.NY
    valid = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
    fx, fy, w = fx[valid], fy[valid], np.asarray(weights, float)[valid]
    if fx.size == 0:
        return rho
    ix0, iy0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    wx, wy = fx - ix0, fy - iy0
    inside = (ix0 >= 0) & (ix0 < nx - 1) & (iy0 >= 0) & (iy0 < ny - 1)
    if np.any(inside):
        a, b = ix0[inside], iy0[inside]
        cx, cy, ww = wx[inside], wy[inside], w[inside]
        np.add.at(rho, (b, a), ww * (1 - cx) * (1 - cy))
        np.add.at(rho, (b, a + 1), ww * cx * (1 - cy))
        np.add.at(rho, (b + 1, a), ww * (1 - cx) * cy)
        np.add.at(rho, (b + 1, a + 1), ww * cx * cy)
    edge = ~inside
    if np.any(edge):
        ex = np.clip(np.round(fx[edge]).astype(int), 0, nx - 1)
        ey = np.clip(np.round(fy[edge]).astype(int), 0, ny - 1)
        np.add.at(rho, (ey, ex), w[edge])
    return rho


def nearest_atom_labels(points: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Label each pixel with its nearest atom index (pixel-Voronoi)."""
    pts = _as_xy(points)
    X, Y = pixel_centers(grid)
    XX, YY = np.meshgrid(X, Y)  # (ny, nx)
    q = np.column_stack([XX.ravel(), YY.ravel()])
    _, idx = cKDTree(pts).query(q)
    return idx.reshape(grid.shape)


def voronoi_cell_areas(points: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Pixel-Voronoi cell area per atom (non-dimensional units)."""
    labels = nearest_atom_labels(points, grid)
    counts = np.bincount(labels.ravel(), minlength=len(_as_xy(points)))
    return counts.astype(float) * grid.dx**2


def normalize_atom_count(rho: np.ndarray, grid: GridSpec, n_atoms: int) -> np.ndarray:
    """Scale so ∫ρ dA = n_atoms."""
    total = rho.sum() * grid.dx**2
    if total > 0:
        rho = rho * (n_atoms / total)
    return rho


def _defect_chi(points: np.ndarray, defects_nd: Optional[np.ndarray], radius: float) -> np.ndarray:
    """1.0 for atoms within ``radius`` of any defect centre, else 0.0."""
    pts = _as_xy(points)
    chi = np.zeros(len(pts))
    if defects_nd is None or len(defects_nd) == 0:
        return chi
    d = _as_xy(np.asarray(defects_nd, float))
    dd, _ = cKDTree(d).query(pts)
    chi[dd < radius] = 1.0
    return chi


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
class KernelInterface:
    """Map atom positions (non-dimensional) to a raw density field ρ (ny, nx)."""

    name: str = "K?"
    description: str = ""

    def map(
        self,
        atoms_nd: np.ndarray,
        grid: GridSpec,
        defects_nd: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}: {self.description}>"


class K1DiracLowpass(KernelInterface):
    name = "K1"
    description = "Dirac deposition + spectral hard low-pass"

    def __init__(self, k_cut: Optional[float] = None, sigma_ref: Optional[float] = None):
        # default cutoff matches the resolution of a σ_normal Gaussian: k_cut ~ 1/σ
        self.sigma_ref = sigma_ref if sigma_ref is not None else config.VORONOI_PARAMS["sigma_normal"]
        self.k_cut = k_cut if k_cut is not None else 1.0 / self.sigma_ref

    def map(self, atoms_nd, grid, defects_nd=None):
        pts = _as_xy(atoms_nd)
        rho = deposit_cic(pts, np.ones(len(pts)), grid)
        kx = np.fft.fftfreq(grid.NX, d=grid.dx) * 2 * np.pi
        ky = np.fft.fftfreq(grid.NY, d=grid.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        mask = (KX**2 + KY**2) <= self.k_cut**2
        rho = np.fft.ifft2(np.fft.fft2(rho) * mask).real
        rho = np.clip(rho, 0.0, None)  # remove ringing undershoot
        return normalize_atom_count(rho, grid, len(pts))


class K2ConstGaussian(KernelInterface):
    name = "K2"
    description = "constant-width Gaussian smearing"

    def __init__(self, sigma: Optional[float] = None):
        self.sigma = sigma if sigma is not None else config.VORONOI_PARAMS["sigma_normal"]

    def map(self, atoms_nd, grid, defects_nd=None):
        pts = _as_xy(atoms_nd)
        rho = deposit_cic(pts, np.ones(len(pts)), grid)
        rho = gaussian_filter(rho, sigma=self.sigma / grid.dx, mode="constant")
        return normalize_atom_count(rho, grid, len(pts))


class K3VoronoiGaussian(KernelInterface):
    name = "K3"
    description = "Voronoi-area-modulated Gaussian (Paper 1)"

    def __init__(self, **voronoi_params):
        p = dict(config.VORONOI_PARAMS)
        p.update(voronoi_params)
        self.sigma_n = p["sigma_normal"]
        self.sigma_d = p["sigma_defect"]
        self.w_n = p["w_normal"]
        self.w_d = p["w_defect"]
        self.defect_radius = p["defect_radius"]

    def map(self, atoms_nd, grid, defects_nd=None):
        pts = _as_xy(atoms_nd)
        areas = voronoi_cell_areas(pts, grid)
        avg = areas[areas > 0].mean() if np.any(areas > 0) else 1.0
        chi = _defect_chi(pts, defects_nd, self.defect_radius)
        sigma = self.sigma_n + chi * (self.sigma_d - self.sigma_n)
        sigma = sigma * np.sqrt(np.maximum(areas, 1e-12) / avg)
        weights = self.w_n + chi * (self.w_d - self.w_n)

        # group atoms by (rounded) sigma to limit the number of convolutions
        rho = np.zeros(grid.shape, dtype=np.float64)
        keys = np.round(sigma / (grid.dx * 1e-3)).astype(np.int64)
        for key in np.unique(keys):
            m = keys == key
            s = float(np.mean(sigma[m]))
            base = deposit_cic(pts[m], weights[m], grid)
            rho += gaussian_filter(base, sigma=max(s / grid.dx, 1e-3), mode="constant")
        return normalize_atom_count(rho, grid, len(pts))


class K4CellIndicator(KernelInterface):
    name = "K4"
    description = "Voronoi 1/area piecewise-constant + Gaussian smooth"

    def __init__(self, sigma_smooth: Optional[float] = None):
        self.sigma_smooth = (
            sigma_smooth if sigma_smooth is not None else config.VORONOI_PARAMS["sigma_normal"]
        )

    def map(self, atoms_nd, grid, defects_nd=None):
        pts = _as_xy(atoms_nd)
        labels = nearest_atom_labels(pts, grid)
        areas = voronoi_cell_areas(pts, grid)
        inv_area = np.where(areas > 0, 1.0 / np.maximum(areas, 1e-12), 0.0)
        rho = inv_area[labels]  # piecewise-constant: each cell integrates to ~1 atom
        rho = gaussian_filter(rho, sigma=self.sigma_smooth / grid.dx, mode="nearest")
        return normalize_atom_count(rho, grid, len(pts))


KERNELS = {
    "K1": K1DiracLowpass,
    "K2": K2ConstGaussian,
    "K3": K3VoronoiGaussian,
    "K4": K4CellIndicator,
}


def get_kernel(name: str, **kwargs) -> KernelInterface:
    if name not in KERNELS:
        raise KeyError(f"unknown kernel {name!r}; choose from {list(KERNELS)}")
    return KERNELS[name](**kwargs)
