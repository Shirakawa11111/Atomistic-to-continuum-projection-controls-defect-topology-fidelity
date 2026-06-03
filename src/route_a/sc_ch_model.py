"""SC-CH continuum testbed — the *fixed model* for the Route A benchmark.

This is a clean port of Paper 1's ``GrapheneXPFCModel`` (conserved /
Cahn–Hilliard dynamics with an XPFC two- and three-point structural kernel and a
morphological constraint toward the mapped field ``n0``). The numerics are kept
faithful to the audited reference so that the model is a *frozen control*: in
Route A we vary only the MD→ψ protocol that produces ``n_init``/``n0``.

Free energy (per the reference implementation)::

    F = ∫ [ ½ n² − (η/6) n³ + (χ/12) n⁴ ] dA          (ideal)
        − ½ ∫ n · IFFT(Ĉ₂ n̂) dA                       (two-point, C2)
        − Σ_j ⅓ ∫ n · s_j² dA,  s_j = IFFT(Ĉ₃ⱼ n̂)      (three-point, C3)
        + ½ λ ∫ mask · (n − n0)² dA                    (morphological constraint)

Evolution: ∂n/∂t = M ∇²μ, semi-implicit (C2 linear part treated implicitly).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.fft import fft2, ifft2
from scipy import special

from .config import (
    REFERENCE_XPFC_PARAMS,
    REFERENCE_EVOLUTION_PARAMS,
    REFERENCE_STEPS,
    RELAX_TOL,
)


class SCCHModel:
    """Constrained Cahn–Hilliard / XPFC continuum model (frozen reference)."""

    def __init__(
        self,
        shape: Tuple[int, int],
        dx: float,
        n_init: Optional[np.ndarray],
        a0: float,
        R: float,
        X: float,
        m: int,
        eta: float,
        chi: float,
        r0_a0_ratio: float,
        morphological_constraint_weight: float = 0.0,
        vacuum_threshold_for_mask: float = 0.05,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.ny, self.nx = shape
        self.shape = shape
        self.dx = float(dx)

        self.a0 = a0
        self.R = R
        self.X = X
        self.m = int(m)
        self.eta = eta
        self.chi = chi
        self.r0_a0_ratio = r0_a0_ratio
        self.r0 = self.r0_a0_ratio * self.a0

        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds

        self.n = np.zeros(self.shape, dtype=np.float64)
        if n_init is not None:
            if n_init.shape != self.shape:
                raise ValueError(
                    f"n_init shape {n_init.shape} mismatch with grid shape {self.shape}"
                )
            self.n[:] = n_init
        self.n0 = self.n.copy()  # morphological-constraint target (= the mapped field)

        self.morph_constraint_weight = morphological_constraint_weight
        self.vacuum_threshold_for_mask = vacuum_threshold_for_mask
        if self.morph_constraint_weight > 0:
            self.constraint_mask = (
                self.n0 >= self.vacuum_threshold_for_mask
            ).astype(float)
        else:
            self.constraint_mask = np.ones_like(self.n0)

        self.energy_curve: List[float] = []

        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.ny, d=self.dx) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self._C2k = self._build_C2_hat()
        self._C3 = self._build_C3_kernels()

    # -- factory ---------------------------------------------------------
    @classmethod
    def reference(
        cls,
        field: np.ndarray,
        dx: float,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
        **overrides,
    ) -> "SCCHModel":
        """Build the frozen reference model seeded by ``field``.

        ``overrides`` may patch reference parameters (use sparingly — Route A
        keeps the model fixed; this hook exists only for convergence studies
        that legitimately change ``dx``/``shape``).
        """
        params = dict(REFERENCE_XPFC_PARAMS)
        params.update(overrides)
        return cls(
            shape=field.shape,
            dx=dx,
            n_init=field,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            **params,
        )

    # -- kernels ---------------------------------------------------------
    def _build_C2_hat(self) -> np.ndarray:
        k = np.sqrt(self.K2)
        C2 = np.zeros_like(k)
        nz = k > 0
        C2[nz] = -2 * self.R * special.j1(self.r0 * k[nz]) / (self.r0 * k[nz])
        C2[~nz] = -self.R
        return C2

    def _build_C3_kernels(self) -> List[np.ndarray]:
        k = np.sqrt(self.K2)
        th = np.arctan2(self.KY, self.KX)
        Jm = np.where(k > 0, special.jv(self.m, self.r0 * k), 0.0)
        base = self.X * (1j**self.m) * Jm
        C3a = base * np.cos(self.m * th)
        C3b = base * np.sin(self.m * th)
        return [C3a, C3b]

    # -- thermodynamics --------------------------------------------------
    def free_energy(self, n: Optional[np.ndarray] = None) -> float:
        if n is None:
            n = self.n
        Fid = (
            np.sum(0.5 * n**2 - self.eta * n**3 / 6 + self.chi * n**4 / 12)
            * self.dx**2
        )
        nh = fft2(n)
        F2 = -0.5 * np.sum(n * ifft2(self._C2k * nh).real) * self.dx**2
        F3 = 0.0
        if self.X != 0 and self._C3:
            for k3 in self._C3:
                s = ifft2(k3 * nh).real
                F3 += -np.sum(n * s**2) * self.dx**2 / 3.0
        F_pfc = Fid + F2 + F3

        F_constraint = 0.0
        if self.morph_constraint_weight > 0:
            F_constraint = (
                0.5
                * self.morph_constraint_weight
                * np.sum(self.constraint_mask * (n - self.n0) ** 2)
                * self.dx**2
            )
        return float(F_pfc + F_constraint)

    def chemical_potential(self, n: Optional[np.ndarray] = None) -> np.ndarray:
        if n is None:
            n = self.n
        mu_id = n - 0.5 * self.eta * n**2 + (self.chi / 3.0) * n**3
        mu_2_contrib = ifft2(self._C2k * fft2(n)).real
        mu_3_contrib = np.zeros_like(n)
        if self.X != 0 and self._C3:
            nh = fft2(n)
            for k3 in self._C3:
                s_val = ifft2(k3 * nh).real
                mu_3_contrib += (
                    s_val**2 + ifft2(np.conj(k3) * fft2(2 * n * s_val)).real
                ) / 3.0
        # Three-body term enters with a MINUS sign: since F3 = −⅓ Σ_j ∫ n s_j² dA,
        # its functional derivative μ_C3 = δF3/δn = −⅓ Σ_j [s_j² + 2·IFFT(conj(Ĉ3j)·FFT(n s_j))]
        # is NEGATIVE. (Corrected 2026-06-03 from an earlier "+"; verified vs a
        # finite-difference gradient of free_energy. Audited impact on all reported
        # metrics ≤0.2% — the λ-pin and small dt dominate the dynamics — so prior
        # conclusions are unchanged.)
        mu_pfc = mu_id - mu_2_contrib - mu_3_contrib

        mu_constraint = 0.0
        if self.morph_constraint_weight > 0:
            mu_constraint = (
                self.morph_constraint_weight * self.constraint_mask * (n - self.n0)
            )
        return mu_pfc + mu_constraint

    # -- evolution -------------------------------------------------------
    def evolve(
        self,
        dt: float,
        steps: int,
        scheme: str = "semi-implicit",
        mobility: float = 1.0,
        record_every: int = 0,
        verbose: bool = False,
    ) -> np.ndarray:
        """Advance ``steps`` of conserved (Cahn–Hilliard) dynamics.

        Returns the evolved field. Records ``free_energy`` into
        ``self.energy_curve`` every ``record_every`` steps (0 disables, except
        the final step which is always recorded).
        """
        linear_op_implicit = -self._C2k  # C2 linear part treated implicitly
        for s in range(1, steps + 1):
            mu_total = self.chemical_potential(self.n)
            nh = fft2(self.n)
            if scheme.lower() in {"semi-implicit", "imex"}:
                mu_explicit = fft2(mu_total - ifft2(linear_op_implicit * nh).real)
                denom = 1.0 - mobility * dt * (-self.K2) * linear_op_implicit
                denom[denom == 0] = 1e-16
                nh_new = (nh + mobility * dt * (-self.K2) * mu_explicit) / denom
            elif scheme.lower() == "explicit":
                mu_h = fft2(mu_total)
                nh_new = nh - mobility * dt * (self.K2 * mu_h)
            else:
                raise ValueError(f"Unknown scheme: {scheme}")
            self.n = ifft2(nh_new).real

            if not self.check_density_health():
                if verbose:
                    print(f"[warn] unhealthy field at step {s}, aborting")
                break
            if record_every and (s % record_every == 0):
                self.energy_curve.append(self.free_energy())
        self.energy_curve.append(self.free_energy())
        return self.n

    def relax(
        self,
        tol: float = RELAX_TOL,
        max_steps: int = REFERENCE_STEPS,
        chunk: int = 200,
        dt: Optional[float] = None,
        mobility: Optional[float] = None,
        scheme: Optional[str] = None,
        verbose: bool = False,
    ) -> dict:
        """Evolve in chunks until ``|ΔF|/|F| < tol`` (per chunk) or ``max_steps``.

        Returns a dict with ``converged``, ``steps``, ``F_init``, ``F_final``,
        ``rel_drop`` and the recorded ``energy_curve``.
        """
        dt = REFERENCE_EVOLUTION_PARAMS["dt"] if dt is None else dt
        mobility = (
            REFERENCE_EVOLUTION_PARAMS["mobility"] if mobility is None else mobility
        )
        scheme = REFERENCE_EVOLUTION_PARAMS["scheme"] if scheme is None else scheme

        F_init = self.free_energy()
        self.energy_curve = [F_init]
        F_prev = F_init
        done = 0
        converged = False
        while done < max_steps:
            this = min(chunk, max_steps - done)
            self.evolve(dt=dt, steps=this, scheme=scheme, mobility=mobility)
            done += this
            F_now = self.free_energy()
            rel = abs(F_now - F_prev) / max(abs(F_prev), 1e-300)
            if verbose:
                print(f"  relax step {done:6d}  F={F_now:.6e}  rel={rel:.2e}")
            F_prev = F_now
            if not self.check_density_health():
                break
            if rel < tol:
                converged = True
                break
        F_final = self.free_energy()
        return dict(
            converged=converged,
            steps=done,
            F_init=F_init,
            F_final=F_final,
            rel_drop=(F_init - F_final) / max(abs(F_init), 1e-300),
            energy_curve=list(self.energy_curve),
        )

    def check_density_health(self) -> bool:
        n = self.n
        if np.isnan(n).any() or np.isinf(n).any():
            return False
        if np.abs(n).max() > 10:
            return False
        return True
