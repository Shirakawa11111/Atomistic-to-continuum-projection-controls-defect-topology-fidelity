"""Postprocess axis P — raw density ρ → the field fed to the fixed model.

* **P1** raw ρ (no transform; deliberately naive baseline).
* **P2** strict ψ = ρ/ρ₀ − 1 (the only postprocess that uses the R axis).
* **P3** mean/peak rescale to fixed targets (Paper 1 shape, R-independent).
* **P4** P3 + clip[0,1] + vacuum cutoff (Paper 1 baseline, R-independent).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import config


def shift_and_rescale_real(n: np.ndarray, mean_target: float, peak_target: float) -> np.ndarray:
    """Port of Paper 1 ``_shift_and_rescale_real``: zero-mean → scale (peak−mean) → shift."""
    mu = n.mean()
    mx = n.max()
    if mx == mu:
        return np.full_like(n, mean_target)
    scale = (peak_target - mean_target) / (mx - mu)
    return (n - mu) * scale + mean_target


class PostProcessInterface:
    name: str = "P?"
    description: str = ""
    needs_reference: bool = False

    def apply(
        self,
        rho: np.ndarray,
        *,
        rho0: Optional[np.ndarray] = None,
        mean_target: float = config.FIELD_MEAN_TARGET,
        peak_target: float = config.FIELD_PEAK_TARGET,
    ) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}: {self.description}>"


class P1Raw(PostProcessInterface):
    name = "P1"
    description = "raw density (no transform)"
    needs_reference = False

    def apply(self, rho, *, rho0=None, mean_target=config.FIELD_MEAN_TARGET,
              peak_target=config.FIELD_PEAK_TARGET):
        return rho.copy()


class P2Strict(PostProcessInterface):
    name = "P2"
    description = "strict ψ = ρ/ρ₀ − 1"
    needs_reference = True

    def __init__(self, floor_frac: float = 1e-3):
        self.floor_frac = floor_frac

    def apply(self, rho, *, rho0=None, mean_target=config.FIELD_MEAN_TARGET,
              peak_target=config.FIELD_PEAK_TARGET):
        if rho0 is None:
            raise ValueError("P2 requires a reference density rho0 (set an R axis)")
        floor = self.floor_frac * float(np.max(rho0)) if np.max(rho0) > 0 else 1e-12
        rho0_safe = np.maximum(rho0, floor)
        return rho / rho0_safe - 1.0


class P3Rescale(PostProcessInterface):
    name = "P3"
    description = "mean/peak rescale to fixed targets"
    needs_reference = False

    def apply(self, rho, *, rho0=None, mean_target=config.FIELD_MEAN_TARGET,
              peak_target=config.FIELD_PEAK_TARGET):
        return shift_and_rescale_real(rho, mean_target, peak_target)


class P4RescaleClip(PostProcessInterface):
    name = "P4"
    description = "mean/peak rescale + clip[0,1] + vacuum cutoff (Paper 1)"
    needs_reference = False

    def apply(self, rho, *, rho0=None, mean_target=config.FIELD_MEAN_TARGET,
              peak_target=config.FIELD_PEAK_TARGET):
        n = shift_and_rescale_real(rho, mean_target, peak_target)
        n = np.clip(n, 0.0, 1.0)
        n[n < config.VACUUM_NOISE_CUTOFF] = 0.0
        return n


POSTPROCESSES = {
    "P1": P1Raw,
    "P2": P2Strict,
    "P3": P3Rescale,
    "P4": P4RescaleClip,
}


def get_postprocess(name: str, **kwargs) -> PostProcessInterface:
    if name not in POSTPROCESSES:
        raise KeyError(f"unknown postprocess {name!r}; choose from {list(POSTPROCESSES)}")
    return POSTPROCESSES[name](**kwargs)
